(function () {
  const ROBOT_CONTROL_STORAGE_KEY = "posetestbot.robotControlTarget.v1";
  const ACTIVE_PREVIEW_JOB_STATUSES = ["queued", "running"];
  const state = {
    sensors: [],
    aliases: {},
    previewPollTimer: null,
    previewEntriesBySensor: {},
    previewJobSensorKeys: {},
    pendingPreviewStartsBySensor: {},
    terminalPreviewFetches: {},
  };

  function $(selector) {
    return document.querySelector(selector);
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function runRootValue() {
    return $("#runRoot").value.trim();
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.output || response.statusText);
    }
    return payload;
  }

  function chip(text, status) {
    const node = el("span", "chip " + (status || "neutral"), text);
    return node;
  }

  function renderSidebar(sections) {
    const panel = $("#sidebarStatus");
    panel.innerHTML = "";
    (sections || []).forEach((section) => {
      const row = el("div", "sidebar-status-row");
      row.append(el("span", "", section.label));
      row.append(chip(section.status, section.status));
      panel.append(row);
    });
  }

  function renderSteps(steps) {
    const panel = $("#stepOverview");
    panel.innerHTML = "";
    if (!steps || steps.length === 0) {
      panel.append(el("div", "empty", "No sequence plan for this run yet."));
      return;
    }
    steps.forEach((step) => {
      const card = el("article", "step-card");
      const top = el("div", "step-top");
      top.append(el("strong", "", step.index + ". " + step.label));
      top.append(chip(step.status, step.status));
      card.append(top);
      card.append(el("p", "", step.stage_id));
      const artifacts = el("div", "chip-row");
      (step.artifacts || []).forEach((artifact) => {
        artifacts.append(chip(artifact.path, artifact.exists ? artifact.status || "exists" : "missing"));
      });
      card.append(artifacts);
      panel.append(card);
    });
  }

  function renderRecommendations(items) {
    const panel = $("#recommendationsPanel");
    panel.innerHTML = "";
    if (!items || items.length === 0) {
      panel.append(el("div", "empty", "No recommendations available."));
      return;
    }
    items.slice(0, 6).forEach((item) => {
      const row = el("div", "list-row");
      row.append(el("strong", "", item.label || item.id));
      row.append(el("p", "", item.reason || item.description || ""));
      panel.append(row);
    });
  }

  async function loadOverview() {
    const runRoot = runRootValue();
    const data = await api("/ui/overview?run_root=" + encodeURIComponent(runRoot));
    $("#overviewSummary").textContent = data.config
      ? data.config.pipeline.sequence_id + " · " + (data.artifact_count || 0) + " artifact(s)"
      : data.config_error || "No run config yet.";
    renderSidebar(data.sidebar);
    renderSteps(data.steps);
    renderRecommendations(data.recommendations);
  }

  function familyDevices(status) {
    const devices = [];
    (status.families || []).forEach((family) => {
      (family.devices || []).forEach((device) => devices.push(device));
    });
    return devices;
  }

  function sensorKeyForDevice(device) {
    return device.sensor_type + ":" + device.device_id;
  }

  function sensorDeviceForKey(sensorKey) {
    return state.sensors.find((device) => sensorKeyForDevice(device) === sensorKey) || {};
  }

  function sensorCardByKey(sensorKey) {
    return (
      Array.from(document.querySelectorAll(".sensor-card")).find(
        (card) => card.dataset.sensorKey === sensorKey
      ) || null
    );
  }

  function sensorCard(device) {
    const key = sensorKeyForDevice(device);
    const card = el("article", "sensor-card");
    card.setAttribute("data-testid", "sensor-card");
    card.dataset.sensorKey = key;
    card.dataset.sensorType = device.sensor_type;
    card.dataset.deviceId = device.device_id;

    const header = el("div", "sensor-card-header");
    const select = document.createElement("input");
    select.type = "checkbox";
    select.className = "sensor-select";
    select.setAttribute("data-testid", "sensor-select");
    select.checked = true;
    header.append(select);
    const title = el("div", "");
    title.append(el("strong", "", device.effective_display_name || device.display_name || key));
    title.append(el("span", "", key));
    header.append(title);
    card.append(header);

    const grid = el("div", "sensor-form-grid");
    const aliasLabel = el("label", "", "Alias");
    const alias = document.createElement("input");
    alias.className = "form-control form-control-sm alias-input";
    alias.value = device.alias || "";
    aliasLabel.append(alias);
    grid.append(aliasLabel);

    const mountingLabel = el("label", "", "Mounting");
    const mounting = document.createElement("select");
    mounting.className = "form-select form-select-sm mounting-input";
    ["eye_in_hand", "static"].forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      option.selected = (device.mounting_mode || "eye_in_hand") === value;
      mounting.append(option);
    });
    mountingLabel.append(mounting);
    grid.append(mountingLabel);

    const invertedLabel = el("label", "checkbox-label", "");
    const inverted = document.createElement("input");
    inverted.type = "checkbox";
    inverted.className = "inverted-input";
    inverted.checked = Boolean(device.inverted);
    inverted.disabled = device.sensor_type !== "realsense_d435";
    inverted.addEventListener("change", () => applyPreviewOrientationChange(key));
    invertedLabel.append(inverted);
    invertedLabel.append(document.createTextNode("Inverted RealSense"));
    grid.append(invertedLabel);
    card.append(grid);

    const actions = el("div", "sensor-card-actions");
    const previewLabel = el("label", "preview-toggle form-check form-switch", "");
    const previewToggle = document.createElement("input");
    previewToggle.type = "checkbox";
    previewToggle.className = "form-check-input preview-toggle-input";
    previewToggle.setAttribute("role", "switch");
    previewToggle.setAttribute("aria-label", "RGB preview for " + key);
    previewToggle.setAttribute("data-testid", "sensor-preview-toggle");
    previewToggle.addEventListener("change", () => toggleSensorPreview(key, previewToggle.checked));
    previewLabel.append(previewToggle);
    previewLabel.append(el("span", "preview-toggle-label", "RGB Preview"));
    actions.append(previewLabel);
    card.append(actions);

    const previewSlot = el("div", "sensor-preview-slot");
    previewSlot.hidden = true;
    previewSlot.setAttribute("data-testid", "sensor-preview-slot");
    card.append(previewSlot);

    if (device.metadata && device.metadata.video_accessible === false) {
      card.append(
        el(
          "p",
          "sensor-warning",
          device.metadata.video_permission_hint || "Camera video nodes are not accessible."
        )
      );
    }
    const meta = el("p", "sensor-meta", JSON.stringify(device.metadata || {}));
    card.append(meta);
    updateSensorPreviewControl(card);
    return card;
  }

  async function loadSensors() {
    const status = await api("/sensors/status");
    state.sensors = familyDevices(status);
    $("#sensorSummary").textContent =
      "Detected " + status.total_connected + " connected sensor(s).";
    const panel = $("#sensorCards");
    panel.innerHTML = "";
    if (state.sensors.length === 0) {
      panel.append(el("div", "empty", "No connected sensors detected."));
      return status;
    }
    state.sensors.forEach((device) => panel.append(sensorCard(device)));
    return status;
  }

  function collectAliasPayload() {
    const aliases = {};
    document.querySelectorAll(".sensor-card").forEach((card) => {
      const alias = card.querySelector(".alias-input").value.trim();
      const mounting = card.querySelector(".mounting-input").value;
      const inverted = card.querySelector(".inverted-input").checked;
      aliases[card.dataset.sensorKey] = {
        alias,
        mounting_mode: mounting,
        inverted,
      };
    });
    return aliases;
  }

  async function saveAliases(options) {
    const reload = !(options && options.reload === false);
    const payload = collectAliasPayload();
    await api("/sensors/aliases", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({aliases: payload}),
    });
    if (reload) await loadSensors();
  }

  function selectedSensorKeys() {
    return Array.from(document.querySelectorAll(".sensor-card"))
      .filter((card) => card.querySelector(".sensor-select").checked)
      .map((card) => card.dataset.sensorKey);
  }

  function previewSpecFromCard(card) {
    const alias = card.querySelector(".alias-input").value.trim();
    const device = sensorDeviceForKey(card.dataset.sensorKey);
    const displayName =
      alias ||
      device.effective_display_name ||
      device.display_name ||
      card.dataset.sensorKey;
    return {
      sensor_type: card.dataset.sensorType,
      device_id: card.dataset.deviceId,
      display_name: displayName,
      alias: alias || undefined,
      effective_display_name: displayName,
      mounting_mode: card.querySelector(".mounting-input").value,
      inverted: card.querySelector(".inverted-input").checked,
      metadata: device.metadata || {},
    };
  }

  function previewSpecsForKeys(sensorKeys) {
    const keys = new Set(sensorKeys);
    return Array.from(document.querySelectorAll(".sensor-card"))
      .filter((card) => keys.has(card.dataset.sensorKey))
      .map(previewSpecFromCard);
  }

  async function requestPreviews(sensorKeys) {
    const selectedKeys = Array.isArray(sensorKeys) ? sensorKeys : selectedSensorKeys();
    const sensors = previewSpecsForKeys(selectedKeys);
    if (sensors.length === 0) {
      $("#sensorSummary").textContent = "Select at least one sensor to preview.";
      return;
    }
    const pendingKeys = sensors.map((sensor) => sensor.sensor_type + ":" + sensor.device_id);
    markPreviewStartsPending(pendingKeys, {clearExisting: true});
    try {
      await saveAliases({reload: false});
      await stopPreviewJobsWithStaleOrientation(sensors);
      const data = await api("/sensors/previews", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({sensors, fps: 6, width: 640, height: 480}),
      });
      renderPreviewStatus(data);
      await loadPreviews();
      if (activePreviewEntryCount() > 0) startPreviewPolling();
    } catch (error) {
      clearPendingPreviewStarts(pendingKeys);
      throw error;
    }
  }

  function previewJobs(data) {
    if (Array.isArray(data.jobs)) return data.jobs;
    if (data.job) return [{job: data.job, preview_status: data.preview_status}];
    return [];
  }

  function previewSensorKey(entry) {
    const job = entry.job || {};
    const status = entry.preview_status || {};
    return status.sensor_key || job.parameters?.sensor_key || null;
  }

  function previewJob(entry) {
    return entry?.job || {};
  }

  function previewStatus(entry) {
    return entry?.preview_status || {};
  }

  function previewJobId(entry) {
    return previewJob(entry).id || null;
  }

  function isPreviewEntryActive(entry) {
    return ACTIVE_PREVIEW_JOB_STATUSES.includes(previewJob(entry).status);
  }

  function previewOrientation(entry) {
    const status = previewStatus(entry);
    if (status.inverted !== undefined && status.inverted !== null) return Boolean(status.inverted);
    const parameters = previewJob(entry).parameters || {};
    if (parameters.inverted !== undefined && parameters.inverted !== null) {
      return Boolean(parameters.inverted);
    }
    return null;
  }

  function rememberPreviewEntry(entry) {
    const sensorKey = previewSensorKey(entry);
    if (!sensorKey) return;
    const jobId = previewJobId(entry);
    if (jobId) {
      state.previewJobSensorKeys[jobId] = sensorKey;
      if (isPreviewEntryActive(entry)) delete state.terminalPreviewFetches[jobId];
    }
    state.previewEntriesBySensor[sensorKey] = entry;
    delete state.pendingPreviewStartsBySensor[sensorKey];
  }

  function rememberPreviewErrors(errors) {
    (errors || []).forEach((item) => {
      const sensorKey = item.sensor_key;
      if (!sensorKey) return;
      state.previewEntriesBySensor[sensorKey] = {
        job: {
          id: null,
          status: "failed",
          message: item.error || "Preview could not start.",
          parameters: {sensor_key: sensorKey},
        },
        preview_status: {
          sensor_key: sensorKey,
          status: "failed",
          error: item.error || "Preview could not start.",
          latest_image: null,
          frame_count: 0,
        },
      };
      delete state.pendingPreviewStartsBySensor[sensorKey];
    });
  }

  function rememberPreviewResponse(data) {
    previewJobs(data).forEach(rememberPreviewEntry);
    rememberPreviewErrors(data.errors);
  }

  function activePreviewEntryCount() {
    return Object.values(state.previewEntriesBySensor).filter(isPreviewEntryActive).length;
  }

  function currentPreviewEntry(sensorKey) {
    return state.previewEntriesBySensor[sensorKey] || null;
  }

  function currentPreviewJobId(sensorKey) {
    const entry = currentPreviewEntry(sensorKey);
    return entry && isPreviewEntryActive(entry) ? previewJobId(entry) : null;
  }

  function markPreviewStartsPending(sensorKeys, options) {
    const clearExisting = Boolean(options && options.clearExisting);
    const clearActive = Boolean(options && options.clearActive);
    sensorKeys.forEach((sensorKey) => {
      state.pendingPreviewStartsBySensor[sensorKey] = true;
      const existing = currentPreviewEntry(sensorKey);
      if (clearExisting && (clearActive || !existing || !isPreviewEntryActive(existing))) {
        clearSensorPreviewState(sensorKey, {render: false, keepPending: true});
      }
    });
    renderPreviewSlots(sensorKeys);
  }

  function clearPendingPreviewStarts(sensorKeys) {
    sensorKeys.forEach((sensorKey) => delete state.pendingPreviewStartsBySensor[sensorKey]);
    renderPreviewSlots(sensorKeys);
  }

  function clearSensorPreviewState(sensorKey, options) {
    const render = !(options && options.render === false);
    const keepPending = Boolean(options && options.keepPending);
    const entry = state.previewEntriesBySensor[sensorKey];
    const jobId = previewJobId(entry);
    if (jobId) {
      delete state.previewJobSensorKeys[jobId];
      delete state.terminalPreviewFetches[jobId];
    }
    delete state.previewEntriesBySensor[sensorKey];
    if (!keepPending) delete state.pendingPreviewStartsBySensor[sensorKey];
    if (render) renderPreviewSlots([sensorKey]);
  }

  function clearAllPreviewStates() {
    state.previewEntriesBySensor = {};
    state.previewJobSensorKeys = {};
    state.pendingPreviewStartsBySensor = {};
    state.terminalPreviewFetches = {};
    renderPreviewSlots();
  }

  function setSensorPreviewBusy(sensorKey, busy) {
    const card = sensorCardByKey(sensorKey);
    if (!card) return;
    const toggle = card.querySelector(".preview-toggle-input");
    if (!toggle) return;
    toggle.disabled = busy;
    toggle.dataset.busy = busy ? "true" : "false";
  }

  function updateSensorPreviewControl(card) {
    const toggle = card.querySelector(".preview-toggle-input");
    if (!toggle) return;
    const sensorKey = card.dataset.sensorKey;
    const entry = currentPreviewEntry(sensorKey);
    const active = Boolean(
      state.pendingPreviewStartsBySensor[sensorKey] || (entry && isPreviewEntryActive(entry))
    );
    const busy = toggle.dataset.busy === "true";
    toggle.checked = active;
    toggle.disabled = busy;
    toggle.setAttribute("aria-checked", active ? "true" : "false");
    card.classList.toggle("preview-active", active);
  }

  function updateSensorPreviewControls() {
    document.querySelectorAll(".sensor-card").forEach(updateSensorPreviewControl);
  }

  async function stopSensorPreview(sensorKey, options) {
    const clear = !(options && options.clear === false);
    const keepWaiting = Boolean(options && options.keepWaiting);
    let jobId = options?.jobId || currentPreviewJobId(sensorKey);
    if (!jobId) {
      await loadPreviews();
      jobId = currentPreviewJobId(sensorKey);
      if (!jobId) {
        if (clear) clearSensorPreviewState(sensorKey);
        else if (keepWaiting) markPreviewStartsPending([sensorKey], {clearExisting: true});
        return null;
      }
    }
    setSensorPreviewBusy(sensorKey, true);
    try {
      const data = await api("/sensors/previews/" + encodeURIComponent(jobId) + "/stop", {
        method: "POST",
      });
      if (clear) {
        clearSensorPreviewState(sensorKey);
      } else {
        clearSensorPreviewState(sensorKey, {render: false, keepPending: keepWaiting});
        if (keepWaiting) state.pendingPreviewStartsBySensor[sensorKey] = true;
        renderPreviewSlots([sensorKey]);
      }
      return data;
    } finally {
      setSensorPreviewBusy(sensorKey, false);
    }
  }

  async function toggleSensorPreview(sensorKey, active) {
    setSensorPreviewBusy(sensorKey, true);
    try {
      if (active) {
        await requestPreviews([sensorKey]);
      } else {
        await stopSensorPreview(sensorKey);
      }
    } catch (error) {
      alert(error.message);
      await loadPreviews().catch(() => updateSensorPreviewControls());
    } finally {
      setSensorPreviewBusy(sensorKey, false);
    }
  }

  async function stopPreviewJobsWithStaleOrientation(sensors) {
    for (const sensor of sensors) {
      const sensorKey = sensor.sensor_type + ":" + sensor.device_id;
      const entry = currentPreviewEntry(sensorKey);
      if (!entry || !isPreviewEntryActive(entry)) continue;
      const inverted = previewOrientation(entry);
      if (inverted === null) continue;
      if (inverted !== Boolean(sensor.inverted)) {
        await stopSensorPreview(sensorKey, {clear: false, keepWaiting: true});
      }
    }
  }

  async function applyPreviewOrientationChange(sensorKey) {
    const entry = currentPreviewEntry(sensorKey);
    if (!entry || !isPreviewEntryActive(entry)) return;
    const jobId = previewJobId(entry);
    markPreviewStartsPending([sensorKey], {clearExisting: true, clearActive: true});
    setSensorPreviewBusy(sensorKey, true);
    try {
      await stopSensorPreview(sensorKey, {clear: false, keepWaiting: true, jobId});
      await requestPreviews([sensorKey]);
    } catch (error) {
      alert(error.message);
      await loadPreviews().catch(() => updateSensorPreviewControls());
    } finally {
      setSensorPreviewBusy(sensorKey, false);
    }
  }

  function renderPreviewStatus(data) {
    rememberPreviewResponse(data);
    renderPreviewSlots();
  }

  function renderPreviewSlots(sensorKeys) {
    const cards = Array.from(document.querySelectorAll(".sensor-card")).filter(
      (card) => !sensorKeys || sensorKeys.includes(card.dataset.sensorKey)
    );
    cards.forEach(renderSensorPreviewSlot);
    updateSensorPreviewControls();
  }

  function renderSensorPreviewSlot(card) {
    const sensorKey = card.dataset.sensorKey;
    const slot = card.querySelector(".sensor-preview-slot");
    if (!slot) return;
    const entry = currentPreviewEntry(sensorKey);
    const pending = Boolean(state.pendingPreviewStartsBySensor[sensorKey]);
    slot.innerHTML = "";
    slot.dataset.previewState = "empty";

    if (!entry && !pending) {
      slot.hidden = true;
      return;
    }

    const job = previewJob(entry);
    const status = previewStatus(entry);
    const statusText = pending && !entry ? "queued" : status.status || job.status || "queued";
    const errorText = status.error || (job.status === "failed" ? job.message : null);
    slot.hidden = false;
    slot.dataset.previewState = errorText
      ? "error"
      : pending || isPreviewEntryActive(entry)
        ? "active"
        : statusText;
    if (job.id) slot.dataset.previewJobId = job.id;
    else delete slot.dataset.previewJobId;

    const header = el("div", "sensor-preview-header");
    header.append(el("strong", "", status.effective_display_name || status.display_name || sensorKey));
    header.append(chip(statusText, statusText));
    slot.append(header);

    if (status.latest_image && job.id) {
      const image = document.createElement("img");
      image.alt = "RGB preview for " + sensorKey;
      image.src = "/sensors/previews/" + encodeURIComponent(job.id) + "/latest.jpg?t=" + Date.now();
      image.setAttribute("data-testid", "sensor-preview-image");
      image.className = "sensor-preview-image";
      slot.append(image);
    } else if (!errorText) {
      slot.append(el("div", "empty sensor-preview-empty", "Waiting for RGB frames."));
    }

    if (errorText) {
      const error = el("p", "preview-error", errorText);
      error.setAttribute("data-testid", "sensor-preview-error");
      slot.append(error);
    }

    const meta = [];
    if (status.frame_count !== undefined && status.frame_count !== null) {
      meta.push("frames " + status.frame_count);
    }
    if (status.selected_node && status.selected_node.path) meta.push(status.selected_node.path);
    if (meta.length > 0) {
      const metaNode = el("p", "sensor-preview-meta", meta.join(" · "));
      metaNode.setAttribute("data-testid", "sensor-preview-meta");
      slot.append(metaNode);
    }
  }

  async function fetchMissingTerminalPreviews(activeEntries) {
    const activeJobIds = new Set(activeEntries.map(previewJobId).filter(Boolean));
    const fetches = [];
    Object.entries(state.previewEntriesBySensor).forEach(([sensorKey, entry]) => {
      const jobId = previewJobId(entry);
      if (!jobId || activeJobIds.has(jobId) || !isPreviewEntryActive(entry)) return;
      if (state.terminalPreviewFetches[jobId]) return;
      state.terminalPreviewFetches[jobId] = "pending";
      fetches.push(
        api("/sensors/previews/" + encodeURIComponent(jobId))
          .then((data) => {
            state.terminalPreviewFetches[jobId] = "done";
            rememberPreviewEntry(data);
          })
          .catch((error) => {
            state.terminalPreviewFetches[jobId] = "done";
            state.previewEntriesBySensor[sensorKey] = {
              job: {
                id: jobId,
                status: "failed",
                message: error.message,
                parameters: {sensor_key: sensorKey},
              },
              preview_status: {
                sensor_key: sensorKey,
                status: "failed",
                error: error.message,
                latest_image: null,
                frame_count: 0,
              },
            };
          })
      );
    });
    await Promise.all(fetches);
  }

  async function loadPreviews() {
    const data = await api("/sensors/previews");
    const activeEntries = previewJobs(data);
    rememberPreviewResponse(data);
    await fetchMissingTerminalPreviews(activeEntries);
    renderPreviewSlots();
    return data;
  }

  function stopPreviewPolling() {
    if (!state.previewPollTimer) return;
    clearInterval(state.previewPollTimer);
    state.previewPollTimer = null;
  }

  function startPreviewPolling() {
    if (state.previewPollTimer) clearInterval(state.previewPollTimer);
    state.previewPollTimer = setInterval(async () => {
      try {
        await loadPreviews();
        if (activePreviewEntryCount() === 0) stopPreviewPolling();
      } catch (_error) {
        stopPreviewPolling();
      }
    }, 1000);
  }

  async function stopPreviews(options) {
    const data = await api("/sensors/previews/stop", {method: "POST"});
    stopPreviewPolling();
    clearAllPreviewStates();
    return data;
  }

  function sensorPayloadFromCards() {
    return Array.from(document.querySelectorAll(".sensor-card")).map((card) => {
      const alias = card.querySelector(".alias-input").value.trim();
      return {
        sensor_type: card.dataset.sensorType,
        device_id: card.dataset.deviceId,
        display_name: alias || card.dataset.sensorKey,
        mounting_mode: card.querySelector(".mounting-input").value,
        inverted: card.querySelector(".inverted-input").checked,
      };
    });
  }

  async function createRunConfig() {
    let sequenceOptions = {};
    try {
      sequenceOptions = JSON.parse($("#sequenceOptions").value || "{}");
    } catch (error) {
      $("#runConfigOutput").textContent = "Invalid sequence options JSON: " + error.message;
      return;
    }
    const payload = {
      run_root: runRootValue(),
      run_name: $("#runName").value.trim() || undefined,
      robot_mode: $("#robotMode").value,
      sequence: $("#sequenceId").value,
      fps: Number($("#fps").value || 6),
      velocity: Number($("#velocity").value || 0.2),
      mounting_mode: $("#mountingMode").value,
      plan_only: $("#planOnly").checked,
      sequence_options: sequenceOptions,
      sensors: $("#useDetectedSensors").checked ? sensorPayloadFromCards() : undefined,
      from_detected_sensors: $("#useDetectedSensors").checked && sensorPayloadFromCards().length === 0,
    };
    const data = await api("/run-config", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    $("#runConfigOutput").textContent = JSON.stringify(data, null, 2);
    await loadOverview();
  }

  async function queueStage(stageId) {
    await stopPreviews({silent: true});
    const data = await api("/pipeline/run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({run_root: runRootValue(), stage: stageId, options: {}}),
    });
    await loadJobs();
    alert(data.output);
  }

  async function queueRunConfig() {
    await stopPreviews({silent: true});
    const data = await api("/pipeline/run-config", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({run_root: runRootValue(), allow_missing_preflight: true}),
    });
    await loadJobs();
    alert(data.output);
  }

  async function loadJobs() {
    const data = await api("/jobs");
    const panel = $("#jobsPanel");
    panel.innerHTML = "";
    (data.jobs || []).forEach((job) => {
      const row = el("div", "list-row");
      row.append(el("strong", "", job.name + " · " + job.status));
      row.append(el("p", "", job.id + " · " + (job.message || "")));
      panel.append(row);
    });
    if ((data.jobs || []).length === 0) panel.append(el("div", "empty", "No jobs."));
  }

  async function loadArtifacts() {
    const data = await api("/artifacts?run_root=" + encodeURIComponent(runRootValue()));
    const panel = $("#artifactsPanel");
    panel.innerHTML = "";
    (data.artifacts || []).forEach((artifact) => {
      const row = el("div", "list-row");
      row.append(el("strong", "", artifact.relative_path || artifact.path));
      row.append(el("p", "", artifact.kind + " · " + (artifact.exists ? "exists" : "missing")));
      panel.append(row);
    });
    if ((data.artifacts || []).length === 0) panel.append(el("div", "empty", "No artifacts."));
  }

  async function loadRobotStatus() {
    const data = await api("/robot/status");
    $("#robotStatusText").textContent = data.mode || data.status || "ready";
  }

  function robotControlElements() {
    return {
      panel: document.querySelector(".robot-control-panel"),
      ip: $("#robotControlIp"),
      port: $("#robotControlPort"),
      start: $("#startIiwaBtn"),
      stop: $("#stopIiwaBtn"),
      status: $("#robotControlStatus"),
    };
  }

  function robotControlDefaults(panel) {
    return {
      robot_ip: panel?.dataset.defaultRobotIp || "172.31.1.147",
      robot_port: panel?.dataset.defaultRobotPort || "30300",
    };
  }

  function loadRobotControlTarget() {
    const controls = robotControlElements();
    if (!controls.panel || !controls.ip || !controls.port) return;
    const defaults = robotControlDefaults(controls.panel);
    let stored = {};
    try {
      stored = JSON.parse(localStorage.getItem(ROBOT_CONTROL_STORAGE_KEY) || "{}");
    } catch (_error) {
      stored = {};
    }
    controls.ip.value = stored.robot_ip || defaults.robot_ip;
    controls.port.value = stored.robot_port || defaults.robot_port;
  }

  function currentRobotControlTarget() {
    const controls = robotControlElements();
    return {
      robot_ip: controls.ip.value.trim(),
      robot_port: controls.port.value.trim(),
    };
  }

  function storeRobotControlTarget() {
    const target = currentRobotControlTarget();
    try {
      localStorage.setItem(ROBOT_CONTROL_STORAGE_KEY, JSON.stringify(target));
    } catch (_error) {
      // Browser storage is a convenience; the current target still submits.
    }
    return target;
  }

  function setRobotControlBusy(busy) {
    const controls = robotControlElements();
    if (controls.start) controls.start.disabled = busy;
    if (controls.stop) controls.stop.disabled = busy;
  }

  function setRobotControlStatus(text, status) {
    const controls = robotControlElements();
    if (!controls.status) return;
    controls.status.textContent = text;
    controls.status.dataset.status = status || "neutral";
  }

  async function queueRobotControl(command) {
    const target = storeRobotControlTarget();
    const label = command === "start_iiwa" ? "Start IIWA" : "Stop IIWA";
    if (
      command === "start_iiwa" &&
      !window.confirm("Queue Start IIWA for " + target.robot_ip + ":" + target.robot_port + "?")
    ) {
      return;
    }

    setRobotControlBusy(true);
    setRobotControlStatus("Queueing " + label + "...", "running");
    try {
      const data = await api("/run-command", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          command,
          robot_ip: target.robot_ip,
          robot_port: target.robot_port,
        }),
      });
      setRobotControlStatus("Queued " + label + " as job " + data.job_id, "ok");
      await loadJobs();
    } catch (error) {
      setRobotControlStatus(error.message, "error");
    } finally {
      setRobotControlBusy(false);
    }
  }

  async function loadRuntimeStatus() {
    const data = await api("/runtime/status");
    $("#runtimeStatusText").textContent = data.all_available ? "ready" : "check";
  }

  async function writeHardwareStatus() {
    const data = await api("/hardware/status", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({run_root: runRootValue()}),
    });
    $("#hardwareStatusText").textContent = data.report.overall_status;
    await loadOverview();
  }

  function bindEvents() {
    $("#refreshOverviewBtn").addEventListener("click", loadOverview);
    $("#refreshJobsBtn").addEventListener("click", loadJobs);
    $("#refreshSensorsBtn").addEventListener("click", loadSensors);
    $("#previewSensorsBtn").addEventListener("click", requestPreviews);
    $("#stopPreviewsBtn").addEventListener("click", () => stopPreviews());
    $("#saveAliasesBtn").addEventListener("click", saveAliases);
    $("#createRunConfigBtn").addEventListener("click", createRunConfig);
    $("#runConfigQueueBtn").addEventListener("click", queueRunConfig);
    $("#loadArtifactsBtn").addEventListener("click", loadArtifacts);
    $("#loadJobsBtn").addEventListener("click", loadJobs);
    $("#robotStatusBtn").addEventListener("click", loadRobotStatus);
    $("#runtimeStatusBtn").addEventListener("click", loadRuntimeStatus);
    $("#hardwareStatusBtn").addEventListener("click", writeHardwareStatus);
    $("#writeHardwareStatusBtn").addEventListener("click", writeHardwareStatus);
    $("#startIiwaBtn").addEventListener("click", () => queueRobotControl("start_iiwa"));
    $("#stopIiwaBtn").addEventListener("click", () => queueRobotControl("stop_iiwa"));
    $("#robotControlIp").addEventListener("change", storeRobotControlTarget);
    $("#robotControlPort").addEventListener("change", storeRobotControlTarget);
    document.querySelectorAll("[data-stage]").forEach((button) => {
      button.addEventListener("click", () => queueStage(button.dataset.stage));
    });
  }

  window.addEventListener("DOMContentLoaded", async () => {
    loadRobotControlTarget();
    bindEvents();
    try {
      await loadOverview();
    } catch (error) {
      $("#overviewSummary").textContent = error.message;
    }
    try {
      await loadSensors();
    } catch (error) {
      $("#sensorSummary").textContent = error.message;
    }
    try {
      const data = await loadPreviews();
      if (previewJobs(data).length > 0) startPreviewPolling();
    } catch (_error) {
      updateSensorPreviewControls();
    }
    loadJobs().catch(() => {});
  });
})();
