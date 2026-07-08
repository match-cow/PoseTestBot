(function () {
  const state = {
    sensors: [],
    aliases: {},
    previewPollTimer: null,
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

  function sensorCard(device) {
    const key = device.sensor_type + ":" + device.device_id;
    const card = el("article", "sensor-card");
    card.dataset.sensorKey = key;
    card.dataset.sensorType = device.sensor_type;
    card.dataset.deviceId = device.device_id;

    const header = el("div", "sensor-card-header");
    const select = document.createElement("input");
    select.type = "checkbox";
    select.className = "sensor-select";
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
    invertedLabel.append(inverted);
    invertedLabel.append(document.createTextNode("Inverted RealSense"));
    grid.append(invertedLabel);
    card.append(grid);

    const actions = el("div", "sensor-card-actions");
    const previewButton = el("button", "btn btn-outline-primary btn-sm", "Start RGB");
    previewButton.type = "button";
    previewButton.addEventListener("click", () => requestPreviews([key]));
    actions.append(previewButton);
    card.append(actions);

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

  async function saveAliases() {
    const payload = collectAliasPayload();
    await api("/sensors/aliases", {
      method: "PUT",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({aliases: payload}),
    });
    await loadSensors();
  }

  function selectedSensorKeys() {
    return Array.from(document.querySelectorAll(".sensor-card"))
      .filter((card) => card.querySelector(".sensor-select").checked)
      .map((card) => card.dataset.sensorKey);
  }

  async function requestPreviews(sensorKeys) {
    const selected = sensorKeys || selectedSensorKeys();
    await saveAliases();
    if (selected.length === 0) {
      $("#previewPanel").innerHTML = "";
      $("#previewPanel").append(el("div", "empty", "Select at least one sensor to preview."));
      return;
    }
    const data = await api("/sensors/previews", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({selected, fps: 6, width: 640, height: 480}),
    });
    renderPreviewStatus(data);
    startPreviewPolling();
  }

  function previewJobs(data) {
    if (Array.isArray(data.jobs)) return data.jobs;
    if (data.job) return [{job: data.job, preview_status: data.preview_status}];
    return [];
  }

  function renderPreviewStatus(data) {
    const panel = $("#previewPanel");
    panel.innerHTML = "";
    const jobs = previewJobs(data);
    if (jobs.length === 0) {
      panel.append(el("div", "empty", data.output || "No active RGB previews."));
      return;
    }
    jobs.forEach((entry) => {
      const job = entry.job || {};
      const status = entry.preview_status || {};
      const sensorKey = status.sensor_key || job.parameters?.sensor_key || job.id;
      const card = el("article", "preview-card");
      const header = el("div", "preview-card-header");
      header.append(el("strong", "", status.effective_display_name || sensorKey));
      header.append(chip(status.status || job.status || "queued", status.status || job.status));
      card.append(header);
      if (status.latest_image) {
        const image = document.createElement("img");
        image.alt = "RGB preview for " + sensorKey;
        image.src = "/sensors/previews/" + job.id + "/latest.jpg?t=" + Date.now();
        card.append(image);
      } else {
        card.append(el("div", "empty preview-empty", "Waiting for RGB frames."));
      }
      if (status.error || job.message) {
        card.append(el("p", "preview-error", status.error || job.message));
      }
      const meta = [];
      if (status.frame_count !== undefined) meta.push("frames " + status.frame_count);
      if (status.selected_node && status.selected_node.path) meta.push(status.selected_node.path);
      if (meta.length > 0) {
        card.append(el("p", "sensor-meta", meta.join(" · ")));
      }
      panel.append(card);
    });
  }

  async function loadPreviews() {
    const data = await api("/sensors/previews");
    renderPreviewStatus(data);
  }

  function startPreviewPolling() {
    if (state.previewPollTimer) clearInterval(state.previewPollTimer);
    state.previewPollTimer = setInterval(() => {
      loadPreviews().catch(() => {});
    }, 1000);
  }

  async function stopPreviews(options) {
    const silent = Boolean(options && options.silent);
    const data = await api("/sensors/previews/stop", {method: "POST"});
    if (state.previewPollTimer) {
      clearInterval(state.previewPollTimer);
      state.previewPollTimer = null;
    }
    if (!silent) renderPreviewStatus({jobs: []});
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
    document.querySelectorAll("[data-stage]").forEach((button) => {
      button.addEventListener("click", () => queueStage(button.dataset.stage));
    });
  }

  window.addEventListener("DOMContentLoaded", async () => {
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
      await loadPreviews();
      startPreviewPolling();
    } catch (_error) {
      // Preview polling is best-effort on initial page load.
    }
    loadJobs().catch(() => {});
  });
})();
