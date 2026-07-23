import { useEffect, useMemo, useState } from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Controller, useForm } from "react-hook-form"
import { Camera, CheckCircle2, Gauge, LoaderCircle, Save, ShieldCheck, TriangleAlert } from "lucide-react"
import { Link } from "react-router-dom"
import { toast } from "sonner"
import { z } from "zod"
import { HelpTip } from "@/components/help-tip"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api, errorMessage, query } from "@/lib/api"
import type { RunConfig, SensorStatus } from "@/lib/contracts"
import { loadSelectedSensorKeys } from "@/lib/sensor-selection"
import { useOperator } from "@/providers/operator-provider"

const schema = z.object({
  run_name: z.string().optional(),
  resolution: z.string().min(1),
  fps: z.number().int().min(1).max(60),
  velocity: z.number().positive().max(1),
})

type SetupValues = z.infer<typeof schema>
type RunSensor = RunConfig["capture"]["sensors"][number]
export type WorkflowIntent = "camera_calibration" | "object_dataset"

type CalibrationIssue = { code: string; message: string; sensor_key?: string }

type CalibrationProfileSummary = {
  profile_id: string
  sensor_type: string
  sensor_id: string
  mounting_mode: string
  status?: string
  resolution?: [number, number]
  calibrated_at?: string | null
  method?: string | null
  quality?: { num_observations?: number; num_inliers?: number; mean_reprojection_error_px?: number | null }
}

type CalibrationSource = {
  source_run_root: string
  source_run_name: string
  bundle_sha256: string | null
  valid: boolean
  compatible: boolean
  issues: CalibrationIssue[]
  calibration_profiles: {
    sha256: string
    valid_profile_count: number
    profiles: CalibrationProfileSummary[]
  }
  intrinsic_calibration_profiles: {
    sha256: string
    profile_count: number
    profiles?: Array<{ profile_id: string; sensor_id: string; resolution: [number, number]; orientation: string }>
  }
}

type CalibrationSelectionArtifact = {
  schema_version: string
  selected_at: string
  source: {
    run_root: string
    run_name: string
    bundle_sha256: string
  }
  snapshot: {
    calibration_profiles: { relative_path: string; sha256: string }
    intrinsic_calibration_profiles: { relative_path: string; sha256: string }
  }
  sensor_profiles?: Record<string, string>
}

type CalibrationLibrarySelected =
  | (CalibrationSelectionArtifact & { valid: true; issues: CalibrationIssue[] })
  | { valid: false; issues: CalibrationIssue[] }

type CalibrationLibraryResponse = {
  selected: CalibrationLibrarySelected | null
  replacement_blockers?: string[]
  calibrations?: CalibrationSource[]
  sources?: CalibrationSource[]
}

type CalibrationSelectionResponse = {
  calibration_profiles: string
  intrinsic_calibration_profiles: string
  sensor_profiles?: Record<string, string>
  sensor_profile_mapping: Array<{ sensor_key: string; profile_id: string }>
  selection: CalibrationSelectionArtifact
}

const sensorKey = (sensor: { sensor_type: string; device_id: string }) => `${sensor.sensor_type}:${sensor.device_id}`
const sequenceFor = (intent: WorkflowIntent) => intent === "camera_calibration"
  ? "real_full_capture_validation"
  : "calibrated_capture_to_bop_dataset_dry_run"

function sharedResolutions(status: SensorStatus | undefined, enabledSensors: RunSensor[]) {
  let shared: string[] | null = null
  for (const sensor of enabledSensors) {
    const family = status?.families.find((item) => item.sensor_type === sensor.sensor_type)
    const supported = Array.isArray(family?.supported_resolutions)
      ? family.supported_resolutions.filter((item): item is string => typeof item === "string")
      : []
    if (!supported.length) continue
    shared = shared === null ? supported : shared.filter((item) => supported.includes(item))
  }
  return shared?.length ? shared : ["720p"]
}

export function RunSetup({ intent = "camera_calibration" }: { intent?: WorkflowIntent }) {
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const sensors = useQuery({ queryKey: ["sensors", "status"], queryFn: () => api<SensorStatus>("/sensors/status"), staleTime: 10_000 })
  const existing = useQuery({ queryKey: ["run-config", selectedRun], queryFn: () => api<{ config: RunConfig }>(query("/run-config", { run_root: selectedRun })), retry: false })
  const calibrations = useQuery({
    queryKey: ["calibration-library", selectedRun],
    queryFn: () => api<CalibrationLibraryResponse>(query("/ui/calibrations", { run_root: selectedRun })),
    enabled: intent === "object_dataset",
    retry: false,
  })
  const calibrationSources = calibrations.data?.calibrations ?? calibrations.data?.sources ?? []
  const form = useForm<SetupValues>({ resolver: zodResolver(schema), defaultValues: { run_name: "", resolution: "720p", fps: 6, velocity: 0.2 } })
  const [enabledOverrides, setEnabledOverrides] = useState<Record<string, Record<string, boolean>>>({})
  const [sourceSelection, setSourceSelection] = useState<{ runRoot: string; sourceRunRoot: string } | null>(null)
  const [confirmReplacement, setConfirmReplacement] = useState(false)

  const detectedByKey = useMemo(() => new Map(
    (sensors.data?.families.flatMap((family) => family.devices) ?? []).map((device) => [sensorKey(device), device]),
  ), [sensors.data])
  const configuredSensors = useMemo<RunSensor[]>(() => {
    const configured = existing.data?.config.capture.sensors ?? []
    const rows = configured.map((sensor) => {
      const detected = detectedByKey.get(sensorKey(sensor))
      return {
        ...detected,
        ...sensor,
        display_name: detected?.effective_display_name ?? detected?.display_name ?? sensor.display_name,
        enabled: sensor.enabled ?? true,
      } as RunSensor
    })
    const configuredKeys = new Set(rows.map(sensorKey))
    for (const key of loadSelectedSensorKeys()) {
      if (configuredKeys.has(key)) continue
      const detected = detectedByKey.get(key)
      if (!detected) continue
      rows.push({
        ...detected,
        display_name: detected.effective_display_name ?? detected.display_name ?? detected.device_id,
        mounting_mode: detected.mounting_mode ?? "eye_in_hand",
        enabled: true,
        inverted: Boolean(detected.inverted),
      } as RunSensor)
    }
    return rows
  }, [detectedByKey, existing.data?.config.capture.sensors])

  const enabledBySensor = enabledOverrides[selectedRun] ?? {}
  const isEnabled = (sensor: RunSensor) => enabledBySensor[sensorKey(sensor)] ?? sensor.enabled ?? true
  const enabledSensors = configuredSensors.filter(isEnabled)
  const resolutionOptions = sharedResolutions(sensors.data, enabledSensors)
  const activeSourceRun = sourceSelection?.runRoot === selectedRun ? sourceSelection.sourceRunRoot : ""
  const activeSource = calibrationSources.find((source) => source.source_run_root === activeSourceRun) ?? null
  const existingConfig = intent === "object_dataset" ? existing.data?.config : undefined
  const existingCalibration = existingConfig?.calibration_profiles ?? null
  const existingIntrinsicCalibration = existingConfig?.intrinsic_calibration_profiles ?? null
  const configuredSelectionHash = existingConfig?.calibration_profile_selection?.bundle_sha256 ?? null
  const librarySelection = calibrations.data?.selected ?? null
  const currentSelectedBundleHash = librarySelection?.valid === true
    ? librarySelection.source.bundle_sha256
    : null
  const existingSelectionComplete = Boolean(existingCalibration && existingIntrinsicCalibration && configuredSelectionHash)
  const existingCalibrationReady = existingSelectionComplete
    && librarySelection?.valid === true
    && currentSelectedBundleHash === configuredSelectionHash
  const hasUnverifiedLegacyCalibration = Boolean(existingCalibration || existingIntrinsicCalibration) && !existingSelectionComplete
  const hasInvalidExistingSelection = existingSelectionComplete
    && !calibrations.isPending
    && (!librarySelection || librarySelection.valid === false || currentSelectedBundleHash !== configuredSelectionHash)
  const activeSourceSelectable = Boolean(activeSource?.valid && activeSource.bundle_sha256)
  const replacingSelection = Boolean(activeSource?.bundle_sha256
    && currentSelectedBundleHash
    && activeSource.bundle_sha256 !== currentSelectedBundleHash)
  const replacementBlockers = calibrations.data?.replacement_blockers ?? []
  const replacementBlocked = replacingSelection && replacementBlockers.length > 0
  const replacementConfirmed = !replacingSelection || confirmReplacement
  const calibrationReady = intent === "camera_calibration"
    || (activeSource
      ? activeSourceSelectable && replacementConfirmed && !replacementBlocked
      : existingCalibrationReady)

  useEffect(() => {
    const config = existing.data?.config
    if (!config) return
    form.reset({ run_name: config.run_name, resolution: config.capture.resolution, fps: config.capture.fps, velocity: config.capture.velocity_m_s })
  }, [existing.data, form])

  const save = useMutation({
    mutationFn: async (values: SetupValues) => {
      if (!sensors.data) throw new Error("Camera discovery must finish before saving this setup")
      if (configuredSensors.length === 0) throw new Error("Select at least one camera on the Devices page")
      if (enabledSensors.length === 0) throw new Error("Enable at least one camera for this recording")
      const unavailable = enabledSensors.filter((sensor) => {
        const detected = detectedByKey.get(sensorKey(sensor))
        return !detected || detected.connected === false || detected.capture_ready === false
      })
      if (unavailable.length) throw new Error(`Enabled cameras are not ready: ${unavailable.map(sensorKey).join(", ")}`)

      let calibrationProfiles = existing.data?.config.calibration_profiles ?? ""
      let intrinsicProfiles = intent === "object_dataset"
        ? existing.data?.config.intrinsic_calibration_profiles ?? String(existing.data?.config.pipeline.options?.camera_rectification && typeof existing.data.config.pipeline.options.camera_rectification === "object"
          ? (existing.data.config.pipeline.options.camera_rectification as Record<string, unknown>).intrinsic_profiles ?? ""
          : "")
        : ""
      let sensorProfiles: Record<string, string> = {}
      let expectedCalibrationBundleSha256 = existingCalibrationReady ? configuredSelectionHash : null

      if (intent === "object_dataset" && activeSource) {
        if (!activeSource.bundle_sha256) throw new Error("This calibration bundle has no valid identity hash")
        if (!activeSource.valid) throw new Error("This calibration bundle is not valid")
        if (replacingSelection && !confirmReplacement) throw new Error("Confirm that you want to replace the current calibration selection")
        const selected = await api<CalibrationSelectionResponse>("/ui/calibrations/select", {
          method: "POST",
          body: JSON.stringify({
            run_root: selectedRun,
            source_run_root: activeSource.source_run_root,
            expected_bundle_sha256: activeSource.bundle_sha256,
            expected_current_bundle_sha256: currentSelectedBundleHash,
            confirm_replace: replacingSelection && confirmReplacement,
            resolution: values.resolution,
            sensors: enabledSensors.map((sensor) => ({
              sensor_type: sensor.sensor_type,
              device_id: sensor.device_id,
              mounting_mode: sensor.mounting_mode,
              inverted: Boolean(sensor.inverted),
            })),
          }),
        })
        calibrationProfiles = selected.calibration_profiles
        intrinsicProfiles = selected.intrinsic_calibration_profiles
        sensorProfiles = selected.sensor_profiles ?? selected.selection.sensor_profiles ?? Object.fromEntries(selected.sensor_profile_mapping.map((item) => [item.sensor_key, item.profile_id]))
        expectedCalibrationBundleSha256 = selected.selection.source.bundle_sha256
      }
      if (intent === "object_dataset" && (!calibrationProfiles || !intrinsicProfiles || !expectedCalibrationBundleSha256)) {
        throw new Error("Choose a compatible saved calibration before saving this dataset setup")
      }

      const selectedSensors = configuredSensors.map((sensor) => ({
        ...sensor,
        enabled: isEnabled(sensor),
        ...(sensorProfiles[sensorKey(sensor)] ? { calibration_profile_id: sensorProfiles[sensorKey(sensor)] } : {}),
      }))
      const sequenceOptions = intent === "object_dataset" ? {
        camera_rectification: { intrinsic_profiles: intrinsicProfiles },
      } : {}
      return api("/run-config", {
        method: "POST",
        body: JSON.stringify({
          run_root: selectedRun,
          ...values,
          sensors: selectedSensors,
          from_detected_sensors: false,
          sequence: sequenceFor(intent),
          sequence_options: sequenceOptions,
          calibration_profiles: calibrationProfiles || null,
          ...(intent === "object_dataset" ? { expected_calibration_bundle_sha256: expectedCalibrationBundleSha256 } : {}),
          dataset_mode: intent === "object_dataset" ? "pose_template" : "objectless",
          plan_only: intent === "camera_calibration",
        }),
      })
    },
    onSuccess: () => {
      toast.success(intent === "camera_calibration" ? "Calibration recording setup saved" : "Object dataset setup saved")
      setSourceSelection(null)
      setConfirmReplacement(false)
      void queryClient.invalidateQueries({ queryKey: ["run-config", selectedRun] })
      void queryClient.invalidateQueries({ queryKey: ["calibration-library", selectedRun] })
      void queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] })
      void queryClient.invalidateQueries({ queryKey: ["workflow-status", selectedRun] })
      void queryClient.invalidateQueries({ queryKey: ["runs"] })
    },
    onError: (error) => toast.error("Setup was not saved", { description: errorMessage(error) }),
    onSettled: () => {
      if (intent === "object_dataset") void queryClient.invalidateQueries({ queryKey: ["calibration-library", selectedRun] })
    },
  })

  return <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_340px]" data-testid={`${intent}-run-setup`}>
    <Card>
      <CardHeader><CardTitle>{intent === "camera_calibration" ? "Calibration recording setup" : "Object dataset recording setup"}</CardTitle><CardDescription>{intent === "camera_calibration" ? "Choose the cameras and capture settings used to record the printed grid." : "Choose the cameras, capture settings, and a previously saved calibration that covers every enabled camera."}</CardDescription></CardHeader>
      <CardContent><form className="space-y-6" onSubmit={form.handleSubmit((values) => save.mutate(values))}>
        <div className="grid gap-4 sm:grid-cols-2"><Field id="run-name" label="Run name" error={form.formState.errors.run_name?.message}><Input id="run-name" {...form.register("run_name")} placeholder="Defaults to folder name" /></Field><Field id="resolution" label="Image resolution" hint="Only modes shared by every enabled camera are offered."><Controller control={form.control} name="resolution" render={({ field }) => <Select value={field.value} onValueChange={field.onChange}><SelectTrigger id="resolution"><SelectValue /></SelectTrigger><SelectContent>{resolutionOptions.map((resolution) => <SelectItem value={resolution} key={resolution}>{resolution === "720p" ? "1280 × 720" : resolution === "360p" ? "672 × 376" : resolution}</SelectItem>)}</SelectContent></Select>} /></Field></div>
        <div className="grid gap-4 sm:grid-cols-2"><Field id="fps" label={<span className="inline-flex items-center gap-1">Frames per second <HelpTip label="frames per second">The requested RGB-D frame rate for each enabled camera. Higher rates create more data and may exceed a camera or USB connection's supported mode.</HelpTip></span>} hint="How many RGB-D frames each camera should request per second." error={form.formState.errors.fps?.message}><Input id="fps" type="number" min={1} max={60} {...form.register("fps", { valueAsNumber: true })} /></Field><Field id="velocity" label={<span className="inline-flex items-center gap-1">Robot capture speed (m/s) <HelpTip label="robot capture speed">The Cartesian speed requested from the lab iiwa capture program. This setting does not authorize motion and never overrides controller-side safety limits.</HelpTip></span>} hint="Cartesian speed requested while recording. The robot application and safety configuration remain authoritative." error={form.formState.errors.velocity?.message}><Input id="velocity" type="number" min="0.01" max="1" step="0.01" {...form.register("velocity", { valueAsNumber: true })} /></Field></div>

        {intent === "object_dataset" && <section className="space-y-3" aria-labelledby="saved-calibration-heading">
          <div><h3 id="saved-calibration-heading" className="text-sm font-semibold">Saved camera calibration <span className="text-destructive">Required</span></h3><p className="mt-1 text-xs leading-relaxed text-muted-foreground">A calibration maps each camera into the workcell and supplies its lens model. Selecting one copies hash-verified evidence into this run; the source run is never referenced as mutable input.</p></div>
          {existingCalibrationReady && !activeSource && <div className="flex items-start gap-2 rounded-lg border border-success/35 bg-success/5 p-3 text-xs"><CheckCircle2 className="mt-0.5 size-4 shrink-0 text-success" /><div><div className="font-semibold">A verified calibration snapshot is selected</div><div className="mt-1 break-all font-mono text-[10px] text-muted-foreground">{existingCalibration}</div><p className="mt-1 text-muted-foreground">Both profile files and the selection record agree on bundle <span className="font-mono">{configuredSelectionHash?.slice(0, 16)}…</span>. Choose another source only if you intend to replace it.</p></div></div>}
          {hasUnverifiedLegacyCalibration && !activeSource && <div className="flex items-start gap-2 rounded-lg border border-warning/40 bg-warning/10 p-3 text-xs"><TriangleAlert className="mt-0.5 size-4 shrink-0 text-warning" /><div><div className="font-semibold">Unverified legacy calibration path</div>{existingCalibration && <div className="mt-1 break-all font-mono text-[10px] text-muted-foreground">Camera profiles: {existingCalibration}</div>}{existingIntrinsicCalibration && <div className="mt-1 break-all font-mono text-[10px] text-muted-foreground">Lens profiles: {existingIntrinsicCalibration}</div>}<p className="mt-1 text-muted-foreground">These paths are not backed by a complete, hash-verified selection and do not satisfy this workflow. Choose a saved calibration below.</p></div></div>}
          {hasInvalidExistingSelection && !activeSource && <div className="flex items-start gap-2 rounded-lg border border-destructive/35 bg-destructive/5 p-3 text-xs"><TriangleAlert className="mt-0.5 size-4 shrink-0 text-destructive" /><div><div className="font-semibold">Saved calibration selection needs attention</div><p className="mt-1 text-muted-foreground">{librarySelection?.valid === false && librarySelection.issues[0] ? librarySelection.issues[0].message : "The selected snapshot does not match the bundle recorded in run_config.json."} Choose and validate a saved calibration before recording.</p></div></div>}
          {calibrations.isPending ? <div className="flex items-center gap-2 rounded-lg border p-4 text-xs text-muted-foreground"><LoaderCircle className="size-4 animate-spin" />Loading promoted calibrations…</div> : calibrationSources.length === 0 ? <div className="rounded-lg border border-dashed p-4 text-xs text-muted-foreground">No reusable promoted calibrations were found. Complete and save the camera-calibration workflow first.</div> : <div className="grid gap-3 md:grid-cols-2" role="radiogroup" aria-label="Saved camera calibration">{calibrationSources.map((source) => {
            const selected = activeSourceRun === source.source_run_root
            const current = Boolean(source.bundle_sha256 && source.bundle_sha256 === currentSelectedBundleHash)
            const badge = !source.valid ? "Unavailable" : source.compatible ? "Matches saved setup" : "Validated when saved"
            const badgeClass = !source.valid ? "bg-destructive/10 text-destructive" : source.compatible ? "bg-success/10 text-success" : "bg-warning/10 text-warning"
            return <button type="button" role="radio" aria-checked={selected} disabled={!source.valid || !source.bundle_sha256} key={source.source_run_root} onClick={() => { setSourceSelection({ runRoot: selectedRun, sourceRunRoot: source.source_run_root }); setConfirmReplacement(false) }} className={`rounded-lg border p-4 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${selected ? "border-primary bg-primary/5 ring-1 ring-primary/30" : "hover:bg-muted/30"}`}>
              <div className="flex items-start justify-between gap-3"><div><div className="font-semibold">{source.source_run_name}</div><div className="mt-0.5 max-w-72 truncate font-mono text-[9px] text-muted-foreground" title={source.source_run_root}>{source.source_run_root}</div></div><span className={`rounded-full px-2 py-1 text-[10px] font-semibold ${badgeClass}`}>{badge}</span></div>
              <div className="mt-3 text-[11px] text-muted-foreground">{source.calibration_profiles.valid_profile_count} valid camera profile{source.calibration_profiles.valid_profile_count === 1 ? "" : "s"} · {source.intrinsic_calibration_profiles.profile_count} lens profile{source.intrinsic_calibration_profiles.profile_count === 1 ? "" : "s"}</div>
              {current && <div className="mt-2 text-[10px] font-semibold text-primary-strong">Current snapshot selection</div>}
              {!source.compatible && source.issues[0] && <div className={`mt-2 text-[10px] ${source.valid ? "text-muted-foreground" : "text-destructive"}`}>{source.issues[0].message}</div>}
              <div className="mt-2 truncate font-mono text-[9px] text-muted-foreground" title={source.bundle_sha256 ?? undefined}>{source.bundle_sha256 ? `Bundle SHA-256 ${source.bundle_sha256.slice(0, 16)}…` : "No valid bundle hash"}</div>
            </button>
          })}</div>}
          {calibrationSources.length > 0 && <p className="text-[11px] leading-relaxed text-muted-foreground">“Matches saved setup” describes the last saved camera configuration. The server validates the cameras, mounting modes, orientation, resolution, and lens profiles currently shown here when you save.</p>}
          {activeSource && <div className="rounded-lg border border-primary/25 bg-primary/5 p-3 text-xs"><div className="font-semibold">Ready for server validation</div><p className="mt-1 text-muted-foreground">Saving will verify this bundle against every enabled camera and the selected resolution before changing run_config.json.</p></div>}
          {replacementBlocked && <div className="flex items-start gap-2 rounded-lg border border-destructive/35 bg-destructive/5 p-3 text-xs"><TriangleAlert className="mt-0.5 size-4 shrink-0 text-destructive" /><div><div className="font-semibold">This run can no longer change calibration</div><p className="mt-1 text-muted-foreground">Derived dataset evidence already depends on the current snapshot. Start a new run to use another calibration.</p><ul className="mt-2 list-disc space-y-1 pl-4 text-[11px] text-muted-foreground">{replacementBlockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul></div></div>}
          {replacingSelection && !replacementBlocked && <Label className="flex cursor-pointer items-start gap-3 rounded-lg border border-warning/40 bg-warning/10 p-3 text-xs"><Checkbox aria-label="Confirm replacing the current calibration selection" checked={confirmReplacement} onCheckedChange={(checked) => setConfirmReplacement(checked === true)} /><span><span className="block font-semibold">Replace the current calibration selection</span><span className="mt-1 block font-normal text-muted-foreground">I understand this changes the calibration bundle used by this run from <span className="font-mono">{currentSelectedBundleHash?.slice(0, 12)}…</span> to <span className="font-mono">{activeSource?.bundle_sha256?.slice(0, 12)}…</span>. The server will reject the change if the current selection changed since this page loaded.</span></span></Label>}
        </section>}

        <details className="rounded-lg border bg-muted/20"><summary className="cursor-pointer px-3 py-2 text-xs font-semibold">Advanced pipeline details</summary><div className="border-t px-3 py-3 text-xs text-muted-foreground"><div>Preset: <code>{sequenceFor(intent)}</code></div><div className="mt-1">{intent === "camera_calibration" ? "The reusable preset is stored in prepare-only mode. Physical recording is always a separate, freshly authorized action." : "After recording, this preset synchronizes, rectifies, and prepares a BOP dataset using the selected calibration."}</div></div></details>
        <div className="flex flex-col-reverse gap-3 sm:flex-row sm:items-center sm:justify-between"><p className="text-xs text-muted-foreground">Saving setup never authorizes a camera or robot action.</p><Button type="submit" disabled={save.isPending || sensors.isPending || !calibrationReady}>{save.isPending ? <LoaderCircle className="animate-spin" /> : <Save />}{save.isPending ? (activeSource ? "Validating and saving…" : "Saving…") : (activeSource ? "Validate and save setup" : "Save setup")}</Button></div>
      </form></CardContent>
    </Card>

    <div className="space-y-4">
      <Card><CardHeader><CardTitle className="flex items-center gap-2 text-base"><Camera className="size-4" />Cameras for this recording</CardTitle><CardDescription>Disabled cameras keep their identity and calibration metadata but are not opened or expected.</CardDescription></CardHeader><CardContent><div className="metric-number">{enabledSensors.length}</div><div className="mt-1 text-xs text-muted-foreground">enabled · {configuredSensors.length - enabledSensors.length} disabled · {sensors.data?.total_connected ?? 0} connected</div>{configuredSensors.length === 0 && <div className="mt-3 rounded border border-warning/40 bg-warning/10 p-2 text-xs">Select at least one camera on <Link to="/devices" className="font-semibold underline">Devices</Link>.</div>}<div className="mt-4 space-y-2">{configuredSensors.map((sensor) => {
        const enabled = isEnabled(sensor)
        const detected = detectedByKey.get(sensorKey(sensor))
        const readiness = !detected ? "not detected" : detected.capture_ready === false ? "not ready" : "ready"
        return <div data-testid="run-camera-row" data-sensor-key={sensorKey(sensor)} data-camera-state={enabled ? "enabled" : "disabled"} key={sensorKey(sensor)} className={`rounded border px-3 py-3 text-xs transition-opacity ${enabled ? "border-border bg-muted" : "border-dashed border-border/70 bg-muted/30 opacity-60"}`}><div className="flex items-start justify-between gap-3"><div className="min-w-0"><div className="truncate font-medium">{sensor.display_name || sensor.device_id}</div><div className="mt-0.5 font-mono text-[10px] text-muted-foreground">{sensorKey(sensor)}</div></div><span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${enabled ? "bg-success/15 text-success" : "bg-muted-foreground/15 text-muted-foreground"}`}>{enabled ? "Enabled" : "Disabled"}</span></div><Label className="mt-3 flex cursor-pointer items-center gap-2"><Checkbox aria-label={`Enable ${sensor.display_name || sensor.device_id} for this run`} checked={enabled} onCheckedChange={(checked) => setEnabledOverrides((current) => ({ ...current, [selectedRun]: { ...current[selectedRun], [sensorKey(sensor)]: checked === true } }))} /><span>Use for this recording</span></Label><div className={`mt-2 flex items-center gap-1 text-[10px] ${readiness !== "ready" && enabled ? "text-destructive" : "text-muted-foreground"}`}><Gauge className="size-3" />{sensor.mounting_mode === "static" ? "Static camera" : "Robot-mounted camera"} · {readiness}</div></div>
      })}</div></CardContent></Card>
      <Card className="border-primary/25"><CardHeader><CardTitle className="flex items-center gap-2 text-base"><ShieldCheck className="size-4 text-primary-strong" />Safety boundary</CardTitle></CardHeader><CardContent className="text-xs leading-relaxed text-muted-foreground">Setup stores parameters only. Recording later requires a current readiness check plus fresh camera and robot confirmations. Those confirmations are never saved.</CardContent></Card>
    </div>
  </div>
}

function Field({ id, label, hint, error, children }: { id: string; label: React.ReactNode; hint?: string; error?: string; children: React.ReactNode }) {
  return <div className="space-y-2"><Label htmlFor={id}>{label}</Label>{children}{hint && <p className="text-[11px] leading-relaxed text-muted-foreground">{hint}</p>}{error && <p className="text-xs text-destructive">{error}</p>}</div>
}
