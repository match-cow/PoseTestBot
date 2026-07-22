import { useEffect, useMemo, useState } from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Controller, useForm } from "react-hook-form"
import { Save, ShieldCheck } from "lucide-react"
import { toast } from "sonner"
import { z } from "zod"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api, errorMessage, query } from "@/lib/api"
import type { PipelineSequence, RunConfig, SensorStatus } from "@/lib/contracts"
import { loadSelectedSensorKeys } from "@/lib/sensor-selection"
import { useOperator } from "@/providers/operator-provider"

const schema = z.object({
  run_name: z.string().optional(),
  sequence: z.string().min(1),
  resolution: z.string().min(1),
  fps: z.number().int().min(1).max(60),
  velocity: z.number().positive().max(1),
  calibration_profiles: z.string().optional(),
  plan_only: z.boolean(),
})
type SetupValues = z.infer<typeof schema>
type RunSensor = RunConfig["capture"]["sensors"][number]

const sensorKey = (sensor: { sensor_type: string; device_id: string }) => `${sensor.sensor_type}:${sensor.device_id}`

export function RunSetup() {
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const sequences = useQuery({ queryKey: ["pipeline", "sequences"], queryFn: () => api<{ sequences: PipelineSequence[] }>("/pipeline/sequences") })
  const sensors = useQuery({ queryKey: ["sensors", "status"], queryFn: () => api<SensorStatus>("/sensors/status"), staleTime: 10_000 })
  const existing = useQuery({ queryKey: ["run-config", selectedRun], queryFn: () => api<{ config: RunConfig }>(query("/run-config", { run_root: selectedRun })), retry: false })
  const form = useForm<SetupValues>({ resolver: zodResolver(schema), defaultValues: { run_name: "", sequence: "real_full_capture_validation", resolution: "720p", fps: 6, velocity: 0.2, calibration_profiles: "", plan_only: true } })
  const [enabledOverrides, setEnabledOverrides] = useState<Record<string, Record<string, boolean>>>({})

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
  const enabledSensorCount = configuredSensors.filter(isEnabled).length

  useEffect(() => {
    const config = existing.data?.config
    if (!config) return
    form.reset({ run_name: config.run_name, sequence: config.pipeline.sequence_id, resolution: config.capture.resolution, fps: config.capture.fps, velocity: config.capture.velocity_m_s, calibration_profiles: config.calibration_profiles ?? "", plan_only: config.pipeline.plan_only })
  }, [existing.data, form])

  const save = useMutation({
    mutationFn: (values: SetupValues) => {
      if (!sensors.data) throw new Error("Sensor discovery must finish before writing the run configuration")
      if (configuredSensors.length === 0) throw new Error("Select at least one camera on the Devices page")
      const enabledSensors = configuredSensors.filter(isEnabled)
      if (enabledSensors.length === 0) throw new Error("Enable at least one camera for this run")
      const unavailable = enabledSensors.filter((sensor) => {
        const detected = detectedByKey.get(sensorKey(sensor))
        return !detected || detected.connected === false || detected.capture_ready === false
      })
      if (unavailable.length) throw new Error(`Enabled cameras are not capture-ready: ${unavailable.map(sensorKey).join(", ")}`)
      const selectedSensors = configuredSensors.map((sensor) => ({ ...sensor, enabled: isEnabled(sensor) }))
      return api("/run-config", { method: "POST", body: JSON.stringify({ run_root: selectedRun, ...values, sensors: selectedSensors, from_detected_sensors: false, sequence_options: {} }) })
    },
    onSuccess: () => { toast.success("Run configuration written"); queryClient.invalidateQueries({ queryKey: ["run-config", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["runs"] }) },
    onError: (error) => toast.error("Run configuration was not written", { description: errorMessage(error) }),
  })

  return <div className="grid grid-cols-[minmax(0,1fr)_340px] gap-5">
    <Card><CardHeader><CardTitle>Run setup</CardTitle><CardDescription>Create or update run_config.json from detected sensors. Safety gates are never stored here.</CardDescription></CardHeader><CardContent><form className="space-y-5" onSubmit={form.handleSubmit((values) => save.mutate(values))}>
      <div className="grid grid-cols-2 gap-4"><Field id="run-name" label="Run name" error={form.formState.errors.run_name?.message}><Input id="run-name" {...form.register("run_name")} placeholder="Defaults to folder name" /></Field><Field id="sequence" label="Sequence" error={form.formState.errors.sequence?.message}><Controller control={form.control} name="sequence" render={({ field }) => <Select value={field.value} onValueChange={field.onChange}><SelectTrigger id="sequence"><SelectValue /></SelectTrigger><SelectContent>{sequences.data?.sequences.map((sequence) => <SelectItem key={sequence.id} value={sequence.id}>{sequence.label}</SelectItem>)}</SelectContent></Select>} /></Field></div>
      <div className="grid grid-cols-3 gap-4"><Field id="resolution" label="Resolution"><Controller control={form.control} name="resolution" render={({ field }) => <Select value={field.value} onValueChange={field.onChange}><SelectTrigger id="resolution"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="720p">1280 × 720</SelectItem><SelectItem value="1080p">1920 × 1080</SelectItem></SelectContent></Select>} /></Field><Field id="fps" label="FPS" error={form.formState.errors.fps?.message}><Input id="fps" type="number" min={1} max={60} {...form.register("fps", { valueAsNumber: true })} /></Field><Field id="velocity" label="Velocity m/s" error={form.formState.errors.velocity?.message}><Input id="velocity" type="number" min="0.01" max="1" step="0.01" {...form.register("velocity", { valueAsNumber: true })} /></Field></div>
      <Field id="calibration-profiles" label="Calibration profiles"><Input id="calibration-profiles" {...form.register("calibration_profiles")} placeholder="Optional profile path" /></Field>
      <Controller control={form.control} name="plan_only" render={({ field }) => <Label className="flex items-start gap-3 rounded-lg border border-primary/25 bg-primary/5 p-4"><Checkbox checked={field.value} onCheckedChange={(checked) => field.onChange(checked === true)} /><span><span className="block font-semibold">Plan-only sequence</span><span className="mt-1 block text-xs font-normal text-muted-foreground">Default and recommended. A non-plan-only capture sequence cannot be queued from the run-config endpoint.</span></span></Label>} />
      <div className="flex justify-end"><Button type="submit" disabled={save.isPending || sensors.isPending}>{save.isPending ? null : <Save />}{save.isPending ? "Writing…" : "Write run config"}</Button></div>
    </form></CardContent></Card>
    <div className="space-y-4"><Card><CardHeader><CardTitle className="text-base">Configured camera input</CardTitle><CardDescription>Disable a camera for this run without deleting its identity or calibration-profile selection.</CardDescription></CardHeader><CardContent><div className="metric-number">{enabledSensorCount}</div><div className="mt-1 text-xs text-muted-foreground">enabled · {configuredSensors.length - enabledSensorCount} disabled · {sensors.data?.total_connected ?? 0} connected</div>{configuredSensors.length === 0 && <div className="mt-3 rounded border border-warning/40 bg-warning/10 p-2 text-xs">Select at least one camera on Devices before writing this run.</div>}<div className="mt-4 space-y-2">{configuredSensors.map((sensor) => {
      const enabled = isEnabled(sensor)
      const detected = detectedByKey.get(sensorKey(sensor))
      const readiness = !detected ? "not detected" : detected.capture_ready === false ? "not capture-ready" : "detected"
      return <div data-testid="run-camera-row" data-sensor-key={sensorKey(sensor)} data-camera-state={enabled ? "enabled" : "disabled"} key={sensorKey(sensor)} className={`rounded border px-3 py-3 text-xs transition-opacity ${enabled ? "border-border bg-muted" : "border-dashed border-border/70 bg-muted/30 opacity-60"}`}><div className="flex items-start justify-between gap-3"><div className="min-w-0"><div className="truncate font-medium">{sensor.display_name || sensor.device_id}</div><div className="mt-0.5 font-mono text-[10px] text-muted-foreground">{sensorKey(sensor)}</div></div><span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${enabled ? "bg-success/15 text-success" : "bg-muted-foreground/15 text-muted-foreground"}`}>{enabled ? "Enabled" : "Disabled"}</span></div><Label className="mt-3 flex cursor-pointer items-center gap-2"><Checkbox aria-label={`Enable ${sensor.display_name || sensor.device_id} for this run`} checked={enabled} onCheckedChange={(checked) => setEnabledOverrides((current) => ({ ...current, [selectedRun]: { ...current[selectedRun], [sensorKey(sensor)]: checked === true } }))} /><span>Enabled for capture and calibration</span></Label><div className={`mt-2 text-[10px] ${readiness === "not capture-ready" && enabled ? "text-destructive" : "text-muted-foreground"}`}>{sensor.mounting_mode === "static" ? "Static" : "Eye in hand"} · {readiness}</div></div>
    })}</div></CardContent></Card><Card className="border-primary/25"><CardHeader><CardTitle className="flex items-center gap-2 text-base"><ShieldCheck className="size-4 text-primary-strong" />Execution boundary</CardTitle></CardHeader><CardContent className="text-xs leading-relaxed text-muted-foreground">This form never writes <code>allow_cameras</code> or <code>allow_real_robot</code>. Physical execution is available only from the fresh Advanced Capture dialog.</CardContent></Card></div>
  </div>
}

function Field({ id, label, error, children }: { id: string; label: string; error?: string; children: React.ReactNode }) {
  return <div className="space-y-2"><Label htmlFor={id}>{label}</Label>{children}{error && <p className="text-xs text-destructive">{error}</p>}</div>
}
