import { useEffect, useRef, useState } from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Controller, useForm, useWatch } from "react-hook-form"
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
import type { ObjectRegistryPayload, PipelineSequence, RunConfig, SensorStatus } from "@/lib/contracts"
import { loadSelectedSensorKeys } from "@/lib/sensor-selection"
import { useOperator } from "@/providers/operator-provider"

const schema = z.object({
  run_name: z.string().optional(),
  sequence: z.string().min(1),
  resolution: z.string().min(1),
  fps: z.number().int().min(1).max(60),
  velocity: z.number().positive().max(1),
  object_folder: z.string().min(1),
  calibration_profiles: z.string().optional(),
  selected_objects: z.array(z.string()),
  plan_only: z.boolean(),
})
type SetupValues = z.infer<typeof schema>

export function RunSetup() {
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const sequences = useQuery({ queryKey: ["pipeline", "sequences"], queryFn: () => api<{ sequences: PipelineSequence[] }>("/pipeline/sequences") })
  const sensors = useQuery({ queryKey: ["sensors", "status"], queryFn: () => api<SensorStatus>("/sensors/status"), staleTime: 10_000 })
  const existing = useQuery({ queryKey: ["run-config", selectedRun], queryFn: () => api<{ config: RunConfig }>(query("/run-config", { run_root: selectedRun })), retry: false })
  const form = useForm<SetupValues>({ resolver: zodResolver(schema), defaultValues: { run_name: "", sequence: "real_full_capture_validation", resolution: "720p", fps: 6, velocity: 0.2, object_folder: "object_models", calibration_profiles: "", selected_objects: [], plan_only: true } })
  const objectFolder = useWatch({ control: form.control, name: "object_folder" })
  const registry = useQuery({ queryKey: ["object-registry", selectedRun, objectFolder], queryFn: () => api<ObjectRegistryPayload>(query("/ui/object-registry", { run_root: selectedRun, object_folder: objectFolder })), retry: false, enabled: Boolean(objectFolder) })
  const selectedObjects = useWatch({ control: form.control, name: "selected_objects" }) ?? []
  const previousRegistryKey = useRef<string | null>(null)
  const [missingSelections, setMissingSelections] = useState<string[]>([])

  useEffect(() => {
    const config = existing.data?.config
    if (!config) return
    form.reset({ run_name: config.run_name, sequence: config.pipeline.sequence_id, resolution: config.capture.resolution, fps: config.capture.fps, velocity: config.capture.velocity_m_s, object_folder: config.object_folder, calibration_profiles: config.calibration_profiles ?? "", selected_objects: config.selected_objects ?? [], plan_only: config.pipeline.plan_only })
  }, [existing.data, form])

  useEffect(() => {
    if (!registry.data) return
    const key = `${selectedRun}:${registry.data.object_folder}`
    if (previousRegistryKey.current === null || !previousRegistryKey.current.startsWith(`${selectedRun}:`)) {
      queueMicrotask(() => setMissingSelections(registry.data.missing_selected_objects))
      form.setValue("selected_objects", registry.data.selected_objects, { shouldDirty: true })
    } else if (previousRegistryKey.current !== key) {
      const available = new Set(registry.data.entries.filter((entry) => entry.valid).map((entry) => entry.name))
      const current = form.getValues("selected_objects")
      queueMicrotask(() => setMissingSelections(current.filter((name) => !available.has(name))))
      form.setValue("selected_objects", current.filter((name) => available.has(name)), { shouldDirty: true })
    }
    previousRegistryKey.current = key
  }, [form, registry.data, selectedRun])

  const save = useMutation({
    mutationFn: (values: SetupValues) => {
      if (!sensors.data) throw new Error("Sensor discovery must finish before writing the run configuration")
      const selectedKeys = loadSelectedSensorKeys()
      if (selectedKeys.size === 0) throw new Error("Select at least one camera on the Devices page")
      const detected = new Map(sensors.data.families.flatMap((family) => family.devices).map((device) => [`${device.sensor_type}:${device.device_id}`, device]))
      const missing = [...selectedKeys].filter((key) => !detected.has(key))
      if (missing.length) throw new Error(`Selected cameras are not connected: ${missing.join(", ")}`)
      const selectedSensors = [...selectedKeys].map((key) => detected.get(key)!).map((device) => ({ ...device, display_name: device.effective_display_name ?? device.display_name }))
      return api("/run-config", { method: "POST", body: JSON.stringify({ run_root: selectedRun, ...values, sensors: selectedSensors, from_detected_sensors: false, sequence_options: {} }) })
    },
    onSuccess: () => { toast.success("Run configuration written"); queryClient.invalidateQueries({ queryKey: ["run-config", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["runs"] }) },
    onError: (error) => toast.error("Run configuration was not written", { description: errorMessage(error) }),
  })

  return <div className="grid grid-cols-[minmax(0,1fr)_340px] gap-5">
    <Card><CardHeader><CardTitle>Run setup</CardTitle><CardDescription>Create or update run_config.json from detected sensors. Safety gates are never stored here.</CardDescription></CardHeader><CardContent><form className="space-y-5" onSubmit={form.handleSubmit((values) => save.mutate(values))}>
      <div className="grid grid-cols-2 gap-4"><Field id="run-name" label="Run name" error={form.formState.errors.run_name?.message}><Input id="run-name" {...form.register("run_name")} placeholder="Defaults to folder name" /></Field><Field id="sequence" label="Sequence" error={form.formState.errors.sequence?.message}><Controller control={form.control} name="sequence" render={({ field }) => <Select value={field.value} onValueChange={field.onChange}><SelectTrigger id="sequence"><SelectValue /></SelectTrigger><SelectContent>{sequences.data?.sequences.map((sequence) => <SelectItem key={sequence.id} value={sequence.id}>{sequence.label}</SelectItem>)}</SelectContent></Select>} /></Field></div>
      <div className="grid grid-cols-3 gap-4"><Field id="resolution" label="Resolution"><Controller control={form.control} name="resolution" render={({ field }) => <Select value={field.value} onValueChange={field.onChange}><SelectTrigger id="resolution"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="720p">1280 × 720</SelectItem><SelectItem value="1080p">1920 × 1080</SelectItem></SelectContent></Select>} /></Field><Field id="fps" label="FPS" error={form.formState.errors.fps?.message}><Input id="fps" type="number" min={1} max={60} {...form.register("fps", { valueAsNumber: true })} /></Field><Field id="velocity" label="Velocity m/s" error={form.formState.errors.velocity?.message}><Input id="velocity" type="number" min="0.01" max="1" step="0.01" {...form.register("velocity", { valueAsNumber: true })} /></Field></div>
      <div className="grid grid-cols-2 gap-4"><Field id="object-folder" label="Object folder" error={form.formState.errors.object_folder?.message}><Input id="object-folder" {...form.register("object_folder")} /></Field><Field id="calibration-profiles" label="Calibration profiles"><Input id="calibration-profiles" {...form.register("calibration_profiles")} placeholder="Optional profile path" /></Field></div>
      <div className="rounded-lg border p-4"><div className="mb-3 flex items-center justify-between"><div><div className="text-sm font-semibold">Dataset objects</div><div className="text-xs text-muted-foreground">Selection is snapshotted in run_config.json. No selection is a valid objectless run.</div></div><div className="text-xs text-muted-foreground">{selectedObjects.length} selected</div></div>
        {registry.isError ? <div className="text-xs text-destructive">Object registry unavailable: {registry.error instanceof Error ? registry.error.message : "invalid registry"}</div> : <div className="grid grid-cols-3 gap-2">{registry.data?.entries.map((entry) => { const checked = selectedObjects.includes(entry.name); return <Label key={entry.name} className="flex items-start gap-2 rounded border p-2 text-xs"><Checkbox disabled={!entry.valid} checked={checked} onCheckedChange={(value) => { const current = form.getValues("selected_objects"); form.setValue("selected_objects", value === true ? [...current, entry.name] : current.filter((name) => name !== entry.name), { shouldDirty: true }) }} /><span><span className="block font-medium">{entry.name} <span className="text-muted-foreground">#{entry.obj_id}</span></span>{!entry.valid && <span className="text-destructive">{entry.errors.join("; ")}</span>}</span></Label> })}</div>}
        {missingSelections.length ? <div className="mt-3 text-xs text-amber-600">Missing prior selections require resolution: {missingSelections.join(", ")}</div> : null}
      </div>
      <Controller control={form.control} name="plan_only" render={({ field }) => <Label className="flex items-start gap-3 rounded-lg border border-primary/25 bg-primary/5 p-4"><Checkbox checked={field.value} onCheckedChange={(checked) => field.onChange(checked === true)} /><span><span className="block font-semibold">Plan-only sequence</span><span className="mt-1 block text-xs font-normal text-muted-foreground">Default and recommended. A non-plan-only capture sequence cannot be queued from the run-config endpoint.</span></span></Label>} />
      <div className="flex justify-end"><Button type="submit" disabled={save.isPending || sensors.isPending}>{save.isPending ? null : <Save />}{save.isPending ? "Writing…" : "Write run config"}</Button></div>
    </form></CardContent></Card>
    <div className="space-y-4"><Card><CardHeader><CardTitle className="text-base">Selected camera input</CardTitle><CardDescription>Selection and per-camera mounting come from the saved Devices setup.</CardDescription></CardHeader><CardContent><div className="metric-number">{[...loadSelectedSensorKeys()].length}</div><div className="mt-1 text-xs text-muted-foreground">selected of {sensors.data?.total_connected ?? 0} connected</div>{loadSelectedSensorKeys().size === 0 && <div className="mt-3 rounded border border-warning/40 bg-warning/10 p-2 text-xs">Select at least one camera on Devices before writing this run.</div>}<div className="mt-4 space-y-2">{sensors.data?.families.flatMap((family) => family.devices).filter((device) => loadSelectedSensorKeys().has(`${device.sensor_type}:${device.device_id}`)).map((device) => <div key={`${device.sensor_type}:${device.device_id}`} className="rounded bg-muted px-3 py-2 text-xs"><div className="truncate font-medium">{device.effective_display_name ?? device.display_name ?? device.device_id}</div><div className="mt-0.5 text-muted-foreground">{device.mounting_mode === "static" ? "Static" : "Eye in hand"}</div></div>)}</div></CardContent></Card><Card className="border-primary/25"><CardHeader><CardTitle className="flex items-center gap-2 text-base"><ShieldCheck className="size-4 text-primary-strong" />Execution boundary</CardTitle></CardHeader><CardContent className="text-xs leading-relaxed text-muted-foreground">This form never writes <code>allow_cameras</code> or <code>allow_real_robot</code>. Physical execution is available only from the fresh Advanced Capture dialog.</CardContent></Card></div>
  </div>
}

function Field({ id, label, error, children }: { id: string; label: string; error?: string; children: React.ReactNode }) {
  return <div className="space-y-2"><Label htmlFor={id}>{label}</Label>{children}{error && <p className="text-xs text-destructive">{error}</p>}</div>
}
