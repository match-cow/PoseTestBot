import { useEffect } from "react"
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
import { useOperator } from "@/providers/operator-provider"

const schema = z.object({
  run_name: z.string().optional(),
  sequence: z.string().min(1),
  resolution: z.string().min(1),
  fps: z.number().int().min(1).max(60),
  velocity: z.number().positive().max(1),
  mounting_mode: z.enum(["eye_in_hand", "static"]),
  object_folder: z.string().min(1),
  calibration_profiles: z.string().optional(),
  plan_only: z.boolean(),
})
type SetupValues = z.infer<typeof schema>

export function RunSetup() {
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const sequences = useQuery({ queryKey: ["pipeline", "sequences"], queryFn: () => api<{ sequences: PipelineSequence[] }>("/pipeline/sequences") })
  const sensors = useQuery({ queryKey: ["sensors", "status"], queryFn: () => api<SensorStatus>("/sensors/status"), staleTime: 10_000 })
  const existing = useQuery({ queryKey: ["run-config", selectedRun], queryFn: () => api<{ config: RunConfig }>(query("/run-config", { run_root: selectedRun })), retry: false })
  const form = useForm<SetupValues>({ resolver: zodResolver(schema), defaultValues: { run_name: "", sequence: "real_full_capture_validation", resolution: "720p", fps: 6, velocity: 0.2, mounting_mode: "eye_in_hand", object_folder: "object_models", calibration_profiles: "", plan_only: true } })

  useEffect(() => {
    const config = existing.data?.config
    if (!config) return
    form.reset({ run_name: config.run_name, sequence: config.pipeline.sequence_id, resolution: config.capture.resolution, fps: config.capture.fps, velocity: config.capture.velocity_m_s, mounting_mode: (config.capture.sensors[0]?.mounting_mode as "eye_in_hand" | "static") ?? "eye_in_hand", object_folder: config.object_folder, calibration_profiles: config.calibration_profiles ?? "", plan_only: config.pipeline.plan_only })
  }, [existing.data, form])

  const save = useMutation({
    mutationFn: (values: SetupValues) => {
      const selectedKeys = new Set(JSON.parse(localStorage.getItem("posetestbot.selectedSensors") ?? "[]") as string[])
      const selectedSensors = sensors.data?.families.flatMap((family) => family.devices).filter((device) => selectedKeys.has(`${device.sensor_type}:${device.device_id}`)).map((device) => ({ ...device, display_name: device.effective_display_name ?? device.display_name, mounting_mode: values.mounting_mode })) ?? []
      return api("/run-config", { method: "POST", body: JSON.stringify({ run_root: selectedRun, ...values, sensors: selectedSensors.length ? selectedSensors : undefined, from_detected_sensors: selectedSensors.length === 0, sequence_options: {} }) })
    },
    onSuccess: () => { toast.success("Run configuration written"); queryClient.invalidateQueries({ queryKey: ["run-config", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["artifacts", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["runs"] }) },
    onError: (error) => toast.error("Run configuration was not written", { description: errorMessage(error) }),
  })

  return <div className="grid grid-cols-[minmax(0,1fr)_340px] gap-5">
    <Card><CardHeader><CardTitle>Run setup</CardTitle><CardDescription>Create or update run_config.json from detected sensors. Safety gates are never stored here.</CardDescription></CardHeader><CardContent><form className="space-y-5" onSubmit={form.handleSubmit((values) => save.mutate(values))}>
      <div className="grid grid-cols-2 gap-4"><Field label="Run name" error={form.formState.errors.run_name?.message}><Input {...form.register("run_name")} placeholder="Defaults to folder name" /></Field><Field label="Sequence" error={form.formState.errors.sequence?.message}><Controller control={form.control} name="sequence" render={({ field }) => <Select value={field.value} onValueChange={field.onChange}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{sequences.data?.sequences.map((sequence) => <SelectItem key={sequence.id} value={sequence.id}>{sequence.label}</SelectItem>)}</SelectContent></Select>} /></Field></div>
      <div className="grid grid-cols-4 gap-4"><Field label="Resolution"><Controller control={form.control} name="resolution" render={({ field }) => <Select value={field.value} onValueChange={field.onChange}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="720p">1280 × 720</SelectItem><SelectItem value="1080p">1920 × 1080</SelectItem></SelectContent></Select>} /></Field><Field label="FPS"><Input type="number" {...form.register("fps", { valueAsNumber: true })} /></Field><Field label="Velocity m/s"><Input type="number" step="0.01" {...form.register("velocity", { valueAsNumber: true })} /></Field><Field label="Mounting"><Controller control={form.control} name="mounting_mode" render={({ field }) => <Select value={field.value} onValueChange={field.onChange}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="eye_in_hand">Eye in hand</SelectItem><SelectItem value="static">Static</SelectItem></SelectContent></Select>} /></Field></div>
      <div className="grid grid-cols-2 gap-4"><Field label="Object folder"><Input {...form.register("object_folder")} /></Field><Field label="Calibration profiles"><Input {...form.register("calibration_profiles")} placeholder="Optional profile path" /></Field></div>
      <Controller control={form.control} name="plan_only" render={({ field }) => <Label className="flex items-start gap-3 rounded-lg border border-primary/25 bg-primary/5 p-4"><Checkbox checked={field.value} onCheckedChange={(checked) => field.onChange(checked === true)} /><span><span className="block font-semibold">Plan-only sequence</span><span className="mt-1 block text-xs font-normal text-muted-foreground">Default and recommended. A non-plan-only capture sequence cannot be queued from the run-config endpoint.</span></span></Label>} />
      <div className="flex justify-end"><Button type="submit" disabled={save.isPending}><Save />Write run config</Button></div>
    </form></CardContent></Card>
    <div className="space-y-4"><Card><CardHeader><CardTitle className="text-base">Detected input</CardTitle><CardDescription>Selection comes from the Devices page.</CardDescription></CardHeader><CardContent><div className="metric-number">{sensors.data?.total_connected ?? 0}</div><div className="mt-1 text-xs text-muted-foreground">connected sensors</div><div className="mt-4 space-y-2">{sensors.data?.families.flatMap((family) => family.devices).map((device) => <div key={`${device.sensor_type}:${device.device_id}`} className="truncate rounded bg-muted px-3 py-2 text-xs">{device.effective_display_name ?? device.display_name ?? device.device_id}</div>)}</div></CardContent></Card><Card className="border-primary/25"><CardHeader><CardTitle className="flex items-center gap-2 text-base"><ShieldCheck className="size-4 text-primary-strong" />Execution boundary</CardTitle></CardHeader><CardContent className="text-xs leading-relaxed text-muted-foreground">This form never writes <code>allow_cameras</code> or <code>allow_real_robot</code>. Physical execution is available only from the fresh Advanced Capture dialog.</CardContent></Card></div>
  </div>
}

function Field({ label, error, children }: { label: string; error?: string; children: React.ReactNode }) {
  return <div className="space-y-2"><Label>{label}</Label>{children}{error && <p className="text-xs text-destructive">{error}</p>}</div>
}
