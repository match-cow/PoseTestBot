import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { AlertTriangle, Bot, Camera, Eye, EyeOff, Info, Power, RefreshCw, Save, Square, Webcam } from "lucide-react"
import { toast } from "sonner"
import { PageHeader } from "@/components/page-header"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { api, errorMessage } from "@/lib/api"
import type { PreviewJob, SensorDevice, SensorStatus } from "@/lib/contracts"
import { loadSelectedSensorKeys, saveSelectedSensorKeys } from "@/lib/sensor-selection"
import { useOperator } from "@/providers/operator-provider"

const PREVIEW_ON = new Set(["queued", "running"])
const PREVIEW_BUSY = new Set(["queued", "running", "canceling"])
const sensorKey = (device: SensorDevice) => `${device.sensor_type}:${device.device_id}`
const isCaptureReady = (device: SensorDevice) => device.connected !== false && device.capture_ready !== false

function captureReadinessMessage(device: SensorDevice): string | null {
  if (device.connected === false) return "The camera is disconnected."
  if (device.capture_ready !== false) return null
  const reason = device.capture_readiness_reason?.trim()
  if (reason === "not_enumerated_by_sdk") return "Visible on USB, but unavailable to the camera SDK."
  if (reason === "usb_connection_below_superspeed") return "The USB connection is below SuperSpeed (USB 3). Check the cable and port, then refresh discovery."
  if (reason === "sdk_unavailable") return "The camera SDK is unavailable."
  return reason ? reason.replaceAll("_", " ") : "Sensor status reports that this camera cannot be opened for capture."
}

interface AliasRecord { alias: string; mounting_mode?: string; inverted?: boolean }
interface SnapshotState {
  job: { status: string }
  manifest: { sensors?: Array<{ sensor_key: string; status: string; rgb_thumbnail?: string | null; error?: string | null }> } | null
}

function Preview({ preview }: { preview?: PreviewJob }) {
  if (!preview) return <div className="grid aspect-video place-items-center rounded-lg bg-muted text-xs text-muted-foreground">Preview is off</div>
  const status = preview.preview_status
  const hasLiveFrame = PREVIEW_ON.has(preview.job.status) && status?.status === "running" && Boolean(status.latest_image)
  const source = status?.selected_node?.path ?? status?.selected_node?.device_id ?? ""
  return (
    <div data-testid="sensor-preview-slot" className="relative isolate aspect-video w-full min-w-0 overflow-hidden rounded-lg bg-muted" style={{ contain: "layout paint" }}>
      {hasLiveFrame ? <img data-testid="sensor-preview-image" src={`/sensors/previews/${preview.job.id}/latest.jpg?t=${status?.frame_count}`} className="absolute inset-0 size-full object-contain" alt="Live sensor preview" /> : <div className="sensor-preview-empty absolute inset-0 grid place-items-center px-4 text-center text-xs text-muted-foreground">{status?.error ? <span data-testid="sensor-preview-error" className="max-w-full break-words text-destructive">{status.error}</span> : preview.job.status === "canceling" ? "Stopping preview…" : "Waiting for first frame…"}</div>}
      <div data-testid="sensor-preview-meta" className="absolute inset-x-2 bottom-2 z-10 flex min-w-0 items-center justify-between gap-2 rounded bg-black/65 px-2 py-1 text-[10px] text-white"><span className="shrink-0">{status?.status ?? preview.job.status}</span><span className="min-w-0 truncate">{String(source)}</span></div>
    </div>
  )
}

export function DevicesPage() {
  const queryClient = useQueryClient()
  const { robotTarget, setRobotTarget } = useOperator()
  const [aliasDraft, setAliasDraft] = useState<Record<string, AliasRecord>>({})
  const [selected, setSelected] = useState<Set<string>>(loadSelectedSensorKeys)
  const [detail, setDetail] = useState<SensorDevice | null>(null)
  const [snapshotJobs, setSnapshotJobs] = useState<Record<string, string>>({})
  const [startDialog, setStartDialog] = useState(false)
  const [robotDialogCommand, setRobotDialogCommand] = useState<"start_iiwa" | "stop_iiwa">("start_iiwa")
  const [confirmedRobotTarget, setConfirmedRobotTarget] = useState(robotTarget)
  const [targetConfirmed, setTargetConfirmed] = useState(false)
  const [realRobotAuthorized, setRealRobotAuthorized] = useState(false)
  const [camerasReady, setCamerasReady] = useState(false)
  const [targetDraft, setTargetDraft] = useState(robotTarget)

  const status = useQuery({ queryKey: ["sensors", "status"], queryFn: () => api<SensorStatus>("/sensors/status"), refetchInterval: 10_000 })
  const aliases = useQuery({ queryKey: ["sensors", "aliases"], queryFn: () => api<{ aliases: Record<string, AliasRecord> }>("/sensors/aliases") })
  const previews = useQuery({ queryKey: ["sensors", "previews"], queryFn: () => api<{ jobs: PreviewJob[] }>("/sensors/previews?include_terminal=true"), refetchInterval: 1_000 })
  const devices = useMemo(() => status.data?.families.flatMap((family) => family.devices) ?? [], [status.data])
  const captureReadyCount = useMemo(() => devices.filter(isCaptureReady).length, [devices])
  const previewByKey = useMemo(() => {
    const byKey = new Map<string, PreviewJob>()
    for (const item of previews.data?.jobs ?? []) {
      const key = String(item.job.parameters.sensor_key)
      const current = byKey.get(key)
      if (!current || (PREVIEW_BUSY.has(item.job.status) && !PREVIEW_BUSY.has(current.job.status))) {
        byKey.set(key, item)
      }
    }
    return byKey
  }, [previews.data])
  const snapshotStates = useQuery<Record<string, SnapshotState>>({
    queryKey: ["sensors", "snapshots", snapshotJobs],
    enabled: Object.keys(snapshotJobs).length > 0,
    queryFn: async () => Object.fromEntries(await Promise.all(Object.entries(snapshotJobs).map(async ([key, jobId]) => [key, await api<SnapshotState>(`/sensors/snapshots/${jobId}`)]))) as Record<string, SnapshotState>,
    refetchInterval: (queryState) => Object.values(queryState.state.data ?? {}).some((item) => PREVIEW_BUSY.has(item.job.status)) ? 1_000 : false,
  })

  const startPreview = useMutation({
    mutationFn: (device: SensorDevice) => api("/sensors/previews", { method: "POST", body: JSON.stringify({ sensors: [device] }) }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["sensors", "previews"] }),
    onError: (error) => toast.error("Preview could not start", { description: errorMessage(error) }),
  })
  const stopPreview = useMutation({
    mutationFn: (jobId: string) => api(`/sensors/previews/${jobId}/stop`, { method: "POST", body: "{}" }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["sensors", "previews"] }),
    onError: (error) => toast.error("Preview could not stop", { description: errorMessage(error) }),
  })
  const stopAllPreviews = useMutation({
    mutationFn: () => api<{ jobs: PreviewJob[] }>("/sensors/previews/stop", { method: "POST", body: "{}" }),
    onSuccess: (data) => { toast.success(data.jobs.length ? `Stopping ${data.jobs.length} preview${data.jobs.length === 1 ? "" : "s"}` : "All previews are already off"); queryClient.invalidateQueries({ queryKey: ["sensors", "previews"] }) },
    onError: (error) => toast.error("Previews could not be stopped", { description: errorMessage(error) }),
  })
  const snapshot = useMutation({
    mutationFn: (device: SensorDevice) => api<{ job_id: string }>("/sensors/snapshots", { method: "POST", body: JSON.stringify({ sensors: [device], max_frames: 1 }) }),
    onSuccess: (data, device) => { setSnapshotJobs((current) => ({ ...current, [sensorKey(device)]: data.job_id })); toast.success("Snapshot queued", { description: `Job ${data.job_id}` }); queryClient.invalidateQueries({ queryKey: ["jobs"] }) },
    onError: (error) => toast.error("Snapshot could not be queued", { description: errorMessage(error) }),
  })
  const saveAliases = useMutation({
    mutationFn: (records: Record<string, AliasRecord>) => api("/sensors/aliases", { method: "PUT", body: JSON.stringify({ aliases: records }) }),
    onSuccess: () => { toast.success("Sensor labels saved"); queryClient.invalidateQueries({ queryKey: ["sensors"] }) },
    onError: (error) => toast.error("Sensor labels could not be saved", { description: errorMessage(error) }),
  })
  const robotCommand = useMutation({
    mutationFn: ({ command, target }: { command: "start_iiwa" | "stop_iiwa"; target: { ip: string; port: number } }) => api<{ job_id: string }>("/run-command", { method: "POST", body: JSON.stringify({ command, robot_ip: target.ip, robot_port: target.port, ...(command === "start_iiwa" ? { allow_real_robot: true, allow_cameras: true } : {}) }) }),
    onSuccess: (data, variables) => { toast.success(variables.command === "start_iiwa" ? "IIWA start queued" : "IIWA stop queued", { description: `Job ${data.job_id}` }); setStartDialog(false); setTargetConfirmed(false); setRealRobotAuthorized(false); setCamerasReady(false); queryClient.invalidateQueries({ queryKey: ["jobs"] }) },
    onError: (error) => toast.error("Robot command was not queued", { description: errorMessage(error) }),
  })

  const updateAlias = (device: SensorDevice, patch: Partial<AliasRecord>) => {
    const key = sensorKey(device)
    const initial = aliasDraft[key] ?? aliases.data?.aliases[key] ?? { alias: device.effective_display_name ?? device.display_name ?? "", mounting_mode: device.mounting_mode ?? "eye_in_hand", inverted: Boolean(device.inverted) }
    setAliasDraft((current) => ({ ...current, [key]: { ...initial, ...patch } }))
  }
  const updateOrientation = async (device: SensorDevice, inverted: boolean, preview?: PreviewJob) => {
    updateAlias(device, { inverted })
    if (!preview || !PREVIEW_ON.has(preview.job.status)) return
    try {
      await api(`/sensors/previews/${preview.job.id}/stop`, { method: "POST", body: "{}" })
      await api("/sensors/previews", { method: "POST", body: JSON.stringify({ sensors: [{ ...device, inverted }] }) })
      await queryClient.invalidateQueries({ queryKey: ["sensors", "previews"] })
    } catch (error) { toast.error("Preview could not restart", { description: errorMessage(error) }) }
  }
  const saveAllAliases = () => {
    const records: Record<string, AliasRecord> = {}
    for (const device of devices) {
      const key = sensorKey(device)
      records[key] = aliasDraft[key] ?? aliases.data?.aliases[key] ?? { alias: device.effective_display_name ?? device.display_name ?? "", mounting_mode: device.mounting_mode ?? "eye_in_hand", inverted: Boolean(device.inverted) }
    }
    saveAliases.mutate(records)
  }
  const toggleSelected = (key: string, checked: boolean, captureReady: boolean) => {
    if (checked && !captureReady) return
    setSelected((current) => {
      const next = new Set(current)
      if (checked) next.add(key)
      else next.delete(key)
      saveSelectedSensorKeys(next)
      return next
    })
  }
  const validatedTarget = () => {
    const target = { ip: targetDraft.ip.trim(), port: targetDraft.port }
    if (!target.ip || !Number.isInteger(target.port) || target.port < 1 || target.port > 65535) { toast.error("Enter a valid robot IP and port"); return null }
    return target
  }
  const applyTarget = () => {
    const target = validatedTarget()
    if (!target) return
    setRobotTarget(target)
    setTargetDraft(target)
    toast.success("Robot target saved locally")
  }
  const openRobotDialog = (command: "start_iiwa" | "stop_iiwa") => {
    const target = validatedTarget()
    if (!target) return
    setRobotTarget(target)
    setTargetDraft(target)
    setConfirmedRobotTarget(target)
    setRobotDialogCommand(command)
    setTargetConfirmed(false)
    setRealRobotAuthorized(false)
    setCamerasReady(false)
    setStartDialog(true)
  }

  const previewTransitionPending = startPreview.isPending || stopPreview.isPending
  const anyPreviewBusy = [...previewByKey.values()].some((item) => PREVIEW_BUSY.has(item.job.status))
  const refreshDiscovery = async () => {
    const result = await status.refetch()
    if (result.error) toast.error("Sensor discovery failed", { description: errorMessage(result.error) })
  }

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Lab hardware" title="Devices" description="Readable camera state and deliberately separated robot controls." actions={<><Button variant="outline" onClick={() => void refreshDiscovery()} disabled={status.isFetching}><RefreshCw className={status.isFetching ? "animate-spin" : ""} />Refresh discovery</Button><Button onClick={saveAllAliases} disabled={saveAliases.isPending || status.isPending || aliases.isPending || devices.length === 0}><Save />{saveAliases.isPending ? "Saving…" : "Save sensor setup"}</Button></>} />

      <Card className="border-primary/25">
        <CardHeader className="flex-row items-start justify-between gap-6"><div><CardTitle className="flex items-center gap-2"><Bot className="size-5 text-primary-strong" />KUKA LBR iiwa</CardTitle><CardDescription>Robot commands are independent from camera previews and require explicit target confirmation.</CardDescription></div><StatusBadge status="ready">real lab profile</StatusBadge></CardHeader>
        <CardContent className="grid grid-cols-[1fr_1fr_auto] items-end gap-4">
          <div className="space-y-2"><Label htmlFor="robot-ip">Robot IP</Label><Input id="robot-ip" value={targetDraft.ip} onChange={(event) => setTargetDraft((value) => ({ ...value, ip: event.target.value }))} /></div>
          <div className="space-y-2"><Label htmlFor="robot-port">Command port</Label><Input id="robot-port" type="number" min={1} max={65535} value={targetDraft.port} onChange={(event) => setTargetDraft((value) => ({ ...value, port: Number(event.target.value) }))} /></div>
          <div className="flex gap-2"><Button variant="outline" onClick={applyTarget}>Save target</Button><Button onClick={() => openRobotDialog("start_iiwa")} disabled={robotCommand.isPending}><Power />Start IIWA</Button><Button variant="destructive" onClick={() => openRobotDialog("stop_iiwa")} disabled={robotCommand.isPending}><Square />Stop IIWA</Button></div>
        </CardContent>
      </Card>

      <div className="flex items-center justify-between"><div><h2 className="font-display text-xl font-semibold">RGB-D sensors</h2><p className="text-sm text-muted-foreground">{captureReadyCount} capture-ready · {status.data?.total_connected ?? 0} connected · {selected.size} selected for run setup</p></div><Button variant="outline" size="sm" onClick={() => stopAllPreviews.mutate()} disabled={!anyPreviewBusy || stopAllPreviews.isPending}><EyeOff />{stopAllPreviews.isPending ? "Stopping previews…" : "Stop all previews"}</Button></div>
      {status.isPending ? <div className="grid grid-cols-3 items-start gap-4">{Array.from({ length: 3 }).map((_, index) => <Skeleton className="h-[430px]" key={index} />)}</div> : devices.length === 0 ? <div className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">No RGB-D sensors were detected. Check SDKs, USB connections, and permissions, then refresh.</div> : <div data-testid="sensor-grid" className="grid grid-cols-3 items-start gap-4">
        {devices.map((device) => {
          const key = sensorKey(device)
          const alias = aliasDraft[key] ?? aliases.data?.aliases[key]
          const preview = previewByKey.get(key)
          const previewOn = Boolean(preview && PREVIEW_ON.has(preview.job.status))
          const previewBusy = Boolean(preview && PREVIEW_BUSY.has(preview.job.status))
          const previewStopping = preview?.job.status === "canceling"
          const previewSupported = device.live_rgb_preview_supported ?? device.sensor_type !== "zed_2i"
          const captureReady = isCaptureReady(device)
          const readinessMessage = captureReadinessMessage(device)
          const selectedForRun = selected.has(key)
          const snapshotState = snapshotStates.data?.[key]
          const snapshotRecord = snapshotState?.manifest?.sensors?.find((item) => item.sensor_key === key)
          const snapshotJobId = snapshotJobs[key]
          return <Card data-testid="sensor-card" data-sensor-key={key} data-capture-ready={captureReady ? "true" : "false"} key={key} className="min-w-0 overflow-hidden">
            <CardHeader className="pb-3"><div className="flex items-start justify-between gap-3"><div className="flex min-w-0 items-center gap-3"><div className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted"><Webcam className="size-4 text-primary-strong" /></div><div className="min-w-0"><CardTitle className="truncate text-base">{alias?.alias || device.effective_display_name || device.display_name || device.device_id}</CardTitle><CardDescription className="truncate">{device.sensor_type.replaceAll("_", " ")} · {device.device_id}</CardDescription></div></div><StatusBadge status={device.connected === false ? "disconnected" : captureReady ? "connected" : "warning"}>{device.connected === false ? "Disconnected" : captureReady ? "Capture-ready" : "Not capture-ready"}</StatusBadge></div></CardHeader>
            <CardContent className="space-y-4">
              {readinessMessage && <div data-testid="sensor-capture-readiness" className="flex items-start gap-2 rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-xs leading-relaxed"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-warning" /><span><span className="font-semibold">Capture unavailable.</span> {readinessMessage}</span></div>}
              <Preview preview={previewBusy || preview?.job.status === "failed" ? preview : undefined} />
              {snapshotJobId && <div data-testid="sensor-snapshot" className="overflow-hidden rounded-lg border border-border bg-muted/20">{snapshotRecord?.rgb_thumbnail ? <img src={`/sensors/snapshots/${snapshotJobId}/image?path=${encodeURIComponent(snapshotRecord.rgb_thumbnail)}`} className="aspect-video w-full object-cover" alt="Latest sensor snapshot" /> : <div className="grid h-16 place-items-center px-3 text-center text-xs text-muted-foreground">{snapshotRecord?.error ?? (PREVIEW_BUSY.has(snapshotState?.job.status ?? "queued") ? "Capturing snapshot…" : "Snapshot did not produce an image")}</div>}</div>}
              <div className="space-y-2"><Label htmlFor={`alias-${key}`}>Operator alias</Label><Input id={`alias-${key}`} value={alias?.alias ?? device.effective_display_name ?? device.display_name ?? ""} onChange={(event) => updateAlias(device, { alias: event.target.value })} /></div>
              <div className="grid grid-cols-2 gap-3"><div className="space-y-2"><Label>Mounting</Label><Select value={alias?.mounting_mode ?? device.mounting_mode ?? "eye_in_hand"} onValueChange={(value) => updateAlias(device, { mounting_mode: value })}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="eye_in_hand">Eye in hand</SelectItem><SelectItem value="static">Static</SelectItem></SelectContent></Select></div><div className="space-y-2"><Label>Orientation</Label><Select value={alias?.inverted ?? device.inverted ? "inverted" : "normal"} onValueChange={(value) => void updateOrientation(device, value === "inverted", preview)} disabled={device.sensor_type !== "realsense_d435" || previewStopping || previewTransitionPending}><SelectTrigger data-testid="sensor-orientation"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="normal">Normal</SelectItem><SelectItem value="inverted">Inverted</SelectItem></SelectContent></Select></div></div>
              <div className="flex items-center justify-between rounded-lg bg-muted/55 px-3 py-2"><Label className={`flex items-center gap-2 ${!captureReady && !selectedForRun ? "cursor-not-allowed text-muted-foreground" : ""}`} title={!captureReady && !selectedForRun ? readinessMessage ?? "Camera is not capture-ready" : undefined}><Checkbox data-testid="sensor-run-selection" checked={selectedForRun} disabled={!captureReady && !selectedForRun} onCheckedChange={(value) => toggleSelected(key, value === true, captureReady)} />Use in run</Label><Button variant="ghost" size="sm" onClick={() => setDetail(device)}><Info />Details</Button></div>
              <div className="grid grid-cols-2 gap-2"><Button data-testid="sensor-preview-toggle" aria-label={`Toggle preview for ${alias?.alias || device.effective_display_name || device.display_name || device.device_id}`} aria-pressed={previewOn} variant={previewOn ? "secondary" : "outline"} disabled={!previewSupported || previewStopping || previewTransitionPending || (!captureReady && !previewOn)} title={!captureReady ? readinessMessage ?? "Camera is not capture-ready" : previewSupported ? undefined : "Live RGB preview is unavailable for this sensor family"} onClick={() => previewOn && preview ? stopPreview.mutate(preview.job.id) : startPreview.mutate(device)}>{!captureReady && !previewOn ? <><EyeOff />Not ready</> : !previewSupported ? <><EyeOff />Unavailable</> : previewStopping ? <><EyeOff />Stopping…</> : previewOn ? <><Eye />Preview on</> : <><EyeOff />Preview off</>}</Button><Button variant="outline" title={!captureReady ? readinessMessage ?? "Camera is not capture-ready" : previewBusy ? "Turn this preview off before taking a snapshot" : undefined} onClick={() => snapshot.mutate(device)} disabled={!captureReady || previewBusy || snapshot.isPending || PREVIEW_BUSY.has(snapshotState?.job.status ?? "")}><Camera />{PREVIEW_BUSY.has(snapshotState?.job.status ?? "") ? "Capturing…" : "Snapshot"}</Button></div>
            </CardContent>
          </Card>
        })}
      </div>}

      <Sheet open={Boolean(detail)} onOpenChange={(open) => !open && setDetail(null)}><SheetContent><SheetHeader><SheetTitle className="font-display text-xl font-semibold">Raw sensor metadata</SheetTitle><SheetDescription>Discovery detail for troubleshooting. Routine controls stay on the device card.</SheetDescription></SheetHeader><pre className="mt-4 flex-1 overflow-auto rounded-lg bg-muted p-4 text-xs leading-relaxed">{JSON.stringify(detail, null, 2)}</pre></SheetContent></Sheet>

      <Dialog open={startDialog} onOpenChange={(open) => { setStartDialog(open); if (!open) { setTargetConfirmed(false); setRealRobotAuthorized(false); setCamerasReady(false) } }}><DialogContent><DialogHeader><DialogTitle>Confirm IIWA {robotDialogCommand === "start_iiwa" ? "start" : "stop"}</DialogTitle><DialogDescription>{robotDialogCommand === "start_iiwa" ? "Starting" : "Stopping"} sends a command to the lab robot target. Verify the address before continuing.</DialogDescription></DialogHeader>{robotDialogCommand === "stop_iiwa" && <div data-testid="iiwa-stop-warning" className="flex items-start gap-3 rounded-lg border border-destructive/45 bg-destructive/10 p-4 text-sm"><AlertTriangle className="mt-0.5 size-5 shrink-0 text-destructive" /><div><div className="font-semibold text-destructive">IIWA STOP is not a safety stop</div><p className="mt-1 text-xs leading-relaxed text-muted-foreground">It cannot interrupt active motion. In the current calibration application it exits the waiting program, so Sunrise must be restarted manually before another START.</p></div></div>}<div className="rounded-lg border border-warning/40 bg-warning/10 p-4"><div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Target</div><div className="mt-1 font-mono text-lg font-semibold">{confirmedRobotTarget.ip}:{confirmedRobotTarget.port}</div></div><Label className="flex items-start gap-3 rounded-lg border p-3"><Checkbox checked={targetConfirmed} onCheckedChange={(value) => setTargetConfirmed(value === true)} /><span>I confirm this is the intended lab IIWA target.</span></Label>{robotDialogCommand === "start_iiwa" && <><Label className="flex items-start gap-3 rounded-lg border p-3"><Checkbox checked={realRobotAuthorized} onCheckedChange={(value) => setRealRobotAuthorized(value === true)} /><span>I authorize motion of the real lab IIWA for this start.</span></Label><Label className="flex items-start gap-3 rounded-lg border p-3"><Checkbox checked={camerasReady} onCheckedChange={(value) => setCamerasReady(value === true)} /><span>I confirm the capture cameras and pose receiver are ready.</span></Label></>}<DialogFooter><Button variant="outline" onClick={() => setStartDialog(false)}>Cancel</Button><Button variant={robotDialogCommand === "stop_iiwa" ? "destructive" : "default"} disabled={!targetConfirmed || (robotDialogCommand === "start_iiwa" && (!realRobotAuthorized || !camerasReady)) || robotCommand.isPending} onClick={() => robotCommand.mutate({ command: robotDialogCommand, target: confirmedRobotTarget })}>{robotDialogCommand === "start_iiwa" ? <Power /> : <Square />}{robotCommand.isPending ? "Queueing…" : robotDialogCommand === "start_iiwa" ? "Queue start" : "Queue stop"}</Button></DialogFooter></DialogContent></Dialog>
    </div>
  )
}
