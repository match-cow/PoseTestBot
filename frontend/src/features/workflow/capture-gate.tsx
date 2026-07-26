import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { AlertTriangle, Camera, ShieldAlert } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { api, errorMessage, query } from "@/lib/api"
import type { PreflightSummary, RunConfig } from "@/lib/contracts"
import { useOperator } from "@/providers/operator-provider"
import { readinessBlockerCopy } from "@/features/workflow/readiness-copy"

const CALIBRATION_CAPTURE_TIMEOUTS = {
  timeout_s: 720,
  startup_wait_s: 15,
  receive_start_timeout_s: 120,
  receive_idle_timeout_s: 60,
  camera_metadata_idle_timeout_s: 5,
} as const

export interface CaptureGateReadiness {
  ready: boolean
  message?: string
  onReview?: () => void
}

export interface CaptureGateProps {
  intent?: "legacy" | "calibration" | "dataset"
  readiness?: CaptureGateReadiness
}

const intentCopy = {
  legacy: {
    title: "Advanced Capture",
    description: "Physical robot and camera execution is isolated from ordinary stage forms.",
    open: "Open capture gate",
    dialogTitle: "Authorize physical capture",
    supervision: "Calibration capture supervision",
    queued: "Physical capture queued",
  },
  calibration: {
    title: "Record calibration images",
    description: "Open the selected cameras and run the supervised calibration motion only after readiness passes.",
    open: "Review and start capture",
    dialogTitle: "Authorize calibration capture",
    supervision: "Calibration capture supervision",
    queued: "Calibration capture queued",
  },
  dataset: {
    title: "Record object dataset",
    description: "Open the selected cameras and run the supervised dataset motion using the confirmed calibration and object placement.",
    open: "Review and start capture",
    dialogTitle: "Authorize dataset capture",
    supervision: "Dataset capture supervision",
    queued: "Dataset capture queued",
  },
} as const

export function CaptureGate({ intent = "legacy", readiness }: CaptureGateProps = {}) {
  const { selectedRun, robotTarget } = useOperator()
  const queryClient = useQueryClient()
  const copy = intentCopy[intent]
  const [open, setOpen] = useState(false)
  const [robotAck, setRobotAck] = useState(false)
  const [cameraAck, setCameraAck] = useState(false)
  const config = useQuery({ queryKey: ["run-config", selectedRun], queryFn: () => api<{ config: RunConfig; preflight: PreflightSummary }>(query("/run-config", { run_root: selectedRun })), retry: false, refetchInterval: (state) => state.state.data?.preflight.queue_blocker ? 2_000 : false })
  const internalBlocker = config.data?.preflight.queue_blocker ?? (config.isError ? "missing_run_config" : null)
  const blocker = readiness ? (readiness.ready ? null : readiness.message ?? "readiness_incomplete") : internalBlocker
  const blockerCopy = readinessBlockerCopy(blocker)

  const preflight = useMutation({
    mutationFn: () => api<{ job_id: string }>("/pipeline/run", { method: "POST", body: JSON.stringify({ stage: "run_preflight", run_root: selectedRun, options: { write: true, check: true } }) }),
    onSuccess: (data) => { toast.success("Preflight queued", { description: `Job ${data.job_id}` }); queryClient.invalidateQueries({ queryKey: ["jobs"] }); queryClient.invalidateQueries({ queryKey: ["run-config", selectedRun] }) },
    onError: (error) => toast.error("Preflight was not queued", { description: errorMessage(error) }),
  })
  const capture = useMutation({
    mutationFn: async () => {
      await api("/sensors/previews/stop", { method: "POST", body: "{}" })
      return api<{ job_id: string }>("/pipeline/run-sequence", {
        method: "POST",
        body: JSON.stringify({
          sequence: "real_full_capture_validation",
          run_root: selectedRun,
          plan_only: false,
          options: {
            capture_plan_preflight: { allow_real_robot: true },
            capture_execution_plan: { allow_cameras: true, allow_real_robot: true, include_sensors: true },
            capture_execution: { allow_cameras: true, allow_real_robot: true, include_sensors: true, ...CALIBRATION_CAPTURE_TIMEOUTS },
          },
        }),
      })
    },
    onSuccess: (data) => { toast.success(copy.queued, { description: `Job ${data.job_id}` }); setOpen(false); setRobotAck(false); setCameraAck(false); queryClient.invalidateQueries({ queryKey: ["jobs"] }); queryClient.invalidateQueries({ queryKey: ["capture-jobs", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["calibration", "setup", selectedRun] }) },
    onError: (error) => toast.error("Physical capture was not queued", { description: errorMessage(error) }),
  })
  const resetOpen = (value: boolean) => { setOpen(value); setRobotAck(false); setCameraAck(false) }

  const captureRobotIp = String(config.data?.config.robot_profile.robot_ip ?? robotTarget.ip)
  const captureRobotPort = Number(config.data?.config.robot_profile.command_port ?? robotTarget.port)
  const requestedCaptureSpeedMps = Number(config.data?.config.capture.velocity_m_s ?? 0.01)
  const usesExtendedDatasetSpeed = intent === "dataset" && requestedCaptureSpeedMps > 0.03

  return <Card className="border-warning/50 bg-warning/5"><CardHeader><CardTitle className="flex items-center gap-2"><ShieldAlert aria-hidden="true" className="size-5 text-warning-foreground" />{copy.title}</CardTitle><CardDescription>{copy.description}</CardDescription></CardHeader><CardContent>{config.isPending ? <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground">Checking fresh readiness evidence…</div> : blocker ? <div className="flex flex-col gap-4 rounded-lg border border-destructive/30 bg-destructive/5 p-4 sm:flex-row sm:items-start sm:justify-between"><div className="flex gap-3"><AlertTriangle aria-hidden="true" className="mt-0.5 size-5 shrink-0 text-destructive" /><div><div className="font-semibold">Capture blocked: {blockerCopy.heading}</div><p className="mt-1 text-xs text-muted-foreground">{blockerCopy.description} This console never submits override flags.</p></div></div>{readiness ? readiness.onReview && <Button variant="outline" onClick={readiness.onReview}>Review readiness</Button> : <Button onClick={() => preflight.mutate()} disabled={preflight.isPending}>{preflight.isPending ? "Queueing…" : "Run preflight"}</Button>}</div> : <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"><div><div className="font-semibold">Readiness evidence is current</div><p className="mt-1 text-xs text-muted-foreground">Opening the dialog resets both acknowledgements. Capture repeats plan, sensor, and empty-output checks before startup.</p></div><Button variant="destructive" onClick={() => resetOpen(true)}><Camera aria-hidden="true" />{copy.open}</Button></div>}</CardContent>
      <Dialog open={open} onOpenChange={resetOpen}><DialogContent><DialogHeader><DialogTitle>{copy.dialogTitle}</DialogTitle><DialogDescription>These acknowledgements are fresh for this request and are not stored in run_config.json or local storage.</DialogDescription></DialogHeader><div className="space-y-3"><div className="rounded-lg bg-muted p-4 text-sm"><div><span className="text-muted-foreground">Run</span><div className="mt-1 break-all font-mono font-semibold">{selectedRun}</div></div><div className="mt-3"><span className="text-muted-foreground">Robot target from run config</span><div className="mt-1 font-mono font-semibold">{captureRobotIp}:{captureRobotPort}</div></div><div className="mt-3"><span className="text-muted-foreground">Requested capture speed</span><div className="mt-1 font-mono font-semibold">{requestedCaptureSpeedMps.toFixed(2)} m/s</div></div></div>{usesExtendedDatasetSpeed && <div data-testid="extended-dataset-speed-warning" className="flex items-start gap-3 rounded-lg border border-warning/50 bg-warning/10 p-3 text-xs"><AlertTriangle aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-warning-foreground" /><div><div className="font-semibold">Extended dataset speed request</div><p className="mt-1 leading-relaxed text-muted-foreground">This request exceeds the 0.03 m/s legacy range and will use the structured robot command. Verify that the commissioned Sunrise application is active; it still caps A1 at 3°/s. Speed alone cannot guarantee sharp frames—exposure time and lighting still matter.</p></div></div>}<div data-testid="capture-timeout-envelope" className="rounded-lg border border-primary/25 bg-primary/5 p-3 text-xs"><div className="font-semibold">{copy.supervision}</div><div className="mt-1 text-muted-foreground">720 s total · 15 s sustained camera readiness (3 frames each) · 5 s maximum live camera-metadata pause · 120 s to first robot packet · 60 s between robot packets</div></div><Label className="flex items-start gap-3 rounded-lg border p-4"><Checkbox data-testid="capture-robot-ack" checked={robotAck} onCheckedChange={(value) => setRobotAck(value === true)} /><span>I confirm the robot workcell is clear, the target is correct, and supervised motion is authorized.</span></Label><Label className="flex items-start gap-3 rounded-lg border p-4"><Checkbox data-testid="capture-camera-ack" checked={cameraAck} onCheckedChange={(value) => setCameraAck(value === true)} /><span>I confirm the selected cameras may be opened and active previews will be stopped.</span></Label></div><DialogFooter><Button variant="outline" onClick={() => resetOpen(false)}>Cancel</Button><Button data-testid="capture-submit" variant="destructive" disabled={!robotAck || !cameraAck || capture.isPending} onClick={() => capture.mutate()}>Start recording</Button></DialogFooter></DialogContent></Dialog>
  </Card>
}
