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

export function CaptureGate() {
  const { selectedRun, robotTarget } = useOperator()
  const queryClient = useQueryClient()
  const [open, setOpen] = useState(false)
  const [robotAck, setRobotAck] = useState(false)
  const [cameraAck, setCameraAck] = useState(false)
  const config = useQuery({ queryKey: ["run-config", selectedRun], queryFn: () => api<{ config: RunConfig; preflight: PreflightSummary }>(query("/run-config", { run_root: selectedRun })), retry: false, refetchInterval: (state) => state.state.data?.preflight.queue_blocker ? 2_000 : false })
  const blocker = config.data?.preflight.queue_blocker ?? (config.isError ? "missing_run_config" : null)

  const preflight = useMutation({
    mutationFn: () => api<{ job_id: string }>("/pipeline/run", { method: "POST", body: JSON.stringify({ stage: "run_preflight", run_root: selectedRun, options: { write: true, check: true } }) }),
    onSuccess: (data) => { toast.success("Preflight queued", { description: `Job ${data.job_id}` }); queryClient.invalidateQueries({ queryKey: ["jobs"] }); queryClient.invalidateQueries({ queryKey: ["run-config", selectedRun] }) },
    onError: (error) => toast.error("Preflight was not queued", { description: errorMessage(error) }),
  })
  const capture = useMutation({
    mutationFn: async () => {
      await api("/sensors/previews/stop", { method: "POST", body: "{}" })
      return api<{ job_id: string }>("/pipeline/run", { method: "POST", body: JSON.stringify({ stage: "capture_execution", run_root: selectedRun, options: { allow_cameras: true, allow_real_robot: true, include_sensors: true } }) })
    },
    onSuccess: (data) => { toast.success("Physical capture queued", { description: `Job ${data.job_id}` }); setOpen(false); setRobotAck(false); setCameraAck(false); queryClient.invalidateQueries({ queryKey: ["jobs"] }); queryClient.invalidateQueries({ queryKey: ["capture-jobs", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["artifacts", selectedRun] }) },
    onError: (error) => toast.error("Physical capture was not queued", { description: errorMessage(error) }),
  })
  const resetOpen = (value: boolean) => { setOpen(value); setRobotAck(false); setCameraAck(false) }

  const captureRobotIp = String(config.data?.config.robot_profile.robot_ip ?? robotTarget.ip)
  const captureRobotPort = Number(config.data?.config.robot_profile.command_port ?? robotTarget.port)

  return <Card className="border-warning/50 bg-warning/5"><CardHeader><CardTitle className="flex items-center gap-2"><ShieldAlert className="size-5 text-warning-foreground" />Advanced Capture</CardTitle><CardDescription>Physical robot and camera execution is isolated from ordinary stage forms.</CardDescription></CardHeader><CardContent>{config.isPending ? <div className="rounded-lg border border-border bg-muted/30 p-4 text-sm text-muted-foreground">Checking fresh preflight evidence…</div> : blocker ? <div className="flex items-start justify-between gap-5 rounded-lg border border-destructive/30 bg-destructive/5 p-4"><div className="flex gap-3"><AlertTriangle className="mt-0.5 size-5 shrink-0 text-destructive" /><div><div className="font-semibold">Capture blocked: {String(blocker).replaceAll("_", " ")}</div><p className="mt-1 text-xs text-muted-foreground">Missing, stale, failed, or invalid preflight must be replaced. This console never submits override flags.</p></div></div><Button onClick={() => preflight.mutate()} disabled={preflight.isPending}>{preflight.isPending ? "Queueing…" : "Run preflight"}</Button></div> : <div className="flex items-center justify-between gap-5"><div><div className="font-semibold">Preflight evidence is current</div><p className="mt-1 text-xs text-muted-foreground">Opening the dialog resets both acknowledgements.</p></div><Button variant="destructive" onClick={() => resetOpen(true)}><Camera />Open capture gate</Button></div>}</CardContent>
      <Dialog open={open} onOpenChange={resetOpen}><DialogContent><DialogHeader><DialogTitle>Authorize physical capture</DialogTitle><DialogDescription>These acknowledgements are fresh for this request and are not stored in run_config.json or local storage.</DialogDescription></DialogHeader><div className="space-y-3"><div className="rounded-lg bg-muted p-4 text-sm"><div><span className="text-muted-foreground">Run</span><div className="mt-1 break-all font-mono font-semibold">{selectedRun}</div></div><div className="mt-3"><span className="text-muted-foreground">Robot target from run config</span><div className="mt-1 font-mono font-semibold">{captureRobotIp}:{captureRobotPort}</div></div></div><Label className="flex items-start gap-3 rounded-lg border p-4"><Checkbox data-testid="capture-robot-ack" checked={robotAck} onCheckedChange={(value) => setRobotAck(value === true)} /><span>I confirm the robot workcell is clear, the target is correct, and supervised motion is authorized.</span></Label><Label className="flex items-start gap-3 rounded-lg border p-4"><Checkbox data-testid="capture-camera-ack" checked={cameraAck} onCheckedChange={(value) => setCameraAck(value === true)} /><span>I confirm the selected cameras may be opened and active previews will be stopped.</span></Label></div><DialogFooter><Button variant="outline" onClick={() => resetOpen(false)}>Cancel</Button><Button data-testid="capture-submit" variant="destructive" disabled={!robotAck || !cameraAck || capture.isPending} onClick={() => capture.mutate()}>Send both gates and capture</Button></DialogFooter></DialogContent></Dialog>
  </Card>
}
