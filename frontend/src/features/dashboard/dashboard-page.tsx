import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useState } from "react"
import { Link } from "react-router-dom"
import { AlertTriangle, ArrowRight, Bot, Camera, CheckCircle2, CircleDot, Cpu, Play, Power, RefreshCw, Route, ShieldCheck, Square } from "lucide-react"
import { toast } from "sonner"
import { HelpTip } from "@/components/help-tip"
import { PageHeader } from "@/components/page-header"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Skeleton } from "@/components/ui/skeleton"
import { api, errorMessage, query } from "@/lib/api"
import type { CaptureState, Job, Overview, SensorStatus } from "@/lib/contracts"
import { titleCase } from "@/lib/utils"
import { useOperator } from "@/providers/operator-provider"
import { RoomMonitor } from "@/features/dashboard/room-monitor"

function SummaryCard({ icon: Icon, label, value, status, detail }: { icon: typeof Bot; label: string; value: string; status?: string; detail: string }) {
  return (
    <Card>
      <CardContent className="pt-5">
        <div className="flex items-start justify-between"><div className="grid size-9 place-items-center rounded-lg bg-muted"><Icon className="size-4 text-primary-strong" /></div><StatusBadge status={status ?? value} /></div>
        <div className="mt-5 text-xs font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
        <div className="mt-1 font-display text-lg font-semibold">{value}</div>
        <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">{detail}</p>
      </CardContent>
    </Card>
  )
}

type IiwaCommand = "start_iiwa" | "stop_iiwa"

function IiwaQuickControls({ profileStatus }: { profileStatus: "checking" | "configured" | "error" }) {
  const { robotTarget } = useOperator()
  const queryClient = useQueryClient()
  const [command, setCommand] = useState<IiwaCommand | null>(null)
  const [targetConfirmed, setTargetConfirmed] = useState(false)
  const [realRobotAuthorized, setRealRobotAuthorized] = useState(false)
  const [camerasReady, setCamerasReady] = useState(false)
  const robotCommand = useMutation({
    mutationFn: (nextCommand: IiwaCommand) => api<{ job_id: string }>("/run-command", {
      method: "POST",
      body: JSON.stringify({ command: nextCommand, robot_ip: robotTarget.ip, robot_port: robotTarget.port, ...(nextCommand === "start_iiwa" ? { allow_real_robot: true, allow_cameras: true } : {}) }),
    }),
    onSuccess: (data, nextCommand) => {
      toast.success(nextCommand === "start_iiwa" ? "IIWA start queued" : "IIWA stop queued", { description: `Job ${data.job_id}` })
      setCommand(null)
      setTargetConfirmed(false)
      setRealRobotAuthorized(false)
      setCamerasReady(false)
      queryClient.invalidateQueries({ queryKey: ["jobs"] })
    },
    onError: (error) => toast.error("Robot command was not queued", { description: errorMessage(error) }),
  })
  const openCommand = (nextCommand: IiwaCommand) => {
    setTargetConfirmed(false)
    setRealRobotAuthorized(false)
    setCamerasReady(false)
    setCommand(nextCommand)
  }
  const setDialogOpen = (open: boolean) => {
    if (open) return
    setCommand(null)
    setTargetConfirmed(false)
    setRealRobotAuthorized(false)
    setCamerasReady(false)
  }

  return (
    <Card data-testid="iiwa-quick-controls">
      <CardContent className="pt-5">
        <div className="flex items-start justify-between"><div className="grid size-9 place-items-center rounded-lg bg-muted"><Bot className="size-4 text-primary-strong" /></div><StatusBadge status={profileStatus}>{profileStatus}</StatusBadge></div>
        <div className="mt-5 flex items-center gap-1 text-xs font-semibold uppercase tracking-wider text-muted-foreground">Robot control <HelpTip label="robot control status">Configured means the fixed lab profile loaded. It does not contact the robot or prove that the Sunrise application is running.</HelpTip></div>
        <div className="mt-1 font-display text-lg font-semibold">Lab IIWA</div>
        <p className="mt-1 text-xs text-muted-foreground">Quick commands use the configured lab target and require confirmation.</p>
        <div className="mt-4 grid grid-cols-2 gap-2"><Button size="sm" onClick={() => openCommand("start_iiwa")} disabled={robotCommand.isPending}><Power />Start IIWA</Button><Button size="sm" variant="destructive" onClick={() => openCommand("stop_iiwa")} disabled={robotCommand.isPending}><Square />Stop IIWA</Button></div>
      </CardContent>
      <Dialog open={command !== null} onOpenChange={setDialogOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Confirm IIWA {command === "stop_iiwa" ? "stop" : "start"}</DialogTitle><DialogDescription>{command === "stop_iiwa" ? "Stopping" : "Starting"} sends a command to the configured lab robot target.</DialogDescription></DialogHeader>
          {command === "stop_iiwa" && <div data-testid="iiwa-stop-warning" className="flex items-start gap-3 rounded-lg border border-destructive/45 bg-destructive/10 p-4 text-sm"><AlertTriangle className="mt-0.5 size-5 shrink-0 text-destructive" /><div><div className="font-semibold text-destructive">IIWA STOP is not a safety stop</div><p className="mt-1 text-xs leading-relaxed text-muted-foreground">It cannot interrupt active motion. In the current calibration application it exits the waiting program, so Sunrise must be restarted manually before another START.</p></div></div>}
          <div className="rounded-lg border border-warning/40 bg-warning/10 p-4"><div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Target</div><div className="mt-1 font-mono text-lg font-semibold">{robotTarget.ip}:{robotTarget.port}</div></div>
          <Label className="flex items-start gap-3 rounded-lg border p-3"><Checkbox checked={targetConfirmed} onCheckedChange={(value) => setTargetConfirmed(value === true)} /><span>I confirm this is the intended lab IIWA target.</span></Label>
          {command === "start_iiwa" && <><Label className="flex items-start gap-3 rounded-lg border p-3"><Checkbox checked={realRobotAuthorized} onCheckedChange={(value) => setRealRobotAuthorized(value === true)} /><span>I authorize motion of the real lab IIWA for this start.</span></Label><Label className="flex items-start gap-3 rounded-lg border p-3"><Checkbox checked={camerasReady} onCheckedChange={(value) => setCamerasReady(value === true)} /><span>I confirm the capture cameras and pose receiver are ready.</span></Label></>}
          <DialogFooter><Button variant="outline" onClick={() => setDialogOpen(false)}>Cancel</Button><Button variant={command === "stop_iiwa" ? "destructive" : "default"} disabled={!targetConfirmed || (command === "start_iiwa" && (!realRobotAuthorized || !camerasReady)) || robotCommand.isPending || command === null} onClick={() => command && robotCommand.mutate(command)}>{command === "stop_iiwa" ? <Square /> : <Power />}{robotCommand.isPending ? "Queueing…" : command === "stop_iiwa" ? "Queue stop" : "Queue start"}</Button></DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  )
}

export function DashboardPage() {
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const overview = useQuery({ queryKey: ["overview", selectedRun], queryFn: () => api<Overview>(query("/ui/overview", { run_root: selectedRun })) })
  const sensors = useQuery({ queryKey: ["sensors", "status"], queryFn: () => api<SensorStatus>("/sensors/status"), staleTime: 10_000 })
  const robot = useQuery({ queryKey: ["robot", "status"], queryFn: () => api<Record<string, unknown>>("/robot/status"), staleTime: 10_000 })
  const runtime = useQuery({ queryKey: ["runtime", "status"], queryFn: () => api<Record<string, unknown>>("/runtime/status"), staleTime: 10_000 })
  const capture = useQuery({ queryKey: ["capture-jobs", selectedRun], queryFn: () => api<CaptureState>(query("/capture/jobs", { run_root: selectedRun })), refetchInterval: (state) => state.state.data?.active_count ? 1_000 : 5_000 })
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: () => api<{ jobs: Job[]; resources: Record<string, string> }>("/jobs"), refetchInterval: (state) => state.state.data?.jobs.some((job) => ["queued", "running", "canceling"].includes(job.status)) ? 1_000 : 5_000 })
  const stopCapture = useMutation({
    mutationFn: (jobId: string) => api(`/capture/jobs/${jobId}/stop`, { method: "POST", body: "{}" }),
    onSuccess: () => { toast.success("Capture stop requested"); queryClient.invalidateQueries({ queryKey: ["capture-jobs", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["jobs"] }) },
    onError: (error) => toast.error("Capture could not be stopped", { description: errorMessage(error) }),
  })

  const activeCapture = capture.data?.jobs.find((job) => job.active)
  const activeJob = jobs.data?.jobs.find((job) => ["queued", "running", "canceling"].includes(job.status))
  const sections = overview.data?.sidebar ?? []
  const preflight = sections.find((item) => item.id === "preflight")
  const runtimeItems = Array.isArray(runtime.data?.runtimes) ? runtime.data.runtimes as Array<{ available?: boolean }> : []
  const availableRuntimes = runtimeItems.filter((item) => item.available).length
  const recommendation = overview.data?.recommendations[0]
  const recommendedLabel = typeof recommendation?.label === "string" ? recommendation.label : overview.data?.config ? "Review workflow state" : "Configure this run"
  const recommendedDescription = typeof recommendation?.description === "string" ? recommendation.description : overview.data?.config ? "Open the guided workflow to continue from the first incomplete step." : "Choose camera calibration or object dataset recording, then configure the required inputs."
  const datasetWorkflow = overview.data?.config?.dataset_mode === "pose_template"
  const workflowHref = !overview.data?.config ? "/workflow/setup" : datasetWorkflow ? "/workflow/dataset" : "/workflow/calibration"
  const workflowLabel = !overview.data?.config ? "Workflow" : datasetWorkflow ? "Object dataset" : "Camera calibration"
  const workflowSections = datasetWorkflow
    ? ["run_setup", "preflight", "capture", "sync", "bop"]
    : ["run_setup", "preflight", "capture", "sync", "calibration"]
  const workflowComplete = sections.filter((section) => workflowSections.includes(section.id))
  const workflowSectionLabels: Record<string, string> = {
    run_setup: "Run & cameras",
    preflight: "Readiness",
    capture: "Recording",
    sync: "Synchronization",
    calibration: "Calibration results",
    bop: "BOP export",
  }

  const refresh = () => queryClient.invalidateQueries({ predicate: (item) => ["overview", "sensors", "robot", "runtime", "capture-jobs", "jobs"].includes(String(item.queryKey[0])) })
  const statusErrors = [
    overview.isError && "run evidence",
    sensors.isError && "sensor discovery",
    robot.isError && "robot profile",
    runtime.isError && "runtime status",
    capture.isError && "capture status",
    jobs.isError && "job status",
  ].filter(Boolean) as string[]
  const robotProfileStatus = robot.isPending ? "checking" : robot.isError ? "error" : "configured"

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Current run" title="Acquisition readiness" description="One place to understand the lab, the run, and the safest next action." actions={<Button variant="outline" onClick={refresh}><RefreshCw />Refresh</Button>} />

      {statusErrors.length > 0 && <div role="alert" className="flex flex-col gap-3 rounded-xl border border-destructive/35 bg-destructive/5 px-4 py-3 sm:flex-row sm:items-center sm:justify-between"><div className="flex items-start gap-3"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" /><div><div className="text-sm font-semibold">Some dashboard status is unavailable</div><p className="mt-1 text-xs text-muted-foreground">Could not load {statusErrors.join(", ")}. Missing responses are not treated as ready; refresh before relying on this overview.</p></div></div><Button variant="outline" size="sm" onClick={refresh}><RefreshCw />Refresh status</Button></div>}
      {activeCapture && <div className="flex flex-col items-start justify-between gap-4 rounded-xl border border-primary/35 bg-primary/10 px-5 py-4 sm:flex-row sm:items-center"><div className="flex items-center gap-3"><span className="relative flex size-3"><span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-60" /><span className="relative inline-flex size-3 rounded-full bg-primary" /></span><div><div className="font-semibold">Capture is {activeCapture.status}</div><div className="text-xs text-muted-foreground">{activeCapture.name} · keep the operator console visible</div></div></div><div className="flex flex-wrap gap-2"><Button variant="destructive" size="sm" onClick={() => stopCapture.mutate(activeCapture.id)} disabled={stopCapture.isPending || activeCapture.status === "canceling"}><Square />{stopCapture.isPending || activeCapture.status === "canceling" ? "Stopping…" : "Stop capture"}</Button><Button asChild size="sm"><Link to="/jobs">Open controls <ArrowRight /></Link></Button></div></div>}
      {!activeCapture && activeJob && <div className="flex flex-col items-start justify-between gap-4 rounded-xl border border-primary/25 bg-primary/5 px-5 py-4 sm:flex-row sm:items-center"><div className="flex items-center gap-3"><span className="relative flex size-3"><span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-50" /><span className="relative inline-flex size-3 rounded-full bg-primary" /></span><div><div className="font-semibold">Job is {activeJob.status}</div><div className="text-xs text-muted-foreground">{activeJob.name} · {activeJob.resources.join(", ") || "no resource locks"}</div></div></div><Button asChild size="sm"><Link to="/jobs">Open job <ArrowRight /></Link></Button></div>}

      <div className="operator-grid">
        <Card className="col-span-12 overflow-hidden xl:col-span-8">
          <CardHeader className="border-b border-border bg-muted/20">
            <div className="flex items-start justify-between gap-4"><div><CardDescription>Recommended next action</CardDescription><CardTitle className="mt-1 text-2xl">{recommendedLabel}</CardTitle></div><div className="grid size-11 place-items-center rounded-full bg-primary text-primary-foreground"><ArrowRight /></div></div>
          </CardHeader>
          <CardContent className="pt-5"><p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">{recommendedDescription}</p><div className="mt-5"><Button asChild><Link to={workflowHref}><Play />Open {workflowLabel.toLowerCase()}</Link></Button></div></CardContent>
        </Card>
        <RoomMonitor />
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {overview.isPending || sensors.isPending ? Array.from({ length: 4 }).map((_, index) => <Skeleton className="h-40" key={index} />) : <>
          <SummaryCard icon={ShieldCheck} label="Readiness check" value={titleCase(preflight?.status ?? "pending")} status={preflight?.status} detail={preflight?.status === "complete" ? "Artifact-backed readiness evidence is present." : "Check or refresh readiness before recording."} />
          <SummaryCard icon={Camera} label="Sensors" value={`${sensors.data?.total_connected ?? 0} connected`} status={sensors.data?.all_expected_connected ? "connected" : "warning"} detail="RealSense, OAK-D Pro, and ZED discovery." />
          <IiwaQuickControls profileStatus={robotProfileStatus} />
          <SummaryCard icon={Cpu} label="Optional runtimes" value={`${availableRuntimes}/${runtimeItems.length} available`} status={availableRuntimes === runtimeItems.length ? "ready" : "warning"} detail="BlenderProc and Stereolabs SDK visibility." />
        </>}
      </div>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><Route className="size-5 text-primary-strong" />{workflowLabel} evidence</CardTitle><CardDescription>Open the guided workflow for required actions. These statuses come from durable run artifacts, not browser memory.</CardDescription></CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {workflowComplete.map((section, index) => <Link to={workflowHref} aria-label={`Open ${workflowLabel.toLowerCase()} workflow from ${workflowSectionLabels[section.id] ?? section.label}`} key={section.id} className="group relative rounded-lg border border-border p-3 transition hover:border-primary/60 hover:bg-primary/5"><div className="mb-5 flex items-center justify-between"><span className="text-[10px] font-bold text-muted-foreground">{String(index + 1).padStart(2, "0")}</span>{section.status === "complete" ? <CheckCircle2 className="size-4 text-success" aria-label="Complete" /> : section.status === "blocked" ? <AlertTriangle className="size-4 text-destructive" aria-label="Blocked" /> : <CircleDot className="size-4 text-muted-foreground" aria-label="Not complete" />}</div><div className="text-sm font-semibold">{workflowSectionLabels[section.id] ?? section.label}</div><div className="mt-1 text-xs capitalize text-muted-foreground">{section.status.replaceAll("_", " ")}</div></Link>)}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
