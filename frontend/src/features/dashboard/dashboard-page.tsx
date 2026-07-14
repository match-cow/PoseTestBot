import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { AlertTriangle, ArrowRight, Bot, Camera, CheckCircle2, CircleDot, Cpu, Play, RefreshCw, Route, ShieldCheck, Square } from "lucide-react"
import { toast } from "sonner"
import { PageHeader } from "@/components/page-header"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
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
  const recommendedDescription = typeof recommendation?.description === "string" ? recommendation.description : overview.data?.config ? "Open the workflow to continue from the first incomplete stage." : "Write a safe, plan-only run configuration before queueing work."
  const workflowComplete = sections.filter((section) => ["run_setup", "preflight", "capture", "sync", "calibration", "bop"].includes(section.id))

  const refresh = () => queryClient.invalidateQueries({ predicate: (item) => ["overview", "sensors", "robot", "runtime", "capture-jobs", "jobs"].includes(String(item.queryKey[0])) })

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Current run" title="Acquisition readiness" description="One place to understand the lab, the run, and the safest next action." actions={<Button variant="outline" onClick={refresh}><RefreshCw />Refresh</Button>} />

      {activeCapture && <div className="flex items-center justify-between rounded-xl border border-primary/35 bg-primary/10 px-5 py-4"><div className="flex items-center gap-3"><span className="relative flex size-3"><span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-60" /><span className="relative inline-flex size-3 rounded-full bg-primary" /></span><div><div className="font-semibold">Capture is {activeCapture.status}</div><div className="text-xs text-muted-foreground">{activeCapture.name} · keep the operator console visible</div></div></div><div className="flex gap-2"><Button variant="destructive" size="sm" onClick={() => stopCapture.mutate(activeCapture.id)}><Square />Stop capture</Button><Button asChild size="sm"><Link to="/jobs">Open controls <ArrowRight /></Link></Button></div></div>}
      {!activeCapture && activeJob && <div className="flex items-center justify-between rounded-xl border border-primary/25 bg-primary/5 px-5 py-4"><div className="flex items-center gap-3"><span className="relative flex size-3"><span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-50" /><span className="relative inline-flex size-3 rounded-full bg-primary" /></span><div><div className="font-semibold">Job is {activeJob.status}</div><div className="text-xs text-muted-foreground">{activeJob.name} · {activeJob.resources.join(", ") || "no resource locks"}</div></div></div><Button asChild size="sm"><Link to="/jobs">Open job <ArrowRight /></Link></Button></div>}

      <div className="operator-grid">
        <Card className="col-span-8 overflow-hidden">
          <CardHeader className="border-b border-border bg-muted/20">
            <div className="flex items-start justify-between gap-4"><div><CardDescription>Recommended next action</CardDescription><CardTitle className="mt-1 text-2xl">{recommendedLabel}</CardTitle></div><div className="grid size-11 place-items-center rounded-full bg-primary text-primary-foreground"><ArrowRight /></div></div>
          </CardHeader>
          <CardContent className="pt-5"><p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">{recommendedDescription}</p><div className="mt-5 flex gap-2"><Button asChild><Link to={overview.data?.config ? "/workflow/preflight" : "/workflow/setup"}><Play />Open workflow</Link></Button><Button asChild variant="outline"><Link to="/artifacts">Inspect evidence</Link></Button></div></CardContent>
        </Card>
        <RoomMonitor />
      </div>

      <div className="grid grid-cols-4 gap-4">
        {overview.isPending || sensors.isPending ? Array.from({ length: 4 }).map((_, index) => <Skeleton className="h-40" key={index} />) : <>
          <SummaryCard icon={ShieldCheck} label="Run preflight" value={titleCase(preflight?.status ?? "pending")} status={preflight?.status} detail={preflight?.status === "complete" ? "Artifact-backed checks are present." : "Run or refresh preflight before execution."} />
          <SummaryCard icon={Camera} label="Sensors" value={`${sensors.data?.total_connected ?? 0} connected`} status={sensors.data?.all_expected_connected ? "connected" : "warning"} detail="RealSense, OAK-D Pro, and ZED discovery." />
          <SummaryCard icon={Bot} label="Robot profile" value="Lab IIWA" status={robot.isSuccess ? "ready" : "warning"} detail="Read-only status; no command was sent." />
          <SummaryCard icon={Cpu} label="Optional runtimes" value={`${availableRuntimes}/${runtimeItems.length} available`} status={availableRuntimes === runtimeItems.length ? "ready" : "warning"} detail="BlenderProc and Stereolabs SDK visibility." />
        </>}
      </div>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><Route className="size-5 text-primary-strong" />Workflow timeline</CardTitle><CardDescription>Statuses come from durable run artifacts, not browser memory.</CardDescription></CardHeader>
        <CardContent>
          <div className="grid grid-cols-6 gap-2">
            {workflowComplete.map((section, index) => <Link to={`/workflow/${section.id === "run_setup" ? "setup" : section.id === "bop" ? "bop-export" : section.id}`} key={section.id} className="group relative rounded-lg border border-border p-3 transition hover:border-primary/60 hover:bg-primary/5"><div className="mb-5 flex items-center justify-between"><span className="text-[10px] font-bold text-muted-foreground">{String(index + 1).padStart(2, "0")}</span>{section.status === "complete" ? <CheckCircle2 className="size-4 text-success" /> : section.status === "blocked" ? <AlertTriangle className="size-4 text-destructive" /> : <CircleDot className="size-4 text-muted-foreground" />}</div><div className="text-sm font-semibold">{section.label}</div><div className="mt-1 text-xs capitalize text-muted-foreground">{section.status.replaceAll("_", " ")}</div></Link>)}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
