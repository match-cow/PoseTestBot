import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Ban, Clock3, FileText, LockKeyhole, RefreshCw, Square, Terminal } from "lucide-react"
import { toast } from "sonner"
import { PageHeader } from "@/components/page-header"
import { EmptyState } from "@/components/empty-state"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { api, errorMessage } from "@/lib/api"
import type { Job } from "@/lib/contracts"
import { formatDate } from "@/lib/utils"

const ACTIVE = new Set(["queued", "running", "canceling"])

export function JobsPage() {
  const queryClient = useQueryClient()
  const [detail, setDetail] = useState<Job | null>(null)
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api<{ jobs: Job[]; resources: Record<string, string> }>("/jobs"),
    refetchInterval: (queryState) => (queryState.state.data?.jobs.some((job) => ACTIVE.has(job.status)) ? 1_000 : 5_000),
  })
  const ordered = useMemo(() => [...(jobs.data?.jobs ?? [])].sort((left, right) => Number(ACTIVE.has(right.status)) - Number(ACTIVE.has(left.status)) || right.created_at.localeCompare(left.created_at)), [jobs.data])
  const currentDetail = detail ? ordered.find((job) => job.id === detail.id) ?? detail : null
  const log = useQuery({ queryKey: ["job-log", currentDetail?.id], queryFn: () => api<string>(`/jobs/${currentDetail!.id}/log`), enabled: Boolean(currentDetail), refetchInterval: currentDetail && ACTIVE.has(currentDetail.status) ? 1_000 : false })
  const cancel = useMutation({
    mutationFn: (job: Job) => api(job.parameters.pipeline_stage === "capture_execution" || job.resources.includes("camera") ? `/capture/jobs/${job.id}/stop` : `/jobs/${job.id}/cancel`, { method: "POST", body: "{}" }),
    onSuccess: () => { toast.success("Cancellation requested"); queryClient.invalidateQueries({ queryKey: ["jobs"] }); queryClient.invalidateQueries({ queryKey: ["capture-jobs"] }) },
    onError: (error) => toast.error("Job could not be canceled", { description: errorMessage(error) }),
  })

  return <div className="space-y-6">
    <PageHeader eyebrow="Local job runner" title="Jobs & resource locks" description="Active work first, live logs on demand, and dedicated stop controls for camera or capture jobs." actions={<Button variant="outline" onClick={() => jobs.refetch()}><RefreshCw />Refresh</Button>} />
    {jobs.data && Object.keys(jobs.data.resources).length > 0 && <Card><CardContent className="flex flex-wrap items-center gap-3 py-4"><span className="flex items-center gap-2 text-xs font-semibold"><LockKeyhole className="size-4 text-warning-foreground" />Held resources</span>{Object.entries(jobs.data.resources).map(([resource, id]) => <StatusBadge key={resource} status="warning">{resource} · {id}</StatusBadge>)}</CardContent></Card>}
    {jobs.isPending ? <div className="space-y-2">{Array.from({ length: 6 }).map((_, index) => <Skeleton className="h-24" key={index} />)}</div> : ordered.length === 0 ? <EmptyState icon={Terminal} title="No jobs yet" description="Queue a preflight, workflow stage, snapshot, or plan-only sequence to see it here." /> : <div className="space-y-2">{ordered.map((job) => <Card key={job.id} className={ACTIVE.has(job.status) ? "border-primary/35" : undefined}><CardContent className="grid grid-cols-[minmax(0,1fr)_140px_240px_auto] items-center gap-5 py-4"><div className="min-w-0"><div className="flex items-center gap-2"><span className="truncate font-semibold">{job.name}</span><StatusBadge status={job.status} /></div><div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground"><span className="font-mono">{job.id}</span><span className="flex items-center gap-1"><Clock3 className="size-3" />{formatDate(job.created_at)}</span></div>{job.message && <p className="mt-2 truncate text-xs text-muted-foreground">{job.message}</p>}</div><div><div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Timing</div><div className="mt-1 text-xs">{job.started_at ? `Started ${formatDate(job.started_at)}` : "Waiting to start"}</div></div><div><div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Resources</div><div className="mt-1 flex flex-wrap gap-1">{job.resources.length ? job.resources.map((resource) => <StatusBadge status={ACTIVE.has(job.status) ? "warning" : "available"} key={resource}>{resource}</StatusBadge>) : <span className="text-xs text-muted-foreground">none</span>}</div></div><div className="flex gap-2"><Button variant="outline" size="sm" onClick={() => setDetail(job)}><FileText />Log</Button>{ACTIVE.has(job.status) && <Button variant="destructive" size="sm" onClick={() => cancel.mutate(job)}>{job.parameters.pipeline_stage === "capture_execution" || job.resources.includes("camera") ? <><Square />Stop capture</> : <><Ban />Cancel</>}</Button>}</div></CardContent></Card>)}</div>}

    <Sheet open={Boolean(detail)} onOpenChange={(open) => !open && setDetail(null)}><SheetContent><SheetHeader><SheetTitle className="font-display text-xl font-semibold">{currentDetail?.name}</SheetTitle><SheetDescription>{currentDetail?.id} · live process log</SheetDescription></SheetHeader><div className="flex items-center justify-between"><StatusBadge status={currentDetail?.status} /><span className="text-xs text-muted-foreground">Return code {currentDetail?.returncode ?? "—"}</span></div><pre data-testid="job-log" className="min-h-0 flex-1 overflow-auto rounded-lg bg-[#11130d] p-4 text-xs leading-relaxed text-[#dce4c4]">{log.data || currentDetail?.tail.join("\n") || "Waiting for log output…"}</pre>{currentDetail && ACTIVE.has(currentDetail.status) && <Button variant="destructive" onClick={() => cancel.mutate(currentDetail)}><Square />{currentDetail.parameters.pipeline_stage === "capture_execution" || currentDetail.resources.includes("camera") ? "Stop capture" : "Cancel job"}</Button>}</SheetContent></Sheet>
  </div>
}
