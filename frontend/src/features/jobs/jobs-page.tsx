import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Ban, ChevronDown, Clock3, Copy, FileText, LockKeyhole, RefreshCw, Search, Square, Terminal, X } from "lucide-react"
import { toast } from "sonner"

import { EmptyState } from "@/components/empty-state"
import { HelpTip } from "@/components/help-tip"
import { PageHeader } from "@/components/page-header"
import { ProcessHandoff } from "@/components/process-handoff"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { api, errorMessage } from "@/lib/api"
import type { Job } from "@/lib/contracts"
import { formatDate } from "@/lib/utils"

const ACTIVE = new Set(["queued", "running", "canceling"])
const FAILED = new Set(["failed"])
const PAGE_SIZE = 20
type StatusFilter = "all" | "active" | "failed" | "finished"

function isCaptureJob(job: Job) {
  return job.parameters.pipeline_stage === "capture_execution" || job.resources.includes("camera")
}

function timing(job: Job) {
  if (job.status === "queued") return "Waiting to start"
  if (job.ended_at) return `Finished ${formatDate(job.ended_at)}`
  if (job.started_at) return `Started ${formatDate(job.started_at)}`
  return "Not started"
}

async function writeClipboard(text: string) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text)
    return
  }
  const textarea = document.createElement("textarea")
  textarea.value = text
  textarea.style.position = "fixed"
  textarea.style.opacity = "0"
  document.body.appendChild(textarea)
  textarea.focus()
  textarea.select()
  const copied = document.execCommand("copy")
  textarea.remove()
  if (!copied) throw new Error("The browser denied clipboard access")
}

async function copyDebugText(label: string, text: string) {
  try {
    await writeClipboard(text)
    toast.success(`${label} copied`)
  } catch (error) {
    toast.error(`${label} could not be copied`, { description: errorMessage(error) })
  }
}

function jobContext(job: Job) {
  const metadata: Partial<Job> = { ...job }
  delete metadata.tail
  return JSON.stringify({
    schema_version: "posetestbot_job_debug_context.v1",
    job: metadata,
  }, null, 2)
}

export function JobsPage() {
  const queryClient = useQueryClient()
  const [detail, setDetail] = useState<Job | null>(null)
  const [search, setSearch] = useState("")
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all")
  const [visibleLimit, setVisibleLimit] = useState(PAGE_SIZE)
  const jobs = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api<{ jobs: Job[]; resources: Record<string, string> }>("/jobs"),
    refetchInterval: (queryState) => (queryState.state.data?.jobs.some((job) => ACTIVE.has(job.status)) ? 1_000 : 5_000),
  })
  const ordered = useMemo(
    () => [...(jobs.data?.jobs ?? [])].sort((left, right) => Number(ACTIVE.has(right.status)) - Number(ACTIVE.has(left.status)) || right.created_at.localeCompare(left.created_at)),
    [jobs.data],
  )
  const filtered = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase()
    return ordered.filter((job) => {
      const matchesStatus = statusFilter === "all"
        || (statusFilter === "active" && ACTIVE.has(job.status))
        || (statusFilter === "failed" && FAILED.has(job.status))
        || (statusFilter === "finished" && !ACTIVE.has(job.status))
      if (!matchesStatus) return false
      if (!needle) return true
      return [
        job.name,
        job.id,
        job.status,
        job.message ?? "",
        job.resources.join(" "),
        String(job.parameters.run_root ?? ""),
      ].join(" ").toLocaleLowerCase().includes(needle)
    })
  }, [ordered, search, statusFilter])
  const visible = filtered.slice(0, visibleLimit)
  const currentDetail = detail ? ordered.find((job) => job.id === detail.id) ?? detail : null
  const log = useQuery({
    queryKey: ["job-log", currentDetail?.id],
    queryFn: () => api<string>(`/jobs/${currentDetail!.id}/log`),
    enabled: Boolean(currentDetail),
    refetchInterval: currentDetail && ACTIVE.has(currentDetail.status) ? 1_000 : false,
  })
  const outputText = log.data || currentDetail?.tail.join("\n") || ""
  const cancel = useMutation({
    mutationFn: (job: Job) => api(isCaptureJob(job) ? `/capture/jobs/${job.id}/stop` : `/jobs/${job.id}/cancel`, { method: "POST", body: "{}" }),
    onSuccess: () => {
      toast.success("Cancellation requested")
      queryClient.invalidateQueries({ queryKey: ["jobs"] })
      queryClient.invalidateQueries({ queryKey: ["capture-jobs"] })
    },
    onError: (error) => toast.error("Job could not be canceled", { description: errorMessage(error) }),
  })

  const setFilter = (value: StatusFilter) => {
    setStatusFilter(value)
    setVisibleLimit(PAGE_SIZE)
  }
  const setSearchValue = (value: string) => {
    setSearch(value)
    setVisibleLimit(PAGE_SIZE)
  }
  const clearFilters = () => {
    setSearch("")
    setStatusFilter("all")
    setVisibleLimit(PAGE_SIZE)
  }
  const activeCount = ordered.filter((job) => ACTIVE.has(job.status)).length
  const failedCount = ordered.filter((job) => FAILED.has(job.status)).length
  const filtersActive = Boolean(search || statusFilter !== "all")

  return <div className="space-y-6">
    <PageHeader eyebrow="Local job runner" title="Jobs & resource locks" description="Monitor background work, inspect live logs, and stop camera or capture jobs from one place." actions={<Button variant="outline" onClick={() => jobs.refetch()} disabled={jobs.isFetching}><RefreshCw className={jobs.isFetching ? "animate-spin" : undefined} />Refresh</Button>} />
    <ProcessHandoff
      title="Jobs continue when you leave their originating page"
      description="Use this page for status, resource ownership, logs, and cancellation. When a job finishes, return to the guided workflow to review its durable evidence and continue."
      to="/workflow/setup"
      action="Open workflow"
    />

    {jobs.data && Object.keys(jobs.data.resources).length > 0 && <Card><CardContent className="flex flex-wrap items-center gap-3 py-4"><span className="flex items-center gap-1 text-xs font-semibold"><LockKeyhole className="size-4 text-warning-foreground" />Held resources <HelpTip label="resource locks">A lock prevents two local jobs from opening the same camera, commanding the robot, or mutating the same managed catalogue at once. It is released when the owning job exits.</HelpTip></span>{Object.entries(jobs.data.resources).map(([resource, id]) => <StatusBadge key={resource} status="warning">{resource} · {id}</StatusBadge>)}</CardContent></Card>}

    {!jobs.isPending && !jobs.isError && ordered.length > 0 && <Card>
      <CardContent className="grid items-end gap-3 py-4 lg:grid-cols-[minmax(260px,1fr)_220px_auto]">
        <div className="space-y-1.5"><Label htmlFor="job-search">Search jobs</Label><div className="relative"><Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" /><Input id="job-search" className="pl-9" value={search} onChange={(event) => setSearchValue(event.target.value)} placeholder="Name, ID, resource, run…" /></div></div>
        <div className="space-y-1.5"><Label htmlFor="job-status-filter">Status</Label><Select value={statusFilter} onValueChange={(value: StatusFilter) => setFilter(value)}><SelectTrigger id="job-status-filter" aria-label="Filter jobs by status"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All jobs</SelectItem><SelectItem value="active">Active ({activeCount})</SelectItem><SelectItem value="failed">Failed ({failedCount})</SelectItem><SelectItem value="finished">Finished</SelectItem></SelectContent></Select></div>
        <div className="flex items-center justify-between gap-3 lg:justify-end"><span className="text-xs tabular-nums text-muted-foreground">Showing {visible.length} of {filtered.length}</span><Button variant="ghost" onClick={clearFilters} disabled={!filtersActive}><X />Clear</Button></div>
      </CardContent>
    </Card>}

    {jobs.isPending
      ? <div className="space-y-2">{Array.from({ length: 6 }).map((_, index) => <Skeleton className="h-24" key={index} />)}</div>
      : jobs.isError
        ? <Card className="border-destructive/40"><CardHeader><CardTitle>Jobs unavailable</CardTitle><CardDescription>{errorMessage(jobs.error)}</CardDescription></CardHeader><CardContent><Button variant="outline" onClick={() => jobs.refetch()}><RefreshCw />Try again</Button></CardContent></Card>
        : ordered.length === 0
          ? <EmptyState icon={Terminal} title="No jobs yet" description="Queue a readiness check, workflow action, snapshot, or plan-only sequence to see it here." />
          : filtered.length === 0
            ? <EmptyState icon={Search} title="No matching jobs" description="Change or clear the search and status filter." action={<Button variant="outline" onClick={clearFilters}><X />Clear filters</Button>} />
            : <div className="space-y-2">
              {visible.map((job) => {
                const cancelPending = job.status === "canceling" || (cancel.isPending && cancel.variables?.id === job.id)
                return <Card key={job.id} className={ACTIVE.has(job.status) ? "border-primary/35" : undefined}>
                  <CardContent className="grid items-center gap-4 py-4 xl:grid-cols-[minmax(0,1fr)_180px_260px_auto]">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2"><span className="min-w-0 truncate font-semibold">{job.name}</span><StatusBadge status={job.status} /></div>
                      <div className="mt-1 flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground"><span className="max-w-full truncate font-mono" title={job.id}>{job.id}</span><span className="flex items-center gap-1"><Clock3 className="size-3" />Queued {formatDate(job.created_at)}</span></div>
                      {job.message && <p className="mt-2 line-clamp-2 text-xs text-muted-foreground" title={job.message}>{job.message}</p>}
                    </div>
                    <div><div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Timing</div><div className="mt-1 text-xs">{timing(job)}</div></div>
                    <div><div className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Resources</div><div className="mt-1 flex flex-wrap gap-1">{job.resources.length ? job.resources.map((resource) => <StatusBadge status={ACTIVE.has(job.status) ? "warning" : "available"} key={resource}>{resource}</StatusBadge>) : <span className="text-xs text-muted-foreground">none</span>}</div></div>
                    <div className="flex flex-wrap gap-2 xl:justify-end"><Button variant="outline" size="sm" onClick={() => setDetail(job)}><FileText />Log</Button>{ACTIVE.has(job.status) && <Button variant="destructive" size="sm" onClick={() => cancel.mutate(job)} disabled={cancelPending}>{isCaptureJob(job) ? <><Square />{cancelPending ? "Stopping…" : "Stop capture"}</> : <><Ban />{cancelPending ? "Canceling…" : "Cancel"}</>}</Button>}</div>
                  </CardContent>
                </Card>
              })}
              {visible.length < filtered.length && <div className="flex justify-center pt-3"><Button variant="outline" onClick={() => setVisibleLimit((value) => value + PAGE_SIZE)}><ChevronDown />Show {Math.min(PAGE_SIZE, filtered.length - visible.length)} older jobs</Button></div>}
            </div>}

    <Sheet open={Boolean(detail)} onOpenChange={(open) => !open && setDetail(null)}>
      <SheetContent>
        <SheetHeader><SheetTitle className="font-display text-xl font-semibold">{currentDetail?.name}</SheetTitle><SheetDescription>{currentDetail?.id} · {ACTIVE.has(currentDetail?.status ?? "") ? "live process log" : "completed process log"}</SheetDescription></SheetHeader>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3"><StatusBadge status={currentDetail?.status} /><span className="text-xs text-muted-foreground">Return code {currentDetail?.returncode ?? "—"}</span></div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" disabled={!outputText || log.isPending} onClick={() => void copyDebugText("Job output", outputText)} title="Copy the complete process output"><Copy />Copy output</Button>
            <Button variant="outline" size="sm" disabled={!currentDetail} onClick={() => currentDetail && void copyDebugText("Job context", jobContext(currentDetail))} title="Copy job context and metadata"><Copy />Copy context</Button>
          </div>
        </div>
        <pre data-testid="job-log" className="min-h-0 flex-1 overflow-auto rounded-lg bg-[#11130d] p-4 text-xs leading-relaxed text-[#dce4c4]">{log.isError ? `Log unavailable: ${errorMessage(log.error)}` : outputText || "Waiting for log output…"}</pre>
        {currentDetail && ACTIVE.has(currentDetail.status) && <Button variant="destructive" onClick={() => cancel.mutate(currentDetail)} disabled={currentDetail.status === "canceling" || (cancel.isPending && cancel.variables?.id === currentDetail.id)}><Square />{currentDetail.status === "canceling" ? "Canceling…" : isCaptureJob(currentDetail) ? "Stop capture" : "Cancel job"}</Button>}
      </SheetContent>
    </Sheet>
  </div>
}
