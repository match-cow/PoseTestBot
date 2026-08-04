import { useEffect, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Link, NavLink, Outlet, useLocation } from "react-router-dom"
import { ArrowRight, BookOpen, Bot, Boxes, ChartNoAxesCombined, Check, Circle, CircleDot, FlaskConical, FolderOpen, Folders, Gauge, Github, Grid3X3, LayoutTemplate, ListChecks, LoaderCircle, LockKeyhole, Moon, PackageSearch, Plus, Route, Sun, Workflow } from "lucide-react"
import { toast } from "sonner"
import { ConsoleGuide } from "@/components/console-guide"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useOperator } from "@/providers/operator-provider"
import { useTheme } from "@/providers/theme-provider"
import { api, query } from "@/lib/api"
import type { CaptureState, Job } from "@/lib/contracts"
import { cn } from "@/lib/utils"
import { activeWorkflowHref, type ActiveWorkflow, type WorkflowProgressStatus } from "@/lib/workflow-session"

const navigationGroups = [
  {
    label: "Operate",
    items: [
      { to: "/dashboard", label: "Dashboard", icon: Gauge },
      { to: "/workflow/setup", label: "Workflow", icon: Workflow, match: "/workflow" },
    ],
  },
  {
    label: "Prepare",
    items: [
      { to: "/devices", label: "Devices", icon: Bot },
      { to: "/calibration-targets", label: "Calibration Targets", icon: Grid3X3 },
      { to: "/workpieces", label: "Workpiece Catalogue", icon: PackageSearch },
      { to: "/pose-templates", label: "Pose Templates", icon: LayoutTemplate },
    ],
  },
  {
    label: "Inspect",
    items: [
      { to: "/cell", label: "Cell View", icon: Boxes },
      { to: "/run-folders", label: "Run folders", icon: Folders },
      { to: "/bop-evaluation", label: "BOP Evaluation", icon: ChartNoAxesCombined },
      { to: "/jobs", label: "Jobs", icon: ListChecks },
    ],
  },
]
const navigation = navigationGroups.flatMap((group) => group.items)

const workflowStatusPresentation: Record<WorkflowProgressStatus, { label: string; className: string; icon: typeof CircleDot }> = {
  complete: { label: "Complete", className: "border-success/30 bg-success/10 text-success", icon: Check },
  current: { label: "Current step", className: "border-primary/35 bg-primary/10 text-primary-strong", icon: CircleDot },
  ready: { label: "Ready", className: "border-primary/35 bg-primary/10 text-primary-strong", icon: CircleDot },
  blocked: { label: "Needs attention", className: "border-destructive/30 bg-destructive/10 text-destructive", icon: LockKeyhole },
  running: { label: "Running", className: "border-warning/35 bg-warning/10 text-warning-foreground", icon: LoaderCircle },
  not_started: { label: "Not started", className: "border-sidebar-border bg-secondary text-sidebar-foreground/60", icon: Circle },
}

interface WorkflowRuntimeStatus {
  value: string
  label: string
}

function runRootForPath(path: string, allowedRoots: string[]) {
  return [...allowedRoots]
    .sort((left, right) => right.length - left.length)
    .find((root) => path === root || path.startsWith(`${root.replace(/\/+$/, "")}/`))
    ?? allowedRoots[0]
}

function runFolderPath(root: string, name: string) {
  return `${root.replace(/\/+$/, "")}/${name.trim()}`
}

function validRunFolderName(name: string) {
  const value = name.trim()
  return Boolean(value) && value !== "." && value !== ".." && !/[\\/\0]/.test(value)
}

function workflowRuntimePresentation(runtime: WorkflowRuntimeStatus) {
  if (["failed", "canceled", "cancelled"].includes(runtime.value)) {
    return { label: runtime.label, className: "border-destructive/30 bg-destructive/10 text-destructive", icon: LockKeyhole }
  }
  if (runtime.value === "succeeded") {
    return { label: runtime.label, className: "border-success/30 bg-success/10 text-success", icon: Check }
  }
  return { label: runtime.label, className: "border-warning/35 bg-warning/10 text-warning-foreground", icon: LoaderCircle }
}

function CurrentWorkflowCard({ workflow, runtime }: { workflow: ActiveWorkflow; runtime?: WorkflowRuntimeStatus | null }) {
  const status = runtime ? workflowRuntimePresentation(runtime) : workflowStatusPresentation[workflow.status]
  const StatusIcon = status.icon
  const href = activeWorkflowHref(workflow)
  return <section data-testid="current-workflow-card" aria-label="Workflow resume position for active run" className="rounded-[10px] border border-primary/35 bg-primary/5 p-3">
    <div className="flex items-start justify-between gap-2">
      <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-primary-strong">Resume position</div>
      <span role="status" className={cn("inline-flex shrink-0 items-center gap-1 rounded-full border px-1.5 py-0.5 text-[9px] font-semibold", status.className)}>
        <StatusIcon aria-hidden="true" className={cn("size-2.5", (workflow.status === "running" || runtime && ["queued", "running", "canceling"].includes(runtime.value)) && "animate-spin")} />
        {status.label}
      </span>
    </div>
    <div className="mt-2 text-xs font-semibold">{workflow.journeyTitle}</div>
    <div className="mt-1 text-[10px] font-bold uppercase tracking-wider text-sidebar-foreground/45">Active run · Viewed step {workflow.stepNumber} of {workflow.stepCount}</div>
    <div className="mt-1 text-[11px] leading-snug text-sidebar-foreground/70">{workflow.stepTitle}</div>
    <Button asChild size="sm" className="mt-3 h-8 w-full text-xs">
      <Link to={href} aria-label={`Resume ${workflow.journeyTitle.toLowerCase()} at step ${workflow.stepNumber}: ${workflow.stepTitle}`}>Resume step {workflow.stepNumber}<ArrowRight aria-hidden="true" /></Link>
    </Button>
    <Link to="/workflow/setup" className="mt-2 block text-center text-[10px] font-semibold text-sidebar-foreground/55 underline-offset-4 hover:text-sidebar-foreground hover:underline">Choose another workflow</Link>
  </section>
}

export function AppShell() {
  const { bootstrap, runs, selectedRun, selectRun, currentWorkflow } = useOperator()
  const { theme, setTheme } = useTheme()
  const location = useLocation()
  const [newRunOpen, setNewRunOpen] = useState(false)
  const [newRunRoot, setNewRunRoot] = useState(() => runRootForPath(bootstrap.default_run_root, bootstrap.allowed_run_roots))
  const [newRunName, setNewRunName] = useState("")
  const [guideOpen, setGuideOpen] = useState(false)
  const workflowHref = currentWorkflow ? activeWorkflowHref(currentWorkflow) : "/workflow/setup"
  const captureState = useQuery({
    queryKey: ["capture-jobs", selectedRun],
    queryFn: () => api<CaptureState>(query("/capture/jobs", { run_root: selectedRun })),
    enabled: currentWorkflow?.stepId === "capture",
    refetchInterval: (state) => state.state.data?.active_count ? 1_000 : 5_000,
  })
  const processingJobs = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api<{ jobs: Job[]; resources: Record<string, string> }>("/jobs"),
    enabled: currentWorkflow?.journey === "dataset" && currentWorkflow.stepId === "sync",
    refetchInterval: (state) => state.state.data?.jobs.some((job) => ["queued", "running", "canceling"].includes(job.status)) ? 1_000 : 5_000,
  })
  const activeCapture = currentWorkflow?.stepId === "capture"
    ? captureState.data?.jobs.find((job) => job.active)
    : undefined
  const datasetProcessingJob = currentWorkflow?.journey === "dataset" && currentWorkflow.stepId === "sync"
    ? [...(processingJobs.data?.jobs ?? [])]
        .filter((job) => job.scope_kind === "run"
          && job.run_root === selectedRun
          && (job.parameters.pipeline_sequence === "calibrated_capture_to_bop_dataset_dry_run"
            || job.name === "pipeline-run-config:calibrated_capture_to_bop_dataset_dry_run"))
        .sort((left, right) => right.created_at.localeCompare(left.created_at))[0]
    : undefined
  const workflowRuntimeStatus: WorkflowRuntimeStatus | null = activeCapture
    ? {
        value: activeCapture.status,
        label: activeCapture.status === "queued"
          ? "Recording queued"
          : activeCapture.status === "canceling"
            ? "Recording stopping"
            : "Recording running",
      }
    : datasetProcessingJob
      ? {
          value: datasetProcessingJob.status,
          label: datasetProcessingJob.status === "queued"
            ? "Processing queued"
            : datasetProcessingJob.status === "running"
              ? "Processing running"
              : datasetProcessingJob.status === "canceling"
                ? "Processing stopping"
                : datasetProcessingJob.status === "succeeded"
                  ? "Processing finished"
                  : ["canceled", "cancelled"].includes(datasetProcessingJob.status)
                    ? "Processing canceled"
                    : "Processing failed",
        }
      : null

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" })
  }, [location.pathname])

  const openRunDialog = () => {
    setNewRunRoot(runRootForPath(selectedRun, bootstrap.allowed_run_roots))
    setNewRunName("")
    setNewRunOpen(true)
  }

  const applyNewRun = () => {
    if (!validRunFolderName(newRunName)) {
      toast.error("Run folder name must be one folder, not a path")
      return
    }
    if (!selectRun(runFolderPath(newRunRoot, newRunName))) {
      toast.error("Run folder must stay inside an allowed run root")
      return
    }
    setNewRunOpen(false)
  }

  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-workspace text-foreground">
        <aside
          aria-label="Application sidebar"
          className="fixed inset-y-0 left-0 z-40 hidden w-[244px] flex-col overflow-y-auto border-r border-sidebar-border bg-sidebar px-4 py-5 text-sidebar-foreground xl:flex"
        >
          <Link to="/dashboard" className="flex items-center gap-3 px-2">
            <img src={bootstrap.brand.logo_urls[theme]} alt={bootstrap.brand.name} className="size-9 rounded-[7px] object-contain" />
            <div><div className="font-display text-[17px] font-semibold tracking-tight">PoseTestBot</div><div className="text-[9px] font-bold uppercase tracking-[0.18em] text-sidebar-foreground/50">Operator console</div></div>
          </Link>
          <nav className="mt-7 space-y-5" aria-label="Primary navigation">
            {navigationGroups.map((group) => <div key={group.label}>
              <div className="mb-1.5 px-3 text-[9px] font-bold uppercase tracking-[0.16em] text-sidebar-foreground/40">{group.label}</div>
              <div className="space-y-1">{group.items.map(({ to, label, icon: Icon, match }) => {
                const active = match ? location.pathname.startsWith(match) : location.pathname === to
                const destination = match === "/workflow" ? workflowHref : to
                return <NavLink key={to} to={destination} className={cn("group flex items-center gap-3 rounded-[8px] border border-transparent px-3 py-2 text-[13px] font-semibold text-sidebar-foreground/65 transition-colors duration-150 hover:bg-secondary hover:text-sidebar-foreground", active && "border-primary/55 bg-sidebar-accent text-sidebar-foreground")}><Icon className={cn("size-[17px]", active && "text-primary-strong")} />{label}</NavLink>
              })}</div>
            </div>)}
          </nav>
          <div className="mt-auto space-y-2 pt-5">
            {currentWorkflow ? <CurrentWorkflowCard workflow={currentWorkflow} runtime={workflowRuntimeStatus} /> : <Link to="/workflow/setup" className="block rounded-[10px] border border-primary/30 bg-primary/5 p-3 transition-colors hover:bg-primary/10">
              <div className="flex items-center gap-2 text-xs font-semibold"><Route className="size-4 text-primary-strong" />Guided acquisition</div>
              <div className="mt-1 text-[10px] leading-relaxed text-sidebar-foreground/55">Start or resume the required operator path.</div>
            </Link>}
            <div className="rounded-[10px] border border-sidebar-border bg-secondary p-3">
              <div className="flex items-center gap-2 text-xs font-semibold"><FlaskConical className="size-4 text-primary" />Trusted lab network</div>
            </div>
          </div>
        </aside>

        <div className="min-w-0 xl:ml-[244px]">
          <header className="sticky top-0 z-30 border-b border-border bg-card/95 px-4 py-3 backdrop-blur-xl sm:px-5 xl:h-[68px] xl:px-7 xl:py-0">
            <div className="flex h-full flex-wrap items-center justify-between gap-3">
              <div className="flex min-w-0 flex-1 items-center gap-2 sm:gap-3">
                <Link to="/dashboard" className="shrink-0 xl:hidden" aria-label="Open dashboard">
                  <img src={bootstrap.brand.logo_urls[theme]} alt="" className="size-8 rounded-[7px] object-contain" />
                </Link>
                <section
                  aria-label="Active run context"
                  className="min-w-0 flex-1 xl:max-w-[780px]"
                  data-testid="active-run-context"
                >
                  <Select value={runs.some((run) => run.path === selectedRun) ? selectedRun : "__custom"} onValueChange={(value) => value === "__new" ? openRunDialog() : value !== "__custom" && selectRun(value)}>
                    <SelectTrigger
                      aria-label="Active run folder"
                      className="h-[50px] min-w-0 gap-2 border-border bg-muted/45 px-3 py-1.5 text-left shadow-none hover:border-primary/45 hover:bg-muted/70 focus:ring-1 focus:ring-ring/55 focus:ring-offset-0"
                      title={selectedRun}
                    >
                      <FolderOpen className="hidden size-[18px] shrink-0 text-primary-strong sm:block" aria-hidden="true" />
                      <div className="min-w-0 flex-1">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="shrink-0 text-[9px] font-bold uppercase tracking-[0.14em] text-foreground">Active run folder</span>
                          <span className="hidden truncate text-[10px] text-muted-foreground md:inline">All run-owned pages and actions use this folder</span>
                        </div>
                        <SelectValue><span className="mt-0.5 block truncate font-mono text-[11px] font-semibold text-foreground">{selectedRun}</span></SelectValue>
                      </div>
                      <span aria-hidden="true" className="hidden shrink-0 rounded-[6px] border bg-card px-2 py-1 text-[10px] font-semibold text-muted-foreground sm:inline">Change</span>
                    </SelectTrigger>
                    <SelectContent className="min-w-[var(--radix-select-trigger-width)] max-w-[min(720px,calc(100vw-2rem))]">
                      {!runs.some((run) => run.path === selectedRun) && <SelectItem value="__custom">{selectedRun}</SelectItem>}
                      {runs.map((run) => <SelectItem value={run.path} textValue={`${run.name} · ${run.config_valid ? run.sequence ?? "configured" : "not configured"} · ${run.path}`} key={run.path}><span className="flex min-w-0 flex-col gap-0.5 py-0.5"><span className="font-medium">{run.name} · {run.config_valid ? run.sequence ?? "configured" : "not configured"}</span><span className="truncate font-mono text-[10px] text-muted-foreground">{run.path}</span></span></SelectItem>)}
                      <SelectItem value="__new"><span className="flex items-center gap-2"><Plus className="size-3.5" />Create or open a run folder…</span></SelectItem>
                    </SelectContent>
                  </Select>
                </section>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Tooltip><TooltipTrigger asChild><Button asChild variant="outline" size="icon" className="hidden size-[34px] sm:inline-flex"><a href="https://github.com/match-cow/PoseTestBot" target="_blank" rel="noreferrer" aria-label="Open PoseTestBot on GitHub"><Github /></a></Button></TooltipTrigger><TooltipContent>GitHub repository</TooltipContent></Tooltip>
                <Tooltip><TooltipTrigger asChild><Button variant="outline" size="icon" className="size-[34px]" onClick={() => setGuideOpen(true)} aria-label="Open operator console guide"><BookOpen /></Button></TooltipTrigger><TooltipContent>Console guide</TooltipContent></Tooltip>
                <Tooltip><TooltipTrigger asChild><Button variant="outline" size="icon" className="size-[34px]" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}>{theme === "light" ? <Moon /> : <Sun />}</Button></TooltipTrigger><TooltipContent>{theme === "light" ? "Use dark theme" : "Use light theme"}</TooltipContent></Tooltip>
              </div>
              <nav className="order-3 flex w-full gap-1 overflow-x-auto pb-0.5 xl:hidden" aria-label="Primary navigation">
                {navigation.map(({ to, label, icon: Icon, match }) => {
                  const active = match ? location.pathname.startsWith(match) : location.pathname === to
                  const destination = match === "/workflow" ? workflowHref : to
                  return <NavLink key={to} to={destination} className={cn("flex shrink-0 items-center gap-2 rounded-[8px] border border-transparent px-2.5 py-2 text-xs font-semibold text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground", active && "border-primary/55 bg-primary/10 text-foreground")}><Icon className={cn("size-4", active && "text-primary-strong")} />{label}</NavLink>
                })}
              </nav>
            </div>
          </header>
          <main className="mx-auto max-w-[1600px] p-4 sm:p-5 xl:p-7"><Outlet /></main>
        </div>
      </div>

      <Dialog open={newRunOpen} onOpenChange={setNewRunOpen}>
        <DialogContent>
          <form onSubmit={(event) => { event.preventDefault(); applyNewRun() }} className="space-y-4">
            <DialogHeader>
              <DialogTitle>Create or open a run folder</DialogTitle>
              <DialogDescription>Each acquisition run is a separate folder directly below an approved storage root. The run name saved during setup is metadata and does not choose this folder.</DialogDescription>
            </DialogHeader>
            <div className="flex gap-3 rounded-lg border bg-muted/40 p-3 text-xs leading-relaxed">
              <FolderOpen className="mt-0.5 size-4 shrink-0 text-primary-strong" aria-hidden="true" />
              <div>
                <strong>Choose one folder per acquisition run.</strong>
                <p className="mt-1 text-muted-foreground">A new folder starts unconfigured. Creating another sibling folder preserves the configuration and evidence of every earlier run.</p>
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="new-run-root">Storage root</Label>
              <Select value={newRunRoot} onValueChange={setNewRunRoot}>
                <SelectTrigger id="new-run-root" aria-label="Run storage root"><SelectValue /></SelectTrigger>
                <SelectContent>{bootstrap.allowed_run_roots.map((root) => <SelectItem value={root} key={root}>{root}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div className="space-y-2"><Label htmlFor="new-run-name">Run folder name</Label><Input id="new-run-name" autoFocus value={newRunName} onChange={(event) => setNewRunName(event.target.value)} placeholder="e.g. object_capture_20260803" /></div>
            <div className="rounded-lg bg-muted p-3 text-xs text-muted-foreground"><strong className="text-foreground">Resulting run folder</strong><div className="mt-1 break-all font-mono" data-testid="new-run-path-preview">{validRunFolderName(newRunName) ? runFolderPath(newRunRoot, newRunName) : `${newRunRoot.replace(/\/+$/, "")}/…`}</div><p className="mt-2">The folder is created when its setup is saved.</p></div>
            <DialogFooter><Button type="button" variant="outline" onClick={() => setNewRunOpen(false)}>Cancel</Button><Button type="submit" disabled={!validRunFolderName(newRunName)}>Use run folder</Button></DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
      <ConsoleGuide open={guideOpen} onOpenChange={setGuideOpen} />
    </TooltipProvider>
  )
}
