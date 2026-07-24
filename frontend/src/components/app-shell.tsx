import { useEffect, useState } from "react"
import { Link, NavLink, Outlet, useLocation } from "react-router-dom"
import { Activity, BookOpen, Bot, Boxes, ChevronDown, FlaskConical, Gauge, Github, Grid3X3, LayoutTemplate, ListChecks, Moon, PackageSearch, Plus, Route, Sun, Workflow } from "lucide-react"
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
import { cn } from "@/lib/utils"

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
      { to: "/jobs", label: "Jobs", icon: ListChecks },
    ],
  },
]
const navigation = navigationGroups.flatMap((group) => group.items)

export function AppShell() {
  const { bootstrap, runs, selectedRun, selectRun } = useOperator()
  const { theme, setTheme } = useTheme()
  const location = useLocation()
  const [newRunOpen, setNewRunOpen] = useState(false)
  const [newRun, setNewRun] = useState(bootstrap.default_run_root)
  const [guideOpen, setGuideOpen] = useState(false)

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: "auto" })
  }, [location.pathname])

  const openRunDialog = () => {
    setNewRun(selectedRun)
    setNewRunOpen(true)
  }

  const applyNewRun = () => {
    if (!selectRun(newRun.trim())) {
      toast.error("Run path must stay inside an allowed run root")
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
                return <NavLink key={to} to={to} className={cn("group flex items-center gap-3 rounded-[8px] border border-transparent px-3 py-2 text-[13px] font-semibold text-sidebar-foreground/65 transition-colors duration-150 hover:bg-secondary hover:text-sidebar-foreground", active && "border-primary/55 bg-sidebar-accent text-sidebar-foreground")}><Icon className={cn("size-[17px]", active && "text-primary-strong")} />{label}</NavLink>
              })}</div>
            </div>)}
          </nav>
          <div className="mt-auto space-y-2 pt-5">
            <Link to="/workflow/setup" className="block rounded-[10px] border border-primary/30 bg-primary/5 p-3 transition-colors hover:bg-primary/10">
              <div className="flex items-center gap-2 text-xs font-semibold"><Route className="size-4 text-primary-strong" />Guided acquisition</div>
              <div className="mt-1 text-[10px] leading-relaxed text-sidebar-foreground/55">Start or resume the required operator path.</div>
            </Link>
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
                <Activity className="hidden size-4 shrink-0 text-primary-strong sm:block" aria-hidden="true" />
                <Select value={runs.some((run) => run.path === selectedRun) ? selectedRun : "__custom"} onValueChange={(value) => value === "__new" ? openRunDialog() : value !== "__custom" && selectRun(value)}>
                  <SelectTrigger className="w-full min-w-0 border-0 bg-muted/60 font-medium shadow-none sm:w-[390px] sm:max-w-[58vw] xl:max-w-[42vw]" aria-label="Selected run"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {!runs.some((run) => run.path === selectedRun) && <SelectItem value="__custom">{selectedRun}</SelectItem>}
                    {runs.map((run) => <SelectItem value={run.path} key={run.path}>{run.name} · {run.config_valid ? run.sequence ?? "configured" : "not configured"}</SelectItem>)}
                    <SelectItem value="__new"><span className="flex items-center gap-2"><Plus className="size-3.5" />Use another run path</span></SelectItem>
                  </SelectContent>
                </Select>
                <span className="hidden truncate text-xs text-muted-foreground 2xl:block">{selectedRun}</span>
              </div>
              <div className="flex shrink-0 items-center gap-2">
                <Tooltip><TooltipTrigger asChild><Button asChild variant="outline" size="icon" className="hidden size-[34px] sm:inline-flex"><a href="https://github.com/match-cow/PoseTestBot" target="_blank" rel="noreferrer" aria-label="Open PoseTestBot on GitHub"><Github /></a></Button></TooltipTrigger><TooltipContent>GitHub repository</TooltipContent></Tooltip>
                <Tooltip><TooltipTrigger asChild><Button variant="outline" size="icon" className="size-[34px]" onClick={() => setGuideOpen(true)} aria-label="Open operator console guide"><BookOpen /></Button></TooltipTrigger><TooltipContent>Console guide</TooltipContent></Tooltip>
                <Tooltip><TooltipTrigger asChild><Button variant="outline" size="icon" className="size-[34px]" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}>{theme === "light" ? <Moon /> : <Sun />}</Button></TooltipTrigger><TooltipContent>{theme === "light" ? "Use dark theme" : "Use light theme"}</TooltipContent></Tooltip>
                <Tooltip><TooltipTrigger asChild><Button variant="outline" size="icon" onClick={openRunDialog} aria-label="Choose run path"><ChevronDown /></Button></TooltipTrigger><TooltipContent>Use another run folder</TooltipContent></Tooltip>
              </div>
              <nav className="order-3 flex w-full gap-1 overflow-x-auto pb-0.5 xl:hidden" aria-label="Primary navigation">
                {navigation.map(({ to, label, icon: Icon, match }) => {
                  const active = match ? location.pathname.startsWith(match) : location.pathname === to
                  return <NavLink key={to} to={to} className={cn("flex shrink-0 items-center gap-2 rounded-[8px] border border-transparent px-2.5 py-2 text-xs font-semibold text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground", active && "border-primary/55 bg-primary/10 text-foreground")}><Icon className={cn("size-4", active && "text-primary-strong")} />{label}</NavLink>
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
            <DialogHeader><DialogTitle>Use a run folder</DialogTitle><DialogDescription>Choose an existing run or enter a new folder inside one of the server-approved roots. A new folder is configured from Workflow.</DialogDescription></DialogHeader>
            <div className="space-y-2"><Label htmlFor="new-run-path">Run path</Label><Input id="new-run-path" autoFocus value={newRun} onChange={(event) => setNewRun(event.target.value)} /></div>
            <div className="rounded-lg bg-muted p-3 text-xs text-muted-foreground"><strong className="text-foreground">Allowed roots</strong>{bootstrap.allowed_run_roots.map((root) => <div className="mt-1 font-mono" key={root}>{root}</div>)}</div>
            <DialogFooter><Button type="button" variant="outline" onClick={() => setNewRunOpen(false)}>Cancel</Button><Button type="submit">Use run</Button></DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
      <ConsoleGuide open={guideOpen} onOpenChange={setGuideOpen} />
    </TooltipProvider>
  )
}
