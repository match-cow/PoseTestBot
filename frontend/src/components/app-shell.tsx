import { useState } from "react"
import { Link, NavLink, Outlet, useLocation } from "react-router-dom"
import { Activity, Bot, Boxes, ChevronDown, FlaskConical, Gauge, Github, Grid3X3, LayoutTemplate, ListChecks, Moon, Plus, Sun, Workflow } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useOperator } from "@/providers/operator-provider"
import { useTheme } from "@/providers/theme-provider"
import { cn } from "@/lib/utils"

const navigation = [
  { to: "/dashboard", label: "Dashboard", icon: Gauge },
  { to: "/devices", label: "Devices", icon: Bot },
  { to: "/cell", label: "Cell", icon: Boxes },
  { to: "/calibration-targets", label: "Calibration Targets", icon: Grid3X3 },
  { to: "/pose-templates", label: "Pose Templates", icon: LayoutTemplate },
  { to: "/workflow/setup", label: "Workflow", icon: Workflow, match: "/workflow" },
  { to: "/jobs", label: "Jobs", icon: ListChecks },
]

export function AppShell() {
  const { bootstrap, runs, selectedRun, selectRun } = useOperator()
  const { theme, setTheme } = useTheme()
  const location = useLocation()
  const [newRunOpen, setNewRunOpen] = useState(false)
  const [newRun, setNewRun] = useState(bootstrap.default_run_root)

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
        <aside className="fixed inset-y-0 left-0 z-40 flex w-[244px] flex-col border-r border-sidebar-border bg-sidebar px-4 py-5 text-sidebar-foreground">
          <Link to="/dashboard" className="flex items-center gap-3 px-2">
            <img src={bootstrap.brand.logo_urls[theme]} alt={bootstrap.brand.name} className="size-9 rounded-[7px] object-contain" />
            <div><div className="font-display text-[17px] font-semibold tracking-tight">PoseTestBot</div><div className="text-[9px] font-bold uppercase tracking-[0.18em] text-sidebar-foreground/50">Operator console</div></div>
          </Link>
          <nav className="mt-9 space-y-1" aria-label="Primary navigation">
            {navigation.map(({ to, label, icon: Icon, match }) => {
              const active = match ? location.pathname.startsWith(match) : location.pathname === to
              return <NavLink key={to} to={to} className={cn("group flex items-center gap-3 rounded-[8px] border border-transparent px-3 py-2.5 text-[13px] font-semibold text-sidebar-foreground/65 transition-colors duration-150 hover:bg-secondary hover:text-sidebar-foreground", active && "border-primary/55 bg-sidebar-accent text-sidebar-foreground")}><Icon className={cn("size-[17px]", active && "text-primary-strong")} />{label}</NavLink>
            })}
          </nav>
          <div className="mt-auto rounded-[10px] border border-sidebar-border bg-secondary p-3">
            <div className="flex items-center gap-2 text-xs font-semibold"><FlaskConical className="size-4 text-primary" />Trusted lab network</div>
          </div>
        </aside>

        <div className="ml-[244px] min-w-0">
          <header className="sticky top-0 z-30 flex h-[68px] items-center justify-between gap-4 border-b border-border bg-card/95 px-7 backdrop-blur-xl">
            <div className="flex min-w-0 items-center gap-3">
              <Activity className="size-4 shrink-0 text-primary-strong" />
              <Select value={runs.some((run) => run.path === selectedRun) ? selectedRun : "__custom"} onValueChange={(value) => value === "__new" ? setNewRunOpen(true) : value !== "__custom" && selectRun(value)}>
                <SelectTrigger className="w-[390px] max-w-[42vw] border-0 bg-muted/60 font-medium shadow-none" aria-label="Selected run"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {!runs.some((run) => run.path === selectedRun) && <SelectItem value="__custom">{selectedRun}</SelectItem>}
                  {runs.map((run) => <SelectItem value={run.path} key={run.path}>{run.name} · {run.config_valid ? run.sequence ?? "configured" : "not configured"}</SelectItem>)}
                  <SelectItem value="__new"><span className="flex items-center gap-2"><Plus className="size-3.5" />Use another run path</span></SelectItem>
                </SelectContent>
              </Select>
              <span className="hidden truncate text-xs text-muted-foreground xl:block">{selectedRun}</span>
            </div>
            <div className="flex items-center gap-2">
              <Tooltip><TooltipTrigger asChild><Button asChild variant="outline" size="icon" className="size-[34px]"><a href="https://github.com/match-cow/PoseTestBot" target="_blank" rel="noreferrer" aria-label="Open PoseTestBot on GitHub"><Github /></a></Button></TooltipTrigger><TooltipContent>GitHub repository</TooltipContent></Tooltip>
              <Tooltip><TooltipTrigger asChild><Button variant="outline" size="icon" className="size-[34px]" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}>{theme === "light" ? <Moon /> : <Sun />}</Button></TooltipTrigger><TooltipContent>{theme === "light" ? "Use dark theme" : "Use light theme"}</TooltipContent></Tooltip>
              <Button variant="outline" size="icon" onClick={() => setNewRunOpen(true)} aria-label="Choose run path"><ChevronDown /></Button>
            </div>
          </header>
          <main className="mx-auto max-w-[1600px] p-7"><Outlet /></main>
        </div>
      </div>

      <Dialog open={newRunOpen} onOpenChange={setNewRunOpen}>
        <DialogContent>
          <DialogHeader><DialogTitle>Use a run folder</DialogTitle><DialogDescription>Choose a path inside one of the server-approved roots. The folder can be configured from Workflow → Setup.</DialogDescription></DialogHeader>
          <div className="space-y-2"><Label htmlFor="new-run-path">Run path</Label><Input id="new-run-path" value={newRun} onChange={(event) => setNewRun(event.target.value)} /></div>
          <div className="rounded-lg bg-muted p-3 text-xs text-muted-foreground"><strong className="text-foreground">Allowed roots</strong>{bootstrap.allowed_run_roots.map((root) => <div className="mt-1 font-mono" key={root}>{root}</div>)}</div>
          <DialogFooter><Button variant="outline" onClick={() => setNewRunOpen(false)}>Cancel</Button><Button onClick={applyNewRun}>Use run</Button></DialogFooter>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  )
}
