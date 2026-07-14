import { useState } from "react"
import { Link, NavLink, Outlet, useLocation } from "react-router-dom"
import { Activity, Archive, Bot, ChevronDown, FlaskConical, Gauge, ListChecks, Moon, Plus, Sun, Workflow } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { useOperator } from "@/providers/operator-provider"
import { useTheme, type Theme } from "@/providers/theme-provider"
import { cn } from "@/lib/utils"

const navigation = [
  { to: "/dashboard", label: "Dashboard", icon: Gauge },
  { to: "/devices", label: "Devices", icon: Bot },
  { to: "/workflow/setup", label: "Workflow", icon: Workflow, match: "/workflow" },
  { to: "/artifacts", label: "Artifacts", icon: Archive },
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
      <div className="min-h-screen bg-background text-foreground">
        <aside className="fixed inset-y-0 left-0 z-40 flex w-[244px] flex-col border-r border-sidebar-border bg-sidebar px-4 py-5 text-sidebar-foreground">
          <Link to="/dashboard" className="flex items-center gap-3 px-2">
            <img src={bootstrap.brand.logo_url} alt="PoseTestBot" className="size-9 rounded-lg bg-white object-contain p-1" />
            <div><div className="font-display font-semibold tracking-tight">PoseTestBot</div><div className="text-[10px] font-semibold uppercase tracking-[0.2em] text-sidebar-foreground/50">Operator console</div></div>
          </Link>
          <nav className="mt-9 space-y-1" aria-label="Primary navigation">
            {navigation.map(({ to, label, icon: Icon, match }) => {
              const active = match ? location.pathname.startsWith(match) : location.pathname === to
              return <NavLink key={to} to={to} className={cn("group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-sidebar-foreground/65 transition-colors hover:bg-sidebar-accent hover:text-sidebar-foreground", active && "bg-sidebar-accent text-sidebar-foreground shadow-sm")}><Icon className={cn("size-[18px]", active && "text-primary")} />{label}</NavLink>
            })}
          </nav>
          <div className="mt-auto rounded-xl border border-sidebar-border bg-sidebar-accent/50 p-3">
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold"><FlaskConical className="size-4 text-primary" />Trusted lab network</div>
            <p className="text-[11px] leading-relaxed text-sidebar-foreground/55">Physical capture always requires fresh operator acknowledgement.</p>
          </div>
        </aside>

        <div className="ml-[244px] min-w-0">
          <header className="sticky top-0 z-30 flex h-16 items-center justify-between gap-4 border-b border-border bg-background/92 px-7 backdrop-blur-xl">
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
              <Select value={theme} onValueChange={(value) => setTheme(value as Theme)}>
                <Tooltip><TooltipTrigger asChild><SelectTrigger className="w-[118px]" aria-label="Theme"><span className="flex items-center gap-2">{theme === "dark" ? <Moon /> : <Sun />}<SelectValue /></span></SelectTrigger></TooltipTrigger><TooltipContent>Color theme</TooltipContent></Tooltip>
                <SelectContent><SelectItem value="system">System</SelectItem><SelectItem value="light">Light</SelectItem><SelectItem value="dark">Dark</SelectItem></SelectContent>
              </Select>
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
