import { useQuery } from "@tanstack/react-query"
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react"
import { api } from "@/lib/api"
import type { Bootstrap, RunIndexItem } from "@/lib/contracts"
import {
  activeWorkflow,
  parseActiveWorkflow,
  type ActiveWorkflow,
  type WorkflowJourneyId,
  type WorkflowProgressStatus,
} from "@/lib/workflow-session"

interface RobotTarget { ip: string; port: number }
interface OperatorContextValue {
  bootstrap: Bootstrap
  runs: RunIndexItem[]
  selectedRun: string
  selectedRunEpoch: number
  selectRun: (path: string) => boolean
  robotTarget: RobotTarget
  setRobotTarget: (target: RobotTarget) => void
  currentWorkflow: ActiveWorkflow | null
  rememberWorkflowStep: (journey: WorkflowJourneyId, stepId: string, status: WorkflowProgressStatus) => void
}

const OperatorContext = createContext<OperatorContextValue | null>(null)
const WORKFLOW_SESSION_STORAGE_KEY = "posetestbot.workflowSessions.v1"
const CUSTOM_RUN_STORAGE_KEY = "posetestbot.customRunFolders.v1"

function loadCustomRunFolders() {
  try {
    const stored = JSON.parse(localStorage.getItem(CUSTOM_RUN_STORAGE_KEY) ?? "[]") as unknown
    if (!Array.isArray(stored)) return new Set<string>()
    return new Set(stored.filter((value): value is string => typeof value === "string").slice(-20))
  } catch {
    return new Set<string>()
  }
}

function loadWorkflowSessions() {
  try {
    const stored = JSON.parse(localStorage.getItem(WORKFLOW_SESSION_STORAGE_KEY) ?? "{}") as unknown
    if (!stored || typeof stored !== "object" || Array.isArray(stored)) return {}
    const sessions: Record<string, ActiveWorkflow> = {}
    for (const [runRoot, value] of Object.entries(stored).slice(-50)) {
      const workflow = parseActiveWorkflow(value)
      if (workflow) sessions[runRoot] = workflow
    }
    return sessions
  } catch {
    return {}
  }
}

function sameWorkflow(left: ActiveWorkflow | undefined, right: ActiveWorkflow) {
  return left?.journey === right.journey
    && left.stepId === right.stepId
    && left.status === right.status
}

function isContained(path: string, roots: string[]) {
  const normalize = (value: string) => {
    const segments: string[] = []
    for (const segment of value.split("/")) {
      if (!segment || segment === ".") continue
      if (segment === "..") segments.pop()
      else segments.push(segment)
    }
    return `${value.startsWith("/") ? "/" : ""}${segments.join("/")}`.replace(/\/+$/, "")
  }
  const normalized = normalize(path)
  return roots.some((root) => {
    const base = normalize(root)
    return normalized === base || normalized.startsWith(`${base}/`)
  })
}

export function OperatorProvider({ children }: { children: React.ReactNode }) {
  const bootstrapQuery = useQuery({ queryKey: ["bootstrap"], queryFn: () => api<Bootstrap>("/ui/bootstrap"), staleTime: Infinity })
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: () => api<{ runs: RunIndexItem[] }>("/ui/runs"),
    enabled: Boolean(bootstrapQuery.data),
  })
  const [selectedOverride, setSelectedOverride] = useState(() => ({
    path: localStorage.getItem("posetestbot.selectedRun") ?? "",
    restored: true,
    epoch: 0,
  }))
  const [customRunFolders, setCustomRunFolders] = useState(loadCustomRunFolders)
  const [robotOverride, setRobotOverride] = useState<RobotTarget | null>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("posetestbot.robotTarget") ?? "null") as Partial<RobotTarget> | null
      return saved && typeof saved.ip === "string" && Number.isInteger(saved.port)
        ? { ip: saved.ip, port: Number(saved.port) }
        : null
    } catch { return null }
  })
  const [workflowSessions, setWorkflowSessions] = useState<Record<string, ActiveWorkflow>>(loadWorkflowSessions)

  const bootstrap = bootstrapQuery.data
  const runs = useMemo(() => runsQuery.data?.runs ?? [], [runsQuery.data])
  const restoredSelectionIsStale = selectedOverride.restored
    && Boolean(selectedOverride.path)
    && Boolean(runsQuery.data)
    && !runs.some((run) => run.path === selectedOverride.path)
    && !customRunFolders.has(selectedOverride.path)
  useEffect(() => {
    if (!restoredSelectionIsStale) return
    localStorage.removeItem("posetestbot.selectedRun")
  }, [restoredSelectionIsStale])
  const selectedRun = bootstrap
    ? selectedOverride.path
      && !restoredSelectionIsStale
      && isContained(selectedOverride.path, bootstrap.allowed_run_roots)
      ? selectedOverride.path
      : runs.find((run) => run.config_valid)?.path ?? bootstrap.default_run_root
    : ""
  const robotTarget = robotOverride ?? bootstrap?.robot ?? null
  const currentWorkflow = workflowSessions[selectedRun] ?? null
  const rememberWorkflowStep = useCallback((journey: WorkflowJourneyId, stepId: string, status: WorkflowProgressStatus) => {
    const workflow = activeWorkflow(journey, stepId, status)
    if (!workflow || !selectedRun) return
    setWorkflowSessions((current) => {
      if (sameWorkflow(current[selectedRun], workflow)) return current
      const next = { ...current, [selectedRun]: workflow }
      try {
        localStorage.setItem(WORKFLOW_SESSION_STORAGE_KEY, JSON.stringify(next))
      } catch {
        // A denied browser-local write must not block the operator workflow.
      }
      return next
    })
  }, [selectedRun])

  if (bootstrapQuery.isError || runsQuery.isError) {
    const error = bootstrapQuery.error ?? runsQuery.error
    return <div className="grid min-h-screen place-items-center bg-background p-8 text-destructive">{error instanceof Error ? error.message : "Unable to load console bootstrap"}</div>
  }
  if (bootstrapQuery.isPending || runsQuery.isPending || !bootstrap || !selectedRun || !robotTarget) {
    return <div className="grid min-h-screen place-items-center bg-background text-sm text-muted-foreground">Loading operator console…</div>
  }

  const selectRun = (path: string) => {
    if (!isContained(path, bootstrap.allowed_run_roots)) return false
    if (!runs.some((run) => run.path === path)) {
      setCustomRunFolders((current) => {
        const next = [...current].filter((item) => item !== path).concat(path).slice(-20)
        try {
          localStorage.setItem(CUSTOM_RUN_STORAGE_KEY, JSON.stringify(next))
        } catch {
          // A denied browser-local write must not block run selection.
        }
        return new Set(next)
      })
    }
    setSelectedOverride((current) => ({ path, restored: false, epoch: current.epoch + 1 }))
    localStorage.setItem("posetestbot.selectedRun", path)
    return true
  }
  const setRobotTarget = (target: RobotTarget) => {
    setRobotOverride(target)
    localStorage.setItem("posetestbot.robotTarget", JSON.stringify(target))
  }

  return <OperatorContext.Provider value={{ bootstrap, runs, selectedRun, selectedRunEpoch: selectedOverride.epoch, selectRun, robotTarget, setRobotTarget, currentWorkflow, rememberWorkflowStep }}>{children}</OperatorContext.Provider>
}

export function useOperator() {
  const value = useContext(OperatorContext)
  if (!value) throw new Error("useOperator must be used inside OperatorProvider")
  return value
}
