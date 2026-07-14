import { useQuery } from "@tanstack/react-query"
import { createContext, useContext, useMemo, useState } from "react"
import { api } from "@/lib/api"
import type { Bootstrap, RunIndexItem } from "@/lib/contracts"

interface RobotTarget { ip: string; port: number }
interface OperatorContextValue {
  bootstrap: Bootstrap
  runs: RunIndexItem[]
  selectedRun: string
  selectRun: (path: string) => boolean
  robotTarget: RobotTarget
  setRobotTarget: (target: RobotTarget) => void
}

const OperatorContext = createContext<OperatorContextValue | null>(null)

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
  const [selectedOverride, setSelectedOverride] = useState(() => localStorage.getItem("posetestbot.selectedRun") ?? "")
  const [robotOverride, setRobotOverride] = useState<RobotTarget | null>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem("posetestbot.robotTarget") ?? "null") as Partial<RobotTarget> | null
      return saved && typeof saved.ip === "string" && Number.isInteger(saved.port)
        ? { ip: saved.ip, port: Number(saved.port) }
        : null
    } catch { return null }
  })

  const bootstrap = bootstrapQuery.data
  const runs = useMemo(() => runsQuery.data?.runs ?? [], [runsQuery.data])
  const selectedRun = bootstrap
    ? selectedOverride && isContained(selectedOverride, bootstrap.allowed_run_roots)
      ? selectedOverride
      : runs.find((run) => run.config_valid)?.path ?? bootstrap.default_run_root
    : ""
  const robotTarget = robotOverride ?? bootstrap?.robot ?? null

  if (bootstrapQuery.isError || runsQuery.isError) {
    const error = bootstrapQuery.error ?? runsQuery.error
    return <div className="grid min-h-screen place-items-center bg-background p-8 text-destructive">{error instanceof Error ? error.message : "Unable to load console bootstrap"}</div>
  }
  if (bootstrapQuery.isPending || runsQuery.isPending || !bootstrap || !selectedRun || !robotTarget) {
    return <div className="grid min-h-screen place-items-center bg-background text-sm text-muted-foreground">Loading operator console…</div>
  }

  const selectRun = (path: string) => {
    if (!isContained(path, bootstrap.allowed_run_roots)) return false
    setSelectedOverride(path)
    localStorage.setItem("posetestbot.selectedRun", path)
    return true
  }
  const setRobotTarget = (target: RobotTarget) => {
    setRobotOverride(target)
    localStorage.setItem("posetestbot.robotTarget", JSON.stringify(target))
  }

  return <OperatorContext.Provider value={{ bootstrap, runs, selectedRun, selectRun, robotTarget, setRobotTarget }}>{children}</OperatorContext.Provider>
}

export function useOperator() {
  const value = useContext(OperatorContext)
  if (!value) throw new Error("useOperator must be used inside OperatorProvider")
  return value
}
