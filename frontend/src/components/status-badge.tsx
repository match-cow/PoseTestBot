import { Badge } from "@/components/ui/badge"

function variant(status: string | null | undefined) {
  const value = status?.toLowerCase() ?? "unknown"
  if (["complete", "connected", "ok", "ready", "running", "succeeded", "valid", "pass", "passed"].includes(value)) return "success" as const
  if (["error", "failed", "blocked", "invalid", "disconnected", "canceled"].includes(value)) return "destructive" as const
  if (["warning", "stale", "queued", "canceling", "in_progress", "waiting", "receiving"].includes(value)) return "warning" as const
  return "outline" as const
}

export function StatusBadge({ status, children }: { status?: string | null; children?: React.ReactNode }) {
  return <Badge variant={variant(status)}>{children ?? status?.replaceAll("_", " ") ?? "unknown"}</Badge>
}
