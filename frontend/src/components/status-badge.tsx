import { Badge } from "@/components/ui/badge"

export type StatusTone = "informational" | "success" | "warning" | "neutral" | "destructive"

const toneStyle = {
  informational: {
    variant: "outline" as const,
    className: "border-primary/35 bg-primary/10 text-primary-strong",
  },
  success: { variant: "success" as const, className: undefined },
  warning: { variant: "warning" as const, className: undefined },
  neutral: { variant: "outline" as const, className: undefined },
  destructive: { variant: "destructive" as const, className: undefined },
}

export function StatusBadge({ status, tone, children }: { status?: string | null; tone: StatusTone; children?: React.ReactNode }) {
  const style = toneStyle[tone]
  return <Badge variant={style.variant} className={style.className} data-status-tone={tone}>{children ?? status?.replaceAll("_", " ") ?? "unknown"}</Badge>
}
