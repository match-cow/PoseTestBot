import { ArrowRight, Route } from "lucide-react"
import { Link } from "react-router-dom"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ProcessHandoffProps {
  title: string
  description: string
  to: string
  action: string
  className?: string
}

/** Compact context for pages that support, but do not replace, a guided workflow step. */
export function ProcessHandoff({ title, description, to, action, className }: ProcessHandoffProps) {
  return (
    <aside
      aria-label="Where this page fits in the operator workflow"
      className={cn("flex flex-col gap-3 rounded-xl border bg-muted/35 px-4 py-3 sm:flex-row sm:items-center sm:justify-between", className)}
    >
      <div className="flex min-w-0 items-start gap-3">
        <span className="grid size-8 shrink-0 place-items-center rounded-lg border bg-card text-primary-strong">
          <Route aria-hidden="true" className="size-4" />
        </span>
        <div className="min-w-0">
          <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-muted-foreground">Workflow handoff</div>
          <div className="mt-0.5 text-sm font-semibold">{title}</div>
          <p className="mt-0.5 max-w-4xl text-xs leading-relaxed text-muted-foreground">{description}</p>
        </div>
      </div>
      <Button asChild variant="outline" size="sm" className="shrink-0 bg-card">
        <Link to={to}>{action}<ArrowRight aria-hidden="true" /></Link>
      </Button>
    </aside>
  )
}
