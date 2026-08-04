import { AlertTriangle, Check, Circle, CircleDot, LoaderCircle, LockKeyhole } from "lucide-react"
import { HelpTip } from "@/components/help-tip"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import type { WorkflowProgressStatus } from "@/lib/workflow-session"

export type WorkflowStepStatus = WorkflowProgressStatus

export interface WorkflowStepDefinition {
  id: string
  number: number
  title: string
  summary: string
  status: WorkflowStepStatus
  required?: boolean
}

const statusPresentation: Record<WorkflowStepStatus, { label: string; className: string; icon: typeof Circle }> = {
  complete: { label: "Complete", className: "border-success/30 bg-success/10 text-success", icon: Check },
  current: { label: "Current step", className: "border-primary/45 bg-primary/10 text-primary-strong", icon: CircleDot },
  ready: { label: "Ready", className: "border-primary/30 bg-primary/5 text-primary-strong", icon: CircleDot },
  blocked: { label: "Needs attention", className: "border-destructive/30 bg-destructive/5 text-destructive", icon: LockKeyhole },
  running: { label: "Running", className: "border-warning/35 bg-warning/10 text-warning-foreground", icon: LoaderCircle },
  not_started: { label: "Not started", className: "border-border bg-muted/40 text-muted-foreground", icon: Circle },
}

export function WorkflowStatus({ status, compact = false }: { status: WorkflowStepStatus; compact?: boolean }) {
  const presentation = statusPresentation[status]
  const Icon = presentation.icon
  return <span className={cn("inline-flex items-center gap-1.5 rounded-full border font-semibold", compact ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs", presentation.className)}>
    <Icon aria-hidden="true" className={cn("size-3", status === "running" && "animate-spin")} />
    {presentation.label}
  </span>
}

export function WorkflowStepper({ steps, selectedStep, onSelect }: { steps: WorkflowStepDefinition[]; selectedStep?: string | null; onSelect: (stepId: string) => void }) {
  return <nav aria-label="Workflow steps" className="rounded-xl border bg-card p-3 xl:sticky xl:top-[88px]">
    <ol className="flex gap-0 overflow-x-auto pb-1 xl:block xl:overflow-visible xl:pb-0">
      {steps.map((step, index) => {
        const active = selectedStep ? step.id === selectedStep : step.status === "current" || step.status === "running"
        return <li key={step.id} className="flex min-w-[210px] flex-1 items-center xl:min-w-0 xl:flex-col xl:items-stretch">
          <button
            type="button"
            aria-current={active ? "step" : undefined}
            onClick={() => onSelect(step.id)}
            className={cn("w-full rounded-lg border border-transparent p-3 text-left transition-colors hover:border-border hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60", active && "border-primary/40 bg-primary/5")}
          >
            <span className="flex items-start gap-3">
              <span data-workflow-step-number className={cn("grid size-7 shrink-0 place-items-center rounded-full border font-mono text-xs font-bold", step.status === "complete" ? "border-success bg-success text-success-foreground" : active ? "border-primary bg-primary text-primary-foreground" : "border-border bg-muted text-muted-foreground")}>
                {step.status === "complete" ? <Check aria-hidden="true" className="size-3.5" /> : step.number}
              </span>
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-1.5 text-sm font-semibold">{step.title}{step.required === false && <span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground">Optional</span>}</span>
                <span className="mt-1 line-clamp-2 block text-[11px] leading-relaxed text-muted-foreground">{step.summary}</span>
                <span className="mt-2 block"><WorkflowStatus status={step.status} compact /></span>
              </span>
            </span>
          </button>
          {index < steps.length - 1 && <span data-workflow-step-connector aria-hidden="true" className="mx-1 h-px min-w-5 flex-1 bg-border xl:ml-[26px] xl:mr-auto xl:h-5 xl:w-px xl:min-w-0 xl:flex-none" />}
        </li>
      })}
    </ol>
  </nav>
}

export interface WorkflowStepCardProps {
  id: string
  number?: number
  title: string
  description: string
  status?: WorkflowStepStatus
  required?: boolean
  help?: React.ReactNode
  children: React.ReactNode
  className?: string
}

export function WorkflowStepCard({ id, number, title, description, status = "not_started", required = true, help, children, className }: WorkflowStepCardProps) {
  return <section id={`workflow-step-${id}`} data-workflow-step={id} className={cn("scroll-mt-24 space-y-3", className)} aria-labelledby={`workflow-step-${id}-title`}>
    <Card className={cn(status === "current" && "border-primary/45", status === "blocked" && "border-destructive/30", status === "running" && "border-warning/40")}>
      <CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <span className={cn("grid size-8 shrink-0 place-items-center rounded-full border font-mono text-xs font-bold", required ? "border-primary/45 bg-primary/10 text-primary-strong" : "border-border bg-muted text-muted-foreground")}>{number ?? "·"}</span>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h2 id={`workflow-step-${id}-title`} className="text-base font-semibold">{title}</h2>
              <Badge variant={required ? "default" : "outline"}>{required ? "Required" : "Optional"}</Badge>
              {help && <HelpTip label={title}>{help}</HelpTip>}
            </div>
            <p className="mt-1 max-w-4xl text-sm leading-relaxed text-muted-foreground">{description}</p>
          </div>
        </div>
        <WorkflowStatus status={status} />
      </CardContent>
    </Card>
    {children}
  </section>
}

export type RequirementStatus = "met" | "missing" | "warning" | "checking"

export interface WorkflowRequirement {
  id: string
  label: string
  description: string
  status: RequirementStatus
  required?: boolean
  fixLabel?: string
  onFix?: () => void
}

export function RequirementList({ requirements }: { requirements: WorkflowRequirement[] }) {
  return <ul className="grid gap-2 md:grid-cols-2" aria-label="Readiness requirements">
    {requirements.map((requirement) => {
      const Icon = requirement.status === "met" ? Check : requirement.status === "checking" ? LoaderCircle : AlertTriangle
      return <li key={requirement.id} className={cn("flex items-start gap-3 rounded-lg border p-3", requirement.status === "met" ? "border-success/25 bg-success/5" : requirement.status === "warning" ? "border-warning/30 bg-warning/5" : requirement.status === "checking" ? "bg-muted/30" : "border-destructive/30 bg-destructive/5")}>
        <span className={cn("mt-0.5 grid size-6 shrink-0 place-items-center rounded-full", requirement.status === "met" ? "bg-success/15 text-success" : requirement.status === "warning" ? "bg-warning/15 text-warning-foreground" : requirement.status === "checking" ? "bg-muted text-muted-foreground" : "bg-destructive/10 text-destructive")}><Icon aria-hidden="true" className={cn("size-3.5", requirement.status === "checking" && "animate-spin")} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2"><span className="text-sm font-semibold">{requirement.label}</span><span className="text-[10px] font-bold uppercase tracking-wide text-muted-foreground">{requirement.required === false ? "Optional" : "Required"}</span></div>
          <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{requirement.description}</p>
          {requirement.onFix && requirement.fixLabel && <button type="button" onClick={requirement.onFix} className="mt-2 text-xs font-semibold text-primary-strong underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60">{requirement.fixLabel}</button>}
        </div>
      </li>
    })}
  </ul>
}
