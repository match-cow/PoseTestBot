import { useMutation, useQueryClient } from "@tanstack/react-query"
import { ChevronDown, Play, TriangleAlert } from "lucide-react"
import { Link } from "react-router-dom"
import { toast } from "sonner"
import { HelpTip } from "@/components/help-tip"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { api, errorMessage } from "@/lib/api"
import type { PipelineStage } from "@/lib/contracts"
import { StageForm } from "@/features/workflow/stage-form"

const stageGroups = [
  { id: "readiness", label: "Planning and readiness", match: new Set(["run_preflight", "hardware_status", "capture_plan", "capture_plan_preflight", "capture_execution_plan", "realsense_capture_smoke"]) },
  { id: "sync", label: "Synchronization", match: new Set(["sync_run", "sync_quality"]) },
  { id: "calibration", label: "Calibration internals", match: new Set(["calibration_preflight", "calibration_target_import", "aruco", "aruco_detection", "intrinsic_calibration", "aruco_pose", "aruco_coverage", "calibration_observations", "calibration_candidates", "calibration_solver", "calibration_validation", "camera_rectification"]) },
  { id: "dataset", label: "Dataset preparation and export", match: new Set(["blenderproc_prepare", "blenderproc_render", "bop_export"]) },
  { id: "audit", label: "Audit", match: new Set(["rewrite_gate", "rewrite_status"]) },
]

export interface AdvancedStageToolsProps {
  runRoot: string
  stages: PipelineStage[]
  artifactStatus: (stageId: string) => string | undefined
  configuredSequence?: { id: string; planOnly: boolean } | null
}

export function AdvancedStageTools({ runRoot, stages, artifactStatus, configuredSequence }: AdvancedStageToolsProps) {
  const queryClient = useQueryClient()
  const queueConfig = useMutation({
    mutationFn: () => api<{ job_id: string }>("/pipeline/run-config", { method: "POST", body: JSON.stringify({ run_root: runRoot }) }),
    onSuccess: (data) => {
      toast.success("Configured plan queued", { description: `Job ${data.job_id}` })
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
      void queryClient.invalidateQueries({ queryKey: ["overview", runRoot] })
    },
    onError: (error) => toast.error("Configured plan was not queued", { description: errorMessage(error) }),
  })
  const visibleStages = stages.filter((stage) => stage.id !== "capture_execution")
  const assigned = new Set(stageGroups.flatMap((group) => [...group.match]))
  const groups = [
    ...stageGroups.map((group) => ({ ...group, stages: visibleStages.filter((stage) => group.match.has(stage.id)) })),
    { id: "other", label: "Other registered stages", match: new Set<string>(), stages: visibleStages.filter((stage) => !assigned.has(stage.id)) },
  ].filter((group) => group.stages.length > 0)

  return <div className="space-y-5" data-testid="advanced-stage-tools">
    <Card className="border-warning/35 bg-warning/5">
      <CardHeader><CardTitle className="flex items-center gap-2 text-base"><TriangleAlert aria-hidden="true" className="size-4 text-warning" />Expert controls</CardTitle><CardDescription>These are individual implementation stages. They do not enforce the guided workflow order and may produce incomplete evidence when run out of sequence.</CardDescription></CardHeader>
      <CardContent className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-xs leading-relaxed text-muted-foreground">Physical capture is intentionally absent here; use a guided journey for its fresh safety gate. Every queued stage continues after navigation; monitor its status, log, and cancellation in <Link className="font-semibold text-primary-strong underline-offset-4 hover:underline" to="/jobs">Jobs</Link>.</div>
        {configuredSequence && <div className="flex shrink-0 flex-col items-end gap-1.5"><div className="flex items-center gap-2"><span className="font-mono text-[10px] text-muted-foreground">{configuredSequence.id}</span><HelpTip label="configured plan">A configured plan is the low-level sequence saved in run_config.json. Guided journeys present its operator-facing purpose instead.</HelpTip><Button size="sm" variant="outline" disabled={!configuredSequence.planOnly || queueConfig.isPending} onClick={() => queueConfig.mutate()}><Play aria-hidden="true" />{queueConfig.isPending ? "Queueing…" : "Queue configured plan"}</Button></div>{!configuredSequence.planOnly && <p className="max-w-xs text-right text-[10px] text-muted-foreground">This saved sequence is an execution plan. Run it through the guided workflow so physical safety gates remain visible.</p>}</div>}
      </CardContent>
    </Card>
    {groups.map((group, index) => <details key={group.id} open={index === 0} className="group rounded-xl border bg-card">
      <summary className="flex cursor-pointer list-none items-center justify-between px-5 py-4 text-sm font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring/60">{group.label}<span className="flex items-center gap-2 text-xs font-normal text-muted-foreground">{group.stages.length} stage{group.stages.length === 1 ? "" : "s"}<ChevronDown aria-hidden="true" className="size-4 transition group-open:rotate-180" /></span></summary>
      <div className="grid gap-4 border-t p-4 xl:grid-cols-2">{group.stages.map((stage) => <StageForm key={stage.id} stage={stage} artifactStatus={artifactStatus(stage.id)} />)}</div>
    </details>)}
  </div>
}
