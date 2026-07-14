import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Navigate, useNavigate, useParams } from "react-router-dom"
import { ListTree, Play, RefreshCw } from "lucide-react"
import { toast } from "sonner"
import { PageHeader } from "@/components/page-header"
import { EmptyState } from "@/components/empty-state"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { api, errorMessage, query } from "@/lib/api"
import type { Overview, PipelineStage, RunConfig } from "@/lib/contracts"
import { useOperator } from "@/providers/operator-provider"
import { CaptureGate } from "@/features/workflow/capture-gate"
import { RunSetup } from "@/features/workflow/run-setup"
import { StageForm } from "@/features/workflow/stage-form"

const phases = [
  { id: "setup", label: "Setup" },
  { id: "preflight", label: "Preflight" },
  { id: "capture", label: "Capture" },
  { id: "sync", label: "Sync" },
  { id: "calibration", label: "Calibration" },
  { id: "bop-export", label: "BOP Export" },
]

const stagePhase: Record<string, string> = {
  run_preflight: "preflight",
  hardware_status: "preflight",
  capture_plan_preflight: "preflight",
  capture_plan: "capture",
  capture_execution_plan: "capture",
  capture_execution: "capture",
  realsense_capture_smoke: "capture",
  sync_run: "sync",
  sync_quality: "sync",
  blenderproc_prepare: "bop-export",
  blenderproc_render: "bop-export",
  bop_export: "bop-export",
  rewrite_gate: "bop-export",
}

function phaseFor(stage: PipelineStage) {
  if (stagePhase[stage.id]) return stagePhase[stage.id]
  if (stage.id.includes("calibration") || stage.id.startsWith("aruco") || stage.id.includes("rectification") || stage.id.includes("intrinsic")) return "calibration"
  return null
}

export function WorkflowPage() {
  const { phase } = useParams()
  const navigate = useNavigate()
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const current = phases.some((item) => item.id === phase) ? phase! : "setup"
  const stages = useQuery({ queryKey: ["pipeline", "stages"], queryFn: () => api<{ stages: PipelineStage[] }>("/pipeline/stages") })
  const overview = useQuery({ queryKey: ["overview", selectedRun], queryFn: () => api<Overview>(query("/ui/overview", { run_root: selectedRun })) })
  const config = useQuery({ queryKey: ["run-config", selectedRun], queryFn: () => api<{ config: RunConfig }>(query("/run-config", { run_root: selectedRun })), retry: false })
  const selectedStages = stages.data?.stages.filter((stage) => phaseFor(stage) === current && stage.id !== "capture_execution") ?? []
  const queueConfig = useMutation({
    mutationFn: () => api<{ job_id: string }>("/pipeline/run-config", { method: "POST", body: JSON.stringify({ run_root: selectedRun }) }),
    onSuccess: (data) => { toast.success("Plan-only sequence queued", { description: `Job ${data.job_id}` }); queryClient.invalidateQueries({ queryKey: ["jobs"] }); queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["artifacts", selectedRun] }) },
    onError: (error) => toast.error("Sequence was not queued", { description: errorMessage(error) }),
  })
  const artifactStatus = (stageId: string) => overview.data?.steps.find((step) => step.stage_id === stageId)?.status

  if (phase && !phases.some((item) => item.id === phase)) return <Navigate to="/workflow/setup" replace />

  return <div className="space-y-6">
    <PageHeader eyebrow="Acquisition pipeline" title="Workflow" description="Artifact-backed phases, friendly stage controls, and a single deliberate path to physical capture." actions={<><Button variant="outline" onClick={() => queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] })}><RefreshCw />Refresh evidence</Button>{config.data?.config.pipeline.plan_only && <Button onClick={() => queueConfig.mutate()} disabled={queueConfig.isPending}><Play />Queue configured plan</Button>}</>} />
    {config.data?.config && !config.data.config.pipeline.plan_only && <div className="rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm"><strong>Execution config detected.</strong> Non-plan-only capture sequences cannot be queued from run config. Use Advanced Capture on the Capture phase.</div>}
    <Tabs value={current} onValueChange={(value) => navigate(`/workflow/${value}`)}><TabsList className="grid h-auto w-full grid-cols-6">{phases.map((item) => <TabsTrigger key={item.id} value={item.id} className="py-2.5">{item.label}</TabsTrigger>)}</TabsList></Tabs>

    {current === "setup" ? <RunSetup /> : stages.isPending ? <div className="grid grid-cols-2 gap-4"><Skeleton className="h-72" /><Skeleton className="h-72" /></div> : !overview.data?.config ? <EmptyState icon={ListTree} title="Configure the run first" description="Stages need a valid run_config.json. Setup defaults to real_full_capture_validation in plan-only mode." action={<Button onClick={() => navigate("/workflow/setup")}>Open setup</Button>} /> : <div className="space-y-4">
      {current === "capture" && <CaptureGate />}
      <div className="grid grid-cols-2 gap-4">{selectedStages.map((stage) => <StageForm key={stage.id} stage={stage} artifactStatus={artifactStatus(stage.id)} />)}</div>
      {selectedStages.length === 0 && <Card><CardContent className="py-12 text-center text-sm text-muted-foreground">No stages are registered for this phase.</CardContent></Card>}
    </div>}
  </div>
}
