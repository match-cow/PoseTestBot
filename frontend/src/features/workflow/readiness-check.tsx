import { useMutation, useQueryClient } from "@tanstack/react-query"
import { RefreshCw, ShieldCheck } from "lucide-react"
import { toast } from "sonner"
import { HelpTip } from "@/components/help-tip"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { api, errorMessage } from "@/lib/api"
import type { PreflightSummary } from "@/lib/contracts"
import { readinessBlockerCopy } from "@/features/workflow/readiness-copy"
import { RequirementList, type WorkflowRequirement } from "@/features/workflow/workflow-steps"

export interface ReadinessCheckProps {
  runRoot: string
  intent: "calibration" | "dataset"
  preflight?: PreflightSummary | null
  loading?: boolean
  requirements: WorkflowRequirement[]
}

export function readinessSatisfied(preflight: PreflightSummary | null | undefined, requirements: WorkflowRequirement[]) {
  return preflight?.queue_blocker === null && requirements.every((item) => item.required === false || item.status === "met")
}

function blockerDescription(blocker: string | null | undefined, loading: boolean) {
  if (loading) return "Reading the latest durable readiness evidence…"
  return readinessBlockerCopy(blocker).description
}

export function ReadinessCheck({ runRoot, intent, preflight, loading = false, requirements }: ReadinessCheckProps) {
  const queryClient = useQueryClient()
  const ready = readinessSatisfied(preflight, requirements)
  const check = useMutation({
    mutationFn: () => api<{ job_id: string }>("/pipeline/run", {
      method: "POST",
      body: JSON.stringify({ stage: "run_preflight", run_root: runRoot, options: { write: true, check: true } }),
    }),
    onSuccess: (data) => {
      toast.success("Readiness check queued", { description: `Job ${data.job_id}` })
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
      void queryClient.invalidateQueries({ queryKey: ["run-config", runRoot] })
      void queryClient.invalidateQueries({ queryKey: ["overview", runRoot] })
    },
    onError: (error) => toast.error("Readiness check was not queued", { description: errorMessage(error) }),
  })

  const coreRequirement: WorkflowRequirement = {
    id: "run-preflight",
    label: loading ? "Checking saved run readiness" : readinessBlockerCopy(preflight?.queue_blocker).heading,
    description: blockerDescription(preflight?.queue_blocker, loading),
    status: loading ? "checking" : preflight?.queue_blocker === null ? "met" : "missing",
    required: true,
  }

  return <Card data-testid={`${intent}-readiness-check`} className={ready ? "border-success/35" : "border-warning/35"}>
    <CardHeader>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2"><CardTitle className="text-base">One readiness check</CardTitle><HelpTip label="readiness check">This single operator step validates saved configuration, provenance, current device/runtime visibility, and the planned software stages. Capture separately checks live device identity, empty output, and safety immediately before hardware starts.</HelpTip></div>
          <CardDescription className="mt-1">This does not reserve cameras or prove that capture output is still empty. Nothing physical starts here.</CardDescription>
        </div>
        <Button onClick={() => check.mutate()} disabled={check.isPending || loading} variant={ready ? "outline" : "default"}>
          <RefreshCw aria-hidden="true" className={check.isPending ? "animate-spin" : ""} />
          {check.isPending ? "Queueing…" : ready ? "Check again" : "Check readiness"}
        </Button>
      </div>
    </CardHeader>
    <CardContent className="space-y-4">
      <RequirementList requirements={[coreRequirement, ...requirements]} />
      <div className={`flex items-start gap-2 rounded-lg border p-3 text-xs ${ready ? "border-success/30 bg-success/5 text-success" : "border-warning/30 bg-warning/5 text-muted-foreground"}`}>
        <ShieldCheck aria-hidden="true" className="mt-0.5 size-4 shrink-0" />
        <span>{ready ? "All required items are ready. Physical capture remains separately gated by two fresh confirmations." : "Resolve every required item, then run this check again. Optional items never block capture."}</span>
      </div>
    </CardContent>
  </Card>
}
