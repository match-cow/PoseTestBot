import { useMutation, useQueryClient } from "@tanstack/react-query"
import { ArrowRight, Check, Circle, LoaderCircle, Play, ShieldCheck } from "lucide-react"
import { Link } from "react-router-dom"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { api, errorMessage } from "@/lib/api"

interface DatasetProcessingProps {
  runRoot: string
  ready: boolean
  captureComplete: boolean
  syncComplete: boolean
  syncQualityComplete: boolean
  calibrationComplete: boolean
  exportComplete: boolean
  onReviewReadiness: () => void
}

const outcomes = [
  { id: "sync", label: "Match camera frames to robot poses" },
  { id: "quality", label: "Verify timestamp and match quality" },
  { id: "calibration", label: "Validate calibration and rectify RGB-D frames" },
  { id: "export", label: "Copy models and write the annotation-free BOP dataset" },
]

export function DatasetProcessing({ runRoot, ready, captureComplete, syncComplete, syncQualityComplete, calibrationComplete, exportComplete, onReviewReadiness }: DatasetProcessingProps) {
  const queryClient = useQueryClient()
  const process = useMutation({
    mutationFn: () => api<{ job_id: string }>("/pipeline/run-config", {
      method: "POST",
      body: JSON.stringify({ run_root: runRoot }),
    }),
    onSuccess: (data) => {
      toast.success("Dataset processing queued", { description: `Job ${data.job_id}` })
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
      void queryClient.invalidateQueries({ queryKey: ["overview", runRoot] })
      void queryClient.invalidateQueries({ queryKey: ["run-config", runRoot] })
    },
    onError: (error) => toast.error("Dataset processing was not queued", { description: errorMessage(error) }),
  })
  const complete = [syncComplete, syncQualityComplete, calibrationComplete, exportComplete]
  const blocked = !ready || !captureComplete

  return <Card data-testid="dataset-processing" className="border-primary/25">
    <CardHeader>
      <CardTitle className="text-base">Process the recorded dataset</CardTitle>
      <CardDescription>One queued job synchronizes, rectifies, and writes the image/model BOP dataset. It does not run BlenderProc or generate rendered GT/masks. Raw camera frames and robot poses are never renamed or replaced.</CardDescription>
    </CardHeader>
    <CardContent className="space-y-5">
      <ol className="grid gap-2 sm:grid-cols-2" aria-label="Automatic dataset processing">
        {outcomes.map((outcome, index) => <li key={outcome.id} className="flex items-start gap-3 rounded-lg border bg-muted/20 p-3">
          <span className={`grid size-6 shrink-0 place-items-center rounded-full ${complete[index] ? "bg-success text-white" : "bg-muted text-muted-foreground"}`}>{complete[index] ? <Check aria-hidden="true" className="size-3.5" /> : <Circle aria-hidden="true" className="size-3" />}</span>
          <span><span className="block font-mono text-[10px] font-bold text-muted-foreground">{index + 1}</span><span className="mt-0.5 block text-xs font-semibold">{outcome.label}</span></span>
        </li>)}
      </ol>
      {blocked ? <div className="flex flex-col gap-3 rounded-lg border border-warning/35 bg-warning/5 p-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-2 text-xs text-muted-foreground"><ShieldCheck aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-warning-foreground" /><span>{!ready ? "Complete the readiness step before processing this run." : "Record the object dataset before processing it."}</span></div>
        {!ready && <Button type="button" variant="outline" size="sm" onClick={onReviewReadiness}>Review readiness</Button>}
      </div> : <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-xs text-muted-foreground">Calibration validation is automatic here; there is no second operator preflight. Rendered annotations remain a separate optional workflow.</p>
        <div className="flex flex-wrap gap-2">
          {process.isSuccess && <Button asChild variant="outline"><Link to="/jobs">Open job <ArrowRight aria-hidden="true" /></Link></Button>}
          <Button type="button" onClick={() => process.mutate()} disabled={process.isPending}>
            {process.isPending ? <LoaderCircle aria-hidden="true" className="animate-spin" /> : <Play aria-hidden="true" />}
            {process.isPending ? "Queueing…" : exportComplete ? "Rebuild dataset" : "Process and export dataset"}
          </Button>
        </div>
      </div>}
    </CardContent>
  </Card>
}
