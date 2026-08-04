import { BookOpen, Boxes, Camera, CircleHelp, Database, ShieldCheck } from "lucide-react"
import { Link } from "react-router-dom"

import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"

interface ConsoleGuideProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const steps = [
  {
    title: "Choose the active run",
    description: "The run selector in the top bar controls every run-owned view and action.",
  },
  {
    title: "Choose an outcome in Workflow",
    description: "Camera calibration and object-dataset recording have separate numbered, required paths.",
  },
  {
    title: "Prepare reusable inputs only when needed",
    description: "Calibration Targets, Workpieces, and Pose Templates author reusable library items; their handoff bars lead back to the correct workflow step.",
  },
  {
    title: "Check readiness, then authorize capture",
    description: "Readiness writes evidence but starts no hardware. Physical capture always asks for fresh camera and real-robot acknowledgements.",
  },
  {
    title: "Monitor background work",
    description: "Queued work continues if you leave its page. Jobs shows status, resource locks, logs, and stop controls.",
  },
]

export function ConsoleGuide({ open, onOpenChange }: ConsoleGuideProps) {
  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent aria-label="Operator console guide">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2 font-display text-xl font-semibold"><BookOpen className="size-5 text-primary-strong" />Operator console guide</SheetTitle>
          <SheetDescription>Use the guided outcome as the primary path. Library, inspection, and job pages support that path without silently changing the active run.</SheetDescription>
        </SheetHeader>

        <div className="min-h-0 flex-1 space-y-6 overflow-y-auto pr-1">
          <ol className="space-y-2" aria-label="Operator process">
            {steps.map((step, index) => (
              <li className="flex gap-3 rounded-lg border bg-muted/20 p-3" key={step.title}>
                <span className="grid size-7 shrink-0 place-items-center rounded-full border border-primary/40 bg-primary/10 font-mono text-xs font-bold text-primary-strong">{index + 1}</span>
                <div><div className="text-sm font-semibold">{step.title}</div><p className="mt-1 text-xs leading-relaxed text-muted-foreground">{step.description}</p></div>
              </li>
            ))}
          </ol>

          <div className="rounded-lg border border-warning/35 bg-warning/10 p-4">
            <div className="flex items-center gap-2 text-sm font-semibold"><ShieldCheck className="size-4 text-warning" />Safety boundary</div>
            <p className="mt-2 text-xs leading-relaxed text-muted-foreground">The console runs on the trusted lab network and exposes deliberate real-robot controls. IIWA STOP is not a safety stop and cannot interrupt active motion. Use the workcell safety system for an emergency.</p>
          </div>

          <section aria-labelledby="guide-terms-heading">
            <h3 id="guide-terms-heading" className="flex items-center gap-2 text-sm font-semibold"><CircleHelp className="size-4 text-primary-strong" />Key terms</h3>
            <dl className="mt-3 divide-y overflow-hidden rounded-lg border text-xs">
              <div className="p-3"><dt className="font-semibold">Run</dt><dd className="mt-1 text-muted-foreground">One folder containing configuration, raw evidence, derived artifacts, and export output for one acquisition intent.</dd></div>
              <div className="p-3"><dt className="font-semibold">Readiness</dt><dd className="mt-1 text-muted-foreground">A saved check of configuration and current prerequisites. It does not reserve devices or authorize capture.</dd></div>
              <div className="p-3"><dt className="font-semibold">Immutable</dt><dd className="mt-1 text-muted-foreground">A published, hash-bound target or template version that later edits cannot silently change.</dd></div>
              <div className="p-3"><dt className="font-semibold">BOP dataset</dt><dd className="mt-1 text-muted-foreground">The portable acquisition output. Estimator execution and result conversion remain in a consumer repository; Inspect can validate an annotation-bearing export against an already compatible BOP19 result CSV.</dd></div>
            </dl>
          </section>
        </div>

        <div className="grid grid-cols-2 gap-2 border-t pt-4">
          <Link onClick={() => onOpenChange(false)} to="/workflow/calibration" className="flex items-center justify-center gap-2 rounded-lg border bg-card px-3 py-2.5 text-xs font-semibold hover:bg-muted"><Camera className="size-4" />Camera calibration</Link>
          <Link onClick={() => onOpenChange(false)} to="/workflow/dataset" className="flex items-center justify-center gap-2 rounded-lg border bg-card px-3 py-2.5 text-xs font-semibold hover:bg-muted"><Database className="size-4" />Object dataset</Link>
          <Link onClick={() => onOpenChange(false)} to="/workpieces" className="flex items-center justify-center gap-2 rounded-lg border bg-card px-3 py-2.5 text-xs font-semibold hover:bg-muted"><Boxes className="size-4" />Workpieces</Link>
          <Link onClick={() => onOpenChange(false)} to="/jobs" className="flex items-center justify-center gap-2 rounded-lg border bg-card px-3 py-2.5 text-xs font-semibold hover:bg-muted"><BookOpen className="size-4" />Jobs & logs</Link>
        </div>
      </SheetContent>
    </Sheet>
  )
}
