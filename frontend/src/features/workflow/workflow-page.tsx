import { useEffect } from "react"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Link, Navigate, useNavigate, useParams, useSearchParams } from "react-router-dom"
import { ArrowLeft, ArrowRight, Boxes, Camera, Database, Grid3X3, ListTree, RefreshCw, Settings2, Sparkles } from "lucide-react"
import { HelpTip } from "@/components/help-tip"
import { PageHeader } from "@/components/page-header"
import { EmptyState } from "@/components/empty-state"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { api, query } from "@/lib/api"
import type { Overview, PipelineStage, PreflightSummary, RunConfig } from "@/lib/contracts"
import { useOperator } from "@/providers/operator-provider"
import { AdvancedStageTools } from "@/features/workflow/advanced-stage-tools"
import { CalibrationWorkflow } from "@/features/workflow/calibration-workflow"
import { CaptureGate } from "@/features/workflow/capture-gate"
import { GroundTruthWorkflow } from "@/features/workflow/ground-truth-workflow"
import { DatasetProcessing } from "@/features/workflow/dataset-processing"
import { ReadinessCheck, readinessSatisfied } from "@/features/workflow/readiness-check"
import { RunSetup } from "@/features/workflow/run-setup"
import { WorkflowStepCard, WorkflowStepper, type WorkflowRequirement, type WorkflowStepDefinition } from "@/features/workflow/workflow-steps"

type JourneyId = "calibration" | "dataset"
type WorkflowPageId = "setup" | JourneyId | "advanced"
type RunConfigResponse = { config: RunConfig; preflight: PreflightSummary }

const legacyAliases: Record<string, { page: JourneyId; step: string }> = {
  preflight: { page: "calibration", step: "readiness" },
  capture: { page: "calibration", step: "capture" },
  sync: { page: "dataset", step: "sync" },
  "ground-truth": { page: "dataset", step: "template" },
  "bop-export": { page: "dataset", step: "export" },
}

const calibrationOutline = [
  "Configure the run and cameras",
  "Choose the printed calibration grid",
  "Check readiness",
  "Record calibration images",
  "Calculate, review, and publish",
]

const datasetOutline = [
  "Configure cameras and select calibration",
  "Choose the object template and placement",
  "Check readiness",
  "Record the object dataset",
  "Synchronize and verify frames",
  "Export the BOP dataset",
]

function stepStatuses(completed: boolean[]): Array<WorkflowStepDefinition["status"]> {
  const firstIncomplete = completed.findIndex((value) => !value)
  return completed.map((value, index) => value ? "complete" : index === firstIncomplete ? "current" : "blocked")
}

function WorkflowChoice({ to, icon: Icon, title, description, steps, output }: { to: string; icon: typeof Camera; title: string; description: string; steps: string[]; output: string }) {
  return <Card className="group flex h-full flex-col transition-colors hover:border-primary/55">
    <CardHeader>
      <div className="mb-2 grid size-11 place-items-center rounded-xl bg-primary/10 text-primary-strong"><Icon aria-hidden="true" className="size-5" /></div>
      <CardTitle>{title}</CardTitle>
      <CardDescription className="leading-relaxed">{description}</CardDescription>
    </CardHeader>
    <CardContent className="flex flex-1 flex-col">
      <ol className="space-y-2 border-l border-border pl-4">{steps.map((step, index) => <li key={step} className="relative text-xs"><span aria-hidden="true" className="absolute -left-[21px] top-0 grid size-3 place-items-center rounded-full bg-muted ring-4 ring-card" /><span className="font-mono text-[10px] font-bold text-muted-foreground">{String(index + 1).padStart(2, "0")}</span><span className="ml-2 text-foreground">{step}</span></li>)}</ol>
      <div className="mt-5 rounded-lg bg-muted/60 p-3 text-xs"><span className="font-semibold">Result:</span> <span className="text-muted-foreground">{output}</span></div>
      <Button asChild className="mt-5 w-full"><Link to={to}>Start this workflow <ArrowRight aria-hidden="true" /></Link></Button>
    </CardContent>
  </Card>
}

function JourneyNavigation({ current }: { current: JourneyId | "advanced" }) {
  return <nav aria-label="Workflow type" className="flex flex-wrap gap-2 rounded-xl border bg-card p-2">
    <Button asChild variant={current === "calibration" ? "default" : "ghost"}><Link to="/workflow/calibration"><Camera aria-hidden="true" />Camera calibration</Link></Button>
    <Button asChild variant={current === "dataset" ? "default" : "ghost"}><Link to="/workflow/dataset"><Database aria-hidden="true" />Object dataset</Link></Button>
    <Button asChild variant={current === "advanced" ? "default" : "ghost"} className="sm:ml-auto"><Link to="/workflow/advanced"><Settings2 aria-hidden="true" />Advanced tools</Link></Button>
  </nav>
}

function OptionalAction({ icon: Icon, title, description, to, action }: { icon: typeof Sparkles; title: string; description: string; to: string; action: string }) {
  return <Card className="border-dashed bg-muted/15"><CardContent className="flex flex-col gap-4 py-4 sm:flex-row sm:items-center sm:justify-between"><div className="flex items-start gap-3"><span className="grid size-8 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground"><Icon aria-hidden="true" className="size-4" /></span><div><div className="flex items-center gap-2 text-sm font-semibold">{title}<span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground">Optional</span></div><p className="mt-1 text-xs leading-relaxed text-muted-foreground">{description}</p></div></div><Button asChild variant="outline" size="sm"><Link to={to}>{action}<ArrowRight aria-hidden="true" /></Link></Button></CardContent></Card>
}

function JourneyShell({ journey, steps, selectedStep, onSelectStep, children }: { journey: JourneyId; steps: WorkflowStepDefinition[]; selectedStep: string | null; onSelectStep: (stepId: string) => void; children: React.ReactNode }) {
  const meta = journey === "calibration" ? {
    eyebrow: "Guided workflow · reusable camera geometry",
    title: "Calibrate cameras",
    description: "Record a known printed ArUco grid, compare camera and robot-camera solutions, then explicitly publish reusable calibration profiles.",
  } : {
    eyebrow: "Guided workflow · acquisition dataset",
    title: "Record an object-template dataset",
    description: "Use a previously published calibration and a confirmed physical object template to record, synchronize, and export a BOP dataset.",
  }
  return <div className="space-y-6">
    <div className="flex items-center gap-2 text-xs"><Button asChild variant="ghost" size="sm"><Link to="/workflow/setup"><ArrowLeft aria-hidden="true" />Choose workflow</Link></Button><span className="text-muted-foreground">/</span><span className="font-semibold">{meta.title}</span></div>
    <PageHeader eyebrow={meta.eyebrow} title={meta.title} description={meta.description} />
    <JourneyNavigation current={journey} />
    <div className="grid items-start gap-6 xl:grid-cols-[270px_minmax(0,1fr)]">
      <WorkflowStepper steps={steps} onSelect={onSelectStep} />
      <div className="min-w-0 space-y-9" data-selected-step={selectedStep ?? ""}>{children}</div>
    </div>
  </div>
}

function artifact(overview: Overview | undefined, path: string) {
  return overview?.sidebar.flatMap((section) => section.artifacts).find((item) => item.path === path)
}

function artifactComplete(overview: Overview | undefined, path: string) {
  const item = artifact(overview, path)
  return Boolean(item?.exists && ["complete", "succeeded", "ok", "warning", "valid", "ready"].includes(item.status ?? ""))
}

export function WorkflowPage() {
  const { phase } = useParams()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const page = (phase ?? "setup") as WorkflowPageId
  const selectedStep = searchParams.get("step")
  const overview = useQuery({
    queryKey: ["overview", selectedRun],
    queryFn: () => api<Overview>(query("/ui/overview", { run_root: selectedRun })),
    refetchInterval: ["calibration", "dataset"].includes(page) ? 2_000 : false,
  })
  const config = useQuery({
    queryKey: ["run-config", selectedRun],
    queryFn: () => api<RunConfigResponse>(query("/run-config", { run_root: selectedRun })),
    retry: false,
    refetchInterval: (state) => state.state.data?.preflight.queue_blocker ? 2_000 : false,
  })
  const stages = useQuery({ queryKey: ["pipeline", "stages"], queryFn: () => api<{ stages: PipelineStage[] }>("/pipeline/stages"), enabled: page === "advanced" })

  useEffect(() => {
    if (!selectedStep || !["calibration", "dataset"].includes(page)) return
    const frame = window.requestAnimationFrame(() => document.getElementById(`workflow-step-${selectedStep}`)?.scrollIntoView({ behavior: "smooth", block: "start" }))
    return () => window.cancelAnimationFrame(frame)
  }, [overview.isPending, page, selectedStep])

  const selectStep = (stepId: string) => {
    setSearchParams({ step: stepId }, { replace: true })
    window.requestAnimationFrame(() => document.getElementById(`workflow-step-${stepId}`)?.scrollIntoView({ behavior: "smooth", block: "start" }))
  }
  const refresh = () => queryClient.invalidateQueries({ predicate: (item) => ["overview", "run-config", "calibration", "pose-template-run"].includes(String(item.queryKey[0])) })

  if (phase && legacyAliases[phase]) return <Navigate to={`/workflow/${legacyAliases[phase].page}?step=${legacyAliases[phase].step}`} replace />
  if (!["setup", "calibration", "dataset", "advanced"].includes(page)) return <Navigate to="/workflow/setup" replace />

  if (page === "setup") return <div className="space-y-6">
    <PageHeader eyebrow="Acquisition workflows" title="What do you want to do?" description="Choose the outcome first. Each guided workflow shows the required order, keeps optional work off the critical path, and exposes low-level stages only under Advanced tools." actions={<Button variant="outline" onClick={refresh}><RefreshCw aria-hidden="true" />Refresh evidence</Button>} />
    <div className="grid gap-5 lg:grid-cols-2">
      <WorkflowChoice to="/workflow/calibration" icon={Camera} title="Calibrate cameras" description="Use a printed calibration grid to calculate and publish camera intrinsics and robot-camera transforms." steps={calibrationOutline} output="A reviewed, reusable calibration profile for every selected camera." />
      <WorkflowChoice to="/workflow/dataset" icon={Boxes} title="Record an object dataset" description="Select a prior calibration and a physical pose template, then record and export an acquisition dataset." steps={datasetOutline} output="Synchronized RGB-D evidence and a BOP dataset with object poses." />
    </div>
    <Card className="border-dashed"><CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between"><div><div className="text-sm font-semibold">Need an individual implementation stage?</div><p className="mt-1 text-xs text-muted-foreground">Advanced tools retain direct stage controls for diagnostics and recovery.</p></div><Button asChild variant="outline"><Link to="/workflow/advanced"><Settings2 aria-hidden="true" />Open Advanced tools</Link></Button></CardContent></Card>
  </div>

  if (overview.isPending || config.isPending) return <div className="space-y-6"><Skeleton className="h-24" /><div className="grid gap-5 xl:grid-cols-[270px_minmax(0,1fr)]"><Skeleton className="h-96" /><div className="space-y-4"><Skeleton className="h-28" /><Skeleton className="h-80" /></div></div></div>

  const runConfig = config.data?.config ?? overview.data?.config ?? null
  const preflight = config.data?.preflight
  const configSaved = Boolean(runConfig)
  const enabledCameras = runConfig?.capture.sensors.filter((sensor) => sensor.enabled !== false) ?? []
  const targetSelected = Boolean(runConfig?.calibration_target)
  const templateSelected = Boolean(runConfig?.pose_template?.placement_confirmed)
  const localCalibration = artifactComplete(overview.data, "calibration_profiles.json")
  const captureComplete = artifactComplete(overview.data, "capture_execution_report.json")
  const syncQualityComplete = artifactComplete(overview.data, "sync_quality_report.json")
  // The run-level quality report validates every enabled camera's per-folder
  // sync report; there is intentionally no mutable root sync_report.json.
  const syncComplete = syncQualityComplete
  const rectificationComplete = artifactComplete(overview.data, "camera_rectification_report.json")
  const calibrationPublished = localCalibration
  const datasetCalibrationSelected = Boolean(
    runConfig?.calibration_profiles
    && runConfig?.intrinsic_calibration_profiles
    && runConfig?.calibration_profile_selection?.bundle_sha256,
  )
  const bopComplete = artifactComplete(overview.data, "bop/bop_export_manifest.json")

  const calibrationRequirements: WorkflowRequirement[] = [
    { id: "config", label: "Run configuration", description: configSaved ? "The run configuration is saved." : "Save the run and camera configuration first.", status: configSaved ? "met" : "missing", onFix: () => selectStep("configure"), fixLabel: "Open step 1" },
    { id: "cameras", label: "At least one enabled camera", description: enabledCameras.length ? `${enabledCameras.length} camera${enabledCameras.length === 1 ? " is" : "s are"} enabled for this calibration.` : "No camera is enabled for capture and calibration.", status: enabledCameras.length ? "met" : "missing", onFix: () => selectStep("configure"), fixLabel: "Choose cameras" },
    { id: "target", label: "Printed grid selected", description: targetSelected ? "The run records an immutable target bundle and geometry hash." : "Select the exact printed grid that will be mounted in the cell.", status: targetSelected ? "met" : "missing", onFix: () => navigate("/calibration-targets"), fixLabel: "Choose calibration grid" },
  ]
  const calibrationReady = readinessSatisfied(preflight, calibrationRequirements)
  const calibrationStatuses = stepStatuses([configSaved, targetSelected, calibrationReady, captureComplete, calibrationPublished])
  const calibrationSteps: WorkflowStepDefinition[] = calibrationOutline.map((title, index) => ({
    id: ["configure", "target", "readiness", "capture", "calculate"][index], number: index + 1, title, summary: ["Choose camera identities and acquisition settings.", "Bind the physical printed board to this run.", "Resolve all blockers in one place.", "Open cameras and authorize supervised robot motion.", "Compare candidates and explicitly publish profiles."][index], status: calibrationStatuses[index], required: true,
  }))

  const datasetRequirements: WorkflowRequirement[] = [
    { id: "config", label: "Run configuration", description: configSaved ? "The dataset run configuration is saved." : "Save the dataset run and camera configuration first.", status: configSaved ? "met" : "missing", onFix: () => selectStep("configure"), fixLabel: "Open step 1" },
    { id: "cameras", label: "At least one enabled camera", description: enabledCameras.length ? `${enabledCameras.length} camera${enabledCameras.length === 1 ? " is" : "s are"} enabled for this dataset.` : "No camera is enabled for capture.", status: enabledCameras.length ? "met" : "missing", onFix: () => selectStep("configure"), fixLabel: "Choose cameras" },
    { id: "calibration", label: "Hash-bound calibration snapshot selected", description: datasetCalibrationSelected ? "Both profile files and their hash-bound selection record are configured for this run. Readiness will revalidate them." : "Choose a previously published calibration that matches every enabled camera.", status: datasetCalibrationSelected ? "met" : "missing", onFix: () => selectStep("configure"), fixLabel: "Select calibration" },
    { id: "template", label: "Object placement confirmed", description: templateSelected ? "The immutable object template and measured placement are confirmed." : "Select an immutable pose template and confirm its measured physical placement.", status: templateSelected ? "met" : "missing", onFix: () => selectStep("template"), fixLabel: "Choose object template" },
  ]
  const datasetReady = readinessSatisfied(preflight, datasetRequirements)
  const datasetConfigured = configSaved && datasetCalibrationSelected
  const datasetStatuses = stepStatuses([datasetConfigured, templateSelected, datasetReady, captureComplete, syncComplete && syncQualityComplete, bopComplete])
  const datasetSteps: WorkflowStepDefinition[] = datasetOutline.map((title, index) => ({
    id: ["configure", "template", "readiness", "capture", "sync", "export"][index], number: index + 1, title, summary: ["Reuse calibration that matches the selected cameras.", "Bind known object poses to the physical scene.", "Resolve all blockers in one place.", "Open cameras and authorize supervised robot motion.", "Create derived synchronized frames and check their quality.", "Write the acquisition result in BOP dataset form."][index], status: datasetStatuses[index], required: true,
  }))

  if (page === "calibration") return <JourneyShell journey="calibration" steps={calibrationSteps} selectedStep={selectedStep} onSelectStep={selectStep}>
    <WorkflowStepCard id="configure" number={1} title="Configure the run and cameras" description="Choose the camera identities, resolution, frame rate, and supervised robot velocity for this calibration recording." status={calibrationStatuses[0]} help="This saves configuration only. It does not open a camera or command the robot.">
      <RunSetup intent="camera_calibration" />
    </WorkflowStepCard>

    <WorkflowStepCard id="target" number={2} title="Choose the printed calibration grid" description="Select the immutable grid bundle that exactly matches the board you will print and place in the workcell." status={calibrationStatuses[1]} help="The target UUID and geometry hash prevent detections from one printed grid being interpreted as another.">
      <Card><CardContent className="flex flex-col gap-4 py-5 sm:flex-row sm:items-center sm:justify-between"><div className="flex items-start gap-3"><span className="grid size-10 shrink-0 place-items-center rounded-lg bg-muted"><Grid3X3 aria-hidden="true" className="size-5 text-primary-strong" /></span><div>{runConfig?.calibration_target ? <><div className="font-semibold">Calibration grid selected</div><div className="mt-1 font-mono text-[11px] text-muted-foreground">{runConfig.calibration_target.target_id}</div><div className="mt-1 text-xs text-muted-foreground">Placement: {runConfig.calibration_target.placement.mode.replaceAll("_", " ")}</div></> : <><div className="font-semibold text-destructive">No grid selected</div><p className="mt-1 text-xs text-muted-foreground">Choose the physical board before readiness and capture.</p></>}</div></div><Button asChild variant={targetSelected ? "outline" : "default"}><Link to="/calibration-targets">{targetSelected ? "Review or change grid" : "Choose grid"}<ArrowRight aria-hidden="true" /></Link></Button></CardContent></Card>
      <OptionalAction icon={Sparkles} title="Create a new printable grid" description="Reuse a saved grid when possible. Generate a new one only when the physical target requirements change." to="/calibration-targets" action="Open target library" />
    </WorkflowStepCard>

    <WorkflowStepCard id="readiness" number={3} title="Check readiness" description="Run one consolidated operator check after cameras and the exact printed grid are selected." status={calibrationStatuses[2]} help="The saved report proves which configuration was checked. Physical capture repeats the time-sensitive safety checks at startup.">
      <ReadinessCheck runRoot={selectedRun} intent="calibration" preflight={preflight} loading={config.isFetching} requirements={calibrationRequirements} />
    </WorkflowStepCard>

    <WorkflowStepCard id="capture" number={4} title="Record calibration images" description="Mount the selected grid as described for the calibration mode, clear the workcell, then authorize the supervised capture." status={calibrationStatuses[3]} help="Eye-in-hand means the camera moves with the robot while the grid remains stationary. Eye-to-hand means the camera is static while the grid moves rigidly with the robot.">
      <CaptureGate intent="calibration" readiness={{ ready: calibrationReady, onReview: () => selectStep("readiness") }} />
    </WorkflowStepCard>

    <WorkflowStepCard id="calculate" number={5} title="Calculate, review, and publish" description="Process the captured grid observations, review the recommendation for every camera, and explicitly publish only passing profiles." status={calibrationStatuses[4]} help="Publishing is deliberate: calculated candidates remain inactive until you accept the reviewed recommendations.">
      <Card className="border-primary/25 bg-primary/5"><CardContent className="py-4 text-xs leading-relaxed"><div className="flex items-center gap-2 font-semibold">Factory and OpenCV intrinsics <HelpTip label="Factory and OpenCV intrinsics">Factory is the per-camera projection supplied by the camera SDK. OpenCV is a new model fitted from this run's grid observations. Existing means an exact compatible profile was already available.</HelpTip></div><p className="mt-1 text-muted-foreground"><strong className="text-foreground">Factory</strong> stays selected when its projection is compatible. The fitted <strong className="text-foreground">OpenCV</strong> model is comparison and fallback evidence; it is activated only when factory projection is unusable and all coverage, held-out, plausibility, and error checks pass. A lower RMS alone does not make it the preferred model.</p></CardContent></Card>
      <CalibrationWorkflow />
    </WorkflowStepCard>
  </JourneyShell>

  if (page === "dataset") {
    return <JourneyShell journey="dataset" steps={datasetSteps} selectedStep={selectedStep} onSelectStep={selectStep}>
      <WorkflowStepCard id="configure" number={1} title="Configure cameras and select calibration" description="Choose the cameras for this recording and select a published calibration made for those exact camera identities and acquisition settings." status={datasetStatuses[0]} help="A calibration profile maps camera pixels into the shared robot/template coordinate system. It is required for an object dataset.">
        <Card className={datasetCalibrationSelected ? "border-success/30" : "border-destructive/30"}><CardContent className="flex flex-col gap-3 py-4 sm:flex-row sm:items-center sm:justify-between"><div><div className="flex items-center gap-2 font-semibold">Hash-bound calibration snapshot <StatusBadge status={datasetCalibrationSelected ? "configured" : "missing"} /></div><p className="mt-1 text-xs text-muted-foreground">{datasetCalibrationSelected ? `Bundle ${runConfig?.calibration_profile_selection?.bundle_sha256.slice(0, 16)}… is bound to both run-owned profile files. Step 3 revalidates it.` : "Required: select and validate a previously published calibration below."}</p></div><HelpTip label="hash-bound calibration snapshot">Use profiles promoted from a completed camera-calibration workflow. PoseTestBot copies both profile files into this run and records their hashes; readiness then rechecks the snapshot and every enabled camera.</HelpTip></CardContent></Card>
        <RunSetup intent="object_dataset" />
      </WorkflowStepCard>

      <WorkflowStepCard id="template" number={2} title="Choose the object template and placement" description="Select the immutable object arrangement that is physically present, enter its measured transform into template base, and confirm it." status={datasetStatuses[1]} help="The object template fixes object identities and relative poses. The measured placement locates the printed template in the robot's dataset reference frame.">
        <GroundTruthWorkflow />
        <div className="grid gap-3 md:grid-cols-2"><OptionalAction icon={Boxes} title="Add or edit workpieces" description="Manage source CAD, canonical geometry, names, tags, and lifecycle before making a new template." to="/workpieces" action="Open catalogue" /><OptionalAction icon={Grid3X3} title="Create a pose template" description="Lay out stable object orientations and publish a new immutable printable version." to="/pose-templates" action="Open templates" /></div>
      </WorkflowStepCard>

      <WorkflowStepCard id="readiness" number={3} title="Check readiness" description="Run one consolidated operator check after calibration and object placement are confirmed." status={datasetStatuses[2]} help="This is the only visible preflight step. The capture supervisor still repeats live checks immediately before hardware starts.">
        <ReadinessCheck runRoot={selectedRun} intent="dataset" preflight={preflight} loading={config.isFetching} requirements={datasetRequirements} />
      </WorkflowStepCard>

      <WorkflowStepCard id="capture" number={4} title="Record the object dataset" description="Place the objects exactly as confirmed, clear the workcell, then authorize supervised camera and robot capture." status={datasetStatuses[3]} help="Raw RGB, depth, timestamp, and robot-pose evidence is preserved. Use a new run folder rather than overwriting a prior capture.">
        <CaptureGate intent="dataset" readiness={{ ready: datasetReady, onReview: () => selectStep("readiness") }} />
      </WorkflowStepCard>

      <WorkflowStepCard id="sync" number={5} title="Synchronize and verify frames" description="Create derived synchronized frames, then pass the timing and match-quality gate. Raw captures are never renamed or deleted." status={datasetStatuses[4]} help="Synchronization matches each camera frame with a robot pose. The quality check catches missing matches, excessive time deltas, and incompatible timestamp sources.">
        <DatasetProcessing runRoot={selectedRun} ready={datasetReady} captureComplete={captureComplete} syncComplete={syncComplete} syncQualityComplete={syncQualityComplete} calibrationComplete={rectificationComplete} exportComplete={bopComplete} onReviewReadiness={() => selectStep("readiness")} />
      </WorkflowStepCard>

      <WorkflowStepCard id="export" number={6} title="Export the BOP dataset" description="Validate the selected calibration, then write the acquisition result in the standard BOP dataset layout." status={datasetStatuses[5]} help="BOP is the portable dataset format produced by PoseTestBot. Estimator execution and metric evaluation belong in a separate consumer repository.">
        <Card className={bopComplete ? "border-success/35 bg-success/5" : "border-dashed"}><CardContent className="flex flex-col gap-4 py-5 sm:flex-row sm:items-center sm:justify-between"><div><div className="font-semibold">{bopComplete ? "BOP dataset is ready" : "BOP export has not completed"}</div><p className="mt-1 text-xs text-muted-foreground">{bopComplete ? "The validated export manifest is stored at bop/bop_export_manifest.json." : "Use Process and export dataset in step 5. It validates calibration, rectifies frames, and creates this export in one ordered job."}</p></div>{bopComplete ? <Button asChild variant="outline"><Link to="/cell">Review dataset in Cell View<ArrowRight aria-hidden="true" /></Link></Button> : <Button type="button" variant="outline" onClick={() => selectStep("sync")}>Open processing step</Button>}</CardContent></Card>
        <OptionalAction icon={Sparkles} title="Generate rendered GT, masks, or COCO annotations" description="Use BlenderProc preparation/rendering only when those derived annotations are needed. The raw recording and synchronization remain valid without them." to="/workflow/advanced" action="Open rendering tools" />
      </WorkflowStepCard>
    </JourneyShell>
  }

  const artifactStatus = (stageId: string) => overview.data?.steps.find((step) => step.stage_id === stageId)?.status
  return <div className="space-y-6">
    <div className="flex items-center gap-2 text-xs"><Button asChild variant="ghost" size="sm"><Link to="/workflow/setup"><ArrowLeft aria-hidden="true" />Choose workflow</Link></Button><span className="text-muted-foreground">/</span><span className="font-semibold">Advanced tools</span></div>
    <PageHeader eyebrow="Expert controls" title="Advanced workflow tools" description="Inspect or queue individual implementation stages for diagnostics and recovery. The guided journeys remain the normal operator path." actions={<Button variant="outline" onClick={refresh}><RefreshCw aria-hidden="true" />Refresh evidence</Button>} />
    <JourneyNavigation current="advanced" />
    {!runConfig ? <EmptyState icon={ListTree} title="Configure the run first" description="Advanced stages still require a valid run_config.json." action={<Button asChild><Link to="/workflow/calibration?step=configure">Open guided setup</Link></Button>} /> : stages.isPending ? <div className="grid gap-4 md:grid-cols-2"><Skeleton className="h-72" /><Skeleton className="h-72" /></div> : <AdvancedStageTools runRoot={selectedRun} stages={stages.data?.stages ?? []} artifactStatus={artifactStatus} configuredSequence={{ id: runConfig.pipeline.sequence_id, planOnly: runConfig.pipeline.plan_only }} />}
  </div>
}
