import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { CheckCircle2, ChevronDown, Grid3X3, LoaderCircle, Save, TriangleAlert } from "lucide-react"
import { Link } from "react-router-dom"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api, errorMessage, query } from "@/lib/api"
import { useOperator } from "@/providers/operator-provider"

type Camera = { sensor_key: string; sensor_name: string; display_name: string; sensor_type: string; device_id: string }
type SavedTarget = { target_id: string; display_name: string; valid: boolean }
type Setup = {
  cameras: Camera[]
  unavailable_cameras: Array<Camera & { errors: string[] }>
  saved_targets: SavedTarget[]
  modes: Array<{ id: "eye_in_hand" | "eye_to_hand"; label: string; primary_transform: string; target_mounting: string }>
  solver: {
    default_pnp_methods: string[]
    default_extrinsic_methods: string[]
    intrinsics_policy: string
    intrinsics_policies: Array<{ id: string; label: string }>
    thresholds: {
      min_pnp_common_inliers: number
      min_pnp_common_inlier_ratio: number
      max_pnp_all_point_mean_reprojection_error_px: number
      min_pnp_supported_markers: number
      min_pnp_grid_rows: number
      min_pnp_grid_columns: number
      min_accepted_views: number
      min_coverage_cells: number
      max_per_view_reprojection_error_px: number
      max_intrinsic_rms_reprojection_error_px: number
      min_motion_poses: number
      min_translation_span_mm: number
      min_rotation_span_deg: number
      min_rotation_axis_second_to_first_ratio: number
      max_nearest_pose_delta_ms: number
    }
  }
  latest_attempt: { attempt_id: string; status: string } | null
}
type Transform = { from: string; to: string; matrix: number[][]; rotation_quaternion_wxyz: number[]; translation_mm: number[] }
type Candidate = {
  candidate_id: string; profile_id?: string; pnp_method: string; extrinsic_method: string; algorithms: string[]
  status: "passing" | "failed" | "error"; validation_state: string; recommended?: boolean; score: number | null
  observation_count: number; inlier_count: number; outlier_count: number; outlier_ratio: number
  mean_reprojection_error_px?: number | null; primary_transform?: Transform; companion_transform?: Transform
  held_out_residuals?: { mean_translation_mm: number; median_translation_mm: number; mean_rotation_deg: number; median_rotation_deg: number }
  error?: string
}
type CameraResult = Camera & { status: "passing" | "failed"; recommended_candidate_id: string | null; recommendation: Candidate | null; candidates: Candidate[] }
type IntrinsicCandidate = {
  profile_id: string
  source: { mode: string }
  quality: { status: string; accepted_view_count: number; coverage_cells: number[]; rms_reprojection_error_px: number | null }
}
type IntrinsicComparison = {
  sensor_key: string
  status: "manual_selected" | "factory_selected" | "existing_selected"
  selected_profile_id: string
  selection_reason: string
  factory_profile_id: string
  manual_profile_id: string | null
  manual_failure: null | { message: string; quality?: { reason?: string } }
  deltas: null | {
    focal_length_delta_px: number[]
    principal_point_delta_px: number[]
    max_abs_distortion_delta: number | null
  }
  candidates: IntrinsicCandidate[]
}
type Attempt = {
  attempt_id: string
  request: { mode: "eye_in_hand" | "eye_to_hand"; sensor_keys: string[]; target_id: string; solver_policy: "auto_compare"; intrinsics_policy: string }
  progress: { status: "queued" | "running" | "complete" | "failed"; message: string; phases: Array<{ id: string; label: string; status: "pending" | "running" | "complete" | "failed" }> }
  results: null | { status: "complete" | "partial" | "failed"; recommended_camera_count: number; failed_camera_count: number; results: CameraResult[] }
  intrinsic_comparison?: null | { policy: string; sensors: IntrinsicComparison[] }
  promotion: null | { status: "queued" | "running" | "promoted" | "failed"; job_id?: string; promoted_profile_ids?: string[]; error?: string }
}

const modeDescriptions = {
  eye_in_hand: "The selected cameras move with the robot while observing a target stationary relative to template_base.",
  eye_to_hand: "The selected cameras remain static while observing a target rigidly attached to robot_flange.",
}

export function CalibrationWorkflow() {
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const setup = useQuery({ queryKey: ["calibration", "setup", selectedRun], queryFn: () => api<Setup>(query("/calibration/setup", { run_root: selectedRun })) })
  const [mode, setMode] = useState<"eye_in_hand" | "eye_to_hand">("eye_in_hand")
  const [sensorSelection, setSensorSelection] = useState<{ runRoot: string; values: string[] } | null>(null)
  const [targetSelection, setTargetSelection] = useState<{ runRoot: string; value: string } | null>(null)
  const [attemptSelection, setAttemptSelection] = useState<{ runRoot: string; attemptId: string } | null>(null)
  const [overrideSelection, setOverrideSelection] = useState<{ runRoot: string; attemptId: string; values: Record<string, string> } | null>(null)
  const sensorKeys = sensorSelection?.runRoot === selectedRun ? sensorSelection.values : setup.data?.cameras.map((camera) => camera.sensor_key) ?? []
  const targetId = targetSelection?.runRoot === selectedRun ? targetSelection.value : setup.data?.saved_targets.find((target) => target.valid)?.target_id ?? ""
  const activeAttemptId = attemptSelection?.runRoot === selectedRun ? attemptSelection.attemptId : setup.data?.latest_attempt?.attempt_id ?? null

  const attempt = useQuery({
    queryKey: ["calibration", "attempt", selectedRun, activeAttemptId],
    queryFn: () => api<Attempt>(query(`/calibration/attempts/${activeAttemptId}`, { run_root: selectedRun })),
    enabled: Boolean(activeAttemptId),
    refetchInterval: (state) => {
      const value = state.state.data
      if (!value || ["queued", "running"].includes(value.progress.status)) return 700
      if (value.promotion && ["queued", "running"].includes(value.promotion.status)) return 700
      return false
    },
  })
  const recommendedOverrides = useMemo(() => Object.fromEntries(
    attempt.data?.results?.results.flatMap((result) => result.recommended_candidate_id ? [[result.sensor_key, result.recommended_candidate_id]] : []) ?? [],
  ), [attempt.data?.results?.results])
  const overrides = overrideSelection?.runRoot === selectedRun && overrideSelection.attemptId === activeAttemptId
    ? overrideSelection.values
    : recommendedOverrides

  const createAttempt = useMutation({
    mutationFn: () => api<{ attempt_id: string; job_id: string }>("/calibration/attempts", { method: "POST", body: JSON.stringify({ run_root: selectedRun, mode, sensor_keys: sensorKeys, target_id: targetId, solver_policy: "auto_compare", intrinsics_policy: "compare_factory_opencv" }) }),
    onSuccess: (value) => { setAttemptSelection({ runRoot: selectedRun, attemptId: value.attempt_id }); setOverrideSelection(null); toast.success("Calibration queued", { description: `Attempt ${value.attempt_id} · job ${value.job_id}` }); queryClient.invalidateQueries({ queryKey: ["calibration", "setup", selectedRun] }); queryClient.invalidateQueries({ queryKey: ["jobs"] }) },
    onError: (error) => toast.error("Calibration was not queued", { description: errorMessage(error) }),
  })
  const promote = useMutation({
    mutationFn: () => api<{ job_id: string }>(`/calibration/attempts/${activeAttemptId}/promote`, { method: "POST", body: JSON.stringify({ run_root: selectedRun, candidate_ids: overrides }) }),
    onSuccess: (value) => { toast.success("Calibration acceptance queued", { description: `Job ${value.job_id}` }); queryClient.invalidateQueries({ queryKey: ["calibration", "attempt", selectedRun, activeAttemptId] }); queryClient.invalidateQueries({ queryKey: ["jobs"] }) },
    onError: (error) => toast.error("Recommendations were not accepted", { description: errorMessage(error) }),
  })
  const canRun = sensorKeys.length > 0 && Boolean(targetId) && !createAttempt.isPending
  const passingSelections = useMemo(() => attempt.data?.results?.results.filter((result) => {
    const selected = overrides[result.sensor_key]
    return result.candidates.some((candidate) => candidate.candidate_id === selected && candidate.status === "passing")
  }).length ?? 0, [attempt.data?.results, overrides])

  if (setup.isPending) return <Card><CardContent className="grid min-h-72 place-items-center text-sm text-muted-foreground"><span className="flex items-center"><LoaderCircle className="mr-2 size-4 animate-spin" />Loading calibration setup…</span></CardContent></Card>
  if (setup.isError || !setup.data) return <Card className="border-destructive/40"><CardContent className="py-8 text-sm text-destructive">{errorMessage(setup.error)}</CardContent></Card>

  return <div className="space-y-5" data-testid="calibration-workflow">
    <Card><CardHeader><CardTitle>Calculate camera calibration</CardTitle><CardDescription>Use captured data from this run. This workflow never initiates physical capture, and results remain inactive until accepted.</CardDescription></CardHeader><CardContent className="space-y-6">
      <fieldset className="space-y-3"><legend className="text-sm font-semibold">Calibration mode</legend><div className="grid grid-cols-2 gap-3">{setup.data.modes.map((item) => <Label key={item.id} className={`cursor-pointer rounded-lg border p-4 ${mode === item.id ? "border-primary bg-primary/5" : "border-border"}`}><span className="flex items-start gap-3"><input type="radio" name="calibration-mode" value={item.id} checked={mode === item.id} onChange={() => setMode(item.id)} className="mt-1" /><span><span className="block font-semibold">{item.label}</span><span className="mt-1 block text-xs font-normal leading-relaxed text-muted-foreground">{modeDescriptions[item.id]}</span><span className="mt-2 block font-mono text-[11px] font-normal text-primary-strong">{item.primary_transform}</span></span></span></Label>)}</div></fieldset>
      <div className="grid grid-cols-2 gap-5"><fieldset className="space-y-3"><legend className="text-sm font-semibold">Cameras from this run</legend><div className="space-y-2 rounded-lg border p-3">{setup.data.cameras.map((camera) => <Label key={camera.sensor_key} className="flex cursor-pointer items-start gap-3 rounded-md p-2 hover:bg-muted/50"><Checkbox checked={sensorKeys.includes(camera.sensor_key)} onCheckedChange={(checked) => setSensorSelection({ runRoot: selectedRun, values: checked === true ? [...sensorKeys, camera.sensor_key] : sensorKeys.filter((key) => key !== camera.sensor_key) })} /><span><span className="block text-sm font-medium">{camera.display_name}</span><span className="block font-mono text-[10px] font-normal text-muted-foreground">{camera.sensor_key}</span></span></Label>)}{!setup.data.cameras.length && <div className="p-3 text-xs text-destructive">No captured camera has complete RGB-D, timestamp, and robot-pose evidence.</div>}</div>{setup.data.unavailable_cameras.length > 0 && <details className="rounded border border-warning/40 p-3 text-xs"><summary className="cursor-pointer font-medium">{setup.data.unavailable_cameras.length} unavailable camera folder(s)</summary><div className="mt-2 space-y-1 text-muted-foreground">{setup.data.unavailable_cameras.map((camera) => <div key={camera.sensor_key}>{camera.display_name}: {camera.errors.join("; ")}</div>)}</div></details>}</fieldset>
        <div className="space-y-5"><div className="space-y-2"><Label htmlFor="calibration-target">Saved calibration target</Label><Select value={targetId} onValueChange={(value) => setTargetSelection({ runRoot: selectedRun, value })}><SelectTrigger id="calibration-target"><SelectValue placeholder="Select saved target" /></SelectTrigger><SelectContent>{setup.data.saved_targets.filter((target) => target.valid).map((target) => <SelectItem value={target.target_id} key={target.target_id}>{target.display_name}</SelectItem>)}</SelectContent></Select><div className="flex items-center justify-between text-xs text-muted-foreground"><span>Saved geometry is reused; mounting is derived from the mode.</span><Link to="/calibration-targets" className="text-primary-strong underline-offset-4 hover:underline">Browse targets</Link></div></div><div className="space-y-2"><Label htmlFor="solver-policy">Solver policy</Label><Select value="auto_compare" disabled><SelectTrigger id="solver-policy"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="auto_compare">Auto compare — recommended</SelectItem></SelectContent></Select><div className="text-xs text-muted-foreground">PnP: {setup.data.solver.default_pnp_methods.join(", ")} · robot-camera: {setup.data.solver.default_extrinsic_methods.join(", ")}</div><div className="rounded border bg-muted/30 p-2 text-[11px] leading-relaxed text-muted-foreground" data-testid="calibration-acceptance-thresholds">Requires ≥{setup.data.solver.thresholds.min_accepted_views} accepted views, ≥{setup.data.solver.thresholds.min_coverage_cells}/9 image cells, ≥{setup.data.solver.thresholds.min_motion_poses} motion poses with ≥{setup.data.solver.thresholds.min_translation_span_mm} mm / ≥{setup.data.solver.thresholds.min_rotation_span_deg}° span and rotation-axis ratio ≥{setup.data.solver.thresholds.min_rotation_axis_second_to_first_ratio}. PnP needs ≥{setup.data.solver.thresholds.min_pnp_common_inliers} corners, ≥{Math.round(setup.data.solver.thresholds.min_pnp_common_inlier_ratio * 100)}% support, and ≥{setup.data.solver.thresholds.min_pnp_supported_markers} markers spanning {setup.data.solver.thresholds.min_pnp_grid_rows}×{setup.data.solver.thresholds.min_pnp_grid_columns} grid rows/columns, with ≤{setup.data.solver.thresholds.max_pnp_all_point_mean_reprojection_error_px} px whole-board error. Sync is exclusive host_received with zero offset and ≤{setup.data.solver.thresholds.max_nearest_pose_delta_ms} ms nearest pose. Compatible factory intrinsics remain the default; manual OpenCV intrinsics require held-out proof, ≤{setup.data.solver.thresholds.max_per_view_reprojection_error_px} px/view, and ≤{setup.data.solver.thresholds.max_intrinsic_rms_reprojection_error_px} px RMS, and activate only when the factory projection is unusable.</div></div><Button className="w-full" size="lg" disabled={!canRun} onClick={() => createAttempt.mutate()}>{createAttempt.isPending ? <LoaderCircle className="animate-spin" /> : <Grid3X3 />}Run calibration</Button></div>
      </div>
    </CardContent></Card>

    {activeAttemptId && <Card className="border-primary/25"><CardHeader><CardTitle className="flex items-center justify-between text-base"><span>Calculation progress</span><span className="font-mono text-[10px] font-normal text-muted-foreground">{activeAttemptId}</span></CardTitle><CardDescription>{attempt.data?.progress.message ?? "Loading attempt…"}</CardDescription></CardHeader><CardContent><div className="grid grid-cols-4 gap-2">{attempt.data?.progress.phases.map((phase, index) => <div key={phase.id} className={`rounded-lg border p-3 ${phase.status === "running" ? "border-primary bg-primary/5" : phase.status === "complete" ? "border-success/40 bg-success/5" : phase.status === "failed" ? "border-destructive/40 bg-destructive/5" : ""}`}><div className="flex items-center gap-2 text-xs font-semibold"><span className="grid size-5 place-items-center rounded-full bg-muted font-mono text-[10px]">{index + 1}</span>{phase.label}</div><div className="mt-2 flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">{phase.status === "running" && <LoaderCircle className="size-3 animate-spin" />}{phase.status === "complete" && <CheckCircle2 className="size-3 text-success" />}{phase.status === "failed" && <TriangleAlert className="size-3 text-destructive" />}{phase.status}</div></div>)}</div></CardContent></Card>}

    {attempt.data?.results && <div className="space-y-4" data-testid="calibration-results"><div><h2 className="text-xl font-semibold">Calibration results</h2><p className="mt-1 text-sm text-muted-foreground">Review each recommendation. A failed camera does not prevent saving another passing camera.</p></div>{attempt.data.results.results.map((result) => <CameraResultCard key={result.sensor_key} result={result} intrinsicComparison={attempt.data?.intrinsic_comparison?.sensors.find((item) => item.sensor_key === result.sensor_key) ?? null} selectedCandidateId={overrides[result.sensor_key] ?? ""} onSelect={(candidateId) => setOverrideSelection({ runRoot: selectedRun, attemptId: activeAttemptId ?? "", values: { ...overrides, [result.sensor_key]: candidateId } })} />)}<Card className="border-primary/30"><CardContent className="flex items-center justify-between py-5"><div><div className="font-semibold">Accept reviewed recommendations</div><div className="mt-1 text-xs text-muted-foreground">{passingSelections} passing camera profile(s) selected. Canonical profiles change only in the queued promotion transaction.</div>{attempt.data.promotion?.status === "failed" && <div className="mt-2 text-xs text-destructive">{attempt.data.promotion.error}</div>}{attempt.data.promotion?.status === "promoted" && <div className="mt-2 flex items-center gap-1 text-xs text-success"><CheckCircle2 className="size-4" />Saved {attempt.data.promotion.promoted_profile_ids?.length ?? 0} camera profile(s).</div>}</div><Button size="lg" disabled={passingSelections === 0 || promote.isPending || ["queued", "running", "promoted"].includes(attempt.data.promotion?.status ?? "")} onClick={() => promote.mutate()}>{promote.isPending || ["queued", "running"].includes(attempt.data.promotion?.status ?? "") ? <LoaderCircle className="animate-spin" /> : <Save />}{attempt.data.promotion?.status === "promoted" ? "Recommendations saved" : "Accept recommendations"}</Button></CardContent></Card></div>}
  </div>
}

function CameraResultCard({ result, intrinsicComparison, selectedCandidateId, onSelect }: { result: CameraResult; intrinsicComparison: IntrinsicComparison | null; selectedCandidateId: string; onSelect: (value: string) => void }) {
  const selected = result.candidates.find((candidate) => candidate.candidate_id === selectedCandidateId) ?? result.recommendation
  const passing = result.candidates.filter((candidate) => candidate.status === "passing")
  const manualIntrinsic = intrinsicComparison?.candidates.find((candidate) => candidate.profile_id === intrinsicComparison.manual_profile_id)
  return <Card className={result.status === "passing" ? "border-success/35" : "border-destructive/35"} data-camera-key={result.sensor_key}><CardHeader><div className="flex items-start justify-between gap-4"><div><CardTitle className="text-base">{result.display_name}</CardTitle><CardDescription className="mt-1 font-mono text-[10px]">{result.sensor_key}</CardDescription></div><span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wide ${result.status === "passing" ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive"}`}>{result.status}</span></div></CardHeader><CardContent className="space-y-4">
    {intrinsicComparison && <div className="rounded-md border bg-muted/20 p-3" data-testid={`intrinsic-comparison-${result.sensor_key}`}><div className="flex items-center justify-between gap-3"><div className="text-xs font-semibold">Factory vs OpenCV intrinsics</div><span className={`rounded-full px-2 py-1 text-[10px] font-semibold uppercase ${intrinsicComparison.status === "manual_selected" ? "bg-success/10 text-success" : "bg-warning/10 text-warning"}`}>{intrinsicComparison.status === "manual_selected" ? "OpenCV selected" : intrinsicComparison.status === "existing_selected" ? "Existing selected" : "Factory selected"}</span></div><div className="mt-3 grid grid-cols-4 gap-3 text-xs"><Metric label="Selected profile" value={intrinsicComparison.selected_profile_id} /><Metric label="Manual views / coverage" value={manualIntrinsic ? `${manualIntrinsic.quality.accepted_view_count} / ${manualIntrinsic.quality.coverage_cells.length}/9` : "—"} /><Metric label="Manual RMS" value={format(manualIntrinsic?.quality.rms_reprojection_error_px, " px")} /><Metric label="Δ fx, fy" value={intrinsicComparison.deltas ? intrinsicComparison.deltas.focal_length_delta_px.map((value) => value.toFixed(3)).join(", ") : "—"} /></div>{intrinsicComparison.deltas && <div className="mt-2 font-mono text-[10px] text-muted-foreground">Δ cx, cy: {intrinsicComparison.deltas.principal_point_delta_px.map((value) => value.toFixed(3)).join(", ")} px · max |Δ distortion|: {intrinsicComparison.deltas.max_abs_distortion_delta === null ? "not comparable" : intrinsicComparison.deltas.max_abs_distortion_delta.toExponential(3)}</div>}<div className="mt-2 text-[11px] text-muted-foreground">{intrinsicComparison.selection_reason}</div>{intrinsicComparison.manual_failure && <div className="mt-2 rounded border border-destructive/35 bg-destructive/5 p-2 text-[11px] text-destructive">OpenCV calibration rejected: {intrinsicComparison.manual_failure.quality?.reason ?? intrinsicComparison.manual_failure.message}</div>}</div>}
    {selected?.primary_transform ? <><div className="grid grid-cols-4 gap-3 text-xs"><Metric label="Transform" value={`${selected.primary_transform.from} → ${selected.primary_transform.to}`} /><Metric label="Observations / inliers" value={`${selected.observation_count} / ${selected.inlier_count}`} /><Metric label="Reprojection" value={format(selected.mean_reprojection_error_px, " px")} /><Metric label="Held-out residual" value={`${format(selected.held_out_residuals?.median_translation_mm, " mm")} · ${format(selected.held_out_residuals?.median_rotation_deg, "°")}`} /></div><div className="grid grid-cols-[1.3fr_1fr] gap-3"><div className="rounded-md bg-muted/60 p-3"><div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">Matrix</div><pre className="overflow-x-auto text-[10px] leading-5">{selected.primary_transform.matrix.map((row) => row.map((value) => value.toFixed(6).padStart(12)).join(" ")).join("\n")}</pre></div><div className="space-y-3 rounded-md border p-3 text-xs"><Datum label="Quaternion WXYZ" value={selected.primary_transform.rotation_quaternion_wxyz.map((value) => value.toFixed(7)).join(", ")} /><Datum label="Translation mm" value={selected.primary_transform.translation_mm.map((value) => value.toFixed(4)).join(", ")} /><Datum label="Algorithms" value={selected.algorithms.join(" + ")} /><Datum label="Validation" value={selected.validation_state} /></div></div></> : <div className="rounded border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">No solver combination passed for this camera. Expand attempts below for concrete failures.</div>}
    {passing.length > 0 && <div className="grid grid-cols-[220px_minmax(0,1fr)] items-center gap-3 border-t pt-4"><Label htmlFor={`override-${result.sensor_key}`}>Candidate override</Label><Select value={selectedCandidateId} onValueChange={onSelect}><SelectTrigger id={`override-${result.sensor_key}`}><SelectValue /></SelectTrigger><SelectContent>{passing.map((candidate) => <SelectItem value={candidate.candidate_id} key={candidate.candidate_id}>{candidate.recommended ? "Recommended · " : ""}{candidate.pnp_method} + {candidate.extrinsic_method} · score {candidate.score?.toFixed(4)}</SelectItem>)}</SelectContent></Select></div>}
    <details className="group rounded-lg border"><summary className="flex cursor-pointer list-none items-center justify-between p-3 text-xs font-semibold">Every attempted candidate and failure <ChevronDown className="size-4 transition group-open:rotate-180" /></summary><div className="overflow-x-auto border-t"><table className="w-full text-left text-[11px]"><thead className="bg-muted/60 text-muted-foreground"><tr><th className="px-3 py-2">PnP</th><th className="px-3 py-2">Extrinsic</th><th className="px-3 py-2">State</th><th className="px-3 py-2">Score</th><th className="px-3 py-2">Inliers</th><th className="px-3 py-2">Reprojection</th><th className="px-3 py-2">Held-out T / R</th><th className="px-3 py-2">Failure</th></tr></thead><tbody>{result.candidates.map((candidate) => <tr key={candidate.candidate_id} className="border-t"><td className="px-3 py-2 font-mono">{candidate.pnp_method}</td><td className="px-3 py-2 font-mono">{candidate.extrinsic_method}</td><td className="px-3 py-2 capitalize">{candidate.status}{candidate.recommended ? " · recommended" : ""}</td><td className="px-3 py-2 tabular-nums">{candidate.score?.toFixed(4) ?? "—"}</td><td className="px-3 py-2 tabular-nums">{candidate.inlier_count}/{candidate.observation_count}</td><td className="px-3 py-2 tabular-nums">{format(candidate.mean_reprojection_error_px, " px")}</td><td className="px-3 py-2 tabular-nums">{format(candidate.held_out_residuals?.mean_translation_mm, " mm")} / {format(candidate.held_out_residuals?.mean_rotation_deg, "°")}</td><td className="max-w-72 px-3 py-2 text-destructive">{candidate.error ?? ""}</td></tr>)}</tbody></table></div></details>
  </CardContent></Card>
}

function Metric({ label, value }: { label: string; value: string }) { return <div className="rounded-md border p-3"><div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div><div className="mt-1 font-mono text-[11px]">{value}</div></div> }
function Datum({ label, value }: { label: string; value: string }) { return <div><span className="text-muted-foreground">{label}</span><div className="mt-1 font-mono text-[10px] capitalize">{value}</div></div> }
function format(value: number | null | undefined, suffix: string) { return value === null || value === undefined ? "—" : `${value.toFixed(3)}${suffix}` }
