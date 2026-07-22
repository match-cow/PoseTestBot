import { Component, Suspense, useEffect, useMemo, useRef, useState } from "react"
import { Canvas, type ThreeEvent, useLoader, useThree } from "@react-three/fiber"
import { OrbitControls, useTexture } from "@react-three/drei"
import { useQuery, useQueryClient } from "@tanstack/react-query"
import { BufferGeometry, Color, Float32BufferAttribute, Line, LineBasicMaterial, LineSegments, Quaternion, Vector3 } from "three"
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js"
import { AlertTriangle, Box, Camera, CirclePause, CirclePlay, Crosshair, Eye, EyeOff, Focus, RotateCcw, Route, ScanLine } from "lucide-react"
import { PageHeader } from "@/components/page-header"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api, query } from "@/lib/api"
import type { CellEntity, CellPose, CellScene, CellTimelinePage, CellTransform } from "@/lib/contracts"
import { titleCase } from "@/lib/utils"
import { useOperator } from "@/providers/operator-provider"

const PAGE_SIZE = 2_000
const LAYERS = ["reference_frame", "template", "robot_base", "robot_flange", "tcp", "camera", "object", "calibration_target"]
type Preset = "perspective" | "top" | "front"

function hasWebGL() {
  try {
    const canvas = document.createElement("canvas")
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"))
  } catch {
    return false
  }
}

function transformProps(transform: CellTransform) {
  const [w, x, y, z] = transform.rotation_quaternion_wxyz
  return {
    position: transform.translation_mm as [number, number, number],
    quaternion: new Quaternion(x, y, z, w),
  }
}

function statusColor(status: CellEntity["status"]) {
  if (status === "unresolved") return "#ef4444"
  if (status === "recorded") return "#22c55e"
  return "#38bdf8"
}

function finiteNumber(value: unknown) {
  if (value === null || value === undefined || value === "") return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function fixed(value: unknown, digits: number) {
  return finiteNumber(value)?.toFixed(digits) ?? "—"
}

function number(value: unknown, digits: number, suffix = "") {
  const formatted = fixed(value, digits)
  return formatted === "—" ? formatted : `${formatted}${suffix}`
}

function vector(values: readonly unknown[] | null | undefined, digits: number) {
  return values?.map((value) => fixed(value, digits)).join(", ") ?? "—"
}

function Detail({ label, value }: { label: string; value: React.ReactNode }) {
  return <div><span className="text-muted-foreground">{label}</span><div className="mt-1 break-all font-mono text-[10px]">{value}</div></div>
}

function Matrix({ values, testId = "cell-calibration-matrix" }: { values: readonly (readonly unknown[])[]; testId?: string }) {
  return <pre data-testid={testId} className="overflow-x-auto rounded bg-muted p-3 text-[9px] leading-5">{values.map((row) => row.map((value) => fixed(value, 6).padStart(12)).join(" ")).join("\n")}</pre>
}

function SelectionDetails({ entity }: { entity: CellEntity | null }) {
  if (!entity) return <div className="py-6 text-center text-sm text-muted-foreground"><Box className="mx-auto mb-2 size-5" />Nothing selected</div>
  const calibration = entity.calibration
  const solver = calibration?.evidence.promotion_solver_provenance
  const solverLabel = solver?.pnp_method || solver?.extrinsic_method
    ? [solver.pnp_method, solver.extrinsic_method].filter(Boolean).join(" + ")
    : null
  return <div className="space-y-4 text-xs">
    <div className="flex items-center justify-between gap-2"><span className="text-sm font-semibold">{entity.label}</span><StatusBadge status={entity.status} /></div>
    <div><span className="text-muted-foreground">Type</span><div>{titleCase(entity.type)}</div></div>
    {entity.unresolved_reason && <div className="rounded border border-destructive/30 bg-destructive/5 p-3 text-destructive">{entity.unresolved_reason}</div>}
    {entity.transform && !calibration && <div className="space-y-3 rounded border p-3">
      <div className="font-semibold">Entity transform</div>
      <Detail label="Frame" value={`${entity.id} → ${entity.transform.parent_frame ?? "root"}`} />
      <Detail label="Translation mm" value={vector(entity.transform.translation_mm, 3)} />
      <Detail label="Quaternion WXYZ" value={vector(entity.transform.rotation_quaternion_wxyz, 7)} />
    </div>}
    {calibration && <div data-testid="cell-calibration-evidence" className="space-y-4 rounded border border-success/30 p-3">
      <div className="flex items-center justify-between gap-2"><div><div className="font-semibold">Calibration extrinsic</div><div className="mt-1 font-mono text-[10px]" data-testid="cell-calibration-transform-frames">{calibration.extrinsics.from} → {calibration.extrinsics.to}</div></div><StatusBadge status={calibration.status} /></div>
      <Matrix values={calibration.extrinsics.matrix} />
      <div className="grid grid-cols-1 gap-3">
        <Detail label="Quaternion WXYZ" value={vector(calibration.extrinsics.rotation_quaternion_wxyz, 7)} />
        <Detail label="Translation mm" value={vector(calibration.extrinsics.translation_mm, 4)} />
      </div>
      {calibration.companion_transform && <div data-testid="cell-calibration-companion" className="space-y-3 border-t pt-3">
        <div><div className="font-semibold">Companion transform estimate</div><div className="mt-1 font-mono text-[10px]" data-testid="cell-calibration-companion-frames">{calibration.companion_transform.from} → {calibration.companion_transform.to}</div></div>
        <Matrix values={calibration.companion_transform.matrix} testId="cell-calibration-companion-matrix" />
        <Detail label="Quaternion WXYZ" value={vector(calibration.companion_transform.rotation_quaternion_wxyz, 7)} />
        <Detail label="Translation mm" value={vector(calibration.companion_transform.translation_mm, 4)} />
      </div>}
      <div className="grid grid-cols-2 gap-2 border-t pt-3">
        <Detail label="Observations / inliers" value={`${calibration.quality.num_observations} / ${calibration.quality.num_inliers}`} />
        <Detail label="Mean reprojection" value={number(calibration.quality.mean_reprojection_error_px, 3, " px")} />
        <Detail label="Max reprojection" value={number(calibration.quality.max_reprojection_error_px, 3, " px")} />
        <Detail label="Outlier count" value={calibration.quality.outlier_count ?? "—"} />
        <Detail label="Outlier ratio" value={number(calibration.quality.outlier_ratio, 4)} />
        <Detail label="Held-out translation" value={number(calibration.quality.residual_translation_mm, 3, " mm")} />
        <Detail label="Held-out rotation" value={number(calibration.quality.residual_rotation_deg, 3, "°")} />
        <Detail label="Held-out residual summary" value={calibration.quality.held_out_residuals ? JSON.stringify(calibration.quality.held_out_residuals) : "—"} />
      </div>
      <div className="space-y-3 border-t pt-3">
        <Detail label="Profile" value={calibration.profile_id} />
        <Detail label="Mount" value={`${calibration.mounting_mode} · ${calibration.rig_position}`} />
        <Detail label="Method" value={calibration.evidence.method ?? "—"} />
        <Detail label="Calibration dataset" value={calibration.evidence.calibration_dataset_id ?? "—"} />
        <Detail label="Sync offset" value={number(calibration.evidence.sync_delta_ms, 3, " ms")} />
        <Detail label="Promoted solver" value={solverLabel ?? "—"} />
        <Detail label="Promotion attempt" value={calibration.evidence.promotion_attempt_id ?? "—"} />
        <Detail label="Promotion candidate" value={calibration.evidence.promotion_candidate_id ?? "—"} />
        <Detail label="Multi-camera bundle" value={calibration.evidence.promotion_multi_camera_bundle_id ?? "—"} />
        <Detail label="Target / intrinsic" value={`${calibration.evidence.target_id ?? "—"} / ${calibration.evidence.intrinsic_profile_id ?? "—"}`} />
        <Detail label="Calibrated" value={calibration.evidence.calibrated_at ?? "—"} />
        <Detail label="Promoted" value={calibration.evidence.promoted_at ?? "—"} />
        <Detail label="Operator / promoted by" value={`${calibration.evidence.operator ?? "—"} / ${calibration.evidence.promoted_by ?? "—"}`} />
        <Detail label="Profile source" value={calibration.evidence.profile_source} />
      </div>
    </div>}
    <details className="rounded border"><summary className="cursor-pointer p-2 font-medium">Raw provenance</summary><pre data-testid="cell-raw-provenance" className="max-h-48 overflow-auto border-t bg-muted p-3 text-[10px]">{JSON.stringify({ entity: entity.provenance, calibration: entity.calibration ?? null }, null, 2)}</pre></details>
  </div>
}

class MeshBoundary extends Component<{ children: React.ReactNode; fallback: React.ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  render() { return this.state.failed ? this.props.fallback : this.props.children }
}

function PlyMesh({ entity, onSelect }: { entity: CellEntity; onSelect: () => void }) {
  const url = String(entity.geometry.mesh_url)
  const geometry = useLoader(PLYLoader, url)
  const textureUrl = entity.geometry.texture_url
  if (typeof textureUrl === "string" && textureUrl) {
    return <TexturedPly geometry={geometry} textureUrl={textureUrl} onSelect={onSelect} />
  }
  const vertexColors = Boolean(geometry.getAttribute("color"))
  return <mesh geometry={geometry} onClick={(event) => selectEvent(event, onSelect)} castShadow receiveShadow>
    <meshStandardMaterial color={vertexColors ? "white" : "#94a3b8"} vertexColors={vertexColors} roughness={0.72} metalness={0.05} />
  </mesh>
}

function TexturedPly({ geometry, textureUrl, onSelect }: { geometry: BufferGeometry; textureUrl: string; onSelect: () => void }) {
  const texture = useTexture(textureUrl)
  return <mesh geometry={geometry} onClick={(event) => selectEvent(event, onSelect)} castShadow receiveShadow>
    <meshStandardMaterial map={texture} color="white" roughness={0.72} />
  </mesh>
}

function selectEvent(event: ThreeEvent<MouseEvent>, callback: () => void) {
  event.stopPropagation()
  callback()
}

function Frustum({ color, onSelect, geometry }: { color: string; onSelect: () => void; geometry: CellEntity["geometry"] }) {
  const object = useMemo(() => {
    const width = Number(geometry.width || 1280)
    const height = Number(geometry.height || 720)
    const fx = Number(geometry.fx || width)
    const fy = Number(geometry.fy || height)
    const cx = Number(geometry.cx || width / 2)
    const cy = Number(geometry.cy || height / 2)
    const depth = Number(geometry.depth_mm || 180)
    const corners = [[0, 0], [width, 0], [width, height], [0, height]].map(([u, v]) => new Vector3((u - cx) * depth / fx, (v - cy) * depth / fy, depth))
    const origin = new Vector3()
    const segments: number[] = []
    const add = (a: Vector3, b: Vector3) => segments.push(a.x, a.y, a.z, b.x, b.y, b.z)
    corners.forEach((corner) => add(origin, corner))
    corners.forEach((corner, index) => add(corner, corners[(index + 1) % corners.length]))
    const buffer = new BufferGeometry()
    buffer.setAttribute("position", new Float32BufferAttribute(segments, 3))
    return new LineSegments(buffer, new LineBasicMaterial({ color }))
  }, [color, geometry])
  return <primitive object={object} onClick={(event: ThreeEvent<MouseEvent>) => selectEvent(event, onSelect)} />
}

function TemplatePlane({ url, onSelect }: { url: string; onSelect: () => void }) {
  const texture = useTexture(url)
  return <mesh position={[0, 0, -0.8]} scale={[1, -1, 1]} onClick={(event) => selectEvent(event, onSelect)} receiveShadow>
    <planeGeometry args={[420, 297]} />
    <meshBasicMaterial map={texture} transparent opacity={0.74} depthWrite={false} />
  </mesh>
}

function TargetGeometry({ entity, color, onSelect }: { entity: CellEntity; color: string; onSelect: () => void }) {
  const grid = entity.geometry.grid_size as number[] | undefined
  const bounds = entity.geometry.target_bounds as { width_mm?: number; height_mm?: number } | undefined
  const marker = Number(entity.geometry.marker_length_mm || entity.geometry.square_length_mm || 40)
  const gap = Number(entity.geometry.marker_separation_mm || marker)
  const width = Number(bounds?.width_mm) || (grid ? marker + Math.max(0, grid[0] - 1) * gap : 200)
  const height = Number(bounds?.height_mm) || (grid ? marker + Math.max(0, grid[1] - 1) * gap : 150)
  return <mesh position={[width / 2, height / 2, 1]} onClick={(event) => selectEvent(event, onSelect)}>
    <boxGeometry args={[width, height, 2]} />
    <meshStandardMaterial color={color} transparent opacity={0.55} />
  </mesh>
}

function EntityVisual({ entity, onSelect }: { entity: CellEntity; onSelect: () => void }) {
  const color = statusColor(entity.status)
  const kind = String(entity.geometry.kind || entity.type)
  if (kind === "axes") return <axesHelper args={[Number(entity.geometry.size_mm || 100)]} onClick={(event) => selectEvent(event, onSelect)} />
  if (kind === "svg_plane") return <TemplatePlane url={String(entity.geometry.asset_url)} onSelect={onSelect} />
  if (kind === "mesh") return <MeshBoundary fallback={<mesh onClick={(event) => selectEvent(event, onSelect)}><boxGeometry args={[30, 30, 30]} /><meshStandardMaterial color="#f59e0b" wireframe /></mesh>}><Suspense fallback={null}><PlyMesh entity={entity} onSelect={onSelect} /></Suspense></MeshBoundary>
  if (kind === "camera_frustum") return <Frustum color={color} onSelect={onSelect} geometry={entity.geometry} />
  if (kind === "calibration_target") return <TargetGeometry entity={entity} color={color} onSelect={onSelect} />
  if (kind === "robot_base") return <mesh position={[0, 0, 45]} onClick={(event) => selectEvent(event, onSelect)}><cylinderGeometry args={[85, 105, 90, 32]} /><meshStandardMaterial color={color} roughness={0.55} /></mesh>
  if (kind === "flange_proxy") return <mesh onClick={(event) => selectEvent(event, onSelect)}><cylinderGeometry args={[42, 42, 30, 24]} /><meshStandardMaterial color={color} /></mesh>
  if (kind === "tcp") return <group onClick={(event) => selectEvent(event, onSelect)}><axesHelper args={[70]} /><mesh position={[0, 0, 25]}><cylinderGeometry args={[8, 3, 50, 12]} /><meshStandardMaterial color={color} /></mesh></group>
  return null
}

function EntityTree({ entity, childrenByParent, visible, pose, onSelect }: { entity: CellEntity; childrenByParent: Map<string | null, CellEntity[]>; visible: Set<string>; pose: CellPose | null; onSelect: (entity: CellEntity) => void }) {
  const transform = entity.id === "robot_flange" && pose ? pose.transform : entity.transform
  if (!transform) return null
  const children = childrenByParent.get(entity.id) ?? []
  return <group name={entity.id} {...transformProps(transform)}>
    {visible.has(entity.type) && <EntityVisual entity={entity} onSelect={() => onSelect(entity)} />}
    {children.map((child) => <EntityTree key={child.id} entity={child} childrenByParent={childrenByParent} visible={visible} pose={pose} onSelect={onSelect} />)}
  </group>
}

function Trajectory({ poses }: { poses: CellPose[] }) {
  const line = useMemo(() => {
    const points = poses.map((pose) => new Vector3(...pose.transform.translation_mm))
    const geometry = new BufferGeometry().setFromPoints(points)
    return new Line(geometry, new LineBasicMaterial({ color: new Color("#a855f7") }))
  }, [poses])
  if (poses.length < 2) return null
  // This connects exact sampled positions visually; it never creates playback poses.
  return <primitive object={line} />
}

function CameraRig({ preset, resetToken }: { preset: Preset; resetToken: number }) {
  const { camera, invalidate } = useThree()
  const controls = useRef<React.ElementRef<typeof OrbitControls>>(null)
  useEffect(() => {
    camera.up.set(0, 0, 1)
    const positions: Record<Preset, [number, number, number]> = {
      perspective: [650, -700, 520],
      top: [0, 0, 1050],
      front: [0, -1050, 180],
    }
    camera.position.set(...positions[preset])
    controls.current?.target.set(0, 0, 80)
    controls.current?.update()
    invalidate()
  }, [camera, invalidate, preset, resetToken])
  return <OrbitControls ref={controls} makeDefault enablePan enableRotate enableZoom dampingFactor={0.08} />
}

function CellCanvas({ scene, visible, pose, trajectory, selected, onSelect, preset, resetToken }: { scene: CellScene; visible: Set<string>; pose: CellPose | null; trajectory: boolean; selected: CellEntity | null; onSelect: (entity: CellEntity) => void; preset: Preset; resetToken: number }) {
  const children = useMemo(() => {
    const map = new Map<string | null, CellEntity[]>()
    for (const entity of scene.entities) {
      if (!entity.transform) continue
      const parent = entity.transform.parent_frame
      map.set(parent, [...(map.get(parent) ?? []), entity])
    }
    return map
  }, [scene.entities])
  const roots = children.get(null) ?? []
  return <Canvas data-testid="cell-webgl-canvas" frameloop="demand" shadows camera={{ position: [650, -700, 520], near: 0.1, far: 10000, fov: 42, up: [0, 0, 1] }} onPointerMissed={() => selected && onSelect(selected)}>
    <color attach="background" args={["#08111f"]} />
    <ambientLight intensity={0.75} />
    <directionalLight position={[350, -250, 700]} intensity={1.8} castShadow />
    <gridHelper args={[1600, 32, "#334155", "#1e293b"]} rotation={[Math.PI / 2, 0, 0]} />
    {roots.map((entity) => <EntityTree key={entity.id} entity={entity} childrenByParent={children} visible={visible} pose={pose} onSelect={onSelect} />)}
    {trajectory && <Trajectory poses={scene.trajectory_preview} />}
    <CameraRig preset={preset} resetToken={resetToken} />
  </Canvas>
}

export function CellPage() {
  const { selectedRun } = useOperator()
  const sceneQuery = useQuery({ queryKey: ["cell-scene", selectedRun], queryFn: () => api<CellScene>(query("/ui/cell-scene", { run_root: selectedRun })) })
  if (sceneQuery.isPending) return <div className="grid min-h-[60vh] place-items-center text-sm text-muted-foreground">Building cell scene…</div>
  if (sceneQuery.isError || !sceneQuery.data) return <Card className="border-destructive/40"><CardHeader><CardTitle>Cell scene unavailable</CardTitle><CardDescription>{sceneQuery.error instanceof Error ? sceneQuery.error.message : "The selected run could not be composed."}</CardDescription></CardHeader></Card>
  return <CellSceneView key={`${selectedRun}:${sceneQuery.data.default_timeline_id ?? "none"}`} selectedRun={selectedRun} scene={sceneQuery.data} />
}

function CellSceneView({ selectedRun, scene }: { selectedRun: string; scene: CellScene }) {
  const client = useQueryClient()
  const [timelineId, setTimelineId] = useState(scene.default_timeline_id ?? "")
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [trajectory, setTrajectory] = useState(true)
  const [visible, setVisible] = useState(() => new Set(LAYERS))
  const [selected, setSelected] = useState<CellEntity | null>(null)
  const [preset, setPreset] = useState<Preset>("perspective")
  const [resetToken, setResetToken] = useState(0)
  const [webgl] = useState(() => hasWebGL())

  const timeline = scene?.timelines.find((item) => item.id === timelineId)
  const offset = Math.floor(frame / PAGE_SIZE) * PAGE_SIZE
  const timelineQuery = useQuery({
    queryKey: ["cell-timeline", selectedRun, timelineId, offset],
    queryFn: () => api<CellTimelinePage>(query("/ui/cell-scene/timeline", { run_root: selectedRun, timeline_id: timelineId, offset, limit: PAGE_SIZE })),
    enabled: Boolean(timelineId),
  })
  const page = timelineQuery.data
  const pose = page?.poses.find((item) => item.index === frame) ?? null

  useEffect(() => {
    if (!page || !timelineId) return
    for (const adjacent of [page.previous_offset, page.next_offset]) {
      if (adjacent === null) continue
      void client.prefetchQuery({ queryKey: ["cell-timeline", selectedRun, timelineId, adjacent], queryFn: () => api<CellTimelinePage>(query("/ui/cell-scene/timeline", { run_root: selectedRun, timeline_id: timelineId, offset: adjacent, limit: PAGE_SIZE })) })
    }
  }, [client, page, selectedRun, timelineId])

  useEffect(() => {
    if (!playing || !timeline) return
    const timer = window.setInterval(() => setFrame((value) => value + 1 >= timeline.frame_count ? 0 : value + 1), 150)
    return () => window.clearInterval(timer)
  }, [playing, timeline])

  const toggleLayer = (layer: string) => setVisible((current) => { const next = new Set(current); if (next.has(layer)) next.delete(layer); else next.add(layer); return next })
  const unresolved = scene.entities.filter((entity) => entity.status === "unresolved")

  return <div className="space-y-5">
    <PageHeader eyebrow="Dataset contents" title="Cell" description="Read-only, millimetre-accurate view of the configured cell and recorded flange trajectory." />
    {(scene.warnings.length > 0 || unresolved.length > 0) && <div className="flex items-start gap-3 rounded-lg border border-amber-500/35 bg-amber-500/8 p-4 text-sm"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-500" /><div><div className="font-semibold">Scene has unresolved provenance</div><div className="mt-1 text-xs text-muted-foreground">{scene.warnings.map((warning) => warning.message).concat(unresolved.map((entity) => `${entity.label}: ${entity.unresolved_reason}`)).join(" · ")}</div></div></div>}
    <div className="grid grid-cols-[minmax(0,1fr)_340px] gap-5">
      <Card className="overflow-hidden"><CardHeader className="border-b"><div className="flex flex-wrap items-center justify-between gap-3"><div><CardTitle>3D cell</CardTitle><CardDescription>Right-handed · Z-up · millimetres · no pose interpolation</CardDescription></div><div className="flex gap-2"><Button size="sm" variant={preset === "perspective" ? "default" : "outline"} onClick={() => setPreset("perspective")}><Focus />Perspective</Button><Button size="sm" variant={preset === "top" ? "default" : "outline"} onClick={() => setPreset("top")}><ScanLine />Top</Button><Button size="sm" variant={preset === "front" ? "default" : "outline"} onClick={() => setPreset("front")}><Crosshair />Front</Button><Button size="sm" variant="outline" aria-label="Reset cell view" onClick={() => { setPreset("perspective"); setResetToken((value) => value + 1) }}><RotateCcw /></Button></div></div></CardHeader>
        <CardContent className="p-0">{webgl ? <div className="h-[620px]"><CellCanvas scene={scene} visible={visible} pose={pose} trajectory={trajectory} selected={selected} onSelect={setSelected} preset={preset} resetToken={resetToken} /></div> : <div data-testid="cell-webgl-fallback" className="grid h-[620px] place-items-center p-10 text-center"><div><AlertTriangle className="mx-auto mb-3 size-8 text-amber-500" /><div className="font-semibold">WebGL is unavailable</div><p className="mt-2 max-w-md text-sm text-muted-foreground">The component and provenance list remains available. Use a browser with WebGL support to orbit the scene.</p></div></div>}</CardContent>
        <div className="space-y-3 border-t bg-muted/20 p-4"><div className="flex items-center gap-3"><Select value={timelineId || "none"} onValueChange={(value) => { setTimelineId(value === "none" ? "" : value); setFrame(0); setPlaying(false) }}><SelectTrigger className="w-[250px]" aria-label="Timeline"><SelectValue placeholder="No trajectory" /></SelectTrigger><SelectContent>{scene.timelines.length ? scene.timelines.map((item) => <SelectItem key={item.id} value={item.id}>{item.label} · {item.frame_count} frames</SelectItem>) : <SelectItem value="none">No trajectory</SelectItem>}</SelectContent></Select><Button size="icon" variant="outline" aria-label={playing ? "Pause timeline" : "Play timeline"} disabled={!timeline} onClick={() => setPlaying((value) => !value)}>{playing ? <CirclePause /> : <CirclePlay />}</Button><div className="min-w-0 flex-1"><input aria-label="Frame scrubber" className="w-full accent-primary" type="range" min={0} max={Math.max(0, (timeline?.frame_count ?? 1) - 1)} value={frame} disabled={!timeline} onChange={(event) => { setFrame(Number(event.target.value)); setPlaying(false) }} /></div><div className="w-28 text-right font-mono text-xs">{timeline ? `${frame + 1} / ${timeline.frame_count}` : "No frames"}</div></div><div className="flex justify-between text-[11px] text-muted-foreground"><span>{pose ? `Exact frame ${pose.frame_id}${pose.motion ? ` · ${pose.motion}` : ""}` : timelineQuery.isFetching ? "Loading exact pose page…" : "Select a recorded frame"}</span><span>Adjacent pages prefetch automatically</span></div></div>
      </Card>

      <div className="space-y-4"><Card><CardHeader><CardTitle className="text-base">Visibility layers</CardTitle></CardHeader><CardContent className="grid grid-cols-2 gap-2">{LAYERS.map((layer) => <Label key={layer} className="flex items-center gap-2 rounded border p-2 text-xs"><Checkbox checked={visible.has(layer)} onCheckedChange={() => toggleLayer(layer)} />{titleCase(layer)}</Label>)}<Label className="col-span-2 flex items-center gap-2 rounded border p-2 text-xs"><Checkbox checked={trajectory} onCheckedChange={(value) => setTrajectory(value === true)} /><Route className="size-3.5" />Recorded trajectory</Label></CardContent></Card>
        <Card><CardHeader><CardTitle className="text-base">Selection details</CardTitle><CardDescription>Click a component in the scene.</CardDescription></CardHeader><CardContent><SelectionDetails entity={selected} /></CardContent></Card>
        <Card><CardHeader><CardTitle className="text-base">Dataset contents</CardTitle><CardDescription>{scene.object_selection.objectless ? "Objectless RGB-D run" : `${scene.object_selection.instance_count} pose-template instance(s)`}</CardDescription></CardHeader><CardContent className="max-h-80 space-y-2 overflow-auto">{scene.entities.map((entity) => <button key={entity.id} type="button" className="flex w-full items-center gap-2 rounded border px-3 py-2 text-left text-xs hover:bg-muted" onClick={() => setSelected(entity)}>{entity.type === "camera" ? <Camera className="size-3.5" /> : entity.status === "unresolved" ? <EyeOff className="size-3.5 text-destructive" /> : <Eye className="size-3.5" />}<span className="min-w-0 flex-1 truncate">{entity.label}</span><StatusBadge status={entity.status} /></button>)}</CardContent></Card>
      </div>
    </div>
  </div>
}
