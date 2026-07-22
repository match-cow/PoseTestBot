import { Component, useEffect, useMemo, useState, type ReactNode } from "react"
import { OrbitControls } from "@react-three/drei"
import { Canvas } from "@react-three/fiber"
import { AlertTriangle, Box, LoaderCircle } from "lucide-react"
import { BufferGeometry, Float32BufferAttribute, Matrix4, Vector3 } from "three"
import { useWorkpieceOrientationThumbnail } from "@/components/geometry-previews"
import type { CatalogObject, Matrix4x4, PoseTemplatePreviewMesh } from "@/lib/contracts"

function hasWebGLSupport() {
  if (typeof document === "undefined") return false
  try {
    const canvas = document.createElement("canvas")
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"))
  } catch {
    return false
  }
}

class PreviewBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function transformMatrix(value: Matrix4x4) {
  return new Matrix4().set(
    value[0][0], value[0][1], value[0][2], value[0][3],
    value[1][0], value[1][1], value[1][2], value[1][3],
    value[2][0], value[2][1], value[2][2], value[2][3],
    value[3][0], value[3][1], value[3][2], value[3][3],
  )
}

function boundedGeometry(mesh: PoseTemplatePreviewMesh, transform: Matrix4x4) {
  const geometry = new BufferGeometry()
  geometry.setAttribute("position", new Float32BufferAttribute(mesh.vertices.flat(), 3))
  geometry.setIndex(mesh.faces.flat())
  geometry.applyMatrix4(transformMatrix(transform))
  geometry.computeBoundingBox()
  const size = geometry.boundingBox?.getSize(new Vector3()) ?? new Vector3(1, 1, 1)
  const largestDimension = Math.max(size.x, size.y, size.z, Number.EPSILON)
  geometry.center()
  geometry.scale(2 / largestDimension, 2 / largestDimension, 2 / largestDimension)
  geometry.computeVertexNormals()
  geometry.computeBoundingSphere()
  return geometry
}

function BoundedWorkpieceMesh({ mesh, transform }: { mesh: PoseTemplatePreviewMesh; transform: Matrix4x4 }) {
  const geometry = useMemo(() => boundedGeometry(mesh, transform), [mesh, transform])
  useEffect(() => () => geometry.dispose(), [geometry])
  return <mesh geometry={geometry} castShadow receiveShadow><meshStandardMaterial color="#aebd92" roughness={.75} metalness={.04} /></mesh>
}

function FailedPreview({ message }: { message: string }) {
  return <div data-testid="workpiece-preview-error" className="grid size-full place-items-center px-5 text-center text-[11px] text-muted-foreground"><div><AlertTriangle className="mx-auto mb-2 size-5 text-warning" />{message}</div></div>
}

function PreviewCanvas({ object }: { object: CatalogObject }) {
  const thumbnail = useWorkpieceOrientationThumbnail(object)
  return <div className="overflow-hidden rounded-lg border bg-[#0b1218]" data-testid="workpiece-preview-interactive">
    <div className="flex h-8 items-center justify-between border-b border-white/10 px-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-300"><span>Interactive bounded preview</span><span className="font-mono text-[9px] font-normal text-slate-500">Z up</span></div>
    <div className="h-64">
      {thumbnail.isPending ? <div className="grid size-full place-items-center text-[11px] text-slate-400"><LoaderCircle className="mb-2 size-4 animate-spin" />Loading bounded mesh…</div>
        : thumbnail.data ? <PreviewBoundary fallback={<FailedPreview message="Bounded 3D preview unavailable" />}><Canvas aria-label={`Interactive 3D preview of ${object.name}`} camera={{ position: [3.1, -3.4, 2.5], up: [0, 0, 1], fov: 34, near: .01, far: 100 }} dpr={[1, 1.5]} frameloop="demand" shadows>
          <color attach="background" args={["#0b1218"]} />
          <hemisphereLight args={["#dbeafe", "#18222c", 1.6]} />
          <directionalLight position={[3, -4, 6]} intensity={2.3} castShadow />
          <directionalLight position={[-4, 2, 1]} intensity={.65} />
          <BoundedWorkpieceMesh mesh={thumbnail.data.preview_mesh} transform={thumbnail.data.orientation.source_to_placed} />
          <OrbitControls makeDefault enablePan={false} minDistance={1.7} maxDistance={8} dampingFactor={.08} />
        </Canvas></PreviewBoundary>
          : <FailedPreview message="Analyze stable orientations in Pose Templates to create this bounded preview." />}
    </div>
  </div>
}

export function WorkpiecePreviews({ object }: { object: CatalogObject }) {
  const [webgl] = useState(hasWebGLSupport)
  if (!webgl) return <div data-testid="workpiece-preview-fallback" className="grid min-h-52 place-items-center rounded-lg border border-dashed bg-muted/30 p-8 text-center"><div className="max-w-sm"><Box className="mx-auto mb-3 size-7 text-muted-foreground" /><div className="text-sm font-semibold">3D preview is unavailable</div><p className="mt-1 text-xs leading-relaxed text-muted-foreground">WebGL is disabled or unsupported. The bounded static catalogue thumbnails and metadata remain available.</p></div></div>
  return <div data-testid="workpiece-previews"><PreviewCanvas object={object} /></div>
}
