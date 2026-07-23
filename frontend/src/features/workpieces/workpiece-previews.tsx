import { Component, Suspense, useEffect, useMemo, useState, type ReactNode } from "react"
import { Html, OrbitControls } from "@react-three/drei"
import { Canvas, useLoader } from "@react-three/fiber"
import { AlertTriangle, Box, LoaderCircle } from "lucide-react"
import { DoubleSide, Vector3 } from "three"
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js"
import type { CatalogObject } from "@/lib/contracts"

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

function canonicalMeshUrl(object: CatalogObject) {
  const revision = object.canonical_ply_sha256 ?? String(object.geometry_revision ?? 1)
  return `/workpieces/catalog/${encodeURIComponent(object.catalog_uuid)}/assets/canonical_ply?revision=${encodeURIComponent(revision)}`
}

function ExactWorkpieceMesh({ object }: { object: CatalogObject }) {
  const url = canonicalMeshUrl(object)
  const geometry = useLoader(PLYLoader, url)
  const placement = useMemo(() => {
    geometry.computeBoundingBox()
    const size = geometry.boundingBox?.getSize(new Vector3()) ?? new Vector3(1, 1, 1)
    const center = geometry.boundingBox?.getCenter(new Vector3()) ?? new Vector3()
    const largestDimension = Math.max(size.x, size.y, size.z, Number.EPSILON)
    if (!geometry.getAttribute("normal")) geometry.computeVertexNormals()
    geometry.computeBoundingSphere()
    const scale = 2 / largestDimension
    return {
      scale,
      position: [-center.x * scale, -center.y * scale, -center.z * scale] as [number, number, number],
    }
  }, [geometry])
  useEffect(() => () => {
    geometry.dispose()
    useLoader.clear(PLYLoader, url)
  }, [geometry, url])
  const vertexColors = Boolean(geometry.getAttribute("color"))
  return <mesh geometry={geometry} scale={placement.scale} position={placement.position}>
    <meshStandardMaterial
      color={vertexColors ? "white" : "#aebd92"}
      vertexColors={vertexColors}
      roughness={.72}
      metalness={.04}
      side={DoubleSide}
    />
  </mesh>
}

function FailedPreview({ message }: { message: string }) {
  return <div data-testid="workpiece-preview-error" className="grid size-full place-items-center px-5 text-center text-[11px] text-muted-foreground"><div><AlertTriangle className="mx-auto mb-2 size-5 text-warning" />{message}</div></div>
}

function PreviewCanvas({ object }: { object: CatalogObject }) {
  return <div className="overflow-hidden rounded-lg border bg-[#0b1218]" data-testid="workpiece-preview-interactive">
    <div className="flex h-8 items-center justify-between border-b border-white/10 px-3 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-300"><span>Full canonical model</span><span className="font-mono text-[9px] font-normal text-slate-500">Exact PLY · Z up</span></div>
    <div className="h-80">
      <PreviewBoundary fallback={<FailedPreview message="Full canonical 3D preview unavailable" />}>
        <Canvas aria-label={`Interactive exact 3D preview of ${object.name}`} camera={{ position: [3.1, -3.4, 2.5], up: [0, 0, 1], fov: 34, near: .01, far: 100 }} dpr={[1, 1.5]} frameloop="demand">
          <color attach="background" args={["#0b1218"]} />
          <hemisphereLight args={["#dbeafe", "#18222c", 1.6]} />
          <directionalLight position={[3, -4, 6]} intensity={2.3} />
          <directionalLight position={[-4, 2, 1]} intensity={.65} />
          <Suspense fallback={<Html center><div className="whitespace-nowrap text-[11px] text-slate-400"><LoaderCircle className="mx-auto mb-2 size-4 animate-spin" />Loading full model…</div></Html>}>
            <ExactWorkpieceMesh object={object} />
          </Suspense>
          <OrbitControls makeDefault enablePan={false} minDistance={1.7} maxDistance={8} dampingFactor={.08} />
        </Canvas>
      </PreviewBoundary>
    </div>
  </div>
}

export function WorkpiecePreviews({ object }: { object: CatalogObject }) {
  const [webgl] = useState(hasWebGLSupport)
  if (!webgl) return <div data-testid="workpiece-preview-fallback" className="grid min-h-52 place-items-center rounded-lg border border-dashed bg-muted/30 p-8 text-center"><div className="max-w-sm"><Box className="mx-auto mb-3 size-7 text-muted-foreground" /><div className="text-sm font-semibold">3D preview is unavailable</div><p className="mt-1 text-xs leading-relaxed text-muted-foreground">WebGL is disabled or unsupported. The bounded static catalogue thumbnails and metadata remain available.</p></div></div>
  return <div data-testid="workpiece-previews"><PreviewCanvas object={object} /></div>
}
