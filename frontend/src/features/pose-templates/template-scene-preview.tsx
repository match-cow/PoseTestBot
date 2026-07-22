import { Component, useEffect, useMemo, useState, type ReactNode } from "react"
import { OrbitControls } from "@react-three/drei"
import { Canvas } from "@react-three/fiber"
import { Box, MousePointer2 } from "lucide-react"
import { BufferGeometry, Float32BufferAttribute, Matrix4 } from "three"
import type { Matrix4x4, PoseTemplateBundle, PoseTemplatePreview, PoseTemplatePreviewMesh } from "@/lib/contracts"

function hasWebGLSupport() {
  if (typeof document === "undefined") return false
  try {
    const canvas = document.createElement("canvas")
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"))
  } catch {
    return false
  }
}

class SceneBoundary extends Component<{ children: ReactNode; fallback: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children
  }
}

function threeMatrix(value: Matrix4x4) {
  return new Matrix4().set(
    value[0][0], value[0][1], value[0][2], value[0][3],
    value[1][0], value[1][1], value[1][2], value[1][3],
    value[2][0], value[2][1], value[2][2], value[2][3],
    value[3][0], value[3][1], value[3][2], value[3][3],
  )
}

function ImmutableMesh({ previewMesh, transform, color }: {
  previewMesh: PoseTemplatePreviewMesh
  transform: Matrix4x4
  color: string
}) {
  const geometry = useMemo(() => {
    const result = new BufferGeometry()
    result.setAttribute("position", new Float32BufferAttribute(previewMesh.vertices.flat(), 3))
    result.setIndex(previewMesh.faces.flat())
    result.computeVertexNormals()
    result.computeBoundingSphere()
    return result
  }, [previewMesh])
  const matrix = useMemo(() => threeMatrix(transform), [transform])

  useEffect(() => () => geometry.dispose(), [geometry])

  return <mesh geometry={geometry} matrix={matrix} matrixAutoUpdate={false} castShadow receiveShadow>
    <meshStandardMaterial color={color} roughness={0.72} metalness={0.04} />
  </mesh>
}

const COLORS = ["#a9bf36", "#4b9bc7", "#d38b3b", "#a471c5", "#5bab7f"]

function EmptyScene({ message }: { message: string }) {
  return <div className="grid size-full place-items-center bg-muted/35 p-6 text-center text-xs text-muted-foreground">
    <div><Box className="mx-auto mb-2 size-6" />{message}</div>
  </div>
}

export function TemplateScenePreview({ bundle, preview }: { bundle: PoseTemplateBundle; preview: PoseTemplatePreview }) {
  const [webgl] = useState(hasWebGLSupport)
  const width = preview.page?.width_mm ?? bundle.page?.width_mm ?? (bundle.page?.orientation === "portrait" ? 297 : 420)
  const height = preview.page?.height_mm ?? bundle.page?.height_mm ?? (bundle.page?.orientation === "portrait" ? 420 : 297)
  const renderable = preview.instances.flatMap((instance) => {
    const transform = instance.pose_template_from_object?.matrix
    const meshHash = instance.preview_mesh_sha256
    const previewMesh = meshHash ? preview.preview_meshes?.[meshHash] : undefined
    return transform && previewMesh ? [{ ...instance, transform, previewMesh }] : []
  })

  if (!webgl) return <EmptyScene message="Interactive 3D is unavailable in this browser. The exact footprint preview remains available above." />
  if (!renderable.length) return <EmptyScene message="This legacy template has no bounded preview mesh. Its exact footprint remains available above." />

  return <div className="relative size-full overflow-hidden bg-[#0b1218]" data-testid="selected-template-scene" data-origin-offset-mm="15,15">
    <SceneBoundary fallback={<EmptyScene message="The immutable 3D assets could not be displayed." />}>
        <Canvas
          aria-label={`Interactive 3D preview of ${bundle.display_name}`}
          camera={{ position: [width * 1.2, -height * 0.85, Math.max(width, height) * 0.9], up: [0, 0, 1], fov: 38, near: .1, far: Math.max(width, height) * 12 }}
          dpr={[1, 1.5]}
          frameloop="demand"
          shadows
        >
          <color attach="background" args={["#0b1218"]} />
          <hemisphereLight args={["#e6eff7", "#12191f", 1.55]} />
          <directionalLight position={[width * .35, -height * .4, Math.max(width, height)]} intensity={2.1} castShadow />
          <directionalLight position={[-width * .2, height, height * .25]} intensity={.55} />
          <mesh position={[width / 2, height / 2, -0.65]} receiveShadow>
            <planeGeometry args={[width, height]} />
            <meshStandardMaterial color="#eef0ea" roughness={.94} />
          </mesh>
          <group position={[15, 15, 0]}>
            {renderable.map((instance, index) => <ImmutableMesh
              key={instance.instance_uuid}
              previewMesh={instance.previewMesh}
              transform={instance.transform}
              color={COLORS[index % COLORS.length]}
            />)}
          </group>
          <OrbitControls makeDefault target={[width / 2, height / 2, 0]} enablePan enableRotate enableZoom dampingFactor={.08} />
        </Canvas>
    </SceneBoundary>
    <div className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-1.5 rounded bg-black/60 px-2 py-1 text-[9px] text-slate-300"><MousePointer2 className="size-3" />Drag to orbit · wheel to zoom</div>
  </div>
}
