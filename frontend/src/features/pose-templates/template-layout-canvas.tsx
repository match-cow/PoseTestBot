import { useMemo, useRef, type KeyboardEvent, type PointerEvent } from "react"
import { contourPoints } from "@/components/geometry-previews"
import type { CatalogObject, PoseTemplateInstanceDraft, PoseTemplateOrientation, PoseTemplatePreviewMesh } from "@/lib/contracts"

export interface PositionedTemplateInstance extends PoseTemplateInstanceDraft {
  object: CatalogObject
  orientation: PoseTemplateOrientation
  preview_mesh: PoseTemplatePreviewMesh
}

type Gesture =
  | { mode: "drag"; pointerId: number; id: string; offsetX: number; offsetY: number }
  | { mode: "rotate"; pointerId: number; id: string }

function pathFor(orientation: PoseTemplateOrientation) {
  return orientation.contours.map((contour) => {
    const [first, ...rest] = contourPoints(contour)
    if (!first) return ""
    return `M ${first.x_mm} ${first.y_mm} ${rest.map((point) => `L ${point.x_mm} ${point.y_mm}`).join(" ")} Z`
  }).join(" ")
}

function rotationFromPointer(origin: { x: number; y: number }, point: { x: number; y: number }) {
  const angle = Math.atan2(point.y - origin.y, point.x - origin.x) * 180 / Math.PI - 90
  const normalized = ((((angle + 180) % 360) + 360) % 360) - 180
  return Object.is(normalized, -0) ? 0 : normalized
}

export function TemplateLayoutCanvas({
  instances,
  selectedId,
  page,
  onSelect,
  onPose,
  onRemove,
}: {
  instances: PositionedTemplateInstance[]
  selectedId: string | null
  page: { width_mm: number; height_mm: number }
  onSelect: (id: string | null) => void
  onPose: (id: string, pose: Partial<PoseTemplateInstanceDraft["pose"]>) => void
  onRemove: (id: string) => void
}) {
  const svgRef = useRef<SVGSVGElement>(null)
  const gesture = useRef<Gesture | null>(null)
  const paths = useMemo(() => new Map(instances.map((item) => [item.instance_uuid, pathFor(item.orientation)])), [instances])

  const pointFromEvent = (event: PointerEvent<SVGElement>) => {
    const svg = svgRef.current
    const matrix = svg?.getScreenCTM()
    if (!svg || !matrix) return null
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse())
    return { x: point.x, y: page.height_mm - point.y }
  }
  const movePointer = (event: PointerEvent<SVGSVGElement>) => {
    const active = gesture.current
    if (!active || active.pointerId !== event.pointerId) return
    const point = pointFromEvent(event)
    const item = instances.find((candidate) => candidate.instance_uuid === active.id)
    if (!point || !item) return
    if (active.mode === "drag") {
      onPose(active.id, { x_mm: point.x - 15 - active.offsetX, y_mm: point.y - 15 - active.offsetY })
    } else {
      onPose(active.id, { rotation_deg: rotationFromPointer({ x: 15 + item.pose.x_mm, y: 15 + item.pose.y_mm }, point) })
    }
  }
  const endPointer = (event: PointerEvent<SVGSVGElement>) => {
    if (gesture.current?.pointerId !== event.pointerId) return
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
    gesture.current = null
  }
  const handleKeyboard = (event: KeyboardEvent<SVGSVGElement>) => {
    if (event.key === "Escape") { event.preventDefault(); onSelect(null); return }
    if (!selectedId) return
    if (event.key === "Delete" || event.key === "Backspace") { event.preventDefault(); onRemove(selectedId); return }
    const distance = event.shiftKey ? 10 : 1
    const delta: Record<string, [number, number]> = { ArrowLeft: [-distance, 0], ArrowRight: [distance, 0], ArrowDown: [0, -distance], ArrowUp: [0, distance] }
    const movement = delta[event.key]
    const item = instances.find((candidate) => candidate.instance_uuid === selectedId)
    if (movement && item) {
      event.preventDefault()
      onPose(selectedId, { x_mm: item.pose.x_mm + movement[0], y_mm: item.pose.y_mm + movement[1] })
    }
  }

  return <div className="surface-grid relative grid min-h-[460px] place-items-center overflow-auto rounded-lg border bg-muted p-6 shadow-inner" data-testid="pose-template-layout-canvas">
    {!instances.length && <div className="pointer-events-none absolute inset-0 grid place-items-center text-center text-xs text-muted-foreground"><div><strong className="block text-sm text-foreground">Choose and orient a workpiece first</strong><span className="mt-1 block">Its exact selected-slice contour will appear here at 1:1 template coordinates.</span></div></div>}
    <svg
      ref={svgRef}
      viewBox={`0 0 ${page.width_mm} ${page.height_mm}`}
      role="application"
      aria-label="Printable pose layout. Select an object, drag to move, use its round handle to rotate, or use arrow keys to nudge."
      tabIndex={0}
      className="max-h-[680px] w-full max-w-[760px] bg-white shadow-[0_18px_40px_rgba(15,23,42,.24)] ring-1 ring-black/20"
      style={{ aspectRatio: `${page.width_mm} / ${page.height_mm}` }}
      onKeyDown={handleKeyboard}
      onPointerMove={movePointer}
      onPointerUp={endPointer}
      onPointerCancel={endPointer}
      onPointerDown={(event) => { if (event.target === event.currentTarget) onSelect(null) }}
    >
      <rect width={page.width_mm} height={page.height_mm} fill="white" />
      <g transform={`translate(0 ${page.height_mm}) scale(1 -1)`}>
        <rect x=".25" y=".25" width={page.width_mm - .5} height={page.height_mm - .5} fill="none" stroke="#cbd0c7" strokeWidth=".5" />
        <rect x="15" y="15" width={page.width_mm - 30} height={page.height_mm - 30} fill="none" stroke="#b2b8ae" strokeWidth=".5" strokeDasharray="3 2" />
        <line x1="15" y1="15" x2={page.width_mm - 15} y2="15" stroke="#d04a42" strokeWidth=".75" />
        <line x1="15" y1="15" x2="15" y2={page.height_mm - 15} stroke="#4f8a52" strokeWidth=".75" />
        {instances.map((item) => {
          const selected = selectedId === item.instance_uuid
          return <g
            key={item.instance_uuid}
            role="button"
            aria-label={`Select and move ${item.object.name}`}
            tabIndex={0}
            transform={`translate(${15 + item.pose.x_mm} ${15 + item.pose.y_mm}) rotate(${item.pose.rotation_deg})`}
            className="cursor-grab focus:outline-none"
            onFocus={() => onSelect(item.instance_uuid)}
            onPointerDown={(event) => {
              event.preventDefault(); event.stopPropagation()
              const point = pointFromEvent(event)
              if (!point || !svgRef.current) return
              svgRef.current.setPointerCapture(event.pointerId)
              gesture.current = { mode: "drag", pointerId: event.pointerId, id: item.instance_uuid, offsetX: point.x - 15 - item.pose.x_mm, offsetY: point.y - 15 - item.pose.y_mm }
              onSelect(item.instance_uuid)
            }}
          >
            <path d={paths.get(item.instance_uuid)} fill={selected ? "rgba(47,114,216,.3)" : "rgba(177,203,33,.28)"} stroke={selected ? "#1762bd" : "#667600"} strokeWidth={selected ? 1.2 : .7} fillRule="evenodd" vectorEffect="non-scaling-stroke" />
            <circle cx="0" cy="0" r="2" fill="#2374d8" stroke="white" strokeWidth=".5" vectorEffect="non-scaling-stroke" />
            {selected && <g><line x1="0" y1="0" x2="0" y2="25" stroke="#1762bd" strokeWidth="1" vectorEffect="non-scaling-stroke" /><circle
              cx="0" cy="25" r="3.5" fill="#1762bd" stroke="white" strokeWidth=".8" vectorEffect="non-scaling-stroke" role="button" aria-label={`Rotate ${item.object.name}`}
              onPointerDown={(event) => { event.preventDefault(); event.stopPropagation(); svgRef.current?.setPointerCapture(event.pointerId); gesture.current = { mode: "rotate", pointerId: event.pointerId, id: item.instance_uuid } }}
            /></g>}
          </g>
        })}
        <circle cx="15" cy="15" r="2.3" fill="#2374d8" />
      </g>
      <text x={page.width_mm - 24} y={page.height_mm - 19} fontSize="7" fontWeight="700" fill="#d04a42">X</text>
      <text x="19" y="24" fontSize="7" fontWeight="700" fill="#4f8a52">Y</text>
      <text x="19" y={page.height_mm - 19} fontSize="7" fontWeight="700" fill="#2374d8">Z</text>
    </svg>
  </div>
}
