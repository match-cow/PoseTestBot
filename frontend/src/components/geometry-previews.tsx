import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import { Box, LoaderCircle, TriangleAlert } from "lucide-react"
import { api } from "@/lib/api"
import type {
  CatalogObject,
  Matrix4x4,
  PoseTemplateBundle,
  PoseTemplateContour,
  PoseTemplateOrientationThumbnail,
  PoseTemplatePreview,
  PoseTemplatePreviewMesh,
  PoseTemplateThumbnail,
} from "@/lib/contracts"
import { IDENTITY_MATRIX_4X4, projectIsometricMesh, type Matrix4x4Tuple } from "@/lib/isometric"
import { cn } from "@/lib/utils"

function faceColor(shade: number) {
  const base = [150, 169, 135]
  const channel = (value: number) => Math.max(0, Math.min(255, Math.round(18 + (value - 18) * shade)))
  return `rgb(${channel(base[0])} ${channel(base[1])} ${channel(base[2])})`
}

export function contourPoints(contour: PoseTemplateContour) {
  return Array.isArray(contour) ? contour : contour.points
}

function footprintPath(contours: Array<Array<{ x_mm: number; y_mm: number }>>) {
  return contours.map((contour) => contour.map((point, index) => `${index ? "L" : "M"} ${point.x_mm} ${point.y_mm}`).join(" ") + " Z").join(" ")
}

export function IsometricMeshPreview({
  mesh,
  transform = IDENTITY_MATRIX_4X4,
  label,
  className,
  testId,
  commonSpan,
}: {
  mesh: PoseTemplatePreviewMesh
  transform?: Matrix4x4 | Matrix4x4Tuple
  label: string
  className?: string
  testId?: string
  commonSpan?: number
}) {
  const result = useMemo(() => {
    try {
      return { projection: projectIsometricMesh(mesh, transform as Matrix4x4Tuple), error: null }
    } catch (error) {
      return { projection: null, error: error instanceof Error ? error.message : "Preview geometry is invalid" }
    }
  }, [mesh, transform])

  if (!result.projection || result.error) {
    return <div data-testid={testId} className={cn("grid size-full min-h-16 place-items-center rounded-md bg-muted/55 px-2 text-center text-[10px] text-muted-foreground", className)} title={result.error ?? undefined}><TriangleAlert className="size-4" /><span className="sr-only">{label} isometric preview unavailable</span></div>
  }
  const { bounds, polygons } = result.projection
  const width = Math.max(0.01, bounds.maxX - bounds.minX)
  const height = Math.max(0.01, bounds.maxY - bounds.minY)
  const span = Math.max(width, height, commonSpan ?? 0.01)
  const centerX = (bounds.minX + bounds.maxX) / 2
  const centerY = (bounds.minY + bounds.maxY) / 2
  const padding = span * 0.09
  return <svg
    data-testid={testId}
    role="img"
    aria-label={`${label} isometric 3D preview`}
    viewBox={`${centerX - span / 2 - padding} ${centerY - span / 2 - padding} ${span + padding * 2} ${span + padding * 2}`}
    preserveAspectRatio="xMidYMid meet"
    className={cn("size-full bg-[#10171d]", className)}
  >
    <title>{label} isometric 3D preview</title>
    {polygons.map((polygon) => <polygon
      key={polygon.faceIndex}
      data-face-index={polygon.faceIndex}
      points={polygon.points.map((point) => `${point.x},${point.y}`).join(" ")}
      fill={faceColor(polygon.shade)}
      stroke="rgba(225,235,218,.42)"
      strokeWidth={Math.max(width, height) / 420}
      strokeLinejoin="round"
    />)}
  </svg>
}

export function orientationAnalysisQueryKey(object: Pick<CatalogObject, "catalog_uuid" | "canonical_ply_sha256" | "geometry_revision">) {
  return ["pose-template-orientations", object.catalog_uuid, object.canonical_ply_sha256 ?? object.geometry_revision ?? 1] as const
}

export function orientationThumbnailQueryKey(object: Pick<CatalogObject, "catalog_uuid" | "canonical_ply_sha256" | "geometry_revision">) {
  return ["pose-template-orientation-thumbnail", object.catalog_uuid, object.canonical_ply_sha256 ?? object.geometry_revision ?? 1] as const
}

export function useWorkpieceOrientationThumbnail(object: CatalogObject, enabled = true) {
  return useQuery({
    queryKey: orientationThumbnailQueryKey(object),
    queryFn: () => api<PoseTemplateOrientationThumbnail>(`/pose-templates/workpieces/${object.catalog_uuid}/orientation-thumbnail`),
    enabled: enabled && Boolean(object.catalog_uuid),
    staleTime: Infinity,
    retry: false,
  })
}

export function WorkpieceIsometricThumbnail({ object, className }: { object: CatalogObject; className?: string }) {
  const thumbnail = useWorkpieceOrientationThumbnail(object)
  const orientation = thumbnail.data?.orientation
  return <div className={cn("grid h-24 w-full place-items-center overflow-hidden rounded-md border bg-muted/30", className)} data-testid={`workpiece-thumbnail-${object.catalog_uuid}`}>
    {thumbnail.isPending ? <LoaderCircle className="size-4 animate-spin text-muted-foreground" />
      : thumbnail.data && orientation ? <IsometricMeshPreview mesh={thumbnail.data.preview_mesh} transform={orientation.source_to_placed} label={object.name} testId={`workpiece-isometric-${object.catalog_uuid}`} />
        : <div className="grid place-items-center gap-1 px-2 text-center text-[9px] text-muted-foreground"><Box className="size-5" /><span>Preview available after orientation analysis</span></div>}
  </div>
}

function unwrapTemplatePreview(value: PoseTemplatePreview | { preview: PoseTemplatePreview }) {
  return "preview" in value ? value.preview : value
}

export function useTemplatePreview(templateUuid: string, enabled = true) {
  return useQuery({
    queryKey: ["pose-template-library-preview", templateUuid],
    queryFn: async () => unwrapTemplatePreview(await api<PoseTemplatePreview | { preview: PoseTemplatePreview }>(`/pose-templates/library/${templateUuid}/preview`)),
    enabled: enabled && Boolean(templateUuid),
    staleTime: Infinity,
    retry: false,
  })
}

export function useTemplateThumbnail(templateUuid: string, enabled = true) {
  return useQuery({
    queryKey: ["pose-template-library-thumbnail", templateUuid],
    queryFn: () => api<PoseTemplateThumbnail>(`/pose-templates/library/${templateUuid}/thumbnail`),
    enabled: enabled && Boolean(templateUuid),
    staleTime: Infinity,
    retry: false,
  })
}

function fallbackPage(bundle?: PoseTemplateBundle) {
  if (bundle?.page?.width_mm && bundle.page.height_mm) return { width_mm: bundle.page.width_mm, height_mm: bundle.page.height_mm }
  const landscape = bundle?.page?.orientation !== "portrait"
  return { width_mm: landscape ? 420 : 297, height_mm: landscape ? 297 : 420 }
}

export function TemplateFootprintThumbnail({ bundle, className }: { bundle: PoseTemplateBundle; className?: string }) {
  const thumbnail = useTemplateThumbnail(bundle.template_uuid)
  const page = thumbnail.data?.page ?? fallbackPage(bundle)
  const pageConfiguration = thumbnail.data?.configuration?.page ?? bundle.configuration?.page ?? bundle.page
  const [originX, originY] = pageConfiguration?.origin_from_lower_left_mm ?? [15, 15]
  const compensation = thumbnail.data?.configuration?.print_compensation ?? bundle.print_compensation ?? { x_scale: 1, y_scale: 1 }
  const compensatedOriginX = page.width_mm / 2 + compensation.x_scale * (originX - page.width_mm / 2)
  const compensatedOriginY = page.height_mm / 2 + compensation.y_scale * (originY - page.height_mm / 2)
  return <div data-testid={`template-thumbnail-${bundle.template_uuid}`} className={cn("surface-grid relative grid h-40 place-items-center overflow-hidden rounded-lg border bg-muted/60 p-3", className)}>
    {thumbnail.isPending ? <LoaderCircle className="size-4 animate-spin text-muted-foreground" /> : thumbnail.data ? <>
      <svg
      viewBox={`0 0 ${page.width_mm} ${page.height_mm}`}
      role="img"
      aria-label={`${bundle.display_name} bounded footprint preview`}
      data-compensated-origin-mm={`${compensatedOriginX.toFixed(3)},${compensatedOriginY.toFixed(3)}`}
      className="max-h-full max-w-full bg-white shadow-sm ring-1 ring-black/15"
    >
      <rect width={page.width_mm} height={page.height_mm} fill="white" stroke="#cbd0c7" />
      <g transform={`translate(0 ${page.height_mm}) scale(1 -1)`}>
        <ellipse cx={compensatedOriginX} cy={compensatedOriginY} rx={2.5 * compensation.x_scale} ry={2.5 * compensation.y_scale} fill="#2374d8" />
        <g transform={`translate(${originX} ${originY})`}>
          {thumbnail.data.instances.map((instance) => <path
            key={instance.instance_uuid}
            d={footprintPath(instance.compensated_contours)}
            fill="rgba(177,203,33,.3)"
            stroke="#667600"
            strokeWidth=".8"
            fillRule="evenodd"
          />)}
        </g>
      </g>
      </svg>
      {thumbnail.data.approximation.truncated ? <span className="absolute bottom-1.5 right-1.5 rounded bg-amber-950/85 px-1.5 py-0.5 text-[8px] font-medium text-amber-100" title={`${thumbnail.data.approximation.included_points} of ${thumbnail.data.approximation.source_points} footprint points shown`}>Simplified</span> : null}
    </> : <div className="grid place-items-center gap-1 text-[10px] text-muted-foreground"><TriangleAlert className="size-4" />Preview unavailable</div>}
  </div>
}
