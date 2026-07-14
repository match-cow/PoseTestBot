import { useMemo, useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { Archive, Download, Eye, File, Folder, Image, Search } from "lucide-react"
import { PageHeader } from "@/components/page-header"
import { EmptyState } from "@/components/empty-state"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Skeleton } from "@/components/ui/skeleton"
import { api, query } from "@/lib/api"
import type { ArtifactRecord, BopFrameDetail, BopSceneDetail, JsonValue } from "@/lib/contracts"
import { formatDate } from "@/lib/utils"
import { useOperator } from "@/providers/operator-provider"

interface ArtifactPreview {
  relative_path: string
  preview_type: string
  kind: string
  text?: string
  truncated?: boolean
  base64?: string
  children?: Array<{ name: string; kind: string; size_bytes: number | null; modified_at: string }>
  summary?: Record<string, JsonValue> | null
}
function isBopScene(record: ArtifactRecord) {
  return record.summary && typeof record.summary === "object" && !Array.isArray(record.summary) && record.summary.type === "bop_scene"
}

export function ArtifactsPage() {
  const { selectedRun } = useOperator()
  const [search, setSearch] = useState("")
  const [kind, setKind] = useState("all")
  const [selected, setSelected] = useState<ArtifactRecord | null>(null)
  const [frameId, setFrameId] = useState(0)
  const artifacts = useQuery({ queryKey: ["artifacts", selectedRun], queryFn: () => api<{ artifacts: ArtifactRecord[] }>(query("/artifacts", { run_root: selectedRun })), retry: false })
  const preview = useQuery({ queryKey: ["artifact-preview", selectedRun, selected?.path], queryFn: () => api<ArtifactPreview>(query("/artifacts/preview", { run_root: selectedRun, path: selected!.relative_path ?? selected!.path })), enabled: Boolean(selected && !isBopScene(selected)) })
  const scene = useQuery({ queryKey: ["bop-scene", selectedRun, selected?.path], queryFn: () => api<BopSceneDetail>(query("/artifacts/bop-scene", { run_root: selectedRun, path: selected!.relative_path ?? selected!.path })), enabled: Boolean(selected && isBopScene(selected)) })
  const frame = useQuery({ queryKey: ["bop-frame", selectedRun, selected?.path, frameId], queryFn: () => api<BopFrameDetail>(query("/artifacts/bop-frame", { run_root: selectedRun, path: selected!.relative_path ?? selected!.path, image_id: frameId })), enabled: Boolean(selected && isBopScene(selected)) })
  const filtered = useMemo(() => (artifacts.data?.artifacts ?? []).filter((artifact) => artifact.exists && (kind === "all" || artifact.preview_type === kind) && `${artifact.key} ${artifact.relative_path ?? artifact.path} ${artifact.source}`.toLowerCase().includes(search.toLowerCase())), [artifacts.data, kind, search])
  const downloadUrl = selected ? query("/artifacts/file", { run_root: selectedRun, path: selected.relative_path ?? selected.path, download: true }) : "#"
  const fileUrl = selected ? query("/artifacts/file", { run_root: selectedRun, path: selected.relative_path ?? selected.path }) : "#"
  const overlayUrl = selected ? query("/artifacts/bop-frame-overlay", { run_root: selectedRun, path: selected.relative_path ?? selected.path, image_id: frameId }) : "#"

  return <div className="space-y-6">
    <PageHeader eyebrow="Run evidence" title="Artifacts" description="Search durable records, preview files safely, and inspect BOP scenes down to RGB, depth, masks, and ground truth." />
    <div className="flex gap-3"><div className="relative flex-1"><Search className="absolute left-3 top-2.5 size-4 text-muted-foreground" /><Input className="pl-9" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search names, paths, or sources" /></div><Select value={kind} onValueChange={setKind}><SelectTrigger className="w-48"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">All preview types</SelectItem><SelectItem value="text">JSON & text</SelectItem><SelectItem value="image">Images</SelectItem><SelectItem value="directory">Directories</SelectItem><SelectItem value="job_log">Job logs</SelectItem></SelectContent></Select></div>
    {artifacts.isPending ? <div className="space-y-2">{Array.from({ length: 8 }).map((_, index) => <Skeleton className="h-16" key={index} />)}</div> : filtered.length === 0 ? <EmptyState icon={Archive} title="No matching artifacts" description={artifacts.isError ? "This run folder does not exist yet. Configure it from Workflow → Setup." : "Try another filter, or run a workflow stage to create evidence."} /> : <Card><CardContent className="p-0"><div className="divide-y divide-border">{filtered.map((artifact) => {
      const Icon = artifact.preview_type === "image" ? Image : artifact.kind === "directory" ? Folder : File
      return <button key={`${artifact.source}:${artifact.path}`} onClick={() => { setSelected(artifact); setFrameId(0) }} className="grid w-full grid-cols-[40px_minmax(0,1fr)_140px_120px_auto] items-center gap-3 px-4 py-3 text-left transition hover:bg-muted/50"><span className="grid size-9 place-items-center rounded-lg bg-muted"><Icon className="size-4 text-primary-strong" /></span><span className="min-w-0"><span className="block truncate text-sm font-semibold">{artifact.key}</span><span className="block truncate font-mono text-[11px] text-muted-foreground">{artifact.relative_path ?? artifact.path}</span></span><StatusBadge status="available">{artifact.source}</StatusBadge><span className="text-xs text-muted-foreground">{artifact.size_bytes == null ? `${artifact.child_count ?? 0} items` : `${Math.max(1, Math.round(artifact.size_bytes / 1024))} KB`}</span><Button variant="ghost" size="sm" asChild={false}><Eye />Preview</Button></button>
    })}</div></CardContent></Card>}

    <Sheet open={Boolean(selected)} onOpenChange={(open) => !open && setSelected(null)}><SheetContent><SheetHeader><SheetTitle className="font-display text-xl font-semibold">{selected?.key}</SheetTitle><SheetDescription className="break-all">{selected?.relative_path ?? selected?.path}</SheetDescription></SheetHeader>{selected && <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-hidden"><div className="flex items-center justify-between"><StatusBadge status="available">{selected.preview_type}</StatusBadge>{selected.kind !== "directory" && <Button variant="outline" size="sm" asChild><a href={downloadUrl}><Download />Download</a></Button>}</div>
      {isBopScene(selected) ? <div className="grid min-h-0 flex-1 grid-cols-[180px_1fr] gap-4"><div className="overflow-auto rounded-lg border p-2"><div className="mb-2 px-2 text-xs font-semibold">{scene.data?.frame_count ?? 0} frames</div>{scene.data?.frames.map((sceneFrame) => <button key={sceneFrame.image_id} onClick={() => setFrameId(sceneFrame.image_id)} className={`mb-1 flex w-full items-center justify-between rounded px-2 py-2 text-xs ${frameId === sceneFrame.image_id ? "bg-primary text-primary-foreground" : "hover:bg-muted"}`}><span>Frame {sceneFrame.image_id}</span><span>{sceneFrame.gt_count} GT</span></button>)}</div><div className="min-w-0 overflow-auto"><img src={overlayUrl} alt={`BOP frame ${frameId} GT and mask overlay`} className="w-full rounded-lg border bg-black object-contain" /><div className="mt-3 grid grid-cols-3 gap-2 rounded-lg bg-muted p-3 text-xs"><span>{frame.data?.gt_count ?? 0} GT object(s)</span><span>{frame.data?.mask_artifacts.length ?? 0} masks</span><span>{frame.data?.mask_visib_artifacts.length ?? 0} visible masks</span></div><details className="mt-3 rounded-lg border p-3 text-xs"><summary className="font-semibold">Frame camera and GT detail</summary><pre className="mt-3 overflow-auto">{JSON.stringify({ camera: frame.data?.camera, gt: frame.data?.gt, gt_info: frame.data?.gt_info }, null, 2)}</pre></details></div></div> : preview.isPending ? <Skeleton className="flex-1" /> : preview.data?.preview_type === "image" ? <img src={preview.data.base64 ? `data:image/png;base64,${preview.data.base64}` : fileUrl} className="min-h-0 flex-1 rounded-lg border object-contain" alt={selected.key} /> : preview.data?.text !== undefined ? <pre className="min-h-0 flex-1 overflow-auto rounded-lg bg-muted p-4 text-xs leading-relaxed">{preview.data.text}</pre> : <div className="min-h-0 flex-1 overflow-auto rounded-lg border">{preview.data?.children?.map((child) => <div className="flex justify-between border-b px-3 py-2 text-xs" key={child.name}><span>{child.name}</span><span className="text-muted-foreground">{child.kind} · {formatDate(child.modified_at)}</span></div>)}</div>}
    </div>}</SheetContent></Sheet>
  </div>
}
