import { useEffect, useMemo, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Archive, Box, Copy, Download, FileUp, LoaderCircle, Plus, RefreshCw, RotateCcw, Save, Trash2 } from "lucide-react"
import { toast } from "sonner"
import { PageHeader } from "@/components/page-header"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { api, errorMessage } from "@/lib/api"
import type { CatalogObject, Job, PoseTemplateBundle, PoseTemplateInstanceDraft, PoseTemplatePreview, PoseTemplateSourceStatus } from "@/lib/contracts"

const zeroPose = () => ({ x_mm: 40, y_mm: 40, z_mm: 0, roll_deg: 0, pitch_deg: 0, yaw_deg: 0 })

function createInstanceUuid() {
  if (typeof globalThis.crypto?.randomUUID === "function") return globalThis.crypto.randomUUID()
  const bytes = new Uint8Array(16)
  if (typeof globalThis.crypto?.getRandomValues === "function") globalThis.crypto.getRandomValues(bytes)
  else for (let index = 0; index < bytes.length; index += 1) bytes[index] = Math.floor(Math.random() * 256)
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("")
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function PoseTemplatesPage() {
  const client = useQueryClient()
  const fileInput = useRef<HTMLInputElement>(null)
  const textureInput = useRef<HTMLInputElement>(null)
  const [texture, setTexture] = useState<File | null>(null)
  const [name, setName] = useState("Object pose template")
  const [description, setDescription] = useState("")
  const [paperSize, setPaperSize] = useState("A3")
  const [orientation, setOrientation] = useState("landscape")
  const [scaleX, setScaleX] = useState(1)
  const [scaleY, setScaleY] = useState(1)
  const [instances, setInstances] = useState<PoseTemplateInstanceDraft[]>([])
  const [previewRequest, setPreviewRequest] = useState<string | null>(null)
  const [pendingUpload, setPendingUpload] = useState<{ id: string; filename: string; startedAt: number } | null>(null)

  const source = useQuery({ queryKey: ["pose-template-source"], queryFn: () => api<PoseTemplateSourceStatus>("/pose-templates/status") })
  const catalog = useQuery({ queryKey: ["pose-template-catalog"], queryFn: () => api<{ objects: CatalogObject[] }>("/pose-templates/catalog") })
  const library = useQuery({ queryKey: ["pose-template-library"], queryFn: () => api<{ templates: PoseTemplateBundle[] }>("/pose-templates/library") })
  const configuration = useMemo(() => ({
    display_name: name,
    description: description || null,
    page: { size: paperSize, orientation },
    print_compensation: { x_scale: scaleX, y_scale: scaleY },
    instances,
  }), [name, description, paperSize, orientation, scaleX, scaleY, instances])

  const preview = useMutation({
    mutationFn: () => api<{ request_id: string }>("/pose-templates/preview", { method: "POST", body: JSON.stringify({ configuration }) }),
    onSuccess: (value) => setPreviewRequest(value.request_id),
    onError: (error) => toast.error("Preview was not queued", { description: errorMessage(error) }),
  })
  const requestPreview = preview.mutate
  const previewResult = useQuery({
    queryKey: ["pose-template-preview", previewRequest],
    queryFn: () => api<PoseTemplatePreview>(`/pose-templates/preview/${previewRequest}`),
    enabled: Boolean(previewRequest),
    retry: true,
    retryDelay: 500,
    staleTime: Infinity,
  })
  useEffect(() => {
    if (!source.data?.available || instances.length === 0) return
    const timer = window.setTimeout(() => requestPreview(), 550)
    return () => window.clearTimeout(timer)
  }, [configuration, instances.length, requestPreview, source.data?.available])

  const upload = useMutation({
    mutationFn: async ({ cad, texture }: { cad: File; texture: File | null }) => {
      const form = new FormData()
      form.set("cad", cad)
      form.set("name", cad.name.replace(/\.[^.]+$/, ""))
      if (texture) form.set("texture", texture)
      return api<{ job_id: string }>("/pose-templates/catalog/upload", { method: "POST", body: form })
    },
    onSuccess: (value, variables) => {
      setTexture(null)
      if (textureInput.current) textureInput.current.value = ""
      setPendingUpload({ id: value.job_id, filename: variables.cad.name, startedAt: Date.now() })
      toast.success("CAD inspection queued", { description: `${variables.cad.name} · job ${value.job_id}` })
    },
    onError: (error) => toast.error("CAD upload failed", { description: errorMessage(error) }),
  })
  const uploadJob = useQuery({
    queryKey: ["pose-template-upload-job", pendingUpload?.id],
    queryFn: () => api<{ job: Job }>(`/jobs/${pendingUpload!.id}`),
    enabled: Boolean(pendingUpload),
    refetchInterval: (state) => ["succeeded", "failed", "canceled"].includes(state.state.data?.job.status ?? "") ? false : 500,
  })
  useEffect(() => {
    const result = uploadJob.data?.job
    if (!result || !pendingUpload || !["succeeded", "failed", "canceled"].includes(result.status)) return
    if (result.status === "succeeded") {
      const seconds = Math.max(1, Math.round((Date.now() - pendingUpload.startedAt) / 1000))
      toast.success("Object added to catalog", { description: `${pendingUpload.filename} · ${seconds} s` })
      client.invalidateQueries({ queryKey: ["pose-template-catalog"] })
    } else {
      toast.error("CAD inspection did not complete", { description: result.message ?? result.tail.at(-1) })
    }
    queueMicrotask(() => setPendingUpload(null))
  }, [client, pendingUpload, uploadJob.data])
  const catalogState = useMutation({
    mutationFn: ({ id, action }: { id: string; action: "archive" | "restore" }) => api(`/pose-templates/catalog/${id}/${action}`, { method: "POST" }),
    onSuccess: () => client.invalidateQueries({ queryKey: ["pose-template-catalog"] }),
    onError: (error) => toast.error("Catalog action failed", { description: errorMessage(error) }),
  })
  const generate = useMutation({
    mutationFn: () => api<{ job_id: string }>("/pose-templates/generate", { method: "POST", body: JSON.stringify({ configuration }) }),
    onSuccess: (value) => toast.success("Immutable template generation queued", { description: `Job ${value.job_id}` }),
    onError: (error) => toast.error("Generation failed", { description: errorMessage(error) }),
  })
  const libraryAction = useMutation({
    mutationFn: ({ id, action }: { id: string; action: "archive" | "restore" | "clone" }) => api(`/pose-templates/library/${id}/${action}`, { method: "POST", body: action === "clone" ? JSON.stringify({}) : undefined }),
    onSuccess: (_, variables) => { toast.success(variables.action === "clone" ? "Clone generation queued" : `Template ${variables.action}d`); client.invalidateQueries({ queryKey: ["pose-template-library"] }) },
    onError: (error) => toast.error("Template action failed", { description: errorMessage(error) }),
  })

  const addInstance = (item: CatalogObject) => setInstances((current) => [...current, { instance_uuid: createInstanceUuid(), catalog_uuid: item.catalog_uuid, pose: zeroPose() }])
  const updatePose = (id: string, key: keyof PoseTemplateInstanceDraft["pose"], raw: string) => setInstances((current) => current.map((item) => item.instance_uuid === id ? { ...item, pose: { ...item.pose, [key]: Number(raw) } } : item))
  const activeObjects = catalog.data?.objects.filter((item) => item.state === "active") ?? []
  const page = previewResult.data?.page ?? { width_mm: orientation === "landscape" ? 420 : 297, height_mm: orientation === "landscape" ? 297 : 420 }

  return <div className="space-y-6" data-testid="pose-templates-page">
    <PageHeader eyebrow="Object ground truth" title="Pose Templates" description="Manage canonical CAD assets, place physical object instances in six degrees of freedom, and generate immutable printable versions." actions={<Button variant="outline" onClick={() => { source.refetch(); catalog.refetch(); library.refetch() }}><RefreshCw />Refresh</Button>} />

    <Card className={source.data?.available ? "border-success/35" : "border-warning/50"}>
      <CardContent className="flex items-start justify-between gap-5 pt-4">
        <div><div className="flex items-center gap-2 text-sm font-semibold">PoseTemplateCreator <StatusBadge status={source.data?.status} /></div><p className="mt-1 text-xs text-muted-foreground">{source.data?.available ? `Pinned revision ${source.data.revision?.slice(0, 12)} · exact closed-contour slicing available` : source.data?.reason ?? "Checking source checkout…"}</p>{!source.data?.available && <code className="mt-2 block rounded bg-muted px-2 py-1 text-[11px]">bash scripts/install.sh --with-posetemplatecreator</code>}</div>
        {source.data?.capabilities && <div className="text-right text-[11px] text-muted-foreground">PLY · STL · OBJ<br />up to {source.data.capabilities.limits.instances} instances</div>}
      </CardContent>
    </Card>

    <section className="space-y-3">
      <div className="flex items-end justify-between"><div><h2 className="text-lg font-semibold">Managed object catalog</h2><p className="text-xs text-muted-foreground">Stable BOP IDs and retained canonical millimetre assets.</p></div><div className="flex items-center gap-2"><input ref={fileInput} type="file" accept=".ply,.stl,.obj" className="hidden" onChange={(event) => { const cad = event.target.files?.[0]; if (cad) upload.mutate({ cad, texture }); event.target.value = "" }} /><input ref={textureInput} type="file" accept="image/png,.png" className="hidden" onChange={(event) => setTexture(event.target.files?.[0] ?? null)} /><Button variant="outline" onClick={() => textureInput.current?.click()} disabled={!source.data?.available || upload.isPending || Boolean(pendingUpload)}><FileUp />{texture ? texture.name : "Optional PNG"}</Button><Button onClick={() => fileInput.current?.click()} disabled={!source.data?.available || upload.isPending || Boolean(pendingUpload)}>{upload.isPending || pendingUpload ? <LoaderCircle className="animate-spin" /> : <FileUp />}{pendingUpload ? "Inspecting CAD…" : "Upload CAD"}</Button></div></div>
      {pendingUpload && <Card className="border-primary/30"><CardContent className="flex items-center justify-between py-4"><div><div className="text-xs uppercase tracking-wide text-muted-foreground">CAD inspection</div><div className="mt-1 flex items-center gap-2 text-sm"><LoaderCircle className="size-4 animate-spin" /><span className="font-medium">{pendingUpload.filename}</span><span className="font-mono text-xs text-muted-foreground">· {uploadJob.data?.job.status ?? "queued"}</span></div><p className="mt-1 text-xs text-muted-foreground">The catalog refreshes automatically when inspection finishes.</p></div></CardContent></Card>}
      <div className="grid grid-cols-3 gap-3">{catalog.data?.objects.map((item) => <Card key={item.catalog_uuid} draggable={item.state === "active"} onDragStart={(event) => event.dataTransfer.setData("application/x-posetestbot-catalog-uuid", item.catalog_uuid)} className={item.state === "archived" ? "opacity-65" : ""}><CardHeader><div className="flex items-start justify-between"><div><CardTitle>{item.name}</CardTitle><CardDescription className="mt-1">obj_{String(item.obj_id).padStart(6, "0")} · {item.source_format.toUpperCase()}</CardDescription></div><StatusBadge status={item.state} /></div></CardHeader><CardContent><div className="mb-3 text-[11px] text-muted-foreground">{item.extraction.vertices.toLocaleString()} vertices · {item.extraction.faces.toLocaleString()} faces · {item.extraction.watertight ? "watertight" : "open mesh"}{item.description ? <><br />{item.description}</> : null}</div><div className="flex flex-wrap gap-2"><Button size="sm" onClick={() => addInstance(item)} disabled={item.state !== "active" || instances.length >= 20}><Plus />Add instance</Button><Button asChild size="sm" variant="outline"><a href={`/pose-templates/catalog/${item.catalog_uuid}/assets/canonical_ply`}><Download />PLY</a></Button><Button asChild size="sm" variant="outline"><a href={`/pose-templates/catalog/${item.catalog_uuid}/assets/source`}><Download />Source</a></Button>{item.assets.texture && <Button asChild size="sm" variant="outline"><a href={`/pose-templates/catalog/${item.catalog_uuid}/assets/texture`}><Download />PNG</a></Button>}<Button size="sm" variant="ghost" onClick={() => catalogState.mutate({ id: item.catalog_uuid, action: item.state === "active" ? "archive" : "restore" })}>{item.state === "active" ? <Archive /> : <RotateCcw />}{item.state === "active" ? "Archive" : "Restore"}</Button></div></CardContent></Card>)}</div>
      {!catalog.isPending && catalog.data?.objects.length === 0 && <Card><CardContent className="py-10 text-center text-sm text-muted-foreground">No managed objects yet. Upload a PLY, STL, or OBJ file to begin.</CardContent></Card>}
    </section>

    <section className="grid grid-cols-[minmax(0,1.15fr)_minmax(380px,.85fr)] gap-4">
      <Card onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); const catalogUuid = event.dataTransfer.getData("application/x-posetestbot-catalog-uuid"); const item = activeObjects.find((candidate) => candidate.catalog_uuid === catalogUuid); if (item && instances.length < 20) addInstance(item) }} aria-label="Pose template instance drop zone"><CardHeader><CardTitle>Template editor</CardTitle><CardDescription>Drag or add catalog objects. Full rigid poses use Rz(yaw) · Ry(pitch) · Rx(roll). CAD and transforms stay nominal; print scaling affects outlines only.</CardDescription></CardHeader><CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3"><div><Label>Name</Label><Input value={name} onChange={(e) => setName(e.target.value)} /></div><div><Label>Description</Label><Textarea className="min-h-[34px]" value={description} onChange={(e) => setDescription(e.target.value)} /></div></div>
        <div className="grid grid-cols-4 gap-3"><div><Label>Paper</Label><Select value={paperSize} onValueChange={setPaperSize}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent>{["A0","A1","A2","A3","A4"].map((value) => <SelectItem value={value} key={value}>{value}</SelectItem>)}</SelectContent></Select></div><div><Label>Orientation</Label><Select value={orientation} onValueChange={setOrientation}><SelectTrigger><SelectValue /></SelectTrigger><SelectContent><SelectItem value="landscape">Landscape</SelectItem><SelectItem value="portrait">Portrait</SelectItem></SelectContent></Select></div><div><Label>X print %</Label><Input aria-label="X print %" type="number" step="0.1" min="50" max="150" value={Number((scaleX * 100).toFixed(3))} onChange={(e) => setScaleX(Number(e.target.value) / 100)} /></div><div><Label>Y print %</Label><Input aria-label="Y print %" type="number" step="0.1" min="50" max="150" value={Number((scaleY * 100).toFixed(3))} onChange={(e) => setScaleY(Number(e.target.value) / 100)} /></div></div>
        <div className="grid grid-cols-3 gap-3"><div><Label>Printable margin mm</Label><Input value="15" readOnly aria-readonly="true" /></div><div><Label>Origin X mm</Label><Input value="15" readOnly aria-readonly="true" /></div><div><Label>Origin Y mm</Label><Input value="15" readOnly aria-readonly="true" /></div></div>
        <div className="rounded-lg border bg-muted/35 p-3 text-[11px] text-muted-foreground">Blue origin: 15 mm from lower-left · +X right · +Y up · +Z out of page. Border, title, and origin stay unscaled.</div>
        <div className="space-y-2">{instances.map((instance, index) => { const object = catalog.data?.objects.find((item) => item.catalog_uuid === instance.catalog_uuid); return <div key={instance.instance_uuid} className="rounded-lg border p-3"><div className="mb-2 flex items-center justify-between"><div className="text-xs font-semibold">{index + 1}. {object?.name ?? instance.catalog_uuid}</div><Button variant="ghost" size="icon" onClick={() => setInstances((current) => current.filter((item) => item.instance_uuid !== instance.instance_uuid))} aria-label="Remove instance"><Trash2 /></Button></div><div className="grid grid-cols-6 gap-2">{(["x_mm","y_mm","z_mm","roll_deg","pitch_deg","yaw_deg"] as const).map((key) => <div key={key}><Label className="text-[10px]">{key.replace("_mm", " mm").replace("_deg", " °")}</Label><Input aria-label={`${object?.name ?? instance.catalog_uuid} ${key}`} type="number" step="0.1" value={instance.pose[key]} onChange={(e) => updatePose(instance.instance_uuid, key, e.target.value)} /></div>)}</div></div> })}</div>
        {instances.length === 0 && <div className="rounded-lg border border-dashed py-9 text-center text-xs text-muted-foreground"><Box className="mx-auto mb-2 size-5" />Add one or more physical instances from the catalog.</div>}
        <div className="flex items-center justify-between"><div className="text-xs">{previewResult.isFetching || preview.isPending ? "Computing exact intersections…" : previewResult.data ? <><StatusBadge status={previewResult.data.valid ? "valid" : "invalid"} /> <span className="ml-2 text-muted-foreground">{previewResult.data.errors[0]?.message ?? "All contours fit the printable work area."}</span></> : "Waiting for geometry"}</div><Button onClick={() => generate.mutate()} disabled={!source.data?.available || !previewResult.data?.valid || generate.isPending}><Save />Generate immutable version</Button></div>
      </CardContent></Card>

      <Card><CardHeader><CardTitle>Exact printable footprint</CardTitle><CardDescription>Server-sliced posed meshes at template Z=0. No projection or bounding-box fallback.</CardDescription></CardHeader><CardContent><div data-testid="pose-template-preview-canvas" className="surface-grid grid min-h-96 place-items-center overflow-auto rounded-lg border bg-muted p-6 shadow-inner"><div data-testid="pose-template-preview-page" className="w-full overflow-hidden bg-white ring-1 ring-black/20 shadow-[0_18px_40px_rgba(15,23,42,.24)]" style={{ aspectRatio: `${page.width_mm} / ${page.height_mm}`, maxWidth: "700px" }}><svg viewBox={`0 0 ${page.width_mm} ${page.height_mm}`} className="h-full w-full" aria-label="Exact pose template preview"><rect x="0" y="0" width={page.width_mm} height={page.height_mm} fill="white" stroke="#cbd0c7" /><circle cx="15" cy={page.height_mm - 15} r="2.5" fill="#2374d8" />{previewResult.data?.instances.flatMap((item) => item.compensated_contours.map((contour, index) => <polygon key={`${item.instance_uuid}-${index}`} points={contour.map((point) => `${15 + point.x_mm},${page.height_mm - 15 - point.y_mm}`).join(" ")} fill="rgba(177,203,33,.28)" stroke="#667600" strokeWidth=".6" />))}</svg></div></div>{previewResult.data?.errors.length ? <div className="mt-3 space-y-1 text-xs text-destructive">{previewResult.data.errors.map((item, index) => <div key={index}>{item.code}: {item.message}</div>)}</div> : null}</CardContent></Card>
    </section>

    <section className="space-y-3"><div><h2 className="text-lg font-semibold">Immutable template library</h2><p className="text-xs text-muted-foreground">Clone to edit. Archive is reversible and never changes selected run snapshots.</p></div><div className="grid grid-cols-3 gap-3">{library.data?.templates.map((item) => <Card key={item.template_uuid}><CardHeader><div className="flex justify-between"><div><CardTitle>{item.display_name}</CardTitle><CardDescription className="mt-1">{item.instances.length} instance{item.instances.length === 1 ? "" : "s"} · {item.template_uuid.slice(0, 8)}</CardDescription></div><StatusBadge status={item.archive.state} /></div></CardHeader><CardContent className="flex flex-wrap gap-2"><Button asChild size="sm"><a href={`/pose-templates/library/${item.template_uuid}/download/pdf`}><Download />PDF</a></Button><Button asChild size="sm" variant="outline"><a href={`/pose-templates/library/${item.template_uuid}/download/manifest`}><Download />Manifest</a></Button><Button size="sm" variant="outline" onClick={() => libraryAction.mutate({ id: item.template_uuid, action: "clone" })}><Copy />Clone</Button><Button size="sm" variant="ghost" onClick={() => libraryAction.mutate({ id: item.template_uuid, action: item.archive.state === "active" ? "archive" : "restore" })}>{item.archive.state === "active" ? <Archive /> : <RotateCcw />}{item.archive.state === "active" ? "Archive" : "Restore"}</Button></CardContent></Card>)}</div></section>
  </div>
}
