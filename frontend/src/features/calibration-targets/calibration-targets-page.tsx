import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Download, FileJson, FileText, Grid3X3, LoaderCircle, ScanLine, Sparkles } from "lucide-react"
import { Link } from "react-router-dom"
import { toast } from "sonner"
import { EmptyState } from "@/components/empty-state"
import { PageHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ApiError, api, errorMessage, query } from "@/lib/api"
import type { Job } from "@/lib/contracts"
import { useOperator } from "@/providers/operator-provider"

type GeneratorStatus = {
  generation_available: boolean
  generator: { reason: string | null; required_revision: string; checkout: string }
}

type Capabilities = {
  paper_sizes_mm: Record<string, [number, number]>
  dictionaries: Record<string, number>
  defaults: Configuration
}

type Configuration = {
  schema_version: "2.0"
  page: { paper_size: string; orientation: "portrait" | "landscape" }
  board: {
    type: "aruco"
    dictionary: string
    rows: number
    columns: number
    marker_size_mm: number
    separation_mm: number
    show_ids: boolean
    id_font_size_pt: number
  }
  print_compensation: { x_percent: number; y_percent: number }
  annotations: { show_ruler: boolean; show_parameters: boolean; show_frame_legend: boolean }
  coordinate_frame: {
    enabled: boolean
    pose: {
      translation_x_m: number
      translation_y_m: number
      translation_z_m: number
      roll_deg: number
      pitch_deg: number
      yaw_deg: number
    }
  }
}

type Bundle = {
  target_id: string
  display_name?: string
  created_at?: string
  valid: boolean
  error?: string
  selected: boolean
  selected_placement?: { mode: string } | null
  geometry_sha256?: string
  target?: {
    target_bounds: { width_mm: number; height_mm: number }
    print_compensation: { x_percent: number; y_percent: number }
    grid_size?: [number, number]
  }
}

const placementLabels = {
  unknown: "Unknown placement",
  template_base_identity: "Aligned to template base",
  posegridgen_board_to_base: "Use PoseGridGen board pose",
}

export function CalibrationTargetsPage() {
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const status = useQuery({ queryKey: ["calibration-targets", "status"], queryFn: () => api<GeneratorStatus>("/calibration-targets/status"), staleTime: 30_000 })
  const capabilities = useQuery({ queryKey: ["calibration-targets", "capabilities"], queryFn: () => api<Capabilities>("/calibration-targets/capabilities"), enabled: status.data?.generation_available === true, staleTime: Infinity })
  const library = useQuery({ queryKey: ["calibration-targets", "bundles", selectedRun], queryFn: () => api<{ bundles: Bundle[] }>(query("/calibration-targets/bundles", { run_root: selectedRun })) })
  const [configurationOverride, setConfiguration] = useState<Configuration | null>(null)
  const [displayName, setDisplayName] = useState("")
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [previewBusy, setPreviewBusy] = useState(false)
  const [selection, setSelection] = useState<Bundle | null>(null)
  const [placement, setPlacement] = useState<keyof typeof placementLabels>("unknown")
  const [pendingJob, setPendingJob] = useState<{ id: string; kind: "generate" | "select" } | null>(null)

  const configuration = configurationOverride ?? capabilities.data?.defaults ?? null

  const serialized = useMemo(() => configuration ? JSON.stringify(configuration) : "", [configuration])
  useEffect(() => {
    if (!configuration) return
    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      setPreviewBusy(true)
      try {
        const response = await fetch("/calibration-targets/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: serialized, signal: controller.signal })
        if (!response.ok) {
          const body = await response.json().catch(() => null) as { output?: string; errors?: Array<{ message: string }> } | null
          throw new Error(body?.output || body?.errors?.map((item) => item.message).join("; ") || `Preview failed (${response.status})`)
        }
        const next = URL.createObjectURL(await response.blob())
        setPreviewUrl((current) => { if (current) URL.revokeObjectURL(current); return next })
        setPreviewError(null)
      } catch (error) {
        if (!controller.signal.aborted) setPreviewError(errorMessage(error))
      } finally {
        if (!controller.signal.aborted) setPreviewBusy(false)
      }
    }, 350)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [configuration, serialized])

  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl) }, [previewUrl])

  const job = useQuery({
    queryKey: ["calibration-target-job", pendingJob?.id],
    queryFn: () => api<{ job: Job }>(`/jobs/${pendingJob!.id}`),
    enabled: Boolean(pendingJob),
    refetchInterval: (state) => ["succeeded", "failed", "canceled"].includes(state.state.data?.job.status ?? "") ? false : 500,
  })
  useEffect(() => {
    const result = job.data?.job
    if (!result || !pendingJob || !["succeeded", "failed", "canceled"].includes(result.status)) return
    if (result.status === "succeeded") {
      toast.success(pendingJob.kind === "generate" ? "Calibration target generated" : "Calibration target selected")
      queryClient.invalidateQueries({ queryKey: ["calibration-targets", "bundles"] })
      queryClient.invalidateQueries({ queryKey: ["run-config", selectedRun] })
      queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] })
    } else toast.error("Calibration target job did not complete", { description: result.message ?? result.tail.at(-1) })
    queueMicrotask(() => {
      if (result.status === "succeeded") {
        setDisplayName("")
        setSelection(null)
      }
      setPendingJob(null)
    })
  }, [job.data, pendingJob, queryClient, selectedRun])

  const fit = useMutation({
    mutationFn: () => api<{ request: Configuration }>("/calibration-targets/fit", { method: "POST", body: serialized }),
    onSuccess: (result) => { setConfiguration(result.request); toast.success("Board fitted to the selected page") },
    onError: (error) => toast.error("Board could not be fitted", { description: errorMessage(error) }),
  })
  const generate = useMutation({
    mutationFn: () => api<{ job_id: string }>("/calibration-targets/generate", { method: "POST", body: JSON.stringify({ display_name: displayName, configuration }) }),
    onSuccess: (result) => { setPendingJob({ id: result.job_id, kind: "generate" }); toast.success("Generation queued", { description: `Job ${result.job_id}` }) },
    onError: (error) => toast.error("Generation was not queued", { description: errorMessage(error) }),
  })
  const select = useMutation({
    mutationFn: () => api<{ job_id: string }>(`/calibration-targets/bundles/${selection!.target_id}/select`, { method: "POST", body: JSON.stringify({ run_root: selectedRun, placement }) }),
    onSuccess: (result) => { setPendingJob({ id: result.job_id, kind: "select" }); toast.success("Selection queued", { description: `Job ${result.job_id}` }) },
    onError: (error) => {
      const body = error instanceof ApiError && typeof error.body === "object" ? error.body as { blockers?: string[] } : null
      const blockers = Array.isArray(body?.blockers) ? body.blockers : []
      toast.error("Target was not selected", {
        description: blockers.length ? `${errorMessage(error)} Blockers: ${blockers.join(", ")}` : errorMessage(error),
      })
    },
  })

  if (status.isPending) return <div className="grid min-h-[60vh] place-items-center text-sm text-muted-foreground">Checking PoseGridGen…</div>
  if (!status.data?.generation_available) return <EmptyState icon={ScanLine} title="Calibration target generation is unavailable" description={status.data?.generator.reason ?? "The pinned PoseGridGen source checkout could not be initialized."} action={<div className="rounded border bg-muted p-3 text-left font-mono text-xs">git submodule update --init third_party/PoseGridGen<br />bash scripts/install.sh --with-posegridgen</div>} />
  if (!configuration || !capabilities.data) return <div className="grid min-h-[60vh] place-items-center text-sm text-muted-foreground">Loading generator capabilities…</div>

  const active = library.data?.bundles.find((item) => item.selected)
  const setBoard = (values: Partial<Configuration["board"]>) => setConfiguration({ ...configuration, board: { ...configuration.board, ...values } })
  const setPage = (values: Partial<Configuration["page"]>) => setConfiguration({ ...configuration, page: { ...configuration.page, ...values } })
  const setCompensation = (values: Partial<Configuration["print_compensation"]>) => setConfiguration({ ...configuration, print_compensation: { ...configuration.print_compensation, ...values } })
  const setAnnotations = (values: Partial<Configuration["annotations"]>) => setConfiguration({ ...configuration, annotations: { ...configuration.annotations, ...values } })
  const setPose = (values: Partial<Configuration["coordinate_frame"]["pose"]>) => setConfiguration({ ...configuration, coordinate_frame: { ...configuration.coordinate_frame, pose: { ...configuration.coordinate_frame.pose, ...values } } })
  const paperDimensions = capabilities.data.paper_sizes_mm[configuration.page.paper_size] ?? [210, 297]
  const [pageWidthMm, pageHeightMm] = configuration.page.orientation === "portrait" ? paperDimensions : [paperDimensions[1], paperDimensions[0]]
  const previewMaxWidthPx = 700 * pageWidthMm / pageHeightMm

  return <div className="space-y-6">
    <PageHeader eyebrow="Printable calibration geometry" title="Calibration Targets" description="Preview compensated ArUco geometry, generate immutable JSON/PDF bundles, then explicitly select one for the active run." />
    {pendingJob && <Card className="border-primary/30"><CardContent className="flex items-center justify-between py-4"><div><div className="text-xs uppercase tracking-wide text-muted-foreground">{pendingJob.kind === "generate" ? "Generation" : "Selection"} job</div><div className="mt-1 flex items-center gap-2 font-mono text-sm"><LoaderCircle className="size-4 animate-spin" />{pendingJob.id} · {job.data?.job.status ?? "queued"}</div></div><Button variant="outline" asChild><Link to="/jobs">Open Jobs</Link></Button></CardContent></Card>}
    {active && <Card className="border-primary/30"><CardContent className="flex items-center justify-between py-4"><div><div className="text-xs uppercase tracking-wide text-muted-foreground">Active for {selectedRun}</div><div className="mt-1 font-semibold">{active.display_name}</div><div className="font-mono text-xs text-muted-foreground">{active.target_id} · {active.selected_placement?.mode}</div></div><Grid3X3 className="size-7 text-primary-strong" /></CardContent></Card>}
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(420px,0.9fr)] gap-5">
      <Card><CardHeader><CardTitle>ArUco specification</CardTitle><CardDescription>All dimensions are millimetres; X and Y compensation is baked into exact marker corners.</CardDescription></CardHeader><CardContent className="space-y-5">
        <div className="grid grid-cols-3 gap-4"><Field label="Dictionary"><Select value={configuration.board.dictionary} onValueChange={(dictionary) => setBoard({ dictionary })}><SelectTrigger aria-label="Dictionary"><SelectValue /></SelectTrigger><SelectContent>{Object.keys(capabilities.data.dictionaries).map((name) => <SelectItem value={name} key={name}>{name}</SelectItem>)}</SelectContent></Select></Field><NumberField label="Columns" value={configuration.board.columns} min={1} max={100} onChange={(columns) => setBoard({ columns })} /><NumberField label="Rows" value={configuration.board.rows} min={1} max={100} onChange={(rows) => setBoard({ rows })} /></div>
        <div className="grid grid-cols-4 gap-4"><NumberField label="Marker size" value={configuration.board.marker_size_mm} min={0.1} step={0.1} onChange={(marker_size_mm) => setBoard({ marker_size_mm })} /><NumberField label="Separation" value={configuration.board.separation_mm} min={0.1} step={0.1} onChange={(separation_mm) => setBoard({ separation_mm })} /><NumberField label="X print %" value={configuration.print_compensation.x_percent} min={0.1} max={200} step={0.1} onChange={(x_percent) => setCompensation({ x_percent })} /><NumberField label="Y print %" value={configuration.print_compensation.y_percent} min={0.1} max={200} step={0.1} onChange={(y_percent) => setCompensation({ y_percent })} /></div>
        <div className="grid grid-cols-2 gap-4"><Field label="Paper"><Select value={configuration.page.paper_size} onValueChange={(paper_size) => setPage({ paper_size })}><SelectTrigger aria-label="Paper"><SelectValue /></SelectTrigger><SelectContent>{Object.keys(capabilities.data.paper_sizes_mm).map((name) => <SelectItem value={name} key={name}>{name}</SelectItem>)}</SelectContent></Select></Field><Field label="Orientation"><Select value={configuration.page.orientation} onValueChange={(orientation: "portrait" | "landscape") => setPage({ orientation })}><SelectTrigger aria-label="Orientation"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="portrait">Portrait</SelectItem><SelectItem value="landscape">Landscape</SelectItem></SelectContent></Select></Field></div>
        <div className="grid grid-cols-2 gap-2"><Check label="Show ruler" checked={configuration.annotations.show_ruler} onChange={(show_ruler) => setAnnotations({ show_ruler })} /><Check label="Show parameters" checked={configuration.annotations.show_parameters} onChange={(show_parameters) => setAnnotations({ show_parameters })} /><Check label="Show marker IDs" checked={configuration.board.show_ids} onChange={(show_ids) => setBoard({ show_ids })} /><Check label="Show board frame" checked={configuration.annotations.show_frame_legend} onChange={(show_frame_legend) => setAnnotations({ show_frame_legend })} /></div>
        <Check label="Attach optional board-to-base pose" checked={configuration.coordinate_frame.enabled} onChange={(enabled) => setConfiguration({ ...configuration, coordinate_frame: { ...configuration.coordinate_frame, enabled } })} />
        {configuration.coordinate_frame.enabled && <div className="grid grid-cols-3 gap-3 rounded border p-3"><NumberField label="X m" value={configuration.coordinate_frame.pose.translation_x_m} step={0.001} onChange={(translation_x_m) => setPose({ translation_x_m })} /><NumberField label="Y m" value={configuration.coordinate_frame.pose.translation_y_m} step={0.001} onChange={(translation_y_m) => setPose({ translation_y_m })} /><NumberField label="Z m" value={configuration.coordinate_frame.pose.translation_z_m} step={0.001} onChange={(translation_z_m) => setPose({ translation_z_m })} /><NumberField label="Roll°" value={configuration.coordinate_frame.pose.roll_deg} step={0.1} onChange={(roll_deg) => setPose({ roll_deg })} /><NumberField label="Pitch°" value={configuration.coordinate_frame.pose.pitch_deg} step={0.1} onChange={(pitch_deg) => setPose({ pitch_deg })} /><NumberField label="Yaw°" value={configuration.coordinate_frame.pose.yaw_deg} step={0.1} onChange={(yaw_deg) => setPose({ yaw_deg })} /></div>}
        <div className="flex gap-3"><Button variant="outline" onClick={() => fit.mutate()} disabled={fit.isPending}>{fit.isPending ? <LoaderCircle className="animate-spin" /> : <Sparkles />}Fit to page</Button><Input aria-label="Target display name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="Target name" /><Button onClick={() => generate.mutate()} disabled={!displayName.trim() || generate.isPending || pendingJob !== null}>{generate.isPending || pendingJob?.kind === "generate" ? <LoaderCircle className="animate-spin" /> : null}Generate bundle</Button></div>
      </CardContent></Card>
      <Card><CardHeader><CardTitle className="flex items-center gap-2">Preview {previewBusy && <LoaderCircle className="size-4 animate-spin" />}</CardTitle><CardDescription>Debounced PNG from the same pinned renderer used for the persistent PDF.</CardDescription></CardHeader><CardContent>{previewError ? <div className="rounded border border-destructive/40 bg-destructive/5 p-4 text-sm text-destructive">{previewError}</div> : previewUrl ? <div className="space-y-3"><div className="flex items-center justify-between gap-3 text-xs text-muted-foreground"><span>{configuration.page.paper_size} · <span className="capitalize">{configuration.page.orientation}</span></span><span className="font-mono tabular-nums">{pageWidthMm} × {pageHeightMm} mm</span></div><div data-testid="calibration-preview-canvas" className="surface-grid grid min-h-96 place-items-center overflow-auto rounded-lg border bg-muted p-6 shadow-inner sm:p-8"><div data-testid="calibration-preview-page" className="w-full overflow-hidden bg-white ring-1 ring-black/20 shadow-[0_20px_50px_-20px_rgba(0,0,0,0.75),0_2px_5px_rgba(0,0,0,0.35)]" style={{ aspectRatio: `${pageWidthMm} / ${pageHeightMm}`, maxWidth: `${previewMaxWidthPx}px` }}><img src={previewUrl} alt="Calibration target preview" className="block size-full object-contain" /></div></div></div> : <div className="grid h-96 place-items-center text-sm text-muted-foreground">Rendering preview…</div>}</CardContent></Card>
    </div>
    <div><h2 className="text-xl font-semibold">Immutable target library</h2><p className="mt-1 text-sm text-muted-foreground">Generation never changes the active run. Download, inspect, then select deliberately.</p></div>
    <div className="grid grid-cols-3 gap-4">{library.data?.bundles.map((bundle) => <Card key={bundle.target_id} className={bundle.selected ? "border-primary/40" : ""}><CardHeader><CardTitle className="text-base">{bundle.display_name ?? bundle.target_id}</CardTitle><CardDescription className="font-mono text-[10px]">{bundle.target_id}</CardDescription></CardHeader><CardContent className="space-y-3">{bundle.valid ? <><div className="text-xs text-muted-foreground">{bundle.target?.grid_size?.join(" × ")} markers · {bundle.target?.target_bounds.width_mm.toFixed(1)} × {bundle.target?.target_bounds.height_mm.toFixed(1)} mm · {bundle.target?.print_compensation.x_percent}% × {bundle.target?.print_compensation.y_percent}%</div><div className="flex flex-wrap gap-2"><DownloadLink bundle={bundle} artifact="source" icon={<FileJson />} /><DownloadLink bundle={bundle} artifact="target" icon={<FileJson />} /><DownloadLink bundle={bundle} artifact="pdf" icon={<FileText />} /></div><Button className="w-full" variant={bundle.selected ? "outline" : "default"} onClick={() => { setSelection(bundle); setPlacement((bundle.selected_placement?.mode as keyof typeof placementLabels) || "unknown") }}>{bundle.selected ? "Review selection" : "Select for run"}</Button></> : <div className="text-xs text-destructive">Invalid bundle: {bundle.error}</div>}</CardContent></Card>)}</div>
    {library.data?.bundles.length === 0 && <EmptyState icon={Grid3X3} title="No generated targets" description="Name and generate a target above; it will appear here without being selected." />}
    <Dialog open={selection !== null} onOpenChange={(open) => { if (!open) setSelection(null) }}><DialogContent><DialogHeader><DialogTitle>Select {selection?.display_name}</DialogTitle><DialogDescription>This snapshots the complete immutable bundle into the selected run. Changing a target later is blocked once target-dependent calibration output exists.</DialogDescription></DialogHeader><Field label="Placement"><Select value={placement} onValueChange={(value: keyof typeof placementLabels) => setPlacement(value)}><SelectTrigger aria-label="Target placement"><SelectValue /></SelectTrigger><SelectContent>{Object.entries(placementLabels).map(([value, label]) => <SelectItem value={value} key={value}>{label}</SelectItem>)}</SelectContent></Select></Field><DialogFooter><Button variant="outline" onClick={() => setSelection(null)}>Cancel</Button><Button onClick={() => select.mutate()} disabled={select.isPending || pendingJob !== null}>{select.isPending || pendingJob?.kind === "select" ? <LoaderCircle className="animate-spin" /> : null}Select target</Button></DialogFooter></DialogContent></Dialog>
  </div>
}

function Field({ label, children }: { label: string; children: React.ReactNode }) { return <div className="space-y-2"><Label>{label}</Label>{children}</div> }
function NumberField({ label, value, onChange, min, max, step = 1 }: { label: string; value: number; onChange: (value: number) => void; min?: number; max?: number; step?: number }) { return <Field label={label}><Input aria-label={label} type="number" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></Field> }
function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) { return <Label className="flex items-center gap-2 rounded border p-3 text-xs"><Checkbox checked={checked} onCheckedChange={(value) => onChange(value === true)} />{label}</Label> }
function DownloadLink({ bundle, artifact, icon }: { bundle: Bundle; artifact: "source" | "target" | "pdf"; icon: React.ReactNode }) { return <Button variant="outline" size="sm" asChild><a href={`/calibration-targets/bundles/${bundle.target_id}/download/${artifact}`} download>{icon}{artifact.toUpperCase()}<Download /></a></Button> }
