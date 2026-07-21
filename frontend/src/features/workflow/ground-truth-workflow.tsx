import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { CheckCircle2, RefreshCw, Send } from "lucide-react"
import { toast } from "sonner"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api, errorMessage, query } from "@/lib/api"
import type { PoseTemplateBundle } from "@/lib/contracts"
import { useOperator } from "@/providers/operator-provider"

interface SelectionStatus {
  selection: null | {
    template_uuid: string
    placement_confirmed: boolean
    instances: Array<{ instance_uuid: string; name: string; obj_id: number; template_base_from_object: { translation_mm: number[] } }>
  }
  replacement_blockers: string[]
  ready: boolean
}

function placementMatrix(values: number[]) {
  const [x, y, z, roll, pitch, yaw] = values
  const [r, p, w] = [roll, pitch, yaw].map((value) => value * Math.PI / 180)
  const [cr, sr, cp, sp, cw, sw] = [Math.cos(r), Math.sin(r), Math.cos(p), Math.sin(p), Math.cos(w), Math.sin(w)]
  return [
    [cw * cp, cw * sp * sr - sw * cr, cw * sp * cr + sw * sr, x],
    [sw * cp, sw * sp * sr + cw * cr, sw * sp * cr - cw * sr, y],
    [-sp, cp * sr, cp * cr, z],
    [0, 0, 0, 1],
  ]
}

export function GroundTruthWorkflow() {
  const { selectedRun } = useOperator()
  const client = useQueryClient()
  const [template, setTemplate] = useState("")
  const [placement, setPlacement] = useState([0, 0, 0, 0, 0, 0])
  const [confirmed, setConfirmed] = useState(false)
  const [operator, setOperator] = useState("operator-console")
  const library = useQuery({ queryKey: ["pose-template-library"], queryFn: () => api<{ templates: PoseTemplateBundle[] }>("/pose-templates/library") })
  const status = useQuery({ queryKey: ["pose-template-run", selectedRun], queryFn: () => api<SelectionStatus>(query("/pose-templates/runs/selection", { run_root: selectedRun })), retry: false })
  const active = useMemo(() => library.data?.templates.filter((item) => item.archive.state === "active") ?? [], [library.data])
  const select = useMutation({
    mutationFn: () => api<{ job_id: string }>("/pose-templates/runs/selection", {
      method: "POST",
      body: JSON.stringify({
        run_root: selectedRun,
        template_uuid: template,
        placement: { matrix: placementMatrix(placement) },
        confirmed,
        operator,
      }),
    }),
    onSuccess: (value) => { toast.success("Ground Truth selection queued", { description: `Job ${value.job_id}` }); client.invalidateQueries({ queryKey: ["jobs"] }) },
    onError: (error) => toast.error("Selection was not queued", { description: errorMessage(error) }),
  })

  return <div className="grid grid-cols-[.8fr_1.2fr] gap-4" data-testid="ground-truth-workflow">
    <Card><CardHeader><div className="flex items-start justify-between"><div><CardTitle>Current Ground Truth</CardTitle><CardDescription className="mt-1">Run-owned immutable template snapshot and resolved object poses.</CardDescription></div><StatusBadge status={status.data?.ready ? "ready" : status.data?.selection ? "waiting" : "missing"} /></div></CardHeader><CardContent className="space-y-3">{status.data?.selection ? <><div className="rounded-lg bg-muted p-3 text-xs"><div className="font-semibold">Template {status.data.selection.template_uuid.slice(0, 8)}</div><div className="mt-1 text-muted-foreground">{status.data.selection.instances.length} resolved physical instance{status.data.selection.instances.length === 1 ? "" : "s"} · placement {status.data.selection.placement_confirmed ? "confirmed" : "not confirmed"}</div></div><div className="space-y-1">{status.data.selection.instances.map((item) => <div key={item.instance_uuid} className="flex justify-between rounded border px-3 py-2 text-xs"><span>{item.name} · obj_{String(item.obj_id).padStart(6, "0")}</span><span className="font-mono text-muted-foreground">{item.template_base_from_object.translation_mm.map((value) => value.toFixed(1)).join(", ")} mm</span></div>)}</div></> : <div className="rounded-lg border border-dashed py-10 text-center text-xs text-muted-foreground">No pose template selected. Acquisition and calibration may continue; objectful rendering and BOP export remain blocked.</div>}{status.data?.replacement_blockers.length ? <div className="rounded-lg border border-destructive/35 bg-destructive/5 p-3 text-xs"><div className="font-semibold text-destructive">Selection replacement blocked</div>{status.data.replacement_blockers.map((item) => <div className="mt-1 font-mono text-muted-foreground" key={item}>{item}</div>)}</div> : null}<Button variant="outline" onClick={() => status.refetch()}><RefreshCw />Refresh selection</Button></CardContent></Card>

    <Card><CardHeader><CardTitle>Select and place a library version</CardTitle><CardDescription>The transform maps the printed blue pose-template origin into template_base. Identity is allowed only after explicit operator confirmation.</CardDescription></CardHeader><CardContent className="space-y-4"><div className="grid grid-cols-2 gap-3"><div><Label>Immutable template</Label><Select value={template} onValueChange={setTemplate}><SelectTrigger aria-label="Immutable template"><SelectValue placeholder="Choose a template" /></SelectTrigger><SelectContent>{active.map((item) => <SelectItem value={item.template_uuid} key={item.template_uuid}>{item.display_name} · {item.instances.length} instance{item.instances.length === 1 ? "" : "s"}</SelectItem>)}</SelectContent></Select></div><div><Label>Operator provenance</Label><Input aria-label="Operator provenance" value={operator} onChange={(event) => setOperator(event.target.value)} /></div></div><div><Label>template_base_from_pose_template</Label><div className="mt-1 grid grid-cols-6 gap-2">{["X mm","Y mm","Z mm","Roll °","Pitch °","Yaw °"].map((label, index) => <div key={label}><Label className="text-[10px]">{label}</Label><Input aria-label={`Template placement ${label}`} type="number" step="0.1" value={placement[index]} onChange={(event) => setPlacement((current) => current.map((value, item) => item === index ? Number(event.target.value) : value))} /></div>)}</div></div><div className="flex items-start gap-2 rounded-lg border bg-muted/40 p-3"><Checkbox id="confirm-placement" checked={confirmed} onCheckedChange={(value) => setConfirmed(value === true)} /><div><Label htmlFor="confirm-placement">I confirm this measured physical placement</Label><p className="mt-1 text-[11px] text-muted-foreground">This rigid placement becomes authoritative Ground Truth. Print compensation never changes it.</p></div></div><Button onClick={() => select.mutate()} disabled={!template || !operator.trim() || !confirmed || select.isPending || Boolean(status.data?.replacement_blockers.length)}><Send />{select.isPending ? "Queueing…" : "Select for run"}</Button>{status.data?.ready && <div className="flex items-center gap-2 text-xs text-success"><CheckCircle2 className="size-4" />Ready for object-instance preparation, rendering, and BOP export.</div>}</CardContent></Card>
  </div>
}
