import { useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { ChevronDown, Play } from "lucide-react"
import { toast } from "sonner"
import { StatusBadge } from "@/components/status-badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api, errorMessage } from "@/lib/api"
import type { JsonValue, PipelineParameter, PipelineStage } from "@/lib/contracts"
import { titleCase } from "@/lib/utils"
import { useOperator } from "@/providers/operator-provider"

const HIDDEN_SAFETY_PARAMETERS = new Set(["allow_cameras", "allow_real_robot"])
const UI_NOISE_PARAMETERS = new Set(["json"])

function initialValue(parameter: PipelineParameter): JsonValue {
  if (parameter.default !== null && parameter.default !== undefined) return parameter.default
  return parameter.kind === "bool" ? false : ""
}

function ParameterField({ parameter, value, setValue }: { parameter: PipelineParameter; value: JsonValue; setValue: (value: JsonValue) => void }) {
  const id = `stage-${parameter.name}`
  if (parameter.kind === "bool") return <Label className="flex items-start gap-2 rounded-lg border border-border p-3"><Checkbox id={id} checked={value === true} onCheckedChange={(checked) => setValue(checked === true)} /><span><span className="block">{titleCase(parameter.name)}</span>{parameter.help && <span className="mt-1 block text-xs font-normal text-muted-foreground">{parameter.help}</span>}</span></Label>
  if (parameter.choices.length) return <div className="space-y-2"><Label htmlFor={id}>{titleCase(parameter.name)}{parameter.required ? " *" : ""}</Label><Select value={String(value || "__unset")} onValueChange={(next) => setValue(next === "__unset" ? "" : next)}><SelectTrigger id={id}><SelectValue /></SelectTrigger><SelectContent>{!parameter.required && <SelectItem value="__unset">Use stage default</SelectItem>}{parameter.choices.map((choice) => <SelectItem value={choice} key={choice}>{titleCase(choice)}</SelectItem>)}</SelectContent></Select>{parameter.help && <p className="text-xs text-muted-foreground">{parameter.help}</p>}</div>
  return <div className="space-y-2"><Label htmlFor={id}>{titleCase(parameter.name)}{parameter.required ? " *" : ""}</Label><Input id={id} type={parameter.kind === "int" || parameter.kind === "float" ? "number" : "text"} step={parameter.kind === "float" ? "any" : undefined} value={Array.isArray(value) ? value.join(", ") : String(value ?? "")} onChange={(event) => setValue(event.target.value)} placeholder={parameter.multiple ? "Comma-separated values" : parameter.path_scope ? `${parameter.path_scope} path` : undefined} />{parameter.help && <p className="text-xs text-muted-foreground">{parameter.help}</p>}</div>
}

function normalizedOptions(parameters: PipelineParameter[], values: Record<string, JsonValue>) {
  const options: Record<string, JsonValue> = {}
  for (const parameter of parameters) {
    if (HIDDEN_SAFETY_PARAMETERS.has(parameter.name) || UI_NOISE_PARAMETERS.has(parameter.name)) continue
    const raw = values[parameter.name]
    if (raw === "" || raw === null || raw === undefined) continue
    if (parameter.multiple && typeof raw === "string") options[parameter.name] = raw.split(",").map((item) => item.trim()).filter(Boolean)
    else if (parameter.kind === "int" && typeof raw === "string") options[parameter.name] = Number.parseInt(raw, 10)
    else if (parameter.kind === "float" && typeof raw === "string") options[parameter.name] = Number.parseFloat(raw)
    else options[parameter.name] = raw
  }
  return options
}

export function StageForm({ stage, artifactStatus }: { stage: PipelineStage; artifactStatus?: string }) {
  const { selectedRun } = useOperator()
  const queryClient = useQueryClient()
  const available = stage.parameters.filter((parameter) => !HIDDEN_SAFETY_PARAMETERS.has(parameter.name) && !UI_NOISE_PARAMETERS.has(parameter.name))
  const [values, setValues] = useState<Record<string, JsonValue>>(() => Object.fromEntries(available.map((parameter) => [parameter.name, initialValue(parameter)])))
  const [advanced, setAdvanced] = useState(false)
  const common = available.slice(0, 3)
  const uncommon = available.slice(3)
  const cameraStage = stage.resources.some((resource) => resource === "camera" || resource.startsWith("camera:"))

  const run = useMutation({
    mutationFn: async () => {
      if (cameraStage) await api("/sensors/previews/stop", { method: "POST", body: "{}" })
      return api<{ job_id: string }>("/pipeline/run", { method: "POST", body: JSON.stringify({ stage: stage.id, run_root: selectedRun, options: normalizedOptions(stage.parameters, values) }) })
    },
    onSuccess: (data) => {
      toast.success(`${stage.label} queued`, { description: `Job ${data.job_id}` })
      queryClient.invalidateQueries({ queryKey: ["jobs"] })
      queryClient.invalidateQueries({ queryKey: ["overview", selectedRun] })
      queryClient.invalidateQueries({ queryKey: ["artifacts", selectedRun] })
    },
    onError: (error) => toast.error(`${stage.label} was not queued`, { description: errorMessage(error) }),
  })

  return (
    <Card data-stage-id={stage.id}>
      <CardHeader><div className="flex items-start justify-between gap-3"><div><CardTitle className="text-base">{stage.label}</CardTitle><CardDescription className="mt-1 leading-relaxed">{stage.description}</CardDescription></div><StatusBadge status={artifactStatus ?? "available"} /></div></CardHeader>
      <CardContent className="space-y-4">
        {common.length > 0 && <div className="grid grid-cols-2 gap-3">{common.map((parameter) => <ParameterField key={parameter.name} parameter={parameter} value={values[parameter.name]} setValue={(value) => setValues((current) => ({ ...current, [parameter.name]: value }))} />)}</div>}
        {uncommon.length > 0 && <div><Button type="button" variant="ghost" size="sm" onClick={() => setAdvanced((value) => !value)}><ChevronDown className={advanced ? "rotate-180 transition" : "transition"} />Advanced · {uncommon.length} options</Button>{advanced && <div className="mt-3 grid grid-cols-2 gap-3 rounded-lg border border-border bg-muted/20 p-4">{uncommon.map((parameter) => <ParameterField key={parameter.name} parameter={parameter} value={values[parameter.name]} setValue={(value) => setValues((current) => ({ ...current, [parameter.name]: value }))} />)}</div>}</div>}
        <div className="flex items-center justify-between border-t border-border pt-4"><div className="flex flex-wrap gap-1">{stage.resources.map((resource) => <StatusBadge status="available" key={resource}>{resource}</StatusBadge>)}</div><Button onClick={() => run.mutate()} disabled={run.isPending}><Play />Queue stage</Button></div>
      </CardContent>
    </Card>
  )
}
