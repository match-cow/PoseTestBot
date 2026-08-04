import { useEffect, useMemo, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import {
  AlertTriangle,
  Boxes,
  Camera,
  Clock3,
  FileWarning,
  FolderOpen,
  HardDrive,
  Link2,
  LoaderCircle,
  MoveRight,
  RefreshCw,
  Trash2,
} from "lucide-react"
import { toast } from "sonner"

import { EmptyState } from "@/components/empty-state"
import { HelpTip } from "@/components/help-tip"
import { PageHeader } from "@/components/page-header"
import { ProcessHandoff } from "@/components/process-handoff"
import { StatusBadge } from "@/components/status-badge"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { api, errorMessage } from "@/lib/api"
import type { Job, RunFolder, RunFolderInventory, RunFolderInventory as Inventory } from "@/lib/contracts"
import { cn, formatDate, titleCase } from "@/lib/utils"
import { activeWorkflowHref } from "@/lib/workflow-session"
import { useOperator } from "@/providers/operator-provider"

const ACTIVE_JOB_STATUSES = new Set(["queued", "running", "canceling"])
const TERMINAL_JOB_STATUSES = new Set(["succeeded", "failed", "canceled", "cancelled"])

type OperationKind = "refresh" | "move" | "delete"

interface JobSubmission {
  job_id: string
  status: string
  job: Job
}

interface RunFolderMutationResponse extends JobSubmission {
  source_run_root: string
  destination_run_root?: string
  compatibility_alias?: string
}

interface TrackedOperation {
  kind: OperationKind
  runName?: string
  sourceRunRoot?: string
  destinationRunRoot?: string
  compatibilityAlias?: string
  job: Job
}

function formatBytes(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Unavailable"
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
  let amount = value
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  const digits = amount >= 100 || unit === 0 ? 0 : amount >= 10 ? 1 : 2
  return `${amount.toFixed(digits)} ${units[unit]}`
}

function plural(value: number, singular: string) {
  const label = value === 1
    ? singular
    : singular.endsWith("y")
      ? `${singular.slice(0, -1)}ies`
      : `${singular}s`
  return `${value.toLocaleString()} ${label}`
}

function normalizeRunPath(value: string) {
  const absolute = value.startsWith("/")
  const segments: string[] = []
  for (const segment of value.split("/")) {
    if (!segment || segment === ".") continue
    if (segment === "..") segments.pop()
    else segments.push(segment)
  }
  const normalized = `${absolute ? "/" : ""}${segments.join("/")}`
  return normalized || (absolute ? "/" : ".")
}

function isSelectedRunFolder(run: RunFolder, selectedRun: string) {
  const selected = normalizeRunPath(selectedRun)
  return [run.path, ...(run.relocation?.aliases ?? [])]
    .some((candidate) => normalizeRunPath(candidate) === selected)
}

function jobTone(status: string) {
  if (status === "succeeded") return "success" as const
  if (["failed", "canceled", "cancelled"].includes(status)) return "destructive" as const
  return "warning" as const
}

function inventoryTone(state: Inventory["inventory_state"]) {
  if (state === "ready") return "success" as const
  if (state === "missing") return "neutral" as const
  return "warning" as const
}

function storageTone(status: string) {
  if (status === "ready") return "success" as const
  if (status === "warning") return "warning" as const
  if (status === "error") return "destructive" as const
  return "neutral" as const
}

function operationTitle(operation: TrackedOperation) {
  if (operation.kind === "refresh") return "Refreshing run-folder inventory"
  if (operation.kind === "move") return `Moving ${operation.runName ?? "run folder"}`
  return `Deleting ${operation.runName ?? "run folder"}`
}

function operationFromJob(job: Job): TrackedOperation | null {
  const kind = job.parameters.run_folder_operation
  if (kind !== "move" && kind !== "delete") return null
  const source = typeof job.parameters.source_run_root === "string"
    ? job.parameters.source_run_root
    : job.run_root ?? undefined
  const destination = typeof job.parameters.destination_run_root === "string"
    ? job.parameters.destination_run_root
    : undefined
  return {
    kind,
    runName: source?.split("/").filter(Boolean).at(-1),
    sourceRunRoot: source,
    destinationRunRoot: destination,
    compatibilityAlias: kind === "move" ? source : undefined,
    job,
  }
}

function operationResult(job: Job): Record<string, unknown> | null {
  for (const line of [...job.tail].reverse()) {
    try {
      const value: unknown = JSON.parse(line)
      if (value && typeof value === "object" && !Array.isArray(value)) return value as Record<string, unknown>
    } catch {
      // Job logs can contain ordinary progress lines before the final JSON result.
    }
  }
  return null
}

function evidenceEntries(run: RunFolder) {
  return [
    ["raw_capture", "Raw capture", run.contents.evidence.raw_capture],
    ["synchronized", "Synchronized", run.contents.evidence.synchronized],
    ["calibration", "Calibration", run.contents.evidence.calibration],
    ["bop_export", "BOP export", run.contents.evidence.bop_export],
    ["bop_evaluation", "BOP evaluation", run.contents.evidence.bop_evaluation],
  ] as const
}

function RootCapacity({ root }: { root: RunFolderInventory["roots"][number] }) {
  const storage = root.storage
  const usedPercent = storage.free_fraction === null
    ? null
    : Math.max(0, Math.min(100, Math.round((1 - storage.free_fraction) * 100)))
  const detail = storage.error
    ?? (usedPercent === null
      ? "Filesystem capacity is unavailable."
      : `${formatBytes(storage.free_bytes)} free of ${formatBytes(storage.total_bytes)}`)

  return <Card data-testid="run-folder-root" data-root-path={root.path}>
    <CardContent className="pt-4">
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="flex min-w-0 items-start gap-3">
          <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-muted"><HardDrive aria-hidden="true" className="size-4 text-primary-strong" /></span>
          <div className="min-w-0">
            <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">Allowed run root</div>
            <div className="mt-1 truncate font-mono text-[11px] font-semibold" title={root.path}>{root.path}</div>
          </div>
        </div>
        <StatusBadge status={root.exists ? storage.status : "missing"} tone={root.exists ? storageTone(storage.status) : "destructive"} />
      </div>
      <div className="mt-4 flex items-end justify-between gap-3">
        <div><div className="font-display text-lg font-semibold">{formatBytes(storage.free_bytes)} free</div><div className="mt-0.5 text-[10px] text-muted-foreground">{detail}</div></div>
        {storage.filesystem_path && <div className="max-w-[42%] truncate text-right font-mono text-[9px] text-muted-foreground" title={storage.filesystem_path}>{storage.filesystem_path}</div>}
      </div>
      {usedPercent === null
        ? <div className="mt-3 rounded bg-muted px-2 py-1 text-[9px] font-medium text-muted-foreground">Capacity unavailable</div>
        : <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted" role="progressbar" aria-label={`Storage used for ${root.path}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={usedPercent}>
            <div className={cn("h-full rounded-full", storage.status === "error" ? "bg-destructive" : storage.status === "warning" ? "bg-warning" : "bg-primary")} style={{ width: `${usedPercent}%` }} />
          </div>}
    </CardContent>
  </Card>
}

function ContentsSummary({ run }: { run: RunFolder }) {
  const configuredSensors = run.contents.sensors
  const objectNames = run.contents.object_names
  const visibleSensors = configuredSensors.slice(0, 4)
  const visibleObjectNames = objectNames.slice(0, 6)
  return <div data-testid="run-folder-contents" className="space-y-3">
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge variant="outline">{run.contents.dataset_mode ? titleCase(run.contents.dataset_mode) : "Dataset mode unknown"}</Badge>
      {run.contents.resolution && <Badge variant="outline">{run.contents.resolution}</Badge>}
      {run.contents.fps !== null && <Badge variant="outline">{run.contents.fps} FPS</Badge>}
      {run.contents.synchronization_mode && <Badge variant="outline">{titleCase(run.contents.synchronization_mode)}</Badge>}
    </div>

    <div>
      <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.1em] text-muted-foreground"><Camera aria-hidden="true" className="size-3" />Sensors · {run.contents.enabled_sensor_count}/{run.contents.sensor_count} enabled</div>
      {configuredSensors.length
        ? <div className="mt-1.5 space-y-1">{visibleSensors.map((sensor) => <div className="min-w-0 text-[11px]" key={`${sensor.sensor_type}:${sensor.device_id}`}>
            <div className={cn("truncate font-semibold", !sensor.enabled && "text-muted-foreground line-through")} title={sensor.name}>{sensor.name}</div>
            <div className="truncate font-mono text-[9px] text-muted-foreground" title={`${sensor.sensor_type}:${sensor.device_id} · ${sensor.mounting_mode}`}>{sensor.sensor_type}:{sensor.device_id} · {titleCase(sensor.mounting_mode)}{sensor.enabled ? "" : " · disabled"}</div>
          </div>)}{configuredSensors.length > visibleSensors.length && <div className="text-[10px] text-muted-foreground">+{configuredSensors.length - visibleSensors.length} more sensor summaries</div>}</div>
        : <div className="mt-1 text-[11px] text-muted-foreground">No configured sensors</div>}
    </div>

    <div>
      <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.1em] text-muted-foreground"><Boxes aria-hidden="true" className="size-3" />Objects · {run.contents.object_count}</div>
      <div className="mt-1 break-words text-[11px] leading-relaxed text-muted-foreground">{objectNames.length ? <>{visibleObjectNames.join(" · ")}{objectNames.length > visibleObjectNames.length ? ` · +${objectNames.length - visibleObjectNames.length} more` : ""}</> : run.contents.dataset_mode === "objectless" ? "Objectless capture" : "No selected object instances"}</div>
      {run.contents.template_uuid && <div className="mt-1 truncate font-mono text-[9px] text-muted-foreground" title={run.contents.template_uuid}>Template {run.contents.template_uuid}</div>}
    </div>
  </div>
}

function EvidenceSummary({ run }: { run: RunFolder }) {
  const available = evidenceEntries(run).filter(([, , exists]) => exists)
  return <div className="flex max-w-[240px] flex-wrap gap-1.5">
    {available.length
      ? available.map(([id, label]) => <StatusBadge key={id} status="available" tone="success">{label}</StatusBadge>)
      : <span className="text-[11px] leading-relaxed text-muted-foreground">No durable capture or processing evidence yet.</span>}
  </div>
}

function RunDetails({ run }: { run: RunFolder }) {
  const breakdown = Object.entries(run.breakdown).sort(([, left], [, right]) => right.size_bytes - left.size_bytes)
  return <details className="group">
    <summary className="cursor-pointer text-[11px] font-semibold text-primary-strong underline-offset-4 hover:underline">Storage breakdown and provenance</summary>
    <div className="mt-3 grid gap-3 border-l-2 border-primary/20 pl-3 xl:grid-cols-2">
      <div>
        <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-muted-foreground">Breakdown</div>
        {breakdown.length
          ? <dl className="mt-1.5 space-y-1">{breakdown.map(([name, value]) => <div className="grid grid-cols-[minmax(0,1fr)_auto] gap-3 text-[10px]" key={name}><dt className="truncate" title={name}>{titleCase(name)}</dt><dd className="font-mono text-muted-foreground">{formatBytes(value.size_bytes)} · {plural(value.file_count, "file")}</dd></div>)}</dl>
          : <div className="mt-1 text-[10px] text-muted-foreground">No category breakdown available.</div>}
      </div>
      <div>
        <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-muted-foreground">Filesystem scan</div>
        <div className="mt-1.5 text-[10px] text-muted-foreground">{plural(run.directory_count, "directory")} · {plural(run.file_count, "file")} · {plural(run.symlink_count, "symlink")} · {formatBytes(run.allocated_bytes)} allocated</div>
        {run.scan_errors.length > 0 && <ul className="mt-2 list-disc space-y-1 pl-4 text-[10px] text-destructive">{run.scan_errors.slice(0, 10).map((error, index) => <li key={`${index}:${error}`}>{error}</li>)}{run.scan_errors.length > 10 && <li>{run.scan_errors.length - 10} more scan errors; inspect the inventory job log for full context.</li>}</ul>}
        {run.relocation && <div className="mt-2 rounded border bg-muted/30 p-2 text-[10px] text-muted-foreground">
          <div className="flex items-center gap-1 font-semibold text-foreground"><Link2 aria-hidden="true" className="size-3" />Relocated {plural(run.relocation.history_count, "time")}</div>
          <div className="mt-1 break-all font-mono">{run.relocation.original_path}</div>
          {run.relocation.aliases.map((alias) => <div className="mt-0.5 break-all font-mono" key={alias}>Alias {alias}</div>)}
        </div>}
      </div>
    </div>
  </details>
}

export function RunFoldersPage() {
  const queryClient = useQueryClient()
  const { currentWorkflow, selectedRun } = useOperator()
  const [moveTarget, setMoveTarget] = useState<RunFolder | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<RunFolder | null>(null)
  const [destinationRoot, setDestinationRoot] = useState("")
  const [trackedOperation, setTrackedOperation] = useState<TrackedOperation | null>(null)
  const automaticRefreshSignature = useRef<string | null>(null)
  const handledTerminalJob = useRef<string | null>(null)
  const workflowHref = currentWorkflow ? activeWorkflowHref(currentWorkflow) : "/workflow/setup"

  const inventory = useQuery({
    queryKey: ["run-folders"],
    queryFn: () => api<RunFolderInventory>("/ui/run-folders"),
    refetchInterval: (state) => state.state.data?.inventory_state === "refreshing" && !state.state.data.refresh_job ? 1_000 : false,
  })

  const refreshInventory = useMutation({
    mutationFn: (automatic: boolean) => api<JobSubmission>("/ui/run-folders/refresh", { method: "POST", body: "{}" }).then((result) => ({ result, automatic })),
    onSuccess: ({ result, automatic }) => {
      setTrackedOperation({ kind: "refresh", job: result.job })
      if (!automatic) toast.success("Run-folder inventory refresh queued", { description: `Job ${result.job_id} continues in the background.` })
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
    },
    onError: (error) => toast.error("Run-folder inventory could not be refreshed", { description: errorMessage(error) }),
  })

  const inventoryRefreshOperation: TrackedOperation | null = inventory.data?.refresh_job && ACTIVE_JOB_STATUSES.has(inventory.data.refresh_job.status)
    ? { kind: "refresh", job: inventory.data.refresh_job }
    : null
  const inventoryStorageOperation = inventory.data?.operation_job && ACTIVE_JOB_STATUSES.has(inventory.data.operation_job.status)
    ? operationFromJob(inventory.data.operation_job)
    : null
  const effectiveOperation = trackedOperation ?? inventoryStorageOperation ?? inventoryRefreshOperation

  useEffect(() => {
    const data = inventory.data
    if (!data) return
    const refreshNeeded = data.inventory_state === "missing" || data.inventory_state === "stale" || data.stale
    if (!refreshNeeded) {
      automaticRefreshSignature.current = null
      return
    }
    const recoveryNeedsAttention = (data.maintenance?.unresolved_count ?? 0) > 0
    if (recoveryNeedsAttention || data.inventory_state === "refreshing" || effectiveOperation || refreshInventory.isPending) return
    const signature = `${data.inventory_state}:${data.generated_at ?? "never"}`
    if (automaticRefreshSignature.current === signature) return
    automaticRefreshSignature.current = signature
    refreshInventory.mutate(true)
  }, [effectiveOperation, inventory.data, refreshInventory])

  const operationJob = useQuery({
    queryKey: ["run-folder-operation-job", effectiveOperation?.job.id],
    queryFn: () => api<{ job: Job }>(`/jobs/${effectiveOperation!.job.id}`),
    enabled: Boolean(effectiveOperation),
    refetchInterval: (state) => ACTIVE_JOB_STATUSES.has(state.state.data?.job.status ?? effectiveOperation?.job.status ?? "") ? 1_000 : false,
  })
  const currentOperationJob = operationJob.data?.job ?? effectiveOperation?.job ?? null
  const operationActive = Boolean(currentOperationJob && ACTIVE_JOB_STATUSES.has(currentOperationJob.status))
  const operationBlocking = Boolean(effectiveOperation)

  useEffect(() => {
    if (!effectiveOperation || !currentOperationJob || !TERMINAL_JOB_STATUSES.has(currentOperationJob.status)) return
    const key = `${currentOperationJob.id}:${currentOperationJob.status}`
    if (handledTerminalJob.current === key) return
    handledTerminalJob.current = key
    const refreshQueries = Promise.all([
      queryClient.invalidateQueries({ queryKey: ["run-folders"] }),
      queryClient.invalidateQueries({ queryKey: ["runs"] }),
      queryClient.invalidateQueries({ queryKey: ["jobs"] }),
    ])
    if (currentOperationJob.status === "succeeded") {
      const result = operationResult(currentOperationJob)
      if (effectiveOperation.kind === "move" && result?.source_cleanup_complete === false) {
        toast.warning("Run folder moved; source cleanup remains", {
          description: typeof result.source_cleanup_warning === "string"
            ? `${result.source_cleanup_warning} Recovery will retry during the next inventory refresh.`
            : "Recovery will retry during the next inventory refresh.",
        })
      } else {
        toast.success(
          effectiveOperation.kind === "refresh" ? "Run-folder inventory refreshed" : effectiveOperation.kind === "move" ? "Run folder moved" : "Run folder deleted",
          { description: effectiveOperation.kind === "move" && effectiveOperation.destinationRunRoot ? effectiveOperation.destinationRunRoot : effectiveOperation.runName },
        )
      }
    } else {
      toast.error(
        effectiveOperation.kind === "refresh" ? "Run-folder inventory refresh did not complete" : effectiveOperation.kind === "move" ? "Run folder was not moved" : "Run folder was not deleted",
        { description: currentOperationJob.message ?? `Job ${currentOperationJob.id} ended with status ${currentOperationJob.status}.` },
      )
    }
    void refreshQueries.finally(() => {
      setTrackedOperation((current) => current?.job.id === currentOperationJob.id ? null : current)
    })
  }, [currentOperationJob, effectiveOperation, queryClient])

  const moveRun = useMutation({
    mutationFn: ({ run, root }: { run: RunFolder; root: string }) => {
      const destination = inventory.data?.roots.find((item) => item.path === root)
      if (!destination?.identity) throw new Error("Refresh inventory before selecting this destination root")
      return api<RunFolderMutationResponse>("/ui/run-folders/move", {
        method: "POST",
        body: JSON.stringify({
          run_root: run.path,
          destination_root: root,
          expected_identity: run.identity,
          expected_destination_root_identity: destination.identity,
        }),
      })
    },
    onSuccess: (result, variables) => {
      setTrackedOperation({
        kind: "move",
        runName: variables.run.name,
        sourceRunRoot: result.source_run_root,
        destinationRunRoot: result.destination_run_root,
        compatibilityAlias: result.compatibility_alias,
        job: result.job,
      })
      setMoveTarget(null)
      toast.success("Run-folder move queued", { description: `Job ${result.job_id} continues in the background and is visible in Jobs.` })
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
    },
    onError: (error) => toast.error("Run folder could not be moved", { description: errorMessage(error) }),
  })

  const deleteRun = useMutation({
    mutationFn: (run: RunFolder) => api<RunFolderMutationResponse>("/ui/run-folders", {
      method: "DELETE",
      body: JSON.stringify({
        run_root: run.path,
        confirm: true,
        expected_identity: run.identity,
      }),
    }),
    onSuccess: (result, run) => {
      setTrackedOperation({ kind: "delete", runName: run.name, sourceRunRoot: result.source_run_root, job: result.job })
      setDeleteTarget(null)
      toast.success("Run-folder deletion queued", { description: `Job ${result.job_id} continues in the background and is visible in Jobs.` })
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
    },
    onError: (error) => toast.error("Run folder could not be deleted", { description: errorMessage(error) }),
  })

  const runs = useMemo(
    () => [...(inventory.data?.runs ?? [])].sort((left, right) => right.size_bytes - left.size_bytes || right.modified_at.localeCompare(left.modified_at)),
    [inventory.data?.runs],
  )
  const totalSize = runs.reduce((total, run) => total + run.size_bytes, 0)
  const totalFiles = runs.reduce((total, run) => total + run.file_count, 0)
  const maintenanceBlocking = (inventory.data?.maintenance?.unresolved_count ?? 0) > 0
  const destinationRoots = (inventory.data?.roots ?? []).filter((root) => root.path !== moveTarget?.root)
  const inventoryRefreshing = inventory.data?.inventory_state === "refreshing"
    || Boolean(inventory.data?.refresh_job && ACTIVE_JOB_STATUSES.has(inventory.data.refresh_job.status))
    || refreshInventory.isPending
    || Boolean(effectiveOperation?.kind === "refresh" && operationActive)
  const inventoryReadyForMutation = inventory.data?.inventory_state === "ready"
    && inventory.data.stale === false
    && !inventoryRefreshing

  const openMove = (run: RunFolder) => {
    const target = (inventory.data?.roots ?? []).find((root) => root.path !== run.root && root.exists && root.identity)
    setDestinationRoot(target?.path ?? "")
    setMoveTarget(run)
  }

  return <div className="space-y-6">
    <PageHeader
      eyebrow="Lab-wide storage"
      title="Run folders"
      description="Compare every discovered acquisition run across the allowed storage roots, including its measured disk footprint, configured sensors and objects, durable evidence, and relocation history."
      actions={<Button variant="outline" onClick={() => refreshInventory.mutate(false)} disabled={inventoryRefreshing || operationBlocking}><RefreshCw className={inventoryRefreshing ? "animate-spin" : undefined} />Refresh inventory</Button>}
    />
    <ProcessHandoff
      title="Manage storage here; acquire and process data in Workflow"
      description="Moving or deleting a run changes its managed storage, not its captured configuration. Return to the guided workflow to configure, record, synchronize, or export the active run."
      to={workflowHref}
      action="Open workflow"
    />

    {inventory.data?.maintenance && (inventory.data.maintenance.recovered_count > 0 || inventory.data.maintenance.unresolved_count > 0) && <Card data-testid="run-folder-maintenance" className={inventory.data.maintenance.unresolved_count > 0 ? "border-destructive/45 bg-destructive/5" : "border-primary/35 bg-primary/5"}>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle>{inventory.data.maintenance.unresolved_count > 0 ? "Storage recovery needs attention" : "Interrupted storage work recovered"}</CardTitle>
            <CardDescription className="mt-1">{inventory.data.maintenance.unresolved_count > 0
              ? "PoseTestBot preserved every path it could not verify. Correct the reported filesystem issue, then refresh inventory to retry the durable recovery record."
              : `${plural(inventory.data.maintenance.recovered_count, "interrupted operation")} recovered before this inventory was measured.`}</CardDescription>
          </div>
          <Button asChild size="sm" variant="outline"><Link to="/jobs">Open Jobs</Link></Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {inventory.data.maintenance.transactions.length > 0 && <div className="flex flex-wrap gap-1.5">{inventory.data.maintenance.transactions.map((item) => <StatusBadge key={item.transaction_id} status="recovered" tone="success">{titleCase(item.action)} · {item.transaction_id.slice(0, 8)}</StatusBadge>)}</div>}
        {inventory.data.maintenance.unresolved.length > 0 && <ul className="space-y-2">{inventory.data.maintenance.unresolved.map((item, index) => <li className="rounded border border-destructive/25 bg-card p-3 text-xs" key={item.transaction_id ?? `invalid:${index}`}>
          <div className="flex flex-wrap items-center gap-2"><StatusBadge status="attention" tone="destructive">{item.operation ? titleCase(item.operation) : "Unknown operation"}</StatusBadge><span className="font-semibold">{item.remnant_bytes !== null ? `${formatBytes(item.remnant_bytes)} retained` : "Retained size unavailable"}</span>{item.transaction_id && <span className="font-mono text-[10px] text-muted-foreground">{item.transaction_id}</span>}</div>
          <p className="mt-2 break-words text-muted-foreground">{item.error}</p>
        </li>)}</ul>}
      </CardContent>
    </Card>}

    {effectiveOperation && operationActive && currentOperationJob && <Card data-testid="run-folder-operation-status" className="border-warning/40 bg-warning/5">
      <CardContent className="flex flex-col gap-4 py-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <LoaderCircle aria-hidden="true" className="mt-0.5 size-5 shrink-0 animate-spin text-warning-foreground" />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2"><span className="font-semibold">{operationTitle(effectiveOperation)}</span><StatusBadge status={currentOperationJob.status} tone={jobTone(currentOperationJob.status)} /></div>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{effectiveOperation.kind === "refresh"
              ? "This background inventory and recovery job continues after navigation and cannot be canceled safely after submission. Jobs shows its resource lock, live output, and final status."
              : "This background storage job continues after navigation and cannot be canceled safely after submission. Jobs shows its resource lock, live output, and final status."}</p>
            {effectiveOperation.kind === "move" && effectiveOperation.sourceRunRoot && <div className="mt-2 break-all font-mono text-[10px] text-muted-foreground">{effectiveOperation.sourceRunRoot} → {effectiveOperation.destinationRunRoot ?? "destination pending"}</div>}
            {effectiveOperation.compatibilityAlias && <div className="mt-1 break-all font-mono text-[10px] text-muted-foreground">Compatibility link: {effectiveOperation.compatibilityAlias}</div>}
          </div>
        </div>
        <Button asChild size="sm" variant="outline" className="shrink-0 bg-card"><Link to="/jobs">Open Jobs</Link></Button>
      </CardContent>
    </Card>}

    {inventory.data && <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_repeat(3,minmax(130px,auto))]">
      <Card>
        <CardContent className="flex h-full items-center justify-between gap-4 py-4">
          <div><div className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">Inventory snapshot</div><div className="mt-1 text-sm font-semibold">{inventory.data.generated_at ? formatDate(inventory.data.generated_at) : "Not generated yet"}</div></div>
          <StatusBadge status={inventory.data.inventory_state} tone={inventoryTone(inventory.data.inventory_state)}>{inventory.data.stale ? "stale" : inventory.data.inventory_state}</StatusBadge>
        </CardContent>
      </Card>
      <Card><CardContent className="py-4"><div className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">Run folders</div><div className="mt-1 font-display text-xl font-semibold">{runs.length.toLocaleString()}</div></CardContent></Card>
      <Card><CardContent className="py-4"><div className="flex items-center gap-1 text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">Measured size <HelpTip label="measured run-folder size">Logical byte size of all regular files in each run. Allocated disk usage and scan errors remain available in row details.</HelpTip></div><div className="mt-1 font-display text-xl font-semibold">{formatBytes(totalSize)}</div></CardContent></Card>
      <Card><CardContent className="py-4"><div className="text-[10px] font-bold uppercase tracking-[0.12em] text-muted-foreground">Files indexed</div><div className="mt-1 font-display text-xl font-semibold">{totalFiles.toLocaleString()}</div></CardContent></Card>
    </div>}

    {inventory.data && <section aria-labelledby="run-folder-roots-heading" className="space-y-3">
      <div><h2 id="run-folder-roots-heading" className="font-display text-lg font-semibold">Allowed roots and capacity</h2><p className="mt-1 text-xs text-muted-foreground">Moves preserve the run-folder name and target one of these configured roots. Capacity is reported by the filesystem containing each root.</p></div>
      <div className="grid gap-3 lg:grid-cols-2">{inventory.data.roots.map((root) => <RootCapacity key={root.path} root={root} />)}</div>
    </section>}

    {inventory.isPending
      ? <div className="space-y-3">{Array.from({ length: 5 }).map((_, index) => <Skeleton className="h-40" key={index} />)}</div>
      : inventory.isError
        ? <Card className="border-destructive/40"><CardHeader><CardTitle>Run-folder inventory unavailable</CardTitle><CardDescription>{errorMessage(inventory.error)}</CardDescription></CardHeader><CardContent><Button variant="outline" onClick={() => inventory.refetch()}><RefreshCw />Try again</Button></CardContent></Card>
        : runs.length === 0
          ? <EmptyState icon={FolderOpen} title={inventoryRefreshing ? "Run-folder inventory is being measured" : "No run folders found"} description={inventoryRefreshing ? "The background inventory job is measuring allowed roots. This page updates when it completes." : "Configured acquisition runs will appear after they contain a run configuration or dataset manifest and the inventory is refreshed."} />
          : <Card>
            <CardHeader className="border-b bg-muted/20">
              <div className="flex flex-wrap items-end justify-between gap-3"><div><CardTitle>Existing run folders</CardTitle><CardDescription className="mt-1">Largest folders are shown first. Actions are disabled for the active run and while another storage operation is running.</CardDescription></div><div className="font-mono text-[10px] text-muted-foreground">{formatBytes(totalSize)} · {plural(totalFiles, "file")}</div></div>
            </CardHeader>
            <CardContent className="p-0">
              <div data-testid="run-folders-table" className="overflow-x-auto">
                <table className="w-full min-w-[1280px] table-fixed text-left">
                  <thead className="border-b bg-muted/35 text-[9px] font-bold uppercase tracking-[0.12em] text-muted-foreground">
                    <tr><th className="w-[19%] px-4 py-3">Run</th><th className="w-[15%] px-4 py-3">Measured size</th><th className="w-[25%] px-4 py-3">Configuration and contents</th><th className="w-[16%] px-4 py-3">Evidence</th><th className="w-[14%] px-4 py-3">Location</th><th className="w-[11%] px-4 py-3 text-right">Actions</th></tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {runs.map((run) => {
                      const active = isSelectedRunFolder(run, selectedRun)
                      const actionDisabled = active || operationBlocking || maintenanceBlocking || !inventoryReadyForMutation
                      const hasDestination = (inventory.data?.roots ?? []).some((root) => root.exists && root.identity && root.path !== run.root)
                      const canMove = hasDestination && !actionDisabled
                      return <tr data-testid="run-folder-row" data-run-path={run.path} className="align-top" key={run.path}>
                        <td className="px-4 py-4">
                          <div className="flex flex-wrap items-center gap-2"><span className="break-words font-semibold">{run.config.run_name || run.name}</span>{active && <StatusBadge status="active" tone="informational">Active run</StatusBadge>}{!run.scan_complete && <StatusBadge status="partial" tone="warning">Partial scan</StatusBadge>}</div>
                          {run.config.run_name && run.config.run_name !== run.name && <div className="mt-1 text-[10px] text-muted-foreground">Folder {run.name}</div>}
                          <div className="mt-2 break-all font-mono text-[9px] leading-relaxed text-muted-foreground">{run.path}</div>
                          <div className="mt-3"><RunDetails run={run} /></div>
                        </td>
                        <td className="px-4 py-4">
                          <div data-testid="run-folder-size" className="font-display text-xl font-semibold tabular-nums">{formatBytes(run.size_bytes)}</div>
                          <div className="mt-1 text-[10px] text-muted-foreground">{formatBytes(run.allocated_bytes)} allocated</div>
                          <div className="mt-2 text-[10px] text-muted-foreground">{plural(run.file_count, "file")} · {plural(run.directory_count, "directory")}</div>
                          {run.scan_error_count > 0 && <div className="mt-2 flex items-start gap-1 text-[10px] leading-relaxed text-warning-foreground"><FileWarning aria-hidden="true" className="mt-0.5 size-3 shrink-0" />{plural(run.scan_error_count, "scan error")}; measured totals may be incomplete.</div>}
                        </td>
                        <td className="px-4 py-4">
                          {run.config.valid
                            ? <><div className="mb-3 flex flex-wrap items-center gap-1.5"><StatusBadge status="configured" tone="success" /><span className="text-[10px] text-muted-foreground">{run.config.sequence ? titleCase(run.config.sequence) : "No sequence"}{run.config.plan_only === true ? " · plan only" : ""}</span></div><ContentsSummary run={run} /></>
                            : <div className="space-y-3">
                                <div className="flex items-start gap-2 rounded border border-destructive/35 bg-destructive/5 p-3 text-[11px] leading-relaxed text-destructive"><AlertTriangle aria-hidden="true" className="mt-0.5 size-4 shrink-0" /><div><div className="font-semibold">Invalid run configuration</div><div className="mt-1 break-words text-muted-foreground">{run.config.error ?? "The configuration could not be read."}</div></div></div>
                                <ContentsSummary run={run} />
                              </div>}
                        </td>
                        <td className="px-4 py-4"><EvidenceSummary run={run} /></td>
                        <td className="px-4 py-4">
                          <div className="truncate font-mono text-[10px] font-semibold" title={run.root}>{run.root}</div>
                          <div className="mt-2 flex items-center gap-1 text-[10px] text-muted-foreground"><Clock3 aria-hidden="true" className="size-3" />{formatDate(run.modified_at)}</div>
                          {run.relocation && <div className="mt-2 flex items-start gap-1 text-[10px] leading-relaxed text-muted-foreground"><Link2 aria-hidden="true" className="mt-0.5 size-3 shrink-0" />Compatibility link retained from an earlier location.</div>}
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex flex-col items-stretch gap-2">
                            <Button size="sm" variant="outline" aria-label={`Move ${run.name}`} disabled={!canMove} onClick={() => openMove(run)}><MoveRight />Move</Button>
                            <Button size="sm" variant="ghost" className="text-destructive hover:text-destructive" aria-label={`Delete ${run.name}`} disabled={actionDisabled} onClick={() => setDeleteTarget(run)}><Trash2 />Delete</Button>
                            {active && <p data-testid="run-folder-active-action-reason" className="text-left text-[10px] leading-relaxed text-warning-foreground">Switch the active run folder before moving or deleting this folder.</p>}
                            {!active && maintenanceBlocking && <p className="text-left text-[10px] leading-relaxed text-destructive">Resolve the storage-recovery issue above before changing run folders.</p>}
                            {!active && !maintenanceBlocking && !inventoryReadyForMutation && <p data-testid="run-folder-inventory-action-reason" className="text-left text-[10px] leading-relaxed text-warning-foreground">Wait for a current inventory before moving or deleting this folder.</p>}
                            {!active && !hasDestination && <p className="text-left text-[10px] leading-relaxed text-muted-foreground">No other available allowed root.</p>}
                          </div>
                        </td>
                      </tr>
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>}

    <Dialog open={moveTarget !== null} onOpenChange={(open) => { if (!open && !moveRun.isPending) setMoveTarget(null) }}>
      <DialogContent data-testid="run-folder-move-dialog" className="max-h-[calc(100vh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Move {moveTarget?.name ?? "run folder"}?</DialogTitle>
          <DialogDescription>Move the complete folder to another allowed storage root. The operation is serialized as disk work, continues after navigation, and cannot be canceled after submission.</DialogDescription>
        </DialogHeader>
        {moveTarget && <div className="space-y-4">
          <div className="rounded-lg border bg-muted/35 p-3 text-xs">
            <div className="text-[9px] font-bold uppercase tracking-[0.12em] text-muted-foreground">Source run folder</div>
            <div className="mt-1 break-all font-mono text-[11px] font-semibold">{moveTarget.path}</div>
            <div className="mt-2 text-muted-foreground">{formatBytes(moveTarget.size_bytes)} measured · {plural(moveTarget.file_count, "file")}</div>
          </div>
          <div className="space-y-2">
            <Label>Destination root</Label>
            <Select value={destinationRoot} onValueChange={setDestinationRoot}>
              <SelectTrigger aria-label="Destination root"><SelectValue placeholder="Choose an allowed root" /></SelectTrigger>
              <SelectContent>{destinationRoots.map((root) => <SelectItem value={root.path} disabled={!root.exists || !root.identity} key={root.path}><span className="flex flex-col gap-0.5"><span className="font-mono">{root.path}</span><span className="text-[9px] text-muted-foreground">{root.exists && root.identity ? `${formatBytes(root.storage.free_bytes)} free` : "Root unavailable; refresh inventory"}</span></span></SelectItem>)}</SelectContent>
            </Select>
          </div>
          {destinationRoot && <div className="rounded-lg border border-primary/30 bg-primary/5 p-3 text-xs"><div className="text-[9px] font-bold uppercase tracking-[0.12em] text-muted-foreground">Resulting path</div><div className="mt-1 break-all font-mono font-semibold">{destinationRoot.replace(/\/+$/, "")}/{moveTarget.name}</div></div>}
          <div className="flex items-start gap-3 rounded-lg border border-warning/35 bg-warning/5 p-3 text-xs leading-relaxed"><Link2 aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-warning-foreground" /><p>The folder name stays the same. After the move, a compatibility link at the original path keeps existing references working.</p></div>
        </div>}
        <DialogFooter><Button variant="outline" onClick={() => setMoveTarget(null)} disabled={moveRun.isPending}>Cancel</Button><Button onClick={() => moveTarget && destinationRoot && moveRun.mutate({ run: moveTarget, root: destinationRoot })} disabled={!moveTarget || !destinationRoot || moveRun.isPending}>{moveRun.isPending ? <LoaderCircle className="animate-spin" /> : <MoveRight />}Queue move</Button></DialogFooter>
      </DialogContent>
    </Dialog>

    <Dialog open={deleteTarget !== null} onOpenChange={(open) => { if (!open && !deleteRun.isPending) setDeleteTarget(null) }}>
      <DialogContent data-testid="run-folder-delete-dialog" className="max-h-[calc(100vh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Delete {deleteTarget?.name ?? "run folder"}?</DialogTitle>
          <DialogDescription>This permanently deletes the entire run folder, including raw capture data and all derived evidence. This action cannot be undone or canceled after submission.</DialogDescription>
        </DialogHeader>
        {deleteTarget && <div className="space-y-3">
          <div className="flex items-start gap-3 rounded-lg border border-destructive/45 bg-destructive/10 p-4"><AlertTriangle aria-hidden="true" className="mt-0.5 size-5 shrink-0 text-destructive" /><div><div className="font-semibold text-destructive">Permanent acquisition-data deletion</div><p className="mt-1 text-xs leading-relaxed text-muted-foreground">The background job removes {formatBytes(deleteTarget.size_bytes)} across {plural(deleteTarget.file_count, "file")} from this exact folder:</p><div className="mt-2 break-all font-mono text-[10px] font-semibold">{deleteTarget.path}</div></div></div>
          {deleteTarget.scan_complete || <div className="flex items-start gap-2 rounded border border-warning/35 bg-warning/5 p-3 text-xs"><FileWarning aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-warning-foreground" /><span>The last inventory scan was incomplete. The deletion target remains the entire folder, not only the measured files.</span></div>}
        </div>}
        <DialogFooter><Button variant="outline" onClick={() => setDeleteTarget(null)} disabled={deleteRun.isPending}>Cancel</Button><Button variant="destructive" onClick={() => deleteTarget && deleteRun.mutate(deleteTarget)} disabled={!deleteTarget || deleteRun.isPending}>{deleteRun.isPending ? <LoaderCircle className="animate-spin" /> : <Trash2 />}Confirm delete</Button></DialogFooter>
      </DialogContent>
    </Dialog>
  </div>
}
