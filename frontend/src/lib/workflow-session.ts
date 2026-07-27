export type WorkflowJourneyId = "calibration" | "dataset"

export type WorkflowProgressStatus =
  | "complete"
  | "current"
  | "ready"
  | "blocked"
  | "running"
  | "not_started"

interface WorkflowStepMetadata {
  id: string
  number: number
  title: string
  required?: boolean
}

interface WorkflowJourneyMetadata {
  title: string
  steps: WorkflowStepMetadata[]
}

export const workflowJourneyMetadata: Record<WorkflowJourneyId, WorkflowJourneyMetadata> = {
  calibration: {
    title: "Camera calibration",
    steps: [
      { id: "configure", number: 1, title: "Configure the run and cameras" },
      { id: "target", number: 2, title: "Choose the printed calibration grid" },
      { id: "readiness", number: 3, title: "Check readiness" },
      { id: "capture", number: 4, title: "Record calibration images" },
      { id: "calculate", number: 5, title: "Calculate, review, and publish" },
    ],
  },
  dataset: {
    title: "Object dataset",
    steps: [
      { id: "configure", number: 1, title: "Configure cameras and select calibration" },
      { id: "template", number: 2, title: "Choose the pose template and placement" },
      { id: "readiness", number: 3, title: "Check readiness" },
      { id: "capture", number: 4, title: "Record the object dataset" },
      { id: "sync", number: 5, title: "Process frames and create the base BOP export" },
      { id: "export", number: 6, title: "Add optional BOP ground-truth evidence", required: false },
    ],
  },
}

export interface ActiveWorkflow {
  journey: WorkflowJourneyId
  journeyTitle: string
  stepId: string
  stepNumber: number
  stepCount: number
  stepTitle: string
  status: WorkflowProgressStatus
}

const progressStatuses = new Set<WorkflowProgressStatus>([
  "complete",
  "current",
  "ready",
  "blocked",
  "running",
  "not_started",
])

export function activeWorkflow(
  journey: WorkflowJourneyId,
  stepId: string,
  status: WorkflowProgressStatus,
): ActiveWorkflow | null {
  const metadata = workflowJourneyMetadata[journey]
  const step = metadata.steps.find((item) => item.id === stepId)
  if (!step) return null
  return {
    journey,
    journeyTitle: metadata.title,
    stepId: step.id,
    stepNumber: step.number,
    stepCount: metadata.steps.length,
    stepTitle: step.title,
    status,
  }
}

export function parseActiveWorkflow(value: unknown): ActiveWorkflow | null {
  if (!value || typeof value !== "object") return null
  const candidate = value as Record<string, unknown>
  if (candidate.journey !== "calibration" && candidate.journey !== "dataset") return null
  if (typeof candidate.stepId !== "string") return null
  const status = typeof candidate.status === "string" && progressStatuses.has(candidate.status as WorkflowProgressStatus)
    ? candidate.status as WorkflowProgressStatus
    : "not_started"
  return activeWorkflow(candidate.journey, candidate.stepId, status)
}

export function activeWorkflowHref(workflow: ActiveWorkflow) {
  return `/workflow/${workflow.journey}?step=${workflow.stepId}`
}
