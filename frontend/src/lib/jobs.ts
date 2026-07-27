import type { Job } from "@/lib/contracts"
import type { StatusTone } from "@/components/status-badge"

const RUNNER_ONLY = [
  /^command (?:failed|exited|returned)/i,
  /^process (?:failed|exited|returned)/i,
  /^exit(?:ed)? (?:code|status)/i,
  /^job .* (?:failed|canceled)$/i,
  /^running command:/i,
]

export function jobFailureDetail(job: Pick<Job, "status" | "message" | "tail">) {
  const meaningful = [...job.tail].reverse().map((line) => line.trim()).find((line) => line && !RUNNER_ONLY.some((pattern) => pattern.test(line)))
  return meaningful ?? job.message ?? job.tail.at(-1) ?? `Job ${job.status}`
}

export function jobStatusTone(status: string | null | undefined): StatusTone {
  switch (status) {
    case "succeeded":
      return "success"
    case "queued":
    case "running":
    case "canceling":
      return "warning"
    case "failed":
      return "destructive"
    case "canceled":
    case "cancelled":
      return "neutral"
    default:
      return "neutral"
  }
}
