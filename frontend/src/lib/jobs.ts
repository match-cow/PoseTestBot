import type { Job } from "@/lib/contracts"

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
