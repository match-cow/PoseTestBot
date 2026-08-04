import { Link } from "react-router-dom"
import { Camera, Grid3X3, TriangleAlert } from "lucide-react"
import { Button } from "@/components/ui/button"

export type CalibrationCameraMounting = "eye_in_hand" | "static"
export type CalibrationMode = "eye_in_hand" | "eye_to_hand"
export type CalibrationTargetMountingFrame = "robot_flange" | "template_base"

export const POSE_TEMPLATE_BASE_SUNRISE_PATH = "/PoseTestBot/PoseTemplateBase"

type CalibrationTargetPlacement = {
  mode?: string | null
  mounting_frame?: string | null
}

type CalibrationSensor = {
  mounting_mode?: string | null
  current_mounting_mode?: string | null
  enabled?: boolean
}

export type CalibrationArrangement =
  | {
      status: "ready"
      mode: CalibrationMode
      cameraMounting: CalibrationCameraMounting
      mountingFrame: CalibrationTargetMountingFrame
      title: string
      cameraSummary: string
      targetSummary: string
      transformSummary: string
      resultSummary: string
    }
  | {
      status: "blocked"
      reason: "no_enabled_cameras" | "mixed_mounting" | "unknown_mounting"
      title: string
      message: string
    }

export function calibrationArrangementForSensors(
  sensors: CalibrationSensor[],
): CalibrationArrangement {
  const enabled = sensors.filter((sensor) => sensor.enabled !== false)
  if (!enabled.length) {
    return {
      status: "blocked",
      reason: "no_enabled_cameras",
      title: "No calibration camera group configured",
      message: "Enable at least one camera in Workflow step 1 before selecting or recording a calibration target.",
    }
  }

  const mountings = new Set(enabled.map((sensor) => sensor.current_mounting_mode ?? sensor.mounting_mode ?? ""))
  if ([...mountings].some((mounting) => mounting !== "eye_in_hand" && mounting !== "static")) {
    return {
      status: "blocked",
      reason: "unknown_mounting",
      title: "Camera mounting is incomplete",
      message: "Every enabled calibration camera needs a saved Static or Robot-mounted value in Workflow step 1.",
    }
  }
  if (mountings.size !== 1) {
    return {
      status: "blocked",
      reason: "mixed_mounting",
      title: "Use one mounting group per calibration recording",
      message: "Static cameras need a robot-carried grid, while robot-mounted cameras need a fixed grid. Disable one group and record the groups in separate runs; their published profiles can be combined later in an object-dataset run.",
    }
  }

  if (mountings.has("static")) {
    return {
      status: "ready",
      mode: "eye_to_hand",
      cameraMounting: "static",
      mountingFrame: "robot_flange",
      title: "Static-camera workcell calibration · moving robot grid",
      cameraSummary: "Every enabled camera remains fixed and will later observe objects on the PoseTemplateBase.",
      targetSummary: "Attach the selected grid rigidly to the robot flange. The robot moves this calibration instrument through many views; it is not tracking the hand as the calibration outcome.",
      transformSummary: "Robot flange poses are reported in PoseTemplateBase. PoseTestBot jointly estimates the supporting grid → robot_flange attachment, so no measured flange-to-grid transform is required.",
      resultSummary: "The reusable result for each camera is camera → PoseTemplateBase; grid → robot_flange is retained only as supporting calibration evidence.",
    }
  }

  return {
    status: "ready",
    mode: "eye_in_hand",
    cameraMounting: "eye_in_hand",
    mountingFrame: "template_base",
    title: "Robot-mounted cameras · fixed target",
    cameraSummary: "Every enabled camera moves rigidly with the robot flange.",
    targetSummary: "Secure the selected grid so it remains fixed relative to PoseTemplateBase throughout recording.",
    transformSummary: "The grid → PoseTemplateBase transform may be estimated, set to identity, or supplied by the target bundle.",
    resultSummary: "Each published profile transforms camera → robot_flange.",
  }
}

export function effectiveCalibrationTargetMountingFrame(
  placement: CalibrationTargetPlacement | null | undefined,
): CalibrationTargetMountingFrame | null {
  if (placement?.mounting_frame === "robot_flange" || placement?.mounting_frame === "template_base") {
    return placement.mounting_frame
  }
  if (placement?.mode === "template_base_identity" || placement?.mode === "posegridgen_board_to_base") {
    return "template_base"
  }
  return null
}

export function CalibrationArrangementCard({
  arrangement,
  editHref,
  editLabel = "Open Workflow step 1",
  testId = "calibration-arrangement",
}: {
  arrangement: CalibrationArrangement
  editHref?: string
  editLabel?: string
  testId?: string
}) {
  if (arrangement.status === "blocked") {
    return <div data-testid={testId} data-arrangement-status="blocked" className="flex items-start justify-between gap-4 rounded-lg border border-destructive/35 bg-destructive/5 p-4 text-xs">
      <div className="flex min-w-0 items-start gap-3">
        <TriangleAlert className="mt-0.5 size-5 shrink-0 text-destructive" />
        <div><div className="font-semibold text-destructive">{arrangement.title}</div><p className="mt-1 leading-relaxed text-muted-foreground">{arrangement.message}</p></div>
      </div>
      {editHref && <Button asChild size="sm" variant="outline" className="shrink-0"><Link to={editHref}>{editLabel}</Link></Button>}
    </div>
  }

  return <div data-testid={testId} data-arrangement-status="ready" data-calibration-mode={arrangement.mode} data-target-mounting-frame={arrangement.mountingFrame} className="rounded-lg border border-primary/30 bg-primary/5 p-4 text-xs">
    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
      <div className="min-w-0">
        <div className="flex items-center gap-2 font-semibold"><Camera className="size-4 text-primary-strong" />{arrangement.title}</div>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          <div className="rounded-md border bg-background/65 p-3"><div className="flex items-center gap-1.5 font-semibold"><Camera className="size-3.5" />Cameras</div><p className="mt-1 leading-relaxed text-muted-foreground">{arrangement.cameraSummary}</p></div>
          <div className="rounded-md border bg-background/65 p-3"><div className="flex items-center gap-1.5 font-semibold"><Grid3X3 className="size-3.5" />Printed grid</div><p className="mt-1 leading-relaxed text-muted-foreground">{arrangement.targetSummary}</p></div>
        </div>
        <p className="mt-3 leading-relaxed text-muted-foreground"><span className="font-semibold text-foreground">Transform policy:</span> {arrangement.transformSummary}</p>
        <p className="mt-1 leading-relaxed text-primary-strong"><span className="font-semibold">Result:</span> {arrangement.resultSummary}</p>
      </div>
      {editHref && <Button asChild size="sm" variant="outline" className="shrink-0"><Link to={editHref}>{editLabel}</Link></Button>}
    </div>
  </div>
}
