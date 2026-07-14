export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue }

export interface Bootstrap {
  schema_version: "web_bootstrap.v1"
  brand: { name: string; logo_url: string }
  robot: { ip: string; port: number }
  default_run_root: string
  allowed_run_roots: string[]
}

export interface RunIndexItem {
  path: string
  name: string
  sequence: string | null
  plan_only: boolean | null
  config_valid: boolean
  config_error: string | null
  modified_at: string
}

export interface OverviewSection {
  id: string
  label: string
  status: string
  artifacts: Array<{ path: string; exists: boolean; status: string | null }>
}

export interface Overview {
  run_root: string
  config: RunConfig | null
  config_error: string | null
  sidebar: OverviewSection[]
  steps: Array<{
    index: number
    id: string
    stage_id: string
    label: string
    description: string
    status: string
    resources: string[]
    artifacts: OverviewSection["artifacts"]
  }>
  recommendations: Array<Record<string, JsonValue>>
  recommendation_error: string | null
  artifact_count: number
}

export interface SensorDevice {
  sensor_type: string
  device_id: string
  display_name?: string
  effective_display_name?: string
  alias?: string
  connected?: boolean
  inverted?: boolean
  mounting_mode?: string
  metadata?: Record<string, JsonValue>
}

export interface SensorStatus {
  schema_version: string
  families: Array<{
    sensor_type: string
    display_name: string
    devices: SensorDevice[]
    [key: string]: unknown
  }>
  total_connected: number
  all_expected_connected?: boolean
  [key: string]: unknown
}

export interface Job {
  id: string
  name: string
  command: string[]
  cwd: string | null
  status: string
  created_at: string
  started_at: string | null
  ended_at: string | null
  returncode: number | null
  message: string | null
  tail: string[]
  resources: string[]
  parameters: Record<string, JsonValue>
  log_path: string
  visibility: "operator" | "service"
  process_pid?: number | null
  process_group_id?: number | null
  process_start_time?: number | null
  supervisor_pid?: number | null
  supervisor_process_group_id?: number | null
  supervisor_start_time?: number | null
}

export interface PreviewJob {
  job: Job
  preview_root: string | null
  preview_status: {
    status: string
    frame_count: number
    latest_image: string | null
    selected_node?: Record<string, JsonValue> | null
    error?: string | null
    sensor_key?: string
    [key: string]: JsonValue | undefined
  } | null
}

export interface CaptureJob {
  id: string
  name: string
  status: string
  kind: string | null
  stage: string | null
  sequence: string | null
  run_root: string | null
  resources: string[]
  message: string | null
  created_at: string
  started_at: string | null
  ended_at: string | null
  active: boolean
  tail: string[]
  log_endpoint: string
  stop_endpoint: string | null
}

export interface CaptureState {
  run_root: string
  jobs: CaptureJob[]
  active_count: number
  resources: Record<string, string>
  status_artifact: Record<string, JsonValue> | null
}

export interface PipelineParameter {
  name: string
  flag: string
  kind: "str" | "path" | "int" | "float" | "bool"
  path_scope: "run" | "input" | "output" | "repository" | null
  required: boolean
  default: JsonValue
  choices: string[]
  multiple: boolean
  help: string
}

export interface PipelineStage {
  id: string
  label: string
  description: string
  resources: string[]
  parameters: PipelineParameter[]
}

export interface PipelineSequence {
  id: string
  label: string
  description: string
  steps: Array<{ id: string; stage_id: string; [key: string]: JsonValue }>
}

export interface RunConfig {
  schema_version: string
  run_name: string
  run_root: string
  robot_profile: Record<string, JsonValue>
  capture: {
    resolution: string
    fps: number
    velocity_m_s: number
    sensors: Array<{
      sensor_type: string
      device_id: string
      display_name: string
      mounting_mode: string
      enabled: boolean
      inverted: boolean
      [key: string]: JsonValue
    }>
  }
  object_folder: string
  selected_objects: string[]
  calibration_profiles: string | null
  pipeline: {
    sequence_id: string
    plan_only: boolean
    options: Record<string, JsonValue>
  }
}

export interface CellTransform {
  semantics: "entity_to_parent"
  parent_frame: string | null
  translation_mm: [number, number, number]
  rotation_quaternion_wxyz: [number, number, number, number]
}

export interface CellEntity {
  id: string
  type: string
  label: string
  status: "planned" | "recorded" | "unresolved"
  transform: CellTransform | null
  unresolved_reason: string | null
  geometry: Record<string, JsonValue>
  provenance: Record<string, JsonValue>
}

export interface CellTimelineMetadata {
  id: string
  label: string
  kind: "synchronized" | "raw"
  frame_count: number
  default: boolean
  exact: true
  interpolation: "none"
  page_limit: number
  source: string
}

export interface CellPose {
  index: number
  frame_index: number
  frame_id: string
  timestamp_ns: number | null
  motion: string | null
  transform: CellTransform
}

export interface CellScene {
  schema_version: "cell_scene.v1"
  coordinate_system: Record<string, JsonValue>
  run_root: string
  entities: CellEntity[]
  warnings: Array<{ code: string; message: string }>
  timelines: CellTimelineMetadata[]
  default_timeline_id: string | null
  trajectory_preview: CellPose[]
  object_selection: {
    selected_objects: string[]
    objectless: boolean
    registry: Record<string, JsonValue>
    bop_export: Record<string, JsonValue>
  }
}

export interface CellTimelinePage {
  schema_version: "cell_timeline.v1"
  timeline: CellTimelineMetadata
  offset: number
  limit: number
  total: number
  next_offset: number | null
  previous_offset: number | null
  poses: CellPose[]
}

export interface ObjectRegistryPayload {
  schema_version: "object_registry.v1"
  run_root: string
  object_folder: string
  selected_objects: string[]
  missing_selected_objects: string[]
  objectless: boolean
  entries: Array<{
    name: string
    obj_id: number
    valid: boolean
    errors: string[]
    selected: boolean
    texture_filename: string | null
  }>
}

export interface PreflightSummary {
  queue_blocker?: string | null
  status?: string
  path?: string
  [key: string]: JsonValue | undefined
}

export interface ArtifactRecord {
  key: string
  source: string
  path: string
  relative_path: string | null
  kind: string
  exists: boolean
  preview_type: string | null
  size_bytes: number | null
  modified_at: string | null
  child_count: number | null
  summary?: Record<string, JsonValue> | null
  [key: string]: unknown
}

export interface BopSceneFrame {
  image_id: number
  image_key?: string
  gt_count: number
  rgb: { exists: boolean; relative_path?: string | null }
  depth: { exists: boolean; relative_path?: string | null }
  mask_files: string[]
  mask_visib_files?: string[]
  camera?: JsonValue
  gt?: JsonValue
}

export interface BopSceneDetail {
  type?: "bop_scene_detail"
  relative_path: string
  frame_count: number
  frames: BopSceneFrame[]
}

export interface BopFrameDetail {
  type: "bop_frame_detail"
  relative_path: string
  image_id: number
  gt_count: number
  rgb: BopSceneFrame["rgb"]
  depth: BopSceneFrame["depth"]
  mask_artifacts: Array<{ relative_path: string; name: string }>
  mask_visib_artifacts: Array<{ relative_path: string; name: string }>
  camera?: JsonValue
  gt?: JsonValue
  gt_info?: JsonValue
}

export interface ApiErrorBody {
  output?: string
  [key: string]: unknown
}
