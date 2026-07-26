"""Operator-facing workflow descriptions built on the stable pipeline API.

These records deliberately sit above pipeline stages and sequences.  Stages
remain useful implementation primitives, while workflows explain the two lab
outcomes an operator is normally trying to achieve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from posetestbot.pipeline.sequences import PIPELINE_SEQUENCES


SCHEMA_VERSION = "operator_workflows.v1"


@dataclass(frozen=True)
class OperatorWorkflowStep:
    number: int
    id: str
    label: str
    description: str
    help: str
    required: bool
    optional: bool
    automatic: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorWorkflowOptionalAction:
    id: str
    label: str
    description: str
    help: str
    required: bool = False
    optional: bool = True
    automatic: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorWorkflow:
    id: str
    label: str
    description: str
    outcome: str
    recommended_sequence_ids: tuple[str, ...]
    steps: tuple[OperatorWorkflowStep, ...]
    optional_actions: tuple[OperatorWorkflowOptionalAction, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "outcome": self.outcome,
            "recommended_sequence_ids": list(self.recommended_sequence_ids),
            "steps": [step.to_dict() for step in self.steps],
            "optional_actions": [action.to_dict() for action in self.optional_actions],
        }


OPERATOR_WORKFLOWS: dict[str, OperatorWorkflow] = {
    "camera_calibration": OperatorWorkflow(
        id="camera_calibration",
        label="Calibrate cameras",
        description=(
            "Record the printed calibration grid, calculate camera intrinsics and "
            "camera-to-robot transforms, then review and save a reusable calibration."
        ),
        outcome=(
            "A promoted calibration profile set that can be selected by later object "
            "dataset runs."
        ),
        recommended_sequence_ids=(
            "real_full_capture_validation",
            "aruco_grid_full_calibration",
        ),
        steps=(
            OperatorWorkflowStep(
                number=1,
                id="configure_cameras",
                label="Choose cameras",
                description="Create a calibration run and enable every camera to calibrate.",
                help=(
                    "Camera serial, mounting mode, resolution, and upright/inverted "
                    "orientation become part of the calibration identity."
                ),
                required=True,
                optional=False,
                automatic=False,
            ),
            OperatorWorkflowStep(
                number=2,
                id="select_grid",
                label="Select the printed grid",
                description="Select the saved grid definition for the board in the cell.",
                help=(
                    "Use the definition generated for the exact printed board. Marker "
                    "size, spacing, dictionary, PDF, and placement are checked together."
                ),
                required=True,
                optional=False,
                automatic=False,
            ),
            OperatorWorkflowStep(
                number=3,
                id="check_readiness",
                label="Check readiness",
                description="Run one combined readiness check before cameras or robot move.",
                help=(
                    "This validates saved configuration, target provenance, current "
                    "device/runtime visibility, and the planned software stages. Live "
                    "camera identity, empty-output, and safety checks repeat at capture."
                ),
                required=True,
                optional=False,
                automatic=True,
            ),
            OperatorWorkflowStep(
                number=4,
                id="record_grid_views",
                label="Record grid views",
                description="Capture varied, sharp views of the grid from every camera.",
                help=(
                    "Physical capture still requires the camera and real-robot safety "
                    "acknowledgements. PoseTestBot never starts it implicitly."
                ),
                required=True,
                optional=False,
                automatic=False,
            ),
            OperatorWorkflowStep(
                number=5,
                id="calculate_review_publish",
                label="Calculate, review, and save",
                description=(
                    "Synchronize frames, solve the calibration, review its evidence, "
                    "and publish the accepted profiles."
                ),
                help=(
                    "Factory intrinsics are the camera SDK values recorded during capture. "
                    "OpenCV intrinsics are re-estimated from the printed-grid images. The "
                    "calibration compares both; an OpenCV result is activated only when the "
                    "Factory projection is unusable and every coverage, held-out, plausibility, "
                    "and reprojection check passes. Only promoted profiles with "
                    "status 'valid' are offered to guided object-dataset runs."
                ),
                required=True,
                optional=False,
                automatic=False,
            ),
        ),
        optional_actions=(
            OperatorWorkflowOptionalAction(
                id="inspect_advanced_evidence",
                label="Inspect advanced solver evidence",
                description="Inspect per-view errors and alternate solver candidates.",
                help=(
                    "Useful when a quality gate fails or two candidates are close; it is "
                    "not required for an accepted calibration."
                ),
            ),
        ),
    ),
    "object_dataset": OperatorWorkflow(
        id="object_dataset",
        label="Record an object dataset",
        description=(
            "Select a previously promoted camera calibration and a pose template, "
            "record the object, then build the synchronized BOP dataset."
        ),
        outcome=(
            "A non-destructively synchronized, calibrated BOP image/model dataset "
            "with immutable calibration and pose-template provenance."
        ),
        recommended_sequence_ids=(
            "real_full_capture_validation",
            "calibrated_capture_to_bop_dataset_dry_run",
        ),
        steps=(
            OperatorWorkflowStep(
                number=1,
                id="configure_dataset_run",
                label="Configure cameras and calibration",
                description=(
                    "Create the dataset run, choose its cameras, and select a saved calibration."
                ),
                help=(
                    "Use the same physical camera identities, mounting modes, resolution, "
                    "and orientation recorded by the calibration. PoseTestBot copies both "
                    "profile files into this run so the source can never change the dataset."
                ),
                required=True,
                optional=False,
                automatic=False,
            ),
            OperatorWorkflowStep(
                number=2,
                id="select_pose_template",
                label="Select and place the object template",
                description="Select the immutable template and confirm its placement in the cell.",
                help=(
                    "The template identifies object geometry and the template-base transforms "
                    "that will be exported with this dataset."
                ),
                required=True,
                optional=False,
                automatic=False,
            ),
            OperatorWorkflowStep(
                number=3,
                id="check_readiness",
                label="Check readiness",
                description="Validate configuration, calibration, template, and current device visibility.",
                help=(
                    "A guided object run fails closed when a selected camera profile is "
                    "missing, incompatible, or not marked valid. Live device identity, "
                    "empty-output, and safety checks repeat at capture startup."
                ),
                required=True,
                optional=False,
                automatic=True,
            ),
            OperatorWorkflowStep(
                number=4,
                id="record_object",
                label="Record the object",
                description="Run the supervised robot-and-camera capture.",
                help=(
                    "Physical capture requires both explicit safety gates. Raw RGB-D frames "
                    "and robot poses remain untouched by later processing."
                ),
                required=True,
                optional=False,
                automatic=False,
            ),
            OperatorWorkflowStep(
                number=5,
                id="process_dataset",
                label="Synchronize and verify",
                description=(
                    "Match timestamps, check sync quality, and rectify from the calibration snapshot."
                ),
                help=(
                    "These are derived artifacts below processed/. They never rename or "
                    "replace the only copy of captured frames."
                ),
                required=True,
                optional=False,
                automatic=True,
            ),
            OperatorWorkflowStep(
                number=6,
                id="export_bop",
                label="Export the BOP dataset",
                description="Validate and write the final BOP dataset artifacts.",
                help=(
                    "The required export writes calibrated RGB-D scenes, canonical "
                    "models, and provenance without BlenderProc or rendered GT. "
                    "Optional masks/ground truth are a separate later action."
                ),
                required=True,
                optional=False,
                automatic=True,
            ),
        ),
        optional_actions=(
            OperatorWorkflowOptionalAction(
                id="generate_ground_truth",
                label="Generate masks and ground truth",
                description="Render optional masks and pose ground truth with BlenderProc.",
                help=(
                    "Skip this when only synchronized calibrated frames are needed. It "
                    "requires a working BlenderProc runtime."
                ),
                automatic=True,
            ),
        ),
    ),
}


def list_operator_workflows() -> list[dict[str, object]]:
    """Return validated, deterministic workflow records for the web console."""

    records: list[dict[str, object]] = []
    for workflow in OPERATOR_WORKFLOWS.values():
        numbers = [step.number for step in workflow.steps]
        if numbers != list(range(1, len(workflow.steps) + 1)):
            raise ValueError(f"Workflow {workflow.id} step numbers must be consecutive")
        missing_sequences = sorted(
            set(workflow.recommended_sequence_ids) - PIPELINE_SEQUENCES.keys()
        )
        if missing_sequences:
            raise ValueError(
                f"Workflow {workflow.id} references unknown sequence(s): "
                + ", ".join(missing_sequences)
            )
        records.append(workflow.to_dict())
    return sorted(records, key=lambda item: str(item["id"]))
