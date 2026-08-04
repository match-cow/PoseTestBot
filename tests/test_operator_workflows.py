from posetestbot.pipeline.sequences import PIPELINE_SEQUENCES
from posetestbot.pipeline.workflows import list_operator_workflows
from posetestbot.web.app import create_app


def test_operator_workflows_are_numbered_and_reference_existing_sequences() -> None:
    workflows = {item["id"]: item for item in list_operator_workflows()}

    assert set(workflows) == {"camera_calibration", "object_dataset"}
    assert len(workflows["camera_calibration"]["steps"]) == 5
    assert len(workflows["object_dataset"]["steps"]) == 6
    for workflow in workflows.values():
        steps = workflow["steps"]
        assert [step["number"] for step in steps] == list(range(1, len(steps) + 1))
        assert set(workflow["recommended_sequence_ids"]) <= set(PIPELINE_SEQUENCES)
        for step in steps:
            assert isinstance(step["required"], bool)
            assert isinstance(step["optional"], bool)
            assert step["required"] is not step["optional"]
            assert isinstance(step["automatic"], bool)
            assert step["description"]
            assert step["help"]
        for action in workflow["optional_actions"]:
            assert action["required"] is False
            assert action["optional"] is True
            assert action["description"]
            assert action["help"]

    intrinsic_help = workflows["camera_calibration"]["steps"][4]["help"]
    assert "Factory intrinsics" in intrinsic_help
    assert "OpenCV intrinsics" in intrinsic_help


def test_pipeline_workflows_endpoint_is_additive() -> None:
    client = create_app().test_client()

    response = client.get("/pipeline/workflows")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema_version"] == "operator_workflows.v1"
    assert {item["id"] for item in payload["workflows"]} == {
        "camera_calibration",
        "object_dataset",
    }
    assert client.get("/pipeline/stages").status_code == 200
    assert client.get("/pipeline/sequences").status_code == 200


def test_calibrated_dataset_sequences_require_valid_profiles_and_allow_injection() -> None:
    for sequence_id in (
        "sync_to_bop_calibrated_dry_run",
        "calibrated_capture_to_bop_dataset_dry_run",
    ):
        sequence = PIPELINE_SEQUENCES[sequence_id]
        preflight = next(
            step for step in sequence.steps if step.id == "calibration_preflight"
        )
        assert preflight.options == {"require_valid": True}

    guided = PIPELINE_SEQUENCES["calibrated_capture_to_bop_dataset_dry_run"]
    assert all(
        step.stage_id not in {"blenderproc_prepare", "blenderproc_render"}
        for step in guided.steps
    )
    export = next(step for step in guided.steps if step.id == "bop_export")
    assert export.options == {"annotation_source": "none", "overwrite": True}
