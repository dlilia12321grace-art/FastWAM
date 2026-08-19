import ast
from pathlib import Path

import torch

from fastwam.models.wan22.dynamic_action_gap import (
    DynamicActionGapGate,
    build_compute_matched_action_gap_schedule,
    build_dynamic_action_gap_meta,
    load_dynamic_action_gap_gate,
    make_dynamic_action_gap_checkpoint,
    select_dynamic_action_gap_route,
)
from scripts.train_dynamic_action_gap import load_samples, split_by_chunk


def test_gate_forward_shape_and_nonnegative_output():
    gate = DynamicActionGapGate(hidden_dim=8, bottleneck_dim=4)
    predicted = gate(torch.randn(3, 8), torch.randn(3, 3))
    assert predicted.shape == (3,)
    assert torch.all(predicted >= 0)


def test_meta_features_are_normalized():
    meta = build_dynamic_action_gap_meta(
        batch_size=2,
        timestep=500.0,
        step_index=4,
        num_steps=10,
        steps_since_full=2,
        max_internal_run=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert meta.shape == (2, 3)
    assert torch.allclose(meta[0], torch.tensor([0.5, 4.0 / 9.0, 0.5]))
    assert torch.equal(meta[0], meta[1])


def test_router_enforces_safety_boundaries():
    assert select_dynamic_action_gap_route(
        predicted_gap=0.0,
        threshold=1.0,
        step_index=0,
        num_steps=10,
        steps_since_full=0,
        max_internal_run=3,
    ) == ("full", "first_step")
    assert select_dynamic_action_gap_route(
        predicted_gap=0.0,
        threshold=1.0,
        step_index=9,
        num_steps=10,
        steps_since_full=0,
        max_internal_run=3,
    ) == ("full", "last_step")
    assert select_dynamic_action_gap_route(
        predicted_gap=0.0,
        threshold=1.0,
        step_index=4,
        num_steps=10,
        steps_since_full=3,
        max_internal_run=3,
    ) == ("full", "max_internal_run")


def test_router_uses_predicted_gap_away_from_boundaries():
    safe = select_dynamic_action_gap_route(
        predicted_gap=0.2,
        threshold=0.3,
        step_index=3,
        num_steps=10,
        steps_since_full=1,
        max_internal_run=3,
    )
    risky = select_dynamic_action_gap_route(
        predicted_gap=0.4,
        threshold=0.3,
        step_index=3,
        num_steps=10,
        steps_since_full=1,
        max_internal_run=3,
    )
    assert safe == ("internal", "predicted_safe")
    assert risky == ("full", "predicted_risky")


def test_router_preserves_fixed_schedule_anchor():
    route = select_dynamic_action_gap_route(
        predicted_gap=0.0,
        threshold=1.0,
        step_index=3,
        num_steps=10,
        steps_since_full=1,
        max_internal_run=3,
        anchor_action_gap=4,
    )
    assert route == ("full", "anchor_schedule")


def test_compute_matched_fixed_schedule_alternates_to_sixty_five_percent():
    schedules = [
        build_compute_matched_action_gap_schedule(
            num_steps=10,
            chunk_index=chunk_index,
            target_internal_ratio=0.65,
            mode="fixed",
            max_internal_run=7,
            anchor_action_gap=8,
        )
        for chunk_index in range(2)
    ]
    internal_counts = [schedule.count("internal") for schedule in schedules]
    assert internal_counts == [6, 7]
    assert sum(internal_counts) / 20 == 0.65
    for schedule in schedules:
        assert schedule[0] == "full"
        assert schedule[7] == "full"
        assert schedule[-1] == "full"


def test_compute_matched_random_schedule_is_reproducible_and_budget_matched():
    kwargs = {
        "num_steps": 10,
        "chunk_index": 0,
        "target_internal_ratio": 0.6,
        "mode": "random",
        "max_internal_run": 7,
        "anchor_action_gap": 8,
    }
    first = build_compute_matched_action_gap_schedule(**kwargs, seed=11)
    repeated = build_compute_matched_action_gap_schedule(**kwargs, seed=11)
    alternatives = [
        build_compute_matched_action_gap_schedule(**kwargs, seed=seed)
        for seed in range(12, 24)
    ]
    assert first == repeated
    assert first.count("internal") == 6
    assert any(schedule != first for schedule in alternatives)


def test_compute_matched_schedule_enforces_max_internal_run():
    schedule = build_compute_matched_action_gap_schedule(
        num_steps=20,
        chunk_index=1,
        target_internal_ratio=1.0,
        mode="fixed",
        max_internal_run=3,
        anchor_action_gap=None,
    )
    longest_run = max(len(run) for run in "".join(
        "i" if route == "internal" else "f" for route in schedule
    ).split("f"))
    assert longest_run <= 3
    assert schedule[0] == schedule[-1] == "full"


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(7)
    gate = DynamicActionGapGate(hidden_dim=8, bottleneck_dim=4)
    hidden = torch.randn(2, 8)
    meta = torch.randn(2, 3)
    expected = gate(hidden, meta)
    checkpoint = tmp_path / "gate.pt"
    torch.save(
        make_dynamic_action_gap_checkpoint(gate, default_threshold=0.25),
        checkpoint,
    )

    restored, payload = load_dynamic_action_gap_gate(
        str(checkpoint), device=torch.device("cpu"), dtype=torch.float32
    )
    assert payload["default_threshold"] == 0.25
    assert torch.allclose(restored(hidden, meta), expected)


def test_meta_only_gate_is_independent_of_hidden_state():
    torch.manual_seed(7)
    gate = DynamicActionGapGate(
        hidden_dim=8, bottleneck_dim=4, input_mode="meta_only"
    )
    meta = torch.randn(2, 3)
    first = gate(torch.randn(2, 8), meta)
    second = gate(torch.randn(2, 8), meta)
    assert torch.equal(first, second)


def test_hidden_only_gate_is_independent_of_metadata():
    torch.manual_seed(7)
    gate = DynamicActionGapGate(
        hidden_dim=8, bottleneck_dim=4, input_mode="hidden_only"
    )
    hidden = torch.randn(2, 8)
    first = gate(hidden, torch.randn(2, 3))
    second = gate(hidden, torch.randn(2, 3))
    assert torch.equal(first, second)


def test_checkpoint_loader_defaults_legacy_gate_to_hidden_meta(tmp_path):
    gate = DynamicActionGapGate(hidden_dim=8, bottleneck_dim=4)
    payload = make_dynamic_action_gap_checkpoint(gate, default_threshold=0.25)
    del payload["input_mode"]
    checkpoint = tmp_path / "legacy_gate.pt"
    torch.save(payload, checkpoint)

    restored, _ = load_dynamic_action_gap_gate(
        str(checkpoint), device=torch.device("cpu"), dtype=torch.float32
    )

    assert restored.input_mode == "hidden_meta"


def test_collected_dataset_loading_and_chunk_split(tmp_path):
    dataset = tmp_path / "dynamic_samples.pt"
    torch.save(
        {
            "format_version": 1,
            "hidden_dim": 8,
            "samples": [
                {
                    "pooled_hidden": torch.randn(1, 8),
                    "gap": 0.1,
                    "step_index": 1,
                    "num_steps": 10,
                    "timestep": 800.0,
                    "chunk_index": 0,
                },
                {
                    "pooled_hidden": torch.randn(1, 8),
                    "gap": 0.2,
                    "step_index": 2,
                    "num_steps": 10,
                    "timestep": 700.0,
                    "chunk_index": 1,
                },
            ],
        },
        dataset,
    )
    hidden, meta, target, groups, hidden_dim = load_samples([dataset], 3)
    assert hidden.shape == (6, 8)
    assert meta.shape == (6, 3)
    assert target.shape == (6,)
    assert hidden_dim == 8
    train_indices, validation_indices = split_by_chunk(groups, 0.5, seed=1)
    assert train_indices.numel() == 3
    assert validation_indices.numel() == 3


def test_fastwam_wires_anchor_to_router_not_metadata_builder():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fastwam"
        / "models"
        / "wan22"
        / "fastwam.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    keywords_by_call = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id in {
            "build_dynamic_action_gap_meta",
            "select_dynamic_action_gap_route",
        }:
            keywords_by_call[node.func.id] = {
                keyword.arg for keyword in node.keywords
            }
    assert "anchor_action_gap" not in keywords_by_call[
        "build_dynamic_action_gap_meta"
    ]
    assert "anchor_action_gap" in keywords_by_call[
        "select_dynamic_action_gap_route"
    ]


def test_fastwam_true_gap_oracle_routes_with_oracle_score():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fastwam"
        / "models"
        / "wan22"
        / "fastwam.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    infer_action = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "infer_action"
    )
    argument_names = {argument.arg for argument in infer_action.args.args}
    assert "enable_oracle_action_gap" in argument_names
    assert "oracle_action_gap_threshold" in argument_names

    oracle_router_calls = []
    for node in ast.walk(infer_action):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "select_dynamic_action_gap_route":
            continue
        predicted_gap = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "predicted_gap"),
            None,
        )
        if isinstance(predicted_gap, ast.Name) and predicted_gap.id == "oracle_score":
            oracle_router_calls.append(node)
    assert len(oracle_router_calls) == 1

    oracle_guard = next(
        node
        for node in ast.walk(infer_action)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "enable_oracle_action_gap"
    )
    clears_explicit_steps = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "internal_lora_inference_steps_set"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "set"
        for node in ast.walk(oracle_guard)
    )
    assert clears_explicit_steps
