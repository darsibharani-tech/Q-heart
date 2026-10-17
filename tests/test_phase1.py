import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_processor_graphs():
    expected = {"processor_A.json": (5, 4), "processor_B.json": (9, 12)}
    for filename, shape in expected.items():
        data = json.loads((ROOT / "processors" / filename).read_text())
        assert len(data["nodes"]) == shape[0]
        assert len(data["coupling_edges"]) == shape[1]

def test_results_schema_and_rows():
    with (ROOT / "results" / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"model", "architecture", "evaluation_mode", "seed", "shots", "accuracy", "loss", "iterations", "depth", "cx_count", "swap_count", "routing_metric"}
    assert required <= set(rows[0])
    assert any(row["evaluation_mode"] == "noisy_shot_based" for row in rows)

def test_required_pngs_exist():
    for name in ["processor_A_coupling.png", "processor_B_coupling.png", "architecture_comparison.png", "training_convergence.png", "accuracy_comparison.png"]:
        path = ROOT / "results" / name
        assert path.exists() and path.stat().st_size > 1000
