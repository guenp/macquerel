import importlib.util
import json
from pathlib import Path


def _load_plot_steps_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "plot_steps.py"
    spec = importlib.util.spec_from_file_location("plot_steps", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_step(path: Path, step: str, qubits: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "step": step,
                "commit": step,
                "results": {
                    "ghz": {
                        "macquerel-mlx": [[q, 1.0] for q in qubits],
                    },
                },
            }
        )
    )


def test_plot_steps_ignores_unordered_step_families(tmp_path):
    plot_steps = _load_plot_steps_module()

    _write_step(tmp_path / "step20-baseline-abc-mlx.json", "step20-baseline", [6, 12])
    _write_step(tmp_path / "step36-ghi-mlx.json", "step36", [24, 26, 28])
    _write_step(tmp_path / "step34-def-mlx.json", "step34", [6, 12])
    # step38 (expectation_pauli) is not in STEP_ORDER, so it is dropped even
    # though it carries results.
    _write_step(tmp_path / "step38-mno-mlx.json", "step38", [6, 12])
    # step40 (density matrix) carries no statevector results and is dropped too.
    (tmp_path / "step40-jkl.json").write_text(
        json.dumps({"step": "step40", "benchmark": "density_matrix_runtime"})
    )

    steps, by_step, commits = plot_steps.load_steps(tmp_path)

    # Returned in STEP_ORDER sequence (step34 before step36), filename sort aside.
    assert steps == ["step20-baseline", "step34", "step36"]
    assert "step38" not in by_step
    assert "step40" not in by_step
    assert commits == {
        "step20-baseline": "step20-baseline",
        "step34": "step34",
        "step36": "step36",
    }
