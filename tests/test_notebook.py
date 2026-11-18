"""Checks on the notebook's structure and on its copy of the model code."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook" / "colombo-dengue-control-timing.ipynb"

REQUIRED_HEADINGS = [
    "## 1. Problem and question",
    "## 2. Model, assumptions and equations",
    "## 3. Implementation",
    "## 4. Verification",
    "## 5. Experiments and results",
    "## 6. Discussion and limitations",
    "## 7. References",
]


def cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]


def test_model_cell_is_identical_to_src_seir():
    model_cells = [src for kind, src in cells() if kind == "code" and "def rk4_step" in src]
    assert len(model_cells) == 1, "expected exactly one model code cell"
    src_file = (ROOT / "src" / "seir.py").read_text()
    assert model_cells[0].rstrip("\n") == src_file.rstrip("\n"), \
        "the notebook's model cell and src/seir.py differ: copy one into the other"


def test_mcmc_cell_is_identical_to_src_mcmc():
    mcmc_cells = [src for kind, src in cells() if kind == "code" and "def metropolis" in src]
    assert len(mcmc_cells) == 1, "expected exactly one MCMC code cell"
    src_file = (ROOT / "src" / "mcmc.py").read_text()
    assert mcmc_cells[0].rstrip("\n") == src_file.rstrip("\n"), \
        "the notebook's MCMC cell and src/mcmc.py differ: copy one into the other"


def test_hostvector_cell_is_identical_to_src_hostvector():
    hv_cells = [src for kind, src in cells() if kind == "code" and "def simulate_hv" in src]
    assert len(hv_cells) == 1, "expected exactly one host-vector code cell"
    src_file = (ROOT / "src" / "hostvector.py").read_text()
    assert hv_cells[0].rstrip("\n") == src_file.rstrip("\n"), \
        "the notebook's host-vector cell and src/hostvector.py differ: copy one into the other"


def test_required_sections_in_order():
    text = "\n".join(src for kind, src in cells() if kind == "markdown")
    positions = [text.find(h) for h in REQUIRED_HEADINGS]
    assert all(p >= 0 for p in positions), [h for h, p in zip(REQUIRED_HEADINGS, positions) if p < 0]
    assert positions == sorted(positions)


def test_first_code_cell_installs_packages_and_sets_seed():
    first = next(src for kind, src in cells() if kind == "code")
    assert "%pip install" in first
    assert "SEED =" in first and "default_rng(SEED)" in first


def test_no_local_paths():
    for kind, src in cells():
        if kind == "code":
            assert "/Users/" not in src and "C:\\" not in src and "/content/drive" not in src
