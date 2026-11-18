"""Checks that the web demo (docs/script.js) matches the Python model and the notebook reference."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import seir  # noqa: E402

SCRIPT = (ROOT / "docs" / "script.js").read_text()


def js_number(name):
    return float(re.search(rf"\b{name}: ([0-9.eE+-]+)[,\s}}]", SCRIPT).group(1))


FIT = {n: js_number(n) for n in ["N", "Re", "amp", "phase", "rho", "s0", "i0", "bg"]}
REF = {n: js_number(n) for n in ["start", "reduction", "prevented", "tolerance"]}


def python_reference():
    p = seir.Params(N=FIT["N"], beta0=FIT["Re"] * seir.GAMMA_DEFAULT / FIT["s0"], amp=FIT["amp"],
                    phase=FIT["phase"], rho=FIT["rho"], s0=FIT["s0"], i0=FIT["i0"],
                    e0=FIT["i0"] * seir.GAMMA_DEFAULT / seir.SIGMA_DEFAULT)
    none = seir.simulate(p, 52)["weekly_reported"].sum()
    cut = seir.simulate(p.with_(start=REF["start"] - 1, reduction=REF["reduction"]), 52)["weekly_reported"].sum()
    return none - cut


def test_reference_value_in_demo_matches_python_model():
    assert REF["tolerance"] == 0.01
    assert abs(python_reference() - REF["prevented"]) <= 1e-9 * REF["prevented"]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_javascript_model_matches_python_model():
    model = SCRIPT.split("// ---------- UI")[0]
    code = model + """
    const p = { ...FIT, beta0: FIT.Re * FIT.gamma / FIT.s0, start: Infinity, end: Infinity, reduction: 0 };
    const out = { none: simulate(p), cut: simulate({ ...p, start: 20, reduction: 0.4 }),
                  campaign: simulate({ ...p, start: 4, end: 12, reduction: 0.4 }), check: referenceCheck() };
    console.log(JSON.stringify(out));
    """
    res = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)
    p = seir.Params(N=FIT["N"], beta0=FIT["Re"] * seir.GAMMA_DEFAULT / FIT["s0"], amp=FIT["amp"],
                    phase=FIT["phase"], rho=FIT["rho"], s0=FIT["s0"], i0=FIT["i0"],
                    e0=FIT["i0"] * seir.GAMMA_DEFAULT / seir.SIGMA_DEFAULT)
    for key, changes in [("none", {}), ("cut", dict(start=20.0, reduction=0.4)),
                         ("campaign", dict(start=4.0, end=12.0, reduction=0.4))]:
        py = seir.simulate(p.with_(**changes), 52)["weekly_reported"] + FIT["bg"]
        assert max(abs(a - b) for a, b in zip(res[key], py)) < 1e-6
    assert res["check"]["ok"] and res["check"]["relDiff"] < 1e-9
