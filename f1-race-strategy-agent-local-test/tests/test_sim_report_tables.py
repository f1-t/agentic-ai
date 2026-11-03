"""SPEC 0022: LaTeX tables render from the committed reports, ASCII only, with provenance."""
import json
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.sim import report_tables as rt

COMMITTED = rt.OUT_DIR
NAMES = ["t1_spain_replay", "t2_midrace_g", "t3_stop_gain", "t4_traffic", "t5_cross_race", "t6_event_rates",
         "t7_pace_prior"]


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("tex")
    done, refused = rt.write(d)
    assert not refused
    return d


def test_every_table_renders_and_sim_tables_inputs_them(out):
    assert {p.stem for p in out.glob("t*.tex")} == set(NAMES)
    index = (out / "sim_tables.tex").read_text()
    for n in NAMES:
        assert f"\\input{{{n}}}" in index


def test_rendered_vs_placeholder(out):
    for n in NAMES[:5]:
        t = (out / f"{n}.tex").read_text()
        assert "\\toprule" in t and "\\bottomrule" in t and "source not committed yet" not in t
    for n in NAMES[5:]:
        t = (out / f"{n}.tex").read_text()
        assert "source not committed yet" in t and "\\toprule" not in t
        assert not re.search(r"\d\.\d", t.split("\\begin{table}")[1])   # no numbers invented


def test_every_file_carries_the_provenance_comment(out):
    for p in out.glob("*.tex"):
        head = p.read_text().splitlines()
        if p.name == "sim_tables.tex":
            continue
        assert head[0].startswith("% source: ") and head[1].startswith("% sim_code_version: ")
        assert head[2].startswith("% config_hash: ")


def test_stamps_come_from_the_sources(out):
    gate = pd.read_csv(rt.CAL_DIR / "2021_Spain_pace_gate.csv")
    t1 = (out / "t1_spain_replay.tex").read_text()
    assert gate["sim_code_version"].iloc[0] in t1 and gate["config_hash"].iloc[0] in t1
    run = json.loads((rt.CAL_DIR / "cross_race" / "run.json").read_text())
    assert run["sim_code_version"] in (out / "t5_cross_race.tex").read_text()


def test_all_bytes_are_ascii(out):
    for p in list(out.glob("*.tex")) + list(COMMITTED.glob("*.tex")):
        p.read_bytes().decode("ascii")


def test_provisional_sources_say_so_in_the_caption(out):
    for n in ("t2_midrace_g", "t3_stop_gain", "t5_cross_race"):
        cap = re.search(r"\\caption\{(.*)\}", (out / f"{n}.tex").read_text()).group(1)
        assert cap.startswith("PROVISIONAL, one race.")


def test_committed_tex_is_current(out):
    for p in out.glob("*.tex"):
        assert (COMMITTED / p.name).read_text() == p.read_text(), f"regenerate reports/tex/: {p.name}"


@pytest.fixture
def synthetic_cal(tmp_path):
    cal = tmp_path / "calibration"
    shutil.copytree(rt.CAL_DIR / "cross_race", cal / "cross_race", ignore=shutil.ignore_patterns("races", "h"))
    for f in ("2021_ESP_linear.csv", "2021_Spain_pace_gate.csv", "2021_04_spain_traffic.csv"):
        shutil.copy(rt.CAL_DIR / f, cal / f)
    run = json.loads((cal / "cross_race" / "run.json").read_text())
    run["races"] = [*run["races"], "2099_01_synthetic"]
    (cal / "cross_race" / "run.json").write_text(json.dumps(run))
    return cal


def test_synthetic_cross_race_source_is_refused_without_the_flag(synthetic_cal, tmp_path):
    with pytest.raises(rt.SyntheticSourceRefused):
        rt.table5(synthetic_cal)
    done, refused = rt.write(tmp_path / "o", synthetic_cal)
    assert any(r.startswith("table 5") for r in refused)
    assert not (tmp_path / "o" / "t5_cross_race.tex").exists()
    assert "t5_cross_race" not in (tmp_path / "o" / "sim_tables.tex").read_text()
    assert rt.main(["--out", str(tmp_path / "o2"), "--calibration-dir", str(synthetic_cal)]) == 2


def test_allow_synthetic_renders_a_captioned_preview(synthetic_cal, tmp_path):
    done, refused = rt.write(tmp_path / "o", synthetic_cal, allow_synthetic=True)
    assert not refused
    cap = re.search(r"\\caption\{(.*)\}", (tmp_path / "o" / "t5_cross_race.tex").read_text()).group(1)
    assert cap.startswith("SYNTHETIC")


@pytest.mark.parametrize("sub,table,name", [("event_rates", 6, "t6_event_rates.tex"),
                                            ("fuel_prior", 7, "t7_pace_prior.tex")])
def test_synthetic_event_rate_and_prior_sources(tmp_path, sub, table, name):
    cal = tmp_path / "cal"
    (cal / sub).mkdir(parents=True)
    pd.DataFrame({"candidate": ["a", "b"], "score": [1.5, 2.5], "source": ["synthetic", "synthetic"]}).to_csv(
        cal / sub / "x.csv", index=False)
    make = rt.table6 if table == 6 else rt.table7
    with pytest.raises(rt.SyntheticSourceRefused):
        make(cal)
    t = make(cal, allow_synthetic=True)
    assert t.name == name and "SYNTHETIC" in t.text and not t.placeholder
    t.text.encode("ascii")


def test_non_ascii_input_is_rejected_not_dropped():
    with pytest.raises(UnicodeEncodeError):
        rt.esc("caf\u00e9")
    assert rt.esc("a_b%") == r"a\_b\%"
