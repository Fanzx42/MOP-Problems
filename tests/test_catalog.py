"""Catalogue, generated documentation and CLI must agree with the public API."""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from mop_problems import (
    PlannedProblemError,
    catalog,
    get_problem,
    get_problem_info,
    get_suite,
    list_problems,
    list_suites,
)
from mop_problems.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FAMILIES = {
    "ZDT",
    "DTLZ",
    "SCH",
    "FON",
    "POL",
    "WFG",
    "IDTLZ",
    "SDTLZ",
    "UF",
    "LZ09",
    "IMOP",
    "MaF",
    "ZCAT",
    "LSMOP",
    "MMF",
    "Omni-test",
    "SYM-PART",
    "FDA",
    "JY",
    "RE_continuous",
    "RWA",
    "BNH",
    "TNK",
    "OSY",
    "SRN",
    "CTP",
    "CF",
    "C-DTLZ",
    "DC-DTLZ",
    "Eq-DTLZ",
    "Eq-IDTLZ",
    "MW",
    "LIR-CMOP",
    "DAS-CMOP",
    "DAS-CMaOP",
    "DOC",
    "NCTP",
    "ZXH_CF",
    "LSCMOP",
    "CMMOP",
    "CRE",
    "Welded-Beam",
    "RCM_continuous",
}


def test_complete_user_taxonomy_and_no_placeholder_registrations():
    data = catalog()
    suites = data["suites"]
    assert data["schema_version"] == 1
    assert {s["name"] for s in suites} == EXPECTED_FAMILIES
    assert len({s["id"] for s in suites}) == len(suites)
    assert len({s["path"].lower() for s in suites}) == len(suites)
    assert len(set(list_problems())) == len(list_problems()) == 19
    for s in suites:
        assert s["status"] in {"implemented", "planned"}
        assert s["source_status"] in {"identified", "pending", "membership-pending"}
        path = (ROOT / s["path"]).resolve()
        assert path.is_relative_to(ROOT / "problems")
        assert (path / "README.md").exists()
        names = list_problems(suite=s["name"])
        assert bool(names) == (s["status"] == "implemented")
        if names:
            assert set(names) == set(s["members"])
            assert s["source_url"].startswith("https://")
        else:
            with pytest.raises(PlannedProblemError):
                get_problem(s["name"])


def test_lookup_filtering_and_detached_metadata():
    assert get_problem(" zdt-1 ").name == "ZDT1"
    assert get_problem("dtlz_2").name == "DTLZ2"
    assert list_problems(category="constrained") == ["BNH", "TNK", "OSY", "SRN"]
    assert list_problems(suite="WFG") == []
    assert len(list_suites(status="implemented")) == 9
    assert len(list_suites(status="planned")) == 34
    info = get_problem_info("ZDT1")
    info["name"] = "changed"
    assert get_problem_info("ZDT1")["name"] == "ZDT1"
    full = catalog()
    full["suites"].clear()
    assert len(catalog()["suites"]) == 43
    assert get_suite("re-continuous")["name"] == "RE_continuous"


@pytest.mark.parametrize("name", ["WFG1", "Eq-IDTLZ2", "LSMOP9"])
def test_known_pending_members_have_useful_errors(name):
    with pytest.raises(PlannedProblemError, match="not implemented"):
        get_problem(name)


def test_invalid_names_filters_and_suite_members():
    for name in ["ZDT5", "ZDT999", "WFG42", "unrelated"]:
        with pytest.raises(KeyError):
            get_problem(name)
    with pytest.raises(KeyError):
        get_suite("missing")
    with pytest.raises(ValueError):
        list_suites(status="unknown")
    for name in ["", None, 123]:
        with pytest.raises(ValueError):
            get_problem(name)


def test_generated_docs_up_to_date():
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_catalog.py"), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_local_document_links_resolve():
    pages = [*ROOT.glob("*.md"), *ROOT.glob("docs/**/*.md"), *ROOT.glob("problems/**/*.md")]
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for destination in re.findall(r"\]\(([^)]+)\)", text):
            if "://" in destination or destination.startswith("#"):
                continue
            path, _, anchor = destination.partition("#")
            target = (page.parent / path).resolve()
            assert target.exists(), f"{page}: missing {destination}"
            if anchor and target.suffix == ".md":
                headings = re.findall(r"^#+\s+(.+)$", target.read_text(encoding="utf-8"), re.M)
                slugs = {re.sub(r"[^\w\s-]", "", h).lower().replace(" ", "-") for h in headings}
                assert anchor in slugs, f"{page}: missing anchor {destination}"


def test_cli(capsys):
    assert main(["list", "--suite", "ZDT", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 5
    main(["suites", "--status", "planned", "--json"])
    assert len(json.loads(capsys.readouterr().out)) == 34
    main(["info", "BNH"])
    assert json.loads(capsys.readouterr().out)["n_ieq_constr"] == 2
    main(["sample", "DTLZ2", "--n", "2", "--n-obj", "5", "--seed", "1"])
    out = json.loads(capsys.readouterr().out)
    assert len(out["X"][0]) == 14 and len(out["F"][0]) == 5
    main(["list", "--category", "constrained"])
    assert capsys.readouterr().out.splitlines() == ["BNH", "TNK", "OSY", "SRN"]
    main(["suites", "--status", "implemented"])
    assert "implemented" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["sample", "WFG1"])
    assert e.value.code == 2


def test_module_cli_from_another_working_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "mop_problems", "list", "--json"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(json.loads(result.stdout)) == 19
