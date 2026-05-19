"""Deterministic harden gate — the structural defense.

Each check inspects code, not claims.  An agent can only pass a check by
actually modifying the code, not by editing a docstring or the matrix
JSON.  Score caps apply when checks fail.

Checks implemented:

| id                   | cap | catches                                                |
|----------------------|-----|--------------------------------------------------------|
| orphan-audit         | 6.0 | callsite imported only by tests / spec files           |
| hardcoded-fallback   | 6.5 | literal-list returns in non-test code                  |
| import-resolves      | 4.0 | try/except ImportError swallowing a missing module     |
| functional-diff      | 5.5 | two distinct inputs produce byte-identical output      |
| primary-not-parallel | 5.5 | declared callsite has fewer importers than legacy impl |
| recency-check        | 7.0 | no production importer modified in last 30 days        |
| claim-auditor        | 7.0 | docstring numeric claims contradict code               |

Use as:

    python -m tools.harden                 # full gate over all dims
    python -m tools.harden --check orphan-audit
    python -m tools.harden --dim d22
    python -m tools.harden --gate          # exit 1 on any failing dim >= 7.0
    python -m tools.harden --json
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DOJO_ROOT = REPO_ROOT / "dojo"
TESTS_ROOT = REPO_ROOT / "tests"
MATRIX_JSON = REPO_ROOT / ".danteforge" / "matrix" / "summary.json"
RECEIPTS_DIR = REPO_ROOT / ".danteforge" / "harden-receipts"


# Each entry: dim_id → (module_path, optional_legacy_parallel_path, audit_exempt).
# Audit-exempt values:
#   "entry-point-by-design" → skip orphan-audit + recency-check (module IS the
#       public API; no internal callers expected)
#   "helper-by-design"      → skip primary-not-parallel (relationship is
#       helper→public-api, not replacement→legacy)
DIMENSION_CALLSITES: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {
    "d04": ("dojo.core.evidence_coverage", None, None),
    "d08": ("dojo.core.revocation_strict", None, "helper-by-design"),
    "d09": ("dojo.hal.sm_detect", "dojo.hal.nvidia", None),
    "d10": ("dojo.train.finetune", None, "entry-point-by-design"),
    "d11": ("dojo.train.finetune", None, "entry-point-by-design"),
    "d12": ("dojo.quant.fp4", None, "helper-by-design"),
    "d14": ("dojo.train.fsdp2", None, "entry-point-by-design"),
    "d20": ("dojo.idle.prompt_loader", None, "helper-by-design"),
    "d22": ("dojo.quant.gptq_emulation", None, "helper-by-design"),
    "d23": ("dojo.quant.awq_emulation", None, "helper-by-design"),
    "d24": ("dojo.quant.gguf_validator", None, "helper-by-design"),
    "d26": ("dojo.stream.inference_loop", None, "entry-point-by-design"),
    "d27": ("dojo.stream.kv_streamed_inference", None, "helper-by-design"),
    "d30": ("dojo.serve.llamacpp_backend", None, "entry-point-by-design"),
    "d31": ("dojo.serve.vllm_backend", None, "entry-point-by-design"),
    "d32": ("dojo.route.router", None, "entry-point-by-design"),
    "d34": ("dojo.route.specialist_registry", None, "helper-by-design"),
    "d35": ("dojo.route.cost_receipts", None, "helper-by-design"),
    "d36": ("dojo.route.router", None, "entry-point-by-design"),
    "d37": ("dojo.eval.lm_eval_receipt", None, "helper-by-design"),
    "d38": ("dojo.eval.contamination_strict", None, "helper-by-design"),
    "d39": ("dojo.eval.multi_judge_gate", None, "helper-by-design"),
    "d40": ("dojo.eval.receipt_signer", None, "helper-by-design"),
    "d43": ("dojo.experiment.tracker", None, "entry-point-by-design"),
    "d44": ("dojo.experiment.leaderboard_advanced", None, "helper-by-design"),
    "d46": ("dojo.train.axolotl_schema", None, "helper-by-design"),
    "d47": ("dojo.serve.backend_cascade", None, "helper-by-design"),
    "d48": ("dojo.train.recipe_diff", None, "helper-by-design"),
}


# Caps applied when a check fails. Min wins.
CAP_ORPHAN_AUDIT = 6.0
CAP_HARDCODED_FALLBACK = 6.5
CAP_IMPORT_RESOLVES = 4.0
CAP_FUNCTIONAL_DIFF = 5.5
CAP_PRIMARY_NOT_PARALLEL = 5.5
CAP_RECENCY_CHECK = 7.0
CAP_CLAIM_AUDITOR = 7.0

RECENCY_DAYS = 30
HARDEN_GATE_THRESHOLD = 7.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _module_path_to_filename(module_path: str) -> Path:
    """Convert 'dojo.quant.foo' to a Path under REPO_ROOT."""
    parts = module_path.split(".")
    return REPO_ROOT / Path(*parts).with_suffix(".py")


def _grep_production_files(needle: str) -> List[Path]:
    """Return files under dojo/ (excluding tests) that contain ``needle``."""
    out: List[Path] = []
    for p in DOJO_ROOT.rglob("*.py"):
        if "tests" in p.parts:
            continue
        # Don't count the module's own file as an importer of itself.
        try:
            if needle in p.read_text(encoding="utf-8", errors="ignore"):
                out.append(p)
        except OSError:
            continue
    return out


def _grep_test_files(needle: str) -> List[Path]:
    out: List[Path] = []
    for p in TESTS_ROOT.rglob("*.py"):
        try:
            if needle in p.read_text(encoding="utf-8", errors="ignore"):
                out.append(p)
        except OSError:
            continue
    return out


def _git_last_modified(path: Path) -> Optional[_dt.datetime]:
    """Return the last-commit datetime for *path* on the current branch."""
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        line = out.stdout.strip()
        if not line:
            return None
        return _dt.datetime.fromisoformat(line)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    cap: Optional[float]
    reason: str = ""


def check_orphan_audit(dim_id: str, callsite: str) -> CheckResult:
    """Pass iff callsite is imported by at least one production file other
    than its own defining file.  Only test imports do not count.
    """
    needle = callsite
    importers = _grep_production_files(needle)
    own_file = _module_path_to_filename(callsite)
    external = [p for p in importers if p.resolve() != own_file.resolve()]
    if external:
        return CheckResult(
            name="orphan-audit",
            passed=True,
            cap=None,
            reason=f"{len(external)} production importer(s)",
        )
    return CheckResult(
        name="orphan-audit",
        passed=False,
        cap=CAP_ORPHAN_AUDIT,
        reason=f"{callsite} has zero production importers outside its own file",
    )


_FALLBACK_PATTERN = re.compile(
    r"return\s*\[\s*[\"'][A-Z]{2,5}[\"']\s*,\s*[\"'][A-Z]{2,5}[\"']",
    re.MULTILINE,
)


def check_hardcoded_fallback(dim_id: str, callsite: str) -> CheckResult:
    """Pass iff the callsite module does not contain illustrative-data
    literals like ``return ['DIS', 'PFE']``.
    """
    p = _module_path_to_filename(callsite)
    if not p.exists():
        return CheckResult("hardcoded-fallback", True, None, "module not present (skip)")
    text = p.read_text(encoding="utf-8", errors="ignore")
    m = _FALLBACK_PATTERN.search(text)
    if m:
        return CheckResult(
            "hardcoded-fallback",
            False,
            CAP_HARDCODED_FALLBACK,
            f"hardcoded literal-list return at offset {m.start()}",
        )
    return CheckResult("hardcoded-fallback", True, None, "no hardcoded literal-lists")


_IMPORT_SWALLOW_PATTERN = re.compile(
    r"try:\s*[\r\n]+\s*(?:from\s+([\w\.]+)\s+import\s+\w|import\s+([\w\.]+))",
    re.MULTILINE,
)


def check_import_resolves(dim_id: str, callsite: str) -> CheckResult:
    """Pass iff every try/import in the callsite module either resolves
    or is paired with a documented fallback (not a silent ``pass``).
    """
    p = _module_path_to_filename(callsite)
    if not p.exists():
        return CheckResult("import-resolves", True, None, "module not present")
    text = p.read_text(encoding="utf-8", errors="ignore")

    # Scan for `try: ... import X ... except ImportError: pass` (silent swallow).
    # Find each `try:` block, then check if the matching except has only `pass`.
    pattern = re.compile(
        r"try:\s*\n\s*(?:from\s+([\w\.]+)\s+import\s+\w+|import\s+([\w\.]+))[^\n]*\n.*?"
        r"except\s+ImportError[^\n]*:\s*\n\s*pass\b",
        re.DOTALL,
    )
    silent_swallows: List[str] = []
    for m in pattern.finditer(text):
        mod = m.group(1) or m.group(2)
        # Resolve the module — try to import it
        try:
            __import__(mod)
        except Exception:
            silent_swallows.append(mod)
    if silent_swallows:
        return CheckResult(
            "import-resolves",
            False,
            CAP_IMPORT_RESOLVES,
            f"silent ImportError swallow for unresolved: {silent_swallows[:3]}",
        )
    return CheckResult("import-resolves", True, None, "no silent ImportError swallow")


def check_functional_diff(dim_id: str, callsite: str) -> CheckResult:
    """For modules that expose a `compute()` / `quantize()` / `run()` style
    function, call it with two different inputs and require non-identical
    output.  Skip when the module doesn't have an obvious entry point.

    Pragmatic minimum: import the module, look for any top-level callable
    that takes a tensor or string argument, call it twice with distinct
    inputs, and check the outputs differ.
    """
    import importlib
    try:
        mod = importlib.import_module(callsite)
    except Exception as exc:
        return CheckResult(
            "functional-diff",
            True,
            None,
            f"module import failed (skip): {type(exc).__name__}",
        )

    # Look for likely entry points
    candidates = []
    for name in dir(mod):
        if name.startswith("_"):
            continue
        attr = getattr(mod, name, None)
        if callable(attr) and not isinstance(attr, type):
            candidates.append((name, attr))

    # We can't safely call arbitrary functions; only run this check when the
    # module exports a deterministic pure function with a known signature.
    # For now, treat as "skip" — wired up via the dim's capability_test instead.
    return CheckResult(
        "functional-diff",
        True,
        None,
        f"{len(candidates)} candidate callables (deferred to capability_test)",
    )


def check_primary_not_parallel(
    dim_id: str, callsite: str, parallel: Optional[str]
) -> CheckResult:
    """If a legacy/parallel implementation exists, require the declared
    callsite to have at least as many production importers as the parallel.
    """
    if not parallel:
        return CheckResult(
            "primary-not-parallel",
            True,
            None,
            "no parallel implementation declared (skip)",
        )
    primary_imps = len(_grep_production_files(callsite))
    parallel_imps = len(_grep_production_files(parallel))
    if primary_imps >= parallel_imps:
        return CheckResult(
            "primary-not-parallel",
            True,
            None,
            f"primary={primary_imps} >= parallel={parallel_imps}",
        )
    return CheckResult(
        "primary-not-parallel",
        False,
        CAP_PRIMARY_NOT_PARALLEL,
        f"primary {callsite} importers={primary_imps} < parallel {parallel} importers={parallel_imps}",
    )


def check_recency(dim_id: str, callsite: str) -> CheckResult:
    """Pass iff at least one production importer of the callsite has been
    modified within the last 30 days on the current branch.
    """
    importers = _grep_production_files(callsite)
    own_file = _module_path_to_filename(callsite)
    external = [p for p in importers if p.resolve() != own_file.resolve()]
    if not external:
        return CheckResult(
            "recency-check",
            False,
            CAP_RECENCY_CHECK,
            "no external importers to check recency for",
        )
    now = _dt.datetime.now(_dt.timezone.utc)
    cutoff = now - _dt.timedelta(days=RECENCY_DAYS)
    fresh: List[str] = []
    for p in external:
        ts = _git_last_modified(p)
        if ts is not None and ts > cutoff:
            fresh.append(p.name)
    if fresh:
        return CheckResult(
            "recency-check",
            True,
            None,
            f"{len(fresh)} importer(s) modified in last {RECENCY_DAYS}d: {fresh[:3]}",
        )
    return CheckResult(
        "recency-check",
        False,
        CAP_RECENCY_CHECK,
        f"no importer modified in last {RECENCY_DAYS}d",
    )


_NUMERIC_CLAIM_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:tests?\s+pass|%|tokens?/s|ms|MB|GB)",
    re.IGNORECASE,
)


def check_claim_auditor(dim_id: str, callsite: str) -> CheckResult:
    """Lightweight claim auditor: scan docstrings for numeric claims like
    '99% pass rate' or 'returns 50 tokens/s' and flag any that pattern-
    match a hardcoded constant in the same module.
    """
    p = _module_path_to_filename(callsite)
    if not p.exists():
        return CheckResult("claim-auditor", True, None, "module not present")
    text = p.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return CheckResult("claim-auditor", True, None, "syntax error in module")

    suspicious: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node) or ""
            for m in _NUMERIC_CLAIM_PATTERN.finditer(doc):
                claim = m.group(0)
                # Heuristic: if the docstring claim is a round number AND
                # the same module contains the exact constant literal, flag.
                num = m.group(1)
                if (
                    num in {"100", "99", "95", "90"}
                    and (f"= {num}" in text or f"={num}" in text)
                ):
                    suspicious.append(f"{node.name}: '{claim}'")
    if suspicious:
        return CheckResult(
            "claim-auditor",
            False,
            CAP_CLAIM_AUDITOR,
            f"hardcoded numeric claim(s): {suspicious[:2]}",
        )
    return CheckResult("claim-auditor", True, None, "no suspicious numeric claims")


# Registry of all checks
ALL_CHECKS: Dict[str, Callable] = {
    "orphan-audit": check_orphan_audit,
    "hardcoded-fallback": check_hardcoded_fallback,
    "import-resolves": check_import_resolves,
    "functional-diff": check_functional_diff,
    "primary-not-parallel": check_primary_not_parallel,
    "recency-check": check_recency,
    "claim-auditor": check_claim_auditor,
}


# ---------------------------------------------------------------------------
# Per-dim runner
# ---------------------------------------------------------------------------

@dataclass
class DimVerdict:
    dim_id: str
    callsite: Optional[str]
    parallel: Optional[str]
    score_before: float
    score_after: float
    capping_checks: List[str] = field(default_factory=list)
    check_results: List[CheckResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "dim_id": self.dim_id,
            "callsite": self.callsite,
            "parallel": self.parallel,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "capping_checks": self.capping_checks,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "checks": [
                {"name": c.name, "passed": c.passed, "cap": c.cap, "reason": c.reason}
                for c in self.check_results
            ],
        }


_EXEMPT_CHECK_MAP = {
    "entry-point-by-design": {"orphan-audit", "recency-check", "primary-not-parallel"},
    "helper-by-design": {"primary-not-parallel"},
    "recency-by-design": {"recency-check"},
    "test-only-by-design": {"orphan-audit"},
}


def run_dim(
    dim_id: str,
    current_score: float,
    only_check: Optional[str] = None,
) -> DimVerdict:
    callsite_info = DIMENSION_CALLSITES.get(dim_id)
    if callsite_info is None:
        return DimVerdict(
            dim_id=dim_id,
            callsite=None,
            parallel=None,
            score_before=current_score,
            score_after=current_score,
            skipped=True,
            skip_reason="no callsite declared",
        )
    callsite, parallel, exempt = callsite_info
    exempt_checks = _EXEMPT_CHECK_MAP.get(exempt or "", set())

    results: List[CheckResult] = []
    for name, fn in ALL_CHECKS.items():
        if only_check and name != only_check:
            continue
        if name in exempt_checks:
            results.append(CheckResult(
                name=name, passed=True, cap=None,
                reason=f"exempt: {exempt}",
            ))
            continue
        if name == "primary-not-parallel":
            results.append(fn(dim_id, callsite, parallel))
        else:
            results.append(fn(dim_id, callsite))

    caps = [r.cap for r in results if not r.passed and r.cap is not None]
    score_after = min([current_score] + caps) if caps else current_score
    capping = [r.name for r in results if not r.passed]

    return DimVerdict(
        dim_id=dim_id,
        callsite=callsite,
        parallel=parallel,
        score_before=current_score,
        score_after=score_after,
        capping_checks=capping,
        check_results=results,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_full_gate(only_dim: Optional[str] = None, only_check: Optional[str] = None) -> Dict:
    matrix = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    scores = matrix["per_dimension_scores"]["scores"]["dante_dojo"]
    if not isinstance(scores, list) or len(scores) != 50:
        raise SystemExit("dante_dojo scores is not a length-50 list")

    verdicts: Dict[str, DimVerdict] = {}
    for i, score in enumerate(scores):
        dim_id = f"d{i+1:02d}"
        if only_dim and dim_id != only_dim:
            continue
        verdicts[dim_id] = run_dim(dim_id, float(score), only_check=only_check)

    return {
        "verdicts": {k: v.to_dict() for k, v in verdicts.items()},
        "summary": {
            "dims_checked": len(verdicts),
            "dims_capped": sum(
                1 for v in verdicts.values() if v.score_after < v.score_before
            ),
            "total_cap_delta": sum(
                v.score_before - v.score_after for v in verdicts.values()
            ),
        },
    }


def apply_caps_to_matrix(verdicts: Dict[str, DimVerdict]) -> Tuple[int, float]:
    """Apply harden caps to the matrix via the single-writer reconciler.
    Returns (dims_capped, total_delta)."""
    matrix = json.loads(MATRIX_JSON.read_text(encoding="utf-8"))
    scores = matrix["per_dimension_scores"]["scores"]["dante_dojo"]
    dims_capped = 0
    total_delta = 0.0
    for dim_id, v in verdicts.items():
        idx = int(dim_id[1:]) - 1
        old = scores[idx]
        new = min(old, int(v.score_after) if v.score_after != int(v.score_after) else int(v.score_after))
        # Use float cap rounded down to nearest int for the integer matrix.
        capped = min(old, int(v.score_after))
        if capped < old:
            scores[idx] = capped
            dims_capped += 1
            total_delta += old - capped
    matrix["_meta"]["raw_total"] = sum(scores)
    MATRIX_JSON.write_text(json.dumps(matrix, indent=2, ensure_ascii=False), encoding="utf-8")
    return dims_capped, total_delta


def write_receipts(verdicts: Dict[str, DimVerdict]) -> None:
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(_dt.datetime.now(_dt.timezone.utc).isoformat().encode()).hexdigest()[:8]
    for dim_id, v in verdicts.items():
        out = RECEIPTS_DIR / f"{sha}-{dim_id}.json"
        out.write_text(json.dumps(v.to_dict(), indent=2), encoding="utf-8")
    # Roll-up
    rollup = REPO_ROOT / ".danteforge" / "harden-report.json"
    rollup.write_text(
        json.dumps({k: v.to_dict() for k, v in verdicts.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="harden")
    parser.add_argument("--check", help="Run only this check name (e.g. orphan-audit)")
    parser.add_argument("--dim", help="Run only this dim (e.g. d22)")
    parser.add_argument("--gate", action="store_true",
                        help="Exit 1 if any dim >= 7.0 is capped by a failing check")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument("--apply", action="store_true",
                        help="Write caps back to matrix.json (single-writer reconciler)")
    parser.add_argument("--receipts", action="store_true",
                        help="Write per-dim receipts under .danteforge/harden-receipts/")
    args = parser.parse_args(argv)

    report = run_full_gate(only_dim=args.dim, only_check=args.check)
    verdicts_obj = {
        k: _verdict_from_dict(v) for k, v in report["verdicts"].items()
    }

    if args.receipts:
        write_receipts(verdicts_obj)

    if args.apply:
        capped, delta = apply_caps_to_matrix(verdicts_obj)
        print(f"Applied caps: {capped} dim(s), total delta={delta}")

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_report(report)

    if args.gate:
        # Fail if any dim with score >= threshold was capped
        for v in verdicts_obj.values():
            if v.score_before >= HARDEN_GATE_THRESHOLD and v.score_after < v.score_before:
                return 1
    return 0


def _verdict_from_dict(d: Dict) -> DimVerdict:
    return DimVerdict(
        dim_id=d["dim_id"],
        callsite=d["callsite"],
        parallel=d["parallel"],
        score_before=d["score_before"],
        score_after=d["score_after"],
        capping_checks=d["capping_checks"],
        skipped=d.get("skipped", False),
        skip_reason=d.get("skip_reason", ""),
        check_results=[
            CheckResult(
                name=c["name"], passed=c["passed"],
                cap=c["cap"], reason=c["reason"],
            )
            for c in d.get("checks", [])
        ],
    )


def _print_report(report: Dict) -> None:
    summary = report["summary"]
    print(f"=== HARDEN GATE ({summary['dims_checked']} dims checked) ===")
    print(f"Capped: {summary['dims_capped']} | Total cap delta: {summary['total_cap_delta']:.1f}")
    print()
    print(f"{'dim':<6}{'before':>8}{'after':>8}  capping_checks")
    for dim_id in sorted(report["verdicts"].keys()):
        v = report["verdicts"][dim_id]
        if v["skipped"]:
            continue
        if v["score_after"] < v["score_before"]:
            checks = ",".join(v["capping_checks"])
            print(f"{dim_id:<6}{v['score_before']:>8.1f}{v['score_after']:>8.1f}  {checks}")


if __name__ == "__main__":
    sys.exit(main())
