#!/usr/bin/env bash
# Portfolio smoke: venv + pytest summary + toy demo + full-plant demo + VERDICT lines.
# Usage (from repo root or any cwd):
#   ./demos/run_portfolio_smoke.sh
#   bash demos/run_portfolio_smoke.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

echo "========================================"
echo "PIMS-ADMM-LLM portfolio smoke"
echo "ROOT=$ROOT"
echo "========================================"

if [[ ! -f .venv/bin/activate ]]; then
  echo "VERDICT: FAIL — missing .venv at $ROOT/.venv (create with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt)"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src

OVERALL=0
PYTEST_RC=0
DEMO_RC=0
FULL_RC=0

echo ""
echo "--- [1/3] pytest -q ---"
set +e
PYTEST_OUT="$(python -m pytest tests/ -q 2>&1)"
PYTEST_RC=$?
set -e
echo "$PYTEST_OUT"
# Summary line (last non-empty pytest summary-ish line)
PYTEST_SUMMARY="$(echo "$PYTEST_OUT" | grep -E 'passed|failed|error|skipped' | tail -n 1 || true)"
if [[ -z "$PYTEST_SUMMARY" ]]; then
  PYTEST_SUMMARY="(no pytest summary line captured; exit=$PYTEST_RC)"
fi
if [[ $PYTEST_RC -eq 0 ]]; then
  echo "VERDICT: PASS — pytest ($PYTEST_SUMMARY)"
else
  echo "VERDICT: FAIL — pytest exit=$PYTEST_RC ($PYTEST_SUMMARY)"
  OVERALL=1
fi

echo ""
echo "--- [2/3] python -m demos.run_demo ---"
set +e
DEMO_OUT="$(python -m demos.run_demo 2>&1)"
DEMO_RC=$?
set -e
echo "$DEMO_OUT"
DEMO_VERDICT="$(echo "$DEMO_OUT" | grep -E '^VERDICT:' | tail -n 1 || true)"
if [[ -z "$DEMO_VERDICT" ]]; then
  if [[ $DEMO_RC -eq 0 ]]; then
    DEMO_VERDICT="VERDICT: PASS — run_demo completed (no VERDICT line in output)"
  else
    DEMO_VERDICT="VERDICT: FAIL — run_demo exit=$DEMO_RC (no VERDICT line in output)"
  fi
fi
echo "$DEMO_VERDICT"
if [[ $DEMO_RC -ne 0 ]] || [[ "$DEMO_VERDICT" == *"FAIL"* ]]; then
  OVERALL=1
fi

echo ""
echo "--- [3/3] python -m demos.run_full_plant_demo ---"
set +e
FULL_OUT="$(python -m demos.run_full_plant_demo 2>&1)"
FULL_RC=$?
set -e
echo "$FULL_OUT"
FULL_VERDICT="$(echo "$FULL_OUT" | grep -E '^VERDICT:' | tail -n 1 || true)"
if [[ -z "$FULL_VERDICT" ]]; then
  if [[ $FULL_RC -eq 0 ]]; then
    FULL_VERDICT="VERDICT: PASS — run_full_plant_demo completed (no VERDICT line in output)"
  else
    FULL_VERDICT="VERDICT: FAIL — run_full_plant_demo exit=$FULL_RC (no VERDICT line in output)"
  fi
fi
echo "$FULL_VERDICT"
if [[ $FULL_RC -ne 0 ]] || [[ "$FULL_VERDICT" == *"FAIL"* ]]; then
  OVERALL=1
fi

echo ""
echo "========================================"
echo "PORTFOLIO SMOKE SUMMARY"
echo "  pytest:            exit=$PYTEST_RC"
echo "  run_demo:          exit=$DEMO_RC"
echo "  run_full_plant:    exit=$FULL_RC"
echo "  $PYTEST_SUMMARY"
echo "  $DEMO_VERDICT"
echo "  $FULL_VERDICT"
if [[ $OVERALL -eq 0 ]]; then
  echo "VERDICT: PASS — portfolio smoke (pytest + run_demo + run_full_plant_demo)"
else
  echo "VERDICT: FAIL — portfolio smoke (see step VERDICT lines above)"
fi
echo "========================================"
exit $OVERALL
