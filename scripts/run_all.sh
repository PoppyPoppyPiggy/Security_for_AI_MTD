#!/bin/bash
# =============================================================================
# Run all 4 demo scripts in sequence
# Usage: bash scripts/run_all.sh
# =============================================================================
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
source .venv/bin/activate

echo ""
echo "============================================"
echo "  ATLAS-MTD-RAG — Full Live Demonstration"
echo "============================================"
echo ""
echo "Running from: $(pwd)"
echo "Python: $(which python)"
echo ""

echo "Press Enter to start Step 1 (Attack)..."
read

python scripts/demo_attack.py

echo ""
echo "Press Enter to start Step 2 (Defense)..."
read

python scripts/demo_defense.py

echo ""
echo "Press Enter to start Step 3 (Comparison)..."
read

python scripts/demo_compare.py

echo ""
echo "Press Enter to start Step 4 (LLM — takes ~30 seconds)..."
read

python scripts/demo_llm.py

echo ""
echo "============================================"
echo "  All 4 demonstrations complete."
echo "============================================"
