#!/bin/bash
set -e
echo "Running RAG evaluation..."
python -m src.eval.run_ragas
echo "Done! Results saved to data/evaluation/eval_results.json"
