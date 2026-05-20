# Makefile — ACCA Project
# ARC Prize 2026

.PHONY: setup test eval-synthetic eval-ablations eval-universality figures kaggle-test clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete. Activate with: source .venv/bin/activate"

test:
	pytest tests/ -v --tb=short

eval-synthetic:
	python eval/local_eval.py \
		--env_dir envs/synthetic/ \
		--agent src.agent.ACCAAgent \
		--output experiments/ablations/full_results.json
	@echo "Results saved to experiments/ablations/full_results.json"

eval-ablations:
	python eval/ablation_runner.py \
		--env_dir envs/synthetic/ \
		--output_dir experiments/ablations/ \
		--n_seeds 3
	@echo "Ablation table saved to experiments/ablations/ablation_table.md"

eval-universality:
	python experiments/universality/arc2_bridge.py \
		--arc2_dir data/arc-agi-2/training/ \
		--output_dir experiments/universality/
	@echo "Universality results saved to experiments/universality/results.json"

figures:
	python paper/figures/generate_figures.py
	@echo "Figures saved to paper/figures/"

kaggle-test:
	jupyter nbconvert --to notebook --execute kaggle/submission_final.ipynb \
		--output kaggle/submission_final_executed.ipynb
	@echo "Kaggle notebook executed successfully"

verify-synthetic-envs:
	python envs/synthetic/verifier.py
	@echo "All synthetic environments verified"

generate-synthetic-envs:
	python envs/synthetic/generate_envs.py
	$(MAKE) verify-synthetic-envs

lint:
	ruff check src/ eval/ experiments/ tests/
	black --check src/ eval/ experiments/ tests/ --line-length 100

format:
	black src/ eval/ experiments/ tests/ --line-length 100

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete

# Count words in Kaggle writeup (must be ≤1500)
wordcount:
	@wc -w paper/kaggle_writeup.md
	@echo "Target: ≤1500 words"
