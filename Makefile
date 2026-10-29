PY ?= python
A  := analysis

.PHONY: generate nli manifests judge-j1 judge-j2 analysis paper-tables check

# GPU: answers, hidden states and AttentionScore for every model in
# configs/models.tsv (MODELS="phi4 qwen3" for a subset)
generate:
	bash scripts/run_generation.sh $(MODELS)

# GPU or Apple MPS: pairwise NLI matrices over the sampled answers
nli:
	$(PY) pipeline/nli_matrices.py

manifests:
	$(PY) pipeline/make_manifest.py

# Judges run as batch jobs: build, submit, then poll `status` and `fetch`
# (see pipeline/judge_*.py). These targets only build and submit.
judge-j1: manifests
	$(PY) pipeline/judge_openai.py build --tag luna
	$(PY) pipeline/judge_openai.py submit --tag luna

judge-j2: manifests
	for k in gemma3 llama31 phi4 qwen3 sailor2 sealion seallms vinallama; do \
	  $(PY) pipeline/judge_gemini.py build --tag gemini-$$k --only $$k && \
	  $(PY) pipeline/judge_gemini.py submit --tag gemini-$$k; done

# CPU: every table and the figure, from the artifacts
analysis: paper-tables

paper-tables:
	$(PY) $(A)/benchmark.py
	$(PY) $(A)/bootstrap_ci.py
	$(PY) $(A)/within_task_bench.py
	$(PY) $(A)/task_control.py
	$(PY) $(A)/tiers.py
	$(PY) $(A)/cert_sensitivity.py
	$(PY) $(A)/cert_final.py
	$(PY) $(A)/fig_risk_coverage.py
	$(PY) $(A)/supervised_baseline.py
	$(PY) $(A)/sup_interaction.py
	$(PY) $(A)/probe_control.py
	$(PY) $(A)/budget_control.py
	$(PY) $(A)/lineage_within.py

check:
	$(PY) -m viuq.fingerprint --check artifacts/generations
