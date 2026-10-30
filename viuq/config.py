"""Paths shared by the pipeline and the analysis scripts.

Every location can be overridden with an environment variable, so the code
never needs editing to point at a different copy of the data:

  VIUQ_DATASET    dataset CSV               (default data/dataset.csv)
  VIUQ_ARTIFACTS  pipeline outputs          (default artifacts/)
  VIUQ_RESULTS    analysis tables, figure   (default results/)
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET = os.environ.get("VIUQ_DATASET", os.path.join(ROOT, "data", "dataset.csv"))
ARTIFACTS = os.environ.get("VIUQ_ARTIFACTS", os.path.join(ROOT, "artifacts"))
RESULTS = os.environ.get("VIUQ_RESULTS", os.path.join(ROOT, "results"))

GEN_DIR = os.path.join(ARTIFACTS, "generations")    # gen_/arrays_<key>
NLI_DIR = os.path.join(ARTIFACTS, "nli")            # matrices_<key>.npz
JUDGE_DIR = os.path.join(ARTIFACTS, "judge")        # judged_<tag>.jsonl
MANIFEST_DIR = os.path.join(JUDGE_DIR, "manifests")
SCORES = os.path.join(RESULTS, "benchmark_scores.npz")

N = 1200
KEYS = ["gemma3", "llama31", "phi4", "qwen3", "sailor2", "sealion",
        "seallms", "vinallama"]
DISPLAY = {"gemma3": "Gemma-3", "llama31": "LLaMA-3.1", "phi4": "Phi-4",
           "qwen3": "Qwen-3", "sailor2": "Sailor2", "sealion": "SEA-LION",
           "seallms": "SeaLLMs", "vinallama": "VinaLLaMA"}
