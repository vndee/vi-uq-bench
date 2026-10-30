# Dataset

This folder is intentionally empty.

The dataset (1,200 Vietnamese questions, 12 task types, with reference
answers) is not stored in this repository. It is on the Hugging Face Hub at
https://huggingface.co/datasets/vndee/vi-uq-bench with gated access, so
that it does not end up in the training data of future language models,
which would make it useless as a benchmark. Request access on that page;
please do not redistribute the file or post it publicly.

```bash
huggingface-cli download vndee/vi-uq-bench data/test.csv --repo-type dataset --local-dir hf
cp hf/data/test.csv data/dataset.csv
```

Place the file here as `data/dataset.csv`, or point `VIUQ_DATASET` at it.
`python -m viuq.fingerprint` prints its fingerprint; the copy used in the paper
has 1,200 rows and fingerprint
`4c5a0dd1268d7a92946246b60eddf31735948bf05ef9b7fa4f5ef59fe2b60b2e`.

Expected columns: `id`, `input` (the prompt given to the model), `output`
(reference answer), `task_type`, `subtype`.
