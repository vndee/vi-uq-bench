# Dataset

This folder is intentionally empty.

The dataset (1,200 Vietnamese questions, 12 task types, with reference
answers) is not published in this repository. It is kept off the public web so
that it does not end up in the training data of future language models, which
would make it useless as a benchmark.

To get a copy for research use, email the authors at `duyhv@pm.me` with
your name, affiliation and intended use. Please do not redistribute it or post
it publicly.

Place the file here as `data/dataset.csv`, or point `VIUQ_DATASET` at it.
`python -m viuq.fingerprint` prints its fingerprint; the copy used in the paper
has 1,200 rows and fingerprint
`4c5a0dd1268d7a92946246b60eddf31735948bf05ef9b7fa4f5ef59fe2b60b2e`.

Expected columns: `id`, `input` (the prompt given to the model), `output`
(reference answer), `task_type`, `subtype`.
