# Artifacts

Pipeline outputs are written here and are not committed or distributed.
Run the pipeline (see the main README) to produce them:

```
artifacts/
  generations/   gen_<key>.jsonl, arrays_<key>.npz, attn_<key>.npz, MANIFEST.json
  nli/           matrices_<key>.npz
  judge/         manifests/, judged_<tag>.jsonl
```

`VIUQ_ARTIFACTS` moves the whole folder elsewhere.
