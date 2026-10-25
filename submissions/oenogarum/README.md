# oenogarum

Self-run Decision Index 0.3 public score **62.80** as scored (raw 71.62, breadth 62.24). With the 1 trained-on test item
below counted as wrong, it is **62.80** (that item was answered correctly; marking it wrong lowers the index by 0.0004). The run is
complete: 140,620 requests run, 140,178 scored after the official 442 exclusions, 140,178 answered, 0 unsupported, 0 errors. This is not
the private Full score or a maintainer-verified rank.

- Model: https://huggingface.co/plantwaterer/oenogarum/tree/9b684d7da38eb200fe4c5d11c36e3ce819ea2cfb: the uniform average of three seeds of a full fine-tune of
  google/gemma-4-31B-it, with its own last-layer head. Code is mirrored at https://github.com/treewaterer/oenogarum.
- Results: https://huggingface.co/datasets/plantwaterer/oenogarum-decision-index/tree/1469023963c1e1d6d9ed3b3e64d452c3bf1d9745/runs/oenogarum (run files as written by the
  kit's runner and scorer, unmodified).
- Kit: apolinario/decision-index @ 9eb2dbe2a358004c8782c66e40a83ac07b953fec, unchanged runner and scorer.
- Engine: `lenavg_bayes_engine:LenAvgBayesEngine`, in-process, from the model repository (the same engine files as the `allec-oavg` entry):
  - `dikit_engine.py`, SHA-256 63bc2c878e8a4a615699a2ee4c20c679ba208c1cf2e2aabf5f547695317712b9;
  - `garas_vbll_engine.py`, SHA-256 dd1f6d55aa90e4b17c6b9b7b8b582fd610d6a830c4635529270499e5c1bb82f0;
  - `vbll_core.py`, SHA-256 b5ec4a496c608712bae15680f83b4de80c59385776eb8eec453a99c44024b0bd;
  - `fullft_mem.py`, SHA-256 31d02612c4bf896f7abe702caea61fb14c8dab307a393d3845942021abfb6c4f;
  - `garas_engine.py`, SHA-256 b35ee6482dc351aee48dca7cedc48e640e4efffda42a3fb9b648929e47ccd5d4;
  - `garas_core.py`, SHA-256 a7f4d8192d6b287b7c57d22ed7c33065a8d753650fd11e9bcf7f9e378b5c3a6e;
  - `oavg_engine.py`, SHA-256 97b6c8f5fc939c4dfe6ca2229af893f4883918c0c867e5cae06d68794635a843;
  - `lenavg_engine.py`, SHA-256 8a2c5c02afe3d23b10afabf40b7a3d77f453c99c140ee142f07181b6b0218a30;
  - `lenavg_bayes_engine.py`, SHA-256 56dd1545e57e00d40f107c2ed405848493926f688890636fc022643eb142a7e7.

## Settings

```bash
REPO=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('plantwaterer/oenogarum', revision='9b684d7da38eb200fe4c5d11c36e3ce819ea2cfb'))")
PYTHONPATH="$REPO" python -m decision_index run --engine lenavg_bayes_engine:LenAvgBayesEngine --option model="$REPO" --compact --out runs/oenogarum
```

- **Precision:** bf16 weights and backbone, eager PyTorch (torch 2.14.1+cu130, transformers 5.12.1). The head runs on the CPU in float64.
- **Attention:** the model's full-attention layers attend bidirectionally within each prompt and its sliding-window layers causally
  (window 1,024). It is served with the `jev_mem` attention function in `fullft_mem.py`; the engine refuses transformers' stock
  attention, which would run the model causally.
- **Which requests are averaged:** L = the sum over the request's questions of [len(state) + len(instructions) + the sum over the question's criteria of (len(str(key)) + len(str(value)))], Python `len()`: a string state counts its characters, a dictionary state counts its top-level keys (not its text), a missing state counts 0, and the state is counted once per question. Averaged iff L <= 939 (94,198 of the 140,620 requests).
- **Averaged choice questions (up to 52 options):** 4 option orders, the engine's own order and `numpy.random.default_rng(20261021 +
  1000 * j + K).permutation(K)` for j = 1, 2, 3; each order's head output (softmax of the letters' logits / T = 1.080) mapped back to
  the options, averaged with equal weight, then p^(1 / 0.95438) renormalised (0.95438 and 0.93738 are fixed engine constants).
- **Averaged yes / no:** the native letter softmax of the base and the swapped order, averaged, then p^(1 / 0.93738) renormalised; then
  the head's binary correction (log-odds of its readout with its fitted residual and posterior variance minus with both off, from the
  base prompt's hidden state and the log averaged native probabilities) is added to the log-odds; no extra temperature.
- **Everything else** (L > 939, questions with more than 52 options): one prompt per question, the head's output from the hidden state
  and the option letters' logits.
- **Head:** Adds a binary Bayesian correction on averaged yes/no predictions and retains the existing binary posterior on unaveraged yes/no. Ordinary choices use native scores with temperature calibration; their current Bayesian correction controls are zero.
- **Batching:** the prompts of one request, re-lettered copies included, run in packed rows of up to 32,768 tokens.
- **Hardware:** one GPU with at least 75 GB (bf16 weights 62 GB); fits one RTX PRO 6000 (96 GB).

## Run identity

- **Shards:** the full public suite ran as 8 row shards (row i to shard i mod 8), each on one NVIDIA H100 80GB HBM3.
- **Runner:** each shard used the unchanged kit runner with resume; every result line carries the engine `lenavg_bayes_engine:LenAvgBayesEngine`.
- **Merge and scoring:** the shard result lines were concatenated unchanged into one `results.jsonl`; resume passes of the unchanged
  runner then ran 0 rows, and the unchanged scorer scored the file. The runner's `model` option pointed at a local copy of the model
  repository's files; `environment.json` records the engine files' md5s.

## Capacity limits

- **Context:** max_len is 32,768 tokens. A request that does not fit is reported as unsupported, and nothing is truncated.
- **Unsupported rows:** none.

## Training data overlap

- **Train splits used:** no test split of any benchmark. The head's fit and calibration data include train-split rows of CLINC150,
  GSM8K, Amazon ESCI, BANKING77 and WinoGrande; the training data include Lichess puzzles (a different dataset from ChessBench's).
- **Overlap found:** an audit of the training text, the head's data included, against the public suite found 1 test item that is an
  exact or near-exact duplicate of a CLINC150 train-split utterance. Following the board's contamination rule, it is counted as wrong
  in the 62.80 above: `5:CLINC150+OOS:CLINC150+OOS:test:1591`.

## Latency

The kit runner's CUDA-synchronized per-request wall time for `lenavg_bayes_engine:LenAvgBayesEngine` with these weights on one RTX PRO
6000, one request at a time, on 1,000 requests of the public suite (100 drawn from each tenth of the length L; 1,000 / 1,000 ok). These
are sample statistics, not a per-request bound (the slowest request took 26.1 s); the board's own check uses a private sample. A second
pass gave the same figures within 1%.

| median | mean | 80th percentile | 95th percentile |
|---|---|---|---|
| 227 ms | 609 ms | 566 ms | 2,094 ms |
