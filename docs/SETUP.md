# Setup and commands

## Environment

Use Python 3.12 on Linux with CUDA GPUs. The model environments are separate:
INSID3 uses PyTorch 2.7.1/CUDA 12.6, CG-ICS uses PyTorch 2.9.0/CUDA 12.8, and the
language-model server uses vLLM 0.11.2. Source revisions and model locations are
recorded in `configs/source_pins.json`.

```bash
export ICS_RUNTIME="/absolute/path/to/ics-runtime"
python3 -m pip install -r requirements.txt
python3 scripts/bootstrap_runtime.py --runtime "$ICS_RUNTIME" --execute
```

The runtime directory must be outside this source directory. Model downloads
require the relevant access permissions. Place the approved DINOv3 PyTorch
checkpoint at `$ICS_RUNTIME/weights/dinov3_L.pth` before preparation, or provide
its download URL through `ICS_DINOV3_URL`. The bootstrap script creates separate
`insid3`, `cgics`, and `mllm` environments and installs the pinned dependencies.

## Data and episodes

Use existing data or prepare COCO/LVIS with:

```bash
python3 scripts/download_data.py --datasets coco lvis \
  --data-root "$ICS_RUNTIME/data" --execute
python3 scripts/prepare_episodes.py --dataset coco \
  --data-root "$ICS_RUNTIME/data" \
  --output "$ICS_RUNTIME/manifests/coco-f0.json" \
  --fold 0 --queries 100 --references-per-query 3 --shots 1 \
  --query-seed 0 --seed 0 --trust-prepared-pickle
```

For LVIS, use `--dataset lvis` and `--empty-mask-policy exclude`. The pickle flag
is for the trusted author-provided split files. Reuse the same manifest across
methods. `examples/episodes.example.json` shows the input schema; query masks
are used for evaluation, not supplied to the models.

| Dataset | Layout under the data root |
|---|---|
| COCO | `COCO2014/{train2014,val2014,annotations,splits/val/foldN.pkl}` |
| LVIS | `LVIS/lvis_val.pkl`, `LVIS/coco/{train2017,val2017}` |

`configs/suite.*.json` records the experiment matrices. The later fresh-query
cohorts are prepared by `scripts/prepare_batch2.py`; recreating those exact
cohorts also requires the prior-run manifests used to exclude earlier queries.

## Inference

For INSID3, use the command in the README. For CG-ICS, start its local language
model in another terminal on a separate GPU:

```bash
CUDA_VISIBLE_DEVICES=1 "$ICS_RUNTIME/envs/mllm/bin/python" \
  -m vllm.entrypoints.openai.api_server \
  --model "$ICS_RUNTIME/weights/Qwen3-VL-4B-Instruct" \
  --served-model-name qwen3-vl-4b --host 127.0.0.1 --port 22002 \
  --tensor-parallel-size 1
```

Then run segmentation:

```bash
CUDA_VISIBLE_DEVICES=0 "$ICS_RUNTIME/envs/cgics/bin/python" scripts/run_experiment.py \
  --config configs/experiments/cgics_full.json \
  --manifest "$ICS_RUNTIME/manifests/coco-f0.json" \
  --dataset-root "$ICS_RUNTIME/data" \
  --output "$ICS_RUNTIME/results/cgics"
```

For AGS, use `configs/experiments/batch2_cgics_gate_t07.json` and a separate output
directory. The other experiment files specify component ablations, scoring
rules, random budgets, and threshold settings. Predictions are evaluated at the
original query resolution; episode, query, and class aggregations are distinct.

## Analysis

```bash
python3 scripts/analyze_results.py \
  --root "$ICS_RUNTIME/results" --output "$ICS_RUNTIME/analysis" --case-images 0
```

`scripts/analyze_batch2.py` analyzes recorded batch-2 run directories with their
`plan.json`. `analysis/b1/b1_statistics.py` and `analysis/b2/b2_statistics.py`
compute the scoring and search-budget comparisons from exported experiment
records; pass `--evidence` and `--output` explicitly. These two scripts expect the
original export directory structure. The B2 script maps the recorded Orion path
prefix into that export, so it does not require a live Orion connection.
