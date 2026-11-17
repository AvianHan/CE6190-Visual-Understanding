# In-Context Segmentation

**HAN SHUANGYUE · CE6190 Visual Understanding · Individual Project**

I compare INSID3 and CG-ICS on COCO-20i and LVIS-92i using the same reference–query
pairs. The experiments study reference sensitivity, model components, search
budgets, accuracy, and inference cost. I also evaluate Agreement-Gated Search
(AGS), which uses agreement between text-guided and visual-guided predictions to
decide whether further concept search is needed.

## Code

| Location | Contents |
|---|---|
| `ics/adapters/` | INSID3 and CG-ICS implementations |
| `ics/cgics_batch2.py` | AGS and scoring/search controls |
| `ics/runner.py`, `ics_eval.py` | Experiment execution and segmentation metrics |
| `configs/experiments/` | Baselines, ablations, and parameter settings |
| `scripts/`, `analysis/` | Setup, data preparation, and result analysis |

## Run

Use Python 3.12 and a CUDA GPU. [Setup and commands](docs/SETUP.md) cover the model
environments, datasets, and CG-ICS language-model server.

```bash
export ICS_RUNTIME="/path/to/ics-runtime"
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/run_experiment.py \
  --config configs/experiments/insid3_full.json \
  --manifest /path/to/episodes.json \
  --dataset-root "$ICS_RUNTIME/data" \
  --output "$ICS_RUNTIME/results/insid3"
```

Use the corresponding CG-ICS environment and configuration to run CG-ICS or AGS.
Dataset files and model weights are stored outside this source directory.

The report and figures are in `report/`.
