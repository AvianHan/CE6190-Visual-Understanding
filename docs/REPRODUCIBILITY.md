# Reproducing the experiments

## Source and data boundaries

The project supplies the adapters, episode construction, runner, metric
aggregation, experiment configurations, and analysis code. Model-specific
upstream source revisions are pinned in `configs/source_pins.json`; the bootstrap
script checks existing checkouts before reuse. Upstream code is a declared
dependency, not an alternative location for this submitted project code.

Data, model weights, environments, and run outputs belong outside the repository.
Use a writable runtime directory and set `ICS_RUNTIME` to that directory. Official
dataset/model access and any required model authorization remain prerequisites;
neither the code ZIP nor its tests contain those resources.

```bash
export ICS_RUNTIME="/absolute/path/to/ce6190-ics-runtime"
python3 scripts/bootstrap_runtime.py --runtime "$ICS_RUNTIME"
```

This command prints a preparation plan. Review it before adding `--execute`.
Separate `insid3`, `cgics`, and `mllm` environments prevent incompatible PyTorch
and vLLM dependencies from replacing one another. The exact environment versions
are defined in the bootstrap script; `requirements.txt` alone is not the complete
model environment.

COCO/LVIS input preparation and manifests follow [DATA_PROTOCOLS.md](DATA_PROTOCOLS.md).
The example manifest in `examples/episodes.example.json` documents the schema; it
does not contain a real dataset. B1/B2 reuse the same fresh-query cohort. Recreating
the original exclusion set requires the prior-run provenance records; no claim is
made that a fresh machine without those records reproduces the same sampled cohort.

## Execute one explicit experiment

Use an approved GPU allocation and its corresponding model environment. A
representative INSID3 invocation is:

```bash
"$ICS_RUNTIME/envs/insid3/bin/python" scripts/run_experiment.py \
  --config configs/experiments/insid3_full.json \
  --manifest /absolute/path/to/manifest.json \
  --dataset-root "$ICS_RUNTIME/data" \
  --output "$ICS_RUNTIME/results/insid3-example"
```

For CG-ICS, the matching local language-model service must already be running and
its endpoint must match the configuration. The recorded deployment assigns SAM3
and the owned language-model service separate GPUs. This is a measurement/deployment
choice, not evidence that one GPU is impossible or equally fast. Direct runner
execution does not automatically allocate GPUs or start the language-model server.

The model receives query RGB and reference RGB/mask only. Query GT is read by the
evaluator after prediction. Evaluation uses original query resolution and reports
episode, query-equal, and class-pooled aggregations explicitly. Those measures and
the unified protocol must not be conflated with the original papers' protocols.

## Analyze saved outputs

```bash
python3 scripts/analyze_results.py \
  --root "$ICS_RUNTIME/results" --output "$ICS_RUNTIME/analysis" --case-images 0
python3 scripts/analyze_batch2.py --help
```

The runner records errors and missing outputs. Successful queue completion is only
the first gate: accepted results require source/config/manifest identity checks,
complete canonical coverage, and prediction/GT metric checks. B1/B2 compact CSVs
and acceptance receipts are under `results_summary/`; their raw images, masks,
episode records, and bulk predictions are intentionally excluded from submission.
The summaries alone cannot rerun the pixel-level audits.

Compare latency only on matched inputs and recorded GPU models. Startup and owned
process memory are separate from inference time. A runner-time ratio between
two-GPU CG-ICS and one-GPU INSID3 is not a total GPU-efficiency ratio.

## Frozen execution versus repository source

This package is based on repository commit
`86b436ad4677b2a3bbf09a926715ac1f08fe452c`. The original B2 frozen source has 104
overlapping source/configuration files: 100 match byte-for-byte. Four differ:
`ics/batch2.py`, `ics/batch2_data.py`, `ics/rolling_queue.py`, and
`scripts/batch2_ics.py`. Their differences implement SUIM geometry-repair hooks,
scheduling delegation, and controller tick injection. The two model adapters,
`ics/cgics_batch2.py`, `ics/runner.py`, evaluator, and experiment configurations
match the frozen snapshot. The whole repository is therefore not described as
identical to that execution snapshot.

No scientific source or experiment parameter is changed by this cleanup. Cluster
operations are a separate concern: the B3 pause/recovery operation and any follow-up
fix are not certified by this packaging task. Never infer controller readiness or
GPU allocation from the presence of scripts in a ZIP.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

CPU tests use synthetic inputs and mocked scheduler/services where documented.
Optional tensor/upstream tests require their extra dependencies. They do not
replace actual model inference or Orion deployment evidence.
