# CE6190: In-Context Segmentation

Individual Visual Understanding project by **HAN SHUANGYUE**. The project compares
INSID3 and CG-ICS on COCO and LVIS, studies their components and search budgets,
and defines Agreement-Gated Search (AGS) as an input-dependent search policy.

The report separates accepted experiments from proposed or pending comparisons.
Saved task `PASS` counts are not a substitute for output-level acceptance.

## Start here

| Purpose | Location |
|---|---|
| Locally maintained report; synchronized separately | [report/](report/README.md) |
| Setup, execution, and measurement boundaries | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| Current acceptance and review status | [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) |
| Model wrappers and metric implementation | `ics/adapters/`, `ics/runner.py`, `ics_eval.py` |
| Frozen experiment definitions | `configs/experiments/`, `configs/suite.*.json` |
| Data and adapter contracts | [data](docs/DATA_PROTOCOLS.md), [INSID3](docs/INSID3_ADAPTER.md), [CG-ICS](docs/CGICS_ADAPTER.md) |
| Accepted compact B1/B2 summaries | [results_summary/](results_summary/README.md) |
| Regression tests | `tests/` |
| Earlier operational notes and research records | [docs/history/](docs/history/README.md) |

## Scientific scope

- **Initial study:** complete systems, reference sensitivity, component and
  backbone ablations, parameter settings, inference cost, and success/failure cases.
- **B1:** paired scoring rules applied to the same generated candidate pool on
  fresh queries. B1 summaries passed output-level validation.
- **B2:** full search, a single concept, BON=1, and INSID3 on the same fresh cohort.
  B2 summaries passed output-level validation.
- **B4:** COCO AGS at threshold 0.7, initial-score gating, random search budgets,
  and staged always-search. Empirical claims require complete matched output
  validation; pending report cells must remain pending until then.

The complete historical configuration matrix remains available for provenance.
Its presence does not imply that every configuration was accepted for the report.

## Run and analyze

Python 3.12 is used for the model environments. The lightweight requirements cover
data preparation, evaluation, and plotting; model dependencies are prepared
separately using the pinned bootstrap script. See the reproducibility guide before
running a model or using a cluster controller.

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_experiment.py --help
python3 scripts/analyze_results.py --help
python3 scripts/analyze_batch2.py --help
python3 -m unittest discover -s tests -v
```

`scripts/run_experiment.py` executes an explicit configuration and manifest.
`scripts/analyze_results.py` and `scripts/analyze_batch2.py` read saved outputs;
they do not rerun models. Paths, source revisions, prediction records, failures,
and measurement metadata are retained by the runner.

The `scripts/` directory also retains deployment and recovery entry points at
their original paths so existing callers remain valid. Historical instructions
are archived; they are not commands to restart a currently active run. A checkout
update does not alter a run's separately frozen `source/` directory.

## Course submission

Submit two actual files in NTULearn: `HAN_SHUANGYUE_report_project1.pdf` and
`HAN_SHUANGYUE_code_project1.zip`. The code package contains project source,
configuration, setup instructions, tests, and compact accepted summaries. It
excludes datasets, model weights, environments, credentials, cluster run state,
bulk predictions, historical operational records, and the separately submitted
report. Repository or cloud links do not replace either submitted file.

This organization update changes documentation and packaging. It does not change
model code or experiment parameters, deploy a controller, commit, or push.

The existing `report/` directory is preserved in full; final report synchronization
is a separate update after the report has been finalized.
