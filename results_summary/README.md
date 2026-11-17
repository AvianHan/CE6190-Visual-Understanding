# Accepted result summaries

These are compact exports of the accepted B1 and B2 analyses, not new experiments.

| Folder | Comparison | Accepted coverage |
|---|---|---|
| `b1/` | Combined, reference-only, and query-only scoring of a fixed candidate pool | 56 shards; 14 outputs; 4,200 episodes; 12,600 paired predictions |
| `b2/` | INSID3, CG-ICS single, BON1, and full search | 224 shards; 56 outputs; 16,800 predictions |

Both studies use the same 400 COCO and 1,000 LVIS query images with three references
per query. They are not independent fresh-query samples. Query-equal summaries and
paired intervals must be distinguished from class-pooled IoU. Cost comparisons use
matched inputs and recorded GPU models.

`analysis_acceptance.json` records the completed analysis acceptance. CSVs preserve
the exported numerical values. Raw dataset imagery, GT masks, prediction PNGs,
per-episode bulk records, and model files are excluded. Reproducing the pixel-level
audit therefore also requires the separately retained original evidence.

B4 is not included in this directory until its complete matched outputs pass
acceptance. A partial task count is not a performance result.
