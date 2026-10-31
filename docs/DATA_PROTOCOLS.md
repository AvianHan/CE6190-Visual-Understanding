# Dataset preparation and fixed episode protocol

Updated: 2026-11-01 (Singapore). **Implemented and synthetic-tested; real dataset bytes, archive availability, and model scores are not verified yet.**

## Purpose and input/output

This layer answers an evaluation question, not a model-design question: do all
methods receive precisely the same reference image, reference mask, and query
image? For example, a reference dog and its dog mask are fixed together with a
different query image. Both methods receive those three inputs; the query dog
mask is read only by the evaluator.

- Input: author-prepared images, segmentation annotations and class/fold lists.
- Output: a schema-v1 episode manifest, a `.protocol.json` provenance sidecar,
  and content-addressed 0/1 PNG masks under `DATA_ROOT/ics_generated_masks/`.
- Reuse: build each manifest once; pass the same file to every method and ablation.
- Limit: an image hash check detects identical image files, not semantic near
  duplicates, patient identity or unannotated common video provenance.

The old `ics_eval.py` schema and metric functions are unchanged. Images are
referenced inside the supplied data root and are not copied into the code ZIP.
The manifest's `concept_id` is for sampling and scoring only. The runner passes
`inference_view(episode)` to adapters, never the complete episode.

## Commands

Inspect the data download plan (no network requests or data writes):

```bash
python scripts/download_data.py --datasets coco lvis --data-root /path/to/ics-data
```

Download/extract independently downloadable archives concurrently:

```bash
python scripts/download_data.py --datasets coco lvis \
  --data-root /path/to/ics-data --workers 2 --execute
```

The source catalog contains URLs from pinned INSID3 data instructions. HTTP
downloads resume with validated range responses; Google Drive downloads use
`gdown==5.2.0`. Existing extracted assets with completed receipts are reused.
The tool rejects archive path traversal, archive symlinks, and overwriting
differing existing files. It retains archives and per-asset receipts rather
than silently deleting downloaded bytes. No authoritative checksums are
invented: when the source did not publish one, the receipt records an
**observed** SHA256 and `upstream_checksum_verified=false`.

If the data already exist, use them directly and skip downloading. The data
root must contain the relevant layout below; assets outside that root via
symlinks are rejected by the inherited manifest validator.

Prepare a small fixed COCO cohort (the number here is a smoke example, not an
experiment-size requirement):

```bash
python scripts/prepare_episodes.py --dataset coco \
  --data-root /path/to/ics-data --output /path/to/runtime/manifests/coco-f0-smoke.json \
  --fold 0 --queries 20 --references-per-query 1 --shots 1 \
  --query-seed 0 --seed 0 --trust-prepared-pickle
```

Prepare a robustness cohort from the same query draws by keeping `--queries`
and `--query-seed` fixed and increasing `--references-per-query`. Change only
`--seed` to select different reference sets while holding the query sample
sequence fixed. `--queries 0` uses every indexed image in every selected class.

COCO/LVIS author metadata use Python pickle. The explicit
`--trust-prepared-pickle` acknowledges use of metadata from the audited source;
never apply it to an unknown arbitrary pickle. This flag does not change mask
semantics or evaluation.

Existing outputs are refused by default, so an accidental rerun cannot
silently change an experiment's sample list. Prefer a new output and protocol
version when changing a study. `--overwrite` is available for an intentional
replacement before results are collected.

Existing generated mask files are decoded and compared with a freshly mapped
source mask before reuse. A stale, edited or corrupt cache raises an error and
is preserved for inspection; the builder never treats its filename as proof of
correctness or silently overwrites it. HTTP downloads with a declared response
length are checked before promotion from `.part` to a completed archive.

## Exact membership and mask semantics

| Dataset | Expected source layout | Membership, labels and important scope |
|---|---|---|
| COCO-20i | `COCO2014/{train2014,val2014,annotations,splits/val/foldN.pkl}`; `coco/` alias accepted | 4 folds; zero-based class ID modulo4. Uses author-prepared semantic masks, foreground=`class_id+1`. Other labels including255 become background, matching both anchor loaders. |
| LVIS-92i | `LVIS/lvis_val.pkl`, `LVIS/coco/{train2017,val2017}` | 10 folds determined by **pickle key insertion order**, not sorted category IDs. Eligible classes have >1 image in one-shot source. Union every annotation of the selected category in the image. Polygon/RLE decoding uses pycocotools. |
| FSS-1000 | `FSS-1000/data/CLASS/*.jpg,*.png`, `FSS-1000/splits/{trn,val,test}.txt` | Defaulttest, fold0. Split categories are explicit; no scan of all classes in place of the official split. Official image masks threshold>=128. Each class normally has10 images. |
| PASCAL-5i | `VOC2012/JPEGImages`, `VOC2012/SegmentationClassAug`; explicit split list | 4 contiguous groups of5 classes. Use `--split-file /path/to/CG-ICS/datasets/splits/pascal/val/foldN.txt`; each line is `image__1_based_class`. Query255is ignored. This is **not PASCAL-Part**. |
| ISIC2018 | `ISIC/ISIC2018_Task1-2_Training_Input/{1,2,3}`, `ISIC/ISIC2018_Task1_Training_GroundTruth` | Reuses author-reorganized public training images; directories1/2/3 are sampled as **three classes** by the actual loader, not treated as held-out folds. Usefold0. Masks describe the primary lesion. This is not officialchallenge held-outtest. |
| SUIM | `SUIM/images`, `SUIM/masks/{FV,HD,PF,RI,RO,SR,WR}` | Uses merged authorlayout, 7foreground categories, nonempty classmasks. Backgroundwaterbody is not a promptclass. This is not simply the original110-image testset. |
| iSAID-5i | `iSAID/val/images`, `iSAID/val/semantic_png`, `iSAID/splits/val/foldN.txt` | 3 contiguous groups of5 classes. Indexed single-channel semantic labels required despite `_instance_color_RGB.png` filename. Query255ignored. Reference/query must be from different original aerial images. |

FSS and PASCAL split lists can be taken explicitly from the pinned CG-ICS
repository using `--split-file`; the program does not implicitly search the
current working directory or download files during sampling. ISIC preparation
still requires the author DR-Adapter class mapping; the downloader retrieves
images/masks but deliberately does not execute a remote preprocessing script.
The catalog gives the source instructions. FSS/PASCAL downloads are manual
because their final archive URLs/layouts were not validated for this bundle.

## Unified sampler: differences from author reproduction

This package currently implements **unified-manifest evaluation**, not exact
replay of each paper's random episode generator. Source membership and mask
semantics were audited against the pinned official loaders:

- INSID3: `0c165a10cf52ab91f335883d06260de86854adbe`.
- CG-ICS: `8da845743f15c18ec490567fbd86dba58c8a388f`.

Our query draws cycle shuffled classes, with class counts differing by at most
one. Within a class, images are shuffled and used before another cycle. If a
class has fewer unique images than its requested draws, query images repeat;
the sidecar records unique query/concept count and repeat count explicitly.
Such draws are **not independent samples** for confidence intervals or
reference-sensitivity uncertainty. Group by actual query image hash and
concept, not by episode ID alone.

References are sampled without duplicate image contents within a support set.
For repeated reference trials, support sets are distinct. With one-shot, each
reference image occurs at most once for a given query draw. Shortfalls default
to `--insufficient-references all`: use all available distinct sets and record
the actual number, rather than silently repeating references. Choose `error`
to require exactly the requested number. In FSS, a fixed query normally has
at most9 other same-class references. Rare LVIS classes can have even fewer.

For n-shot extensions, LVIS class/fold membership remains the one-shot universe;
the builder fails if a selected query has fewer than `shots` eligible distinct
references. The original loaders instead filter eligible LVIS classes by
`len(class_images)>shot`, which can change the class universe and folds. This
intentional unified-protocol difference must be disclosed if n-shot is studied.

All masks are evaluated at original query resolution. Normalized query PNGs
contain0/1 and, only where the source protocol specifies it, ignore255.
**Reference PNGs contain only0/1**, with void pixels mapped to background;
otherwise an adapter using `mask>0` could incorrectly prompt ignored pixels.
An empty selected source mask or a mismatched mask/image size is an error.

## Source grouping

Use `--group-map groups.json` with keys relative to the data root:

```json
{"iSAID/val/images/P0001_0_800_0_800.png": "P0001"}
```

When the query has a group ID, its references cannot share that ID. Standard
iSAID `P####_...` names can provide the source-image ID; unknown name formats
require an explicit map. If patient/video IDs are available for future data,
include every image in the mapping. Without verified patient IDs, ISIC results
must not be claimed as patient-disjoint generalization.

Group-aware selection here concerns reference/query pairs. If validation/test
subsets are used for parameter selection, their membership must additionally
be split at the same source-group level; passing a group map alone does not
create a separate validation set.

## Validation completed in this package

`python -m unittest discover -s tests -p test_data.py -v` passed13 synthetic
tests: deterministic manifests, query/reference sampling independence, distinct
reference limits, duplicate image exclusion, group exclusion, queryGT input
isolation, ignore-mask handling, LVIS union and author fold order, n-shot
support-set uniqueness, class balancing/repeated-query disclosure, and safe
archive extraction, detection of edited generated-mask caches, and retention of
truncated HTTP transfers as partial files. These checks verify implementation properties. They do
not verify real model quality, actual download links, or full benchmark data.

Sources:
- https://github.com/visinf/INSID3/blob/0c165a10cf52ab91f335883d06260de86854adbe/docs/data.md
- https://github.com/visinf/INSID3/tree/0c165a10cf52ab91f335883d06260de86854adbe/datasets
- https://github.com/Kakarot1103/CG-ICS/tree/8da845743f15c18ec490567fbd86dba58c8a388f/datasets
- https://github.com/HKUSTCV/FSS-1000
- https://irvlab.cs.umn.edu/resources/suim-dataset
- https://challenge.isic-archive.com/landing/2018/45/
