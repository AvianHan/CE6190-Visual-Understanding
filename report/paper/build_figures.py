#!/usr/bin/env python3
"""Regenerate report figures from exact audited inputs; no model execution."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, default=HERE.parent / 'evidence',
                        help='Local figure evidence root; the required six-case subset is included')
    parser.add_argument('--methods-only', action='store_true',
                        help='Regenerate only the two method schematics, without result evidence')
    parser.add_argument('--results-only', action='store_true',
                        help='Regenerate only audited result figures; leave method schematics unchanged')
    args = parser.parse_args()
    if args.methods_only and args.results_only:
        parser.error('--methods-only and --results-only are mutually exclusive')
    ROOT = args.evidence_root.resolve()
    if not args.methods_only:
        required = [ROOT / 'audit_20261116/full_analysis/components_analysis.json']
        cases_root = ROOT / 'audit_20261116/report_inputs_received/qualitative/cases'
        for case in ('case03', 'case04', 'case06', 'case07', 'case01', 'case02'):
            required += [cases_root / case / name for name in (
                'manifest_episode.json', 'assets/query_image.jpg', 'assets/query_mask.png',
                'assets/reference00_image.jpg', 'assets/reference00_mask.png',
                'insid3_full/episode.json', 'insid3_full/prediction.png',
                'cgics_full/episode.json', 'cgics_full/prediction.png')]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            parser.error('Audited figure inputs are missing. Supplied figure PDFs already support '
                         'report compilation. Missing: ' + ', '.join(missing[:6]))
    if not args.results_only:
        subprocess.run([sys.executable, str(HERE / 'build_method_figures.py')], check=True)
    if args.methods_only:
        return
    import numpy as np
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    OUT = HERE / 'figures'
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .55, 'axes.edgecolor': '#747D81',
        'xtick.major.width': .55, 'ytick.major.width': .55,
        'xtick.major.size': 2.4, 'ytick.major.size': 0,
        'text.color': '#20292F', 'axes.labelcolor': '#20292F',
        'xtick.color': '#444D53', 'ytick.color': '#20292F',
    })
    source = ROOT/'audit_20261116/full_analysis/components_analysis.json'
    d = json.loads(source.read_text())
    questions = [
        ('Debias gain with self/cross aggregation', 'INSID3: debiasing'),
        ('Self/cross versus seed-only with debias', 'INSID3: region aggregation'),
        ('Visual boxes at single concept', 'CG-ICS: visual prompting'),
        ('Search beyond single concept, with visual boxes', 'CG-ICS: full search'),
    ]
    fig, ax = plt.subplots(figsize=(6.875, 2.38))
    ax.axhspan(-.48, 1.48, color='#F2F6F8', zorder=0)
    ax.axhspan(1.52, 3.48, color='#FCF7F5', zorder=0)
    for offset, ds, color, marker in [
        (-.13, 'coco', '#466F87', 'o'), (.13, 'lvis', '#E48578', 's')
    ]:
        rows = [next(r for r in d['paired_contrasts']
                     if r['question'] == q and r['dataset'] == ds) for q, _ in questions]
        vals = np.array([r['mean_query_target_delta'] for r in rows])*100
        ci = np.array([r['query_image_cluster_ci95'] for r in rows])*100
        ax.errorbar(vals, np.arange(4)+offset,
                    xerr=np.array([vals-ci[:, 0], ci[:, 1]-vals]),
                    fmt=marker, color=color, ecolor=color, capsize=2.4,
                    ms=4.4, lw=1.15, markeredgecolor='white', markeredgewidth=.5,
                    label=r'COCO-$20^i$' if ds == 'coco' else r'LVIS-$92^i$', zorder=3)
    ax.set_yticks(np.arange(4), [x[1] for x in questions])
    ax.set_ylim(3.52, -.52)
    ax.axvline(0, color='#7B858B', lw=.7, ls=(0, (3, 3)), zorder=2)
    ax.set_xlabel('Paired equal-query IoU gain (percentage points)', labelpad=4)
    ax.set_xlim(-2, 12)
    ax.set_xticks(np.arange(-2, 13, 2))
    ax.grid(axis='x', color='#DDE3E6', linewidth=.45, zorder=1)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', pad=8)
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.01),
              fontsize=8.5, handlelength=1.3, columnspacing=1.8)
    fig.subplots_adjust(left=.253, right=.985, bottom=.18, top=.855)
    fig.savefig(OUT/'component_effects.pdf')
    fig.savefig(OUT/'component_effects.svg')
    fig.savefig(OUT/'component_effects.png', dpi=240)
    plt.close(fig)

    SRC=ROOT/'audit_20261116/report_inputs_received/qualitative/cases'
    def rgb(hex_color):
        return np.array([int(hex_color[i:i+2], 16)/255 for i in (1, 3, 5)])
    GREEN = rgb('#98AF1E')
    TP = rgb('#78B6C8')
    FP = rgb('#E48578')
    FN = rgb('#F2D98E')
    def overlay(im, gt, pred=None):
        a = im.astype(float)/255
        gt = gt > 0
        regions = ([(gt, GREEN)] if pred is None else
                   [(gt & pred, TP), (~gt & pred, FP), (gt & ~pred, FN)])
        for region, color in regions:
            a[region] = .42*a[region]+.58*color
        return a

    cases = ['case03', 'case04', 'case06', 'case07', 'case01', 'case02']
    labels = ['(a) COCO: wrong food', '(b) COCO: display',
              '(c) COCO: joint success', '(d) COCO: joint failure',
              '(e) LVIS: oversegmentation', '(f) LVIS: book']
    fig = plt.figure(figsize=(6.875, 6.72))
    records = []
    left, gap, width, image_height = .014, .014, .2325, .119
    row_step, top = .151, .925
    xs = [left+j*(width+gap) for j in range(4)]
    for j, title in enumerate(['Reference + GT', 'Query + GT', 'INSID3', 'CG-ICS']):
        fig.text(xs[j]+width/2, .982, title, ha='center', va='top',
                 fontsize=9.0, fontweight='bold', color='#466F87')
    for i, (cid, label) in enumerate(zip(cases, labels)):
        p = SRC/cid
        e = json.loads((p/'manifest_episode.json').read_text())['episode']
        q = np.array(Image.open(p/'assets/query_image.jpg').convert('RGB'))
        gt = np.array(Image.open(p/'assets/query_mask.png'))
        r = np.array(Image.open(p/'assets/reference00_image.jpg').convert('RGB'))
        rm = np.array(Image.open(p/'assets/reference00_mask.png'))
        y = top-i*row_step-image_height
        fig.text(left, y+image_height+.010, label, ha='left', va='bottom',
                 fontsize=8.2, fontweight='bold', color='#334F61')
        images = [overlay(r, rm), overlay(q, gt)]
        for j, method in enumerate(['insid3_full', 'cgics_full'], 2):
            row = json.loads((p/method/'episode.json').read_text())
            pred = np.array(Image.open(p/method/'prediction.png')) > 0
            valid = np.ones(gt.shape, bool) if e['ignore_label'] is None else gt != e['ignore_label']
            counts = {'tp': int(np.sum(pred & (gt == 1) & valid)),
                      'fp': int(np.sum(pred & (gt == 0) & valid)),
                      'fn': int(np.sum(~pred & (gt == 1) & valid)),
                      'tn': int(np.sum(~pred & (gt == 0) & valid)),
                      'ignored': int(np.sum(~valid))}
            assert counts == row['counts']
            u = counts['tp']+counts['fp']+counts['fn']
            assert abs(counts['tp']/u-row['iou']) < 1e-12
            images.append(overlay(q, gt, pred))
            fig.text(xs[j]+width/2, y+image_height+.010,
                     f'IoU {100*row["iou"]:.2f}%', ha='center', va='bottom',
                     fontsize=8.0, color='#444D53')
            records.append({'case': cid, 'method': method, 'iou': row['iou'],
                            'episode_id': e['episode_id'],
                            'prediction_sha256': hashlib.sha256((p/method/'prediction.png').read_bytes()).hexdigest()})
        for x, im in zip(xs, images):
            ax = fig.add_axes([x, y, width, image_height])
            # Equal comparison cells; aspect-ratio-preserving letterboxing keeps
            # every exported pixel, with no crop, distortion, or mask alteration.
            h, w = im.shape[:2]
            physical_ratio = (width*6.875)/(image_height*6.72)
            if w/h >= physical_ratio:
                xspan, yspan = w, w/physical_ratio
            else:
                yspan, xspan = h, h*physical_ratio
            ax.set_facecolor('#F1F3F4')
            ax.imshow(im, extent=(0, w, h, 0), interpolation='bilinear')
            ax.set_xlim((w-xspan)/2, (w+xspan)/2)
            ax.set_ylim((h+yspan)/2, (h-yspan)/2)
            ax.set_aspect('auto')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
    fig.legend(handles=[Patch(facecolor=GREEN, label='Ground truth'),
                        Patch(facecolor=TP, label='True positive'),
                        Patch(facecolor=FP, label='False positive'),
                        Patch(facecolor=FN, label='False negative')],
               loc='lower center', ncol=4, frameon=False, fontsize=8,
               handlelength=1.15, handleheight=.65, columnspacing=1.8,
               bbox_to_anchor=(.5, .008))
    fig.savefig(OUT/'qualitative.pdf', dpi=240)
    fig.savefig(OUT/'qualitative.png', dpi=220)
    plt.close(fig)
    (HERE/'figure_provenance.json').write_text(json.dumps({'method_figures':['insid3_pipeline.pdf','cgics_pipeline.pdf'],'method_audit':'paper/figures/method_figure_audit.json','component_source':str(source.relative_to(ROOT)),'component_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'qualitative_scope':'Six purposively selected cases; original aspect ratios retained in equal letterboxed cells; every displayed mask rechecked against original GT','verified_predictions':records},indent=2)+'\n')
    print('Audited result figures generated; 12 original prediction masks verified.')


if __name__ == '__main__':
    main()
