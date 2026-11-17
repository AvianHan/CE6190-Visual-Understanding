# Foundation and Visual In-Context Learning Reference Audit

Audit date: 2026-11-17 (SGT). Checked 14 references and 18 active citation occurrences. Old, unused TeX sections were excluded.

All 14 works were verified through primary paper/proceedings sources. Existing citations support the surrounding clauses. Two entries should preferably cite the published version: DINOv2 (TMLR 2024) and SAM 2 (ICLR 2025). Their existing preprint entries are valid rather than fabricated.

## Retrieval record

Reopened primary paper/proceedings pages; search of official CVF/OpenReview source content used where direct retrieval blocked. NeurIPS official BibTeX retrieved independently through urllib after search tool rejected x-bibtex MIME. No reliance on old audit labels.

Some direct CVF/OpenReview requests were blocked; the official source text was available through search, and author arXiv full text was used for detailed method verification. This is recorded rather than presented as a successful direct fetch.

## `visualprompt` — Visual Prompting via Image Inpainting

- Metadata: 2022; NeurIPS 35; 25005–25017.
- Primary source: [Visual Prompting via Image Inpainting](https://papers.nips.cc/paper_files/paper/2022/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html).
- Finding: Abstract explicitly formulates prediction as completing a missing image region in an example-conditioned concatenated prompt. Title, author order, venue, year, pages and DOI match the official BibTeX endpoint.
- Action: No change.
- Active uses:
  - `paper/sections/related_work.tex:8`; citation group `visualprompt`.

## `goodexamples` — What Makes Good Examples for Visual In-Context Learning?

- Metadata: 2023; NeurIPS 36; 17773–17794.
- Primary source: [What Makes Good Examples for Visual In-Context Learning?](https://proceedings.neurips.cc/paper_files/paper/2023/hash/398ae57ed4fda79d0781c65c926d667b-Abstract-Conference.html).
- Finding: Abstract studies sensitivity to demonstration selection and introduces prompt retrieval. Current reference-variation motivation is supported; it does not imply the current project implemented that retrieval method. Official BibTeX verifies pages and DOI.
- Action: No factual change. Optional editorial replacement: “Prior work studies example selection” in place of the shortened paper name “Good Examples”.
- Active uses:
  - `paper/sections/related_work.tex:12`; citation group `goodexamples`.

## `vrpsam` — VRP-SAM: SAM with Visual Reference Prompt

- Metadata: 2024; CVPR; 23565–23574.
- Primary source: [VRP-SAM: SAM with Visual Reference Prompt](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_VRP-SAM_SAM_with_Visual_Reference_Prompt_CVPR_2024_paper.html).
- Finding: Official abstract introduces the Visual Reference Prompt encoder, trained with meta-learning. The report’s learned reference-prompt encoder description is accurate. Listed author prefix agrees with all nine authors in the proceedings.
- Action: No change.
- Active uses:
  - `paper/sections/related_work.tex:10`; citation group `vrpsam`.

## `unicl` — UNICL-SAM: Uncertainty-Driven In-Context Segmentation with Part Prototype Discovery

- Metadata: 2025; CVPR; 20201–20211.
- Primary source: [UNICL-SAM: Uncertainty-Driven In-Context Segmentation with Part Prototype Discovery](https://openaccess.thecvf.com/content/CVPR2025/html/Sheng_UNICL-SAM_Uncertainty-Driven_In-Context_Segmentation_with_Part_Prototype_Discovery_CVPR_2025_paper.html).
- Finding: Official abstract models unreliable examples caused by variable quality and inaccurate labeling, using uncertainty, feature refinement and part prototypes. Current phrase about uncertainty in imperfect examples is supported. Author prefix and page range match proceedings.
- Action: No change.
- Active uses:
  - `paper/sections/related_work.tex:12`; citation group `unicl`.

## `sine` — A Simple Image Segmentation Framework via In-Context Examples

- Metadata: 2024; NeurIPS 37; 25095–25119.
- Primary source: [A Simple Image Segmentation Framework via In-Context Examples](https://papers.nips.cc/paper_files/paper/2024/hash/2cc0b08447bf9668db268e6c86364a6e-Abstract-Conference.html).
- Finding: Abstract explicitly names the In-context Interaction module for reference/target correlations, followed by a Matching Transformer. Current concise description is supported. Complete authors, pages, venue, year and DOI match official BibTeX.
- Action: No change.
- Active uses:
  - `paper/sections/related_work.tex:10`; citation group `sine`.

## `dinov3` — DINOv3

- Metadata: 2025; arXiv:2508.10104.
- Primary source: [DINOv3](https://arxiv.org/abs/2508.10104).
- Finding: Abstract supports self-supervised scaling and dense representations; §6.1.3 tests geometric and semantic correspondence with frozen features. The co-citations beside INSID3 identify its backbone; INSID3 itself must support debiasing/clustering/aggregation, not DINOv3 alone. First six authors and arXiv identifier/year/DOI match.
- Action: No change.
- Active uses:
  - `paper/sections/introduction.tex:8`; citation group `insid3,dinov3`.
  - `paper/sections/methods.tex:10`; citation group `insid3,dinov3`.
  - `paper/sections/related_work.tex:3`; citation group `dinov2,dinov3`.

## `dinov2` — DINOv2: Learning Robust Visual Features without Supervision

- Metadata: Current entry: 2023 arXiv; published version: TMLR January 2024.
- Primary source: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
- Finding: Abstract establishes scale and generic features. §7.5 Patch matching demonstrates cross-image semantic correspondences. Existing 2023 arXiv entry is valid, but official TMLR paper is published January 2024 and gives author Huy V. Vo.
- Action: Prefer @article, journal={Transactions on Machine Learning Research}, year={2024}, url={https://openreview.net/forum?id=a68SUt6zFt}; use Vo, Huy V. Keep arXiv identifier as an optional access field.
- Active uses:
  - `paper/sections/related_work.tex:3`; citation group `dinov2,dinov3`.

## `sam3` — SAM 3: Segment Anything with Concepts

- Metadata: 2025; arXiv:2511.16719 (v2 revised March 2026).
- Primary source: [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719).
- Finding: Abstract defines concept prompts as noun phrases, exemplars or their combination and returns matching instance masks. This supports concept-level prompt capability. The CG-ICS co-citation must support its own concept-generation/search procedure. Listed author prefix and original arXiv year match.
- Action: No change. Do not interpret the 2026 revision date as the initial publication year.
- Active uses:
  - `paper/sections/introduction.tex:8`; citation group `cgics,sam3`.
  - `paper/sections/related_work.tex:5`; citation group `sam3`.

## `sam` — Segment Anything

- Metadata: 2023; ICCV; 4015–4026.
- Primary source: [Segment Anything](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html).
- Finding: Official metadata verifies author prefix, pages, venue and year. Paper §3 specifies point/box/mask prompts and a mask decoder, supporting the spatial-prompt statement. The report does not claim SAM is exclusively spatial: paper also explores text prompts.
- Action: No change.
- Active uses:
  - `paper/sections/related_work.tex:5`; citation group `sam`.

## `qwen3vl` — Qwen3-VL Technical Report

- Metadata: 2025; arXiv:2511.21631.
- Primary source: [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631).
- Finding: Abstract supports multimodal reasoning and includes dense 4B variants. Official Qwen model card confirms Qwen3-VL-4B-Instruct. Local configs/source_pins.json and cluster.orion.recovery.json identify this checkpoint, independently of the paper citation. Original first six authors/year/title/identifier/DOI match.
- Action: Keep citation attached to model identity. Working resolution, candidate count, loops and GPU allocation are implementation facts, not claims established by the Qwen paper.
- Active uses:
  - `paper/sections/methods.tex:42`; citation group `qwen3vl`.
  - `paper/sections/related_work.tex:5`; citation group `qwen3vl`.

## `dino` — Emerging Properties in Self-Supervised Vision Transformers

- Metadata: 2021; ICCV; 9650–9660.
- Primary source: [Emerging Properties in Self-Supervised Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/html/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.html).
- Finding: Abstract and Figure 1 describe object/segmentation structure emerging in self-supervised ViT features, without labels. All seven authors and publication fields match official proceedings. The report claims capability, not guaranteed correspondence on every image.
- Action: No change.
- Active uses:
  - `paper/sections/related_work.tex:3`; citation group `dino`.

## `vit` — An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale

- Metadata: 2021; ICLR.
- Primary source: [An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy).
- Finding: Official ICLR PDF explicitly applies Transformers to sequences of image patches, supporting patch-token description. First six authors, title and conference year match. Direct forum open was challenge-blocked; official PDF search supplied the accepted paper.
- Action: No change.
- Active uses:
  - `paper/sections/related_work.tex:3`; citation group `vit`.

## `sam2` — SAM 2: Segment Anything in Images and Videos

- Metadata: Current entry: 2024 arXiv; published version: ICLR 2025.
- Primary source: [SAM 2: Segment Anything in Images and Videos](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45c1f6a8cbf2da59ebf2c802b4f742cd-Abstract-Conference.html).
- Finding: Official ICLR proceedings confirms title, authors, 2025 conference publication and promptable segmentation across images/videos. Current 2024 arXiv reference is valid but the published conference entry is preferable.
- Action: Prefer @inproceedings, booktitle={International Conference on Learning Representations}, year={2025}, official proceedings URL. Keep arXiv identifier as an optional access field.
- Active uses:
  - `paper/sections/related_work.tex:5`; citation group `sam2`.

## `clip` — Learning Transferable Visual Models From Natural Language Supervision

- Metadata: 2021; ICML; PMLR 139:8748–8763.
- Primary source: [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html).
- Finding: Official abstract and BibTeX verify image–text matching pretraining, author prefix, title, proceedings, year, volume and pages. This directly supports image–text alignment, without claiming a segmentation architecture.
- Action: No change.
- Active uses:
  - `paper/sections/related_work.tex:5`; citation group `clip`.

## Proposed bibliography replacements

Exact replacement entries are in `bib_foundations_replacements.bib`. No main bibliography or report source has been edited by this audit.

## Citation scope

A foundation-model reference establishes the capabilities and identity of that model. It is not used as evidence that this project ran a particular configuration or achieved its measured results. INSID3/CG-ICS co-citations establish their processing pipelines; this project’s run bindings/configurations establish execution details.
