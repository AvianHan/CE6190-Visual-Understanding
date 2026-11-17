# ICS bibliography and citation-use audit — 17 November 2026

12 references and 18 active citation occurrences checked. All twelve bibliographic identities are supported by primary sources. No invented or unrelated work was found. One implementation-scope qualifier is recommended; two missing optional metadata fields can be completed.

## Entry-by-entry verification

| Key | Finding | Primary sources |
|---|---|---|
| `insid3` | Title, six authors, CVPR 2026 verified. Frozen DINOv3, original-feature grouping, debiased matching and orthogonal projection supported. The report describes the executed implementation: its candidate-coverage factor and thresholded seed differ from paper Eq.13–14. Preserve existing Gaussian-noise/zero-image and CRF distinction. Optional add pages 21638–21648. | [metadata](https://arxiv.org/abs/2603.28480); [method Sections 3.1–3.4 and appendix B](https://arxiv.org/html/2603.28480v1); [official accepted version; first/last page 21638/21648](https://openaccess.thecvf.com/content/CVPR2026/papers/Cuttano_INSID3_Training-Free_In-Context_Segmentation_with_DINOv3_CVPR_2026_paper.pdf) |
| `cgics` | Title, three authors, ECCV 2026, LNCS17074, pp336–352 and DOI verified. Reference robustness claim is explicitly the paper objective. Local author preprint v2 Sections3.2–3.4 confirms reference-IoU × query-presence, tree-search and stitched-image box extraction. General exponents alpha/beta describe project parameterization; default1/1 matches paper. No new measured performance asserted from this citation. | [publisher metadata and abstract](https://link.springer.com/chapter/10.1007/978-3-032-37029-7_20); [author preprint identity; direct current fetch blocked](https://arxiv.org/abs/2606.28149) |
| `persam` | ICLR2024 final paper verified. Existing six listed authors plus others is consistent. Full author list should retain Yu Qiao and Peng Gao from final PDF, rather than a different arXiv author list. Current prose is specifically PerSAM (training-free), not PerSAM-F (two tuned scale parameters). | [metadata and training-free PerSAM vs two-parameter PerSAM-F](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4f44e5c82dc7b0ba4472083e5dedb08e-Abstract-Conference.html); [final paper author order](https://proceedings.iclr.cc/paper_files/paper/2024/file/4f44e5c82dc7b0ba4472083e5dedb08e-Paper-Conference.pdf) |
| `painter` | CVPR2023 pp6830–6839 and five authors verified. Learning multiple dense prediction tasks from image-pair context is supported. | [official metadata and abstract via indexed primary page; direct open 403](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Images_Speak_in_Images_A_Generalist_Painter_for_In-Context_Visual_CVPR_2023_paper.html) |
| `seggpt` | ICCV2023 pp1130–1140 and six authors verified. Context-conditioned general segmentation interface is supported; current prose does not incorrectly call its training phase training-free. | [official metadata and abstract via indexed primary page; direct open 403](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.html) |
| `segic` | ECCV2024 pp203–220, six authors and DOI verified. Current description is semantically supported. Exact paper terminology is geometric, visual, and meta instructions; optionally adopt these terms. Publisher LNCS volume15096 can be added. | [official abstract and authors](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5519_ECCV_2024_paper.php); [publisher metadata and pages](https://link.springer.com/chapter/10.1007/978-3-031-72920-1_12) |
| `matcher` | ICLR2024 six-author identity verified. Training-free feature matching plus promptable segmentation supported. Dataset citation supports COCO four 20-class held-out folds and LVIS ten92-class folds. Project episode counts remain local evidence, not attributed to Matcher. | [metadata and training-free matching](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4df9a5e6bad9e64ebcea453e031142bb-Abstract-Conference.html); [dataset four COCO folds / ten LVIS folds; direct open too large](https://proceedings.iclr.cc/paper_files/paper/2024/file/4df9a5e6bad9e64ebcea453e031142bb-Paper-Conference.pdf); [author-prepared data provenance](https://github.com/aim-uofa/Matcher/blob/main/datasets/README.md) |
| `abcdfss` | Single author Jonas Herzog, CVPR2024 pp23605–23615 verified. Test-time feature-processing adaptation is accurate: small networks attached to a pretrained feature pyramid are learned using augmented-view consistency. | [official metadata and test-time auxiliary-layer adaptation; direct open 403](https://openaccess.thecvf.com/content/CVPR2024/html/Herzog_Adapt_Before_Comparison_A_New_Perspective_on_Cross-Domain_Few-Shot_Segmentation_CVPR_2024_paper.html) |
| `oslsm` | BMVC2017 five authors pp167.1–167.13 DOI verified. One-shot semantic prediction supports the historical clause; report does not incorrectly describe its base training as training-free. | [official title authors venue year pages DOI and abstract](https://www.bmva-archive.org.uk/bmvc/2017/papers/paper167/index.html) |
| `panet` | ICCV2019 five authors pp9197–9206 verified. Support-derived class prototypes and query matching support the historical clause. | [official metadata and prototype matching](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_PANet_Few-Shot_Image_Semantic_Segmentation_With_Prototype_Alignment_ICCV_2019_paper.html) |
| `hsnet` | ICCV2021 three authors pp6941–6952 verified. Multi-level support/query 4D correlation supports dense-correspondence clause. | [official metadata and dense correlations](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Hypercorrelation_Squeeze_for_Few-Shot_Segmentation_ICCV_2021_paper.html) |
| `repri` | CVPR2021 six authors pp13979–13988 verified; spelling/order matches official BibTeX. Transductive inference optimizes support and unlabeled-query terms; report contrasts this optimization with frozen inference correctly. | [official metadata and transductive inference](https://openaccess.thecvf.com/content/CVPR2021/html/Boudiaf_Few-Shot_Segmentation_Without_Meta-Learning_A_Good_Transductive_Inference_Is_All_CVPR_2021_paper.html) |

## Every citation use

### U1 — `insid3` at paper/sections/introduction.tex:8

**Claim context in the report:** Recent foundation models support two distinct solutions. INSID3 constructs a mask from frozen DINOv3 features through debiasing, clustering, and similarity aggregation~\cite{insid3,dinov3}. CG-ICS generates language concepts from the reference, evaluates and searches these concepts, and combines them with visual prompts in SAM~3~\cite{cgics,sam3}. Their central difference is how the reference becomes actionable evidence: direct visual correspondence in one system, and concept-guided segmentation supported by visual transfer in the other.

**Audit:** Title, six authors, CVPR 2026 verified. Frozen DINOv3, original-feature grouping, debiased matching and orthogonal projection supported. The report describes the executed implementation: its candidate-coverage factor and thresholded seed differ from paper Eq.13–14. Preserve existing Gaussian-noise/zero-image and CRF distinction. Optional add pages 21638–21648.

### U2 — `cgics` at paper/sections/introduction.tex:8

**Claim context in the report:** Recent foundation models support two distinct solutions. INSID3 constructs a mask from frozen DINOv3 features through debiasing, clustering, and similarity aggregation~\cite{insid3,dinov3}. CG-ICS generates language concepts from the reference, evaluates and searches these concepts, and combines them with visual prompts in SAM~3~\cite{cgics,sam3}. Their central difference is how the reference becomes actionable evidence: direct visual correspondence in one system, and concept-guided segmentation supported by visual transfer in the other.

**Audit:** Title, three authors, ECCV 2026, LNCS17074, pp336–352 and DOI verified. Reference robustness claim is explicitly the paper objective. Local author preprint v2 Sections3.2–3.4 confirms reference-IoU × query-presence, tree-search and stitched-image box extraction. General exponents alpha/beta describe project parameterization; default1/1 matches paper. No new measured performance asserted from this citation.

### U3 — `oslsm` at paper/sections/related_work.tex:8

**Claim context in the report:** Reference-conditioned segmentation builds on one-shot prediction~\cite{oslsm}, support prototypes~\cite{panet}, and dense support--query correspondence~\cite{hsnet}. Visual in-context learning broadens this idea by expressing tasks through image examples: visual prompting casts prediction as image completion~\cite{visualprompt}, Painter learns across dense prediction tasks~\cite{painter}, and SegGPT provides a general context-conditioned segmentation interface~\cite{seggpt}. These approaches establish the annotated example as a flexible specification of the desired output.

**Audit:** BMVC2017 five authors pp167.1–167.13 DOI verified. One-shot semantic prediction supports the historical clause; report does not incorrectly describe its base training as training-free.

### U4 — `panet` at paper/sections/related_work.tex:8

**Claim context in the report:** Reference-conditioned segmentation builds on one-shot prediction~\cite{oslsm}, support prototypes~\cite{panet}, and dense support--query correspondence~\cite{hsnet}. Visual in-context learning broadens this idea by expressing tasks through image examples: visual prompting casts prediction as image completion~\cite{visualprompt}, Painter learns across dense prediction tasks~\cite{painter}, and SegGPT provides a general context-conditioned segmentation interface~\cite{seggpt}. These approaches establish the annotated example as a flexible specification of the desired output.

**Audit:** ICCV2019 five authors pp9197–9206 verified. Support-derived class prototypes and query matching support the historical clause.

### U5 — `hsnet` at paper/sections/related_work.tex:8

**Claim context in the report:** Reference-conditioned segmentation builds on one-shot prediction~\cite{oslsm}, support prototypes~\cite{panet}, and dense support--query correspondence~\cite{hsnet}. Visual in-context learning broadens this idea by expressing tasks through image examples: visual prompting casts prediction as image completion~\cite{visualprompt}, Painter learns across dense prediction tasks~\cite{painter}, and SegGPT provides a general context-conditioned segmentation interface~\cite{seggpt}. These approaches establish the annotated example as a flexible specification of the desired output.

**Audit:** ICCV2021 three authors pp6941–6952 verified. Multi-level support/query 4D correlation supports dense-correspondence clause.

### U6 — `painter` at paper/sections/related_work.tex:8

**Claim context in the report:** Reference-conditioned segmentation builds on one-shot prediction~\cite{oslsm}, support prototypes~\cite{panet}, and dense support--query correspondence~\cite{hsnet}. Visual in-context learning broadens this idea by expressing tasks through image examples: visual prompting casts prediction as image completion~\cite{visualprompt}, Painter learns across dense prediction tasks~\cite{painter}, and SegGPT provides a general context-conditioned segmentation interface~\cite{seggpt}. These approaches establish the annotated example as a flexible specification of the desired output.

**Audit:** CVPR2023 pp6830–6839 and five authors verified. Learning multiple dense prediction tasks from image-pair context is supported.

### U7 — `seggpt` at paper/sections/related_work.tex:8

**Claim context in the report:** Reference-conditioned segmentation builds on one-shot prediction~\cite{oslsm}, support prototypes~\cite{panet}, and dense support--query correspondence~\cite{hsnet}. Visual in-context learning broadens this idea by expressing tasks through image examples: visual prompting casts prediction as image completion~\cite{visualprompt}, Painter learns across dense prediction tasks~\cite{painter}, and SegGPT provides a general context-conditioned segmentation interface~\cite{seggpt}. These approaches establish the annotated example as a flexible specification of the desired output.

**Audit:** ICCV2023 pp1130–1140 and six authors verified. Context-conditioned general segmentation interface is supported; current prose does not incorrectly call its training phase training-free.

### U8 — `segic` at paper/sections/related_work.tex:10

**Claim context in the report:** Recent methods differ in how they turn that example into a query mask. SegIC learns to use foundation-model correspondences as geometric and semantic guidance~\cite{segic}; SINE introduces in-context interaction~\cite{sine}; and VRP-SAM learns a reference-prompt encoder~\cite{vrpsam}. Frozen-model pipelines instead reuse pretrained representations and segmentation priors: PerSAM derives target-specific information from one example~\cite{persam}, and Matcher links general feature matching to promptable segmentation~\cite{matcher}. INSID3 operates directly on DINOv3 features~\cite{insid3}, whereas CG-ICS combines generated concepts, search, and visual exemplars~\cite{cgics}. We compare these last two complete inference systems and then examine their individual processing stages.

**Audit:** ECCV2024 pp203–220, six authors and DOI verified. Current description is semantically supported. Exact paper terminology is geometric, visual, and meta instructions; optionally adopt these terms. Publisher LNCS volume15096 can be added.

### U9 — `persam` at paper/sections/related_work.tex:10

**Claim context in the report:** Recent methods differ in how they turn that example into a query mask. SegIC learns to use foundation-model correspondences as geometric and semantic guidance~\cite{segic}; SINE introduces in-context interaction~\cite{sine}; and VRP-SAM learns a reference-prompt encoder~\cite{vrpsam}. Frozen-model pipelines instead reuse pretrained representations and segmentation priors: PerSAM derives target-specific information from one example~\cite{persam}, and Matcher links general feature matching to promptable segmentation~\cite{matcher}. INSID3 operates directly on DINOv3 features~\cite{insid3}, whereas CG-ICS combines generated concepts, search, and visual exemplars~\cite{cgics}. We compare these last two complete inference systems and then examine their individual processing stages.

**Audit:** ICLR2024 final paper verified. Existing six listed authors plus others is consistent. Full author list should retain Yu Qiao and Peng Gao from final PDF, rather than a different arXiv author list. Current prose is specifically PerSAM (training-free), not PerSAM-F (two tuned scale parameters).

### U10 — `matcher` at paper/sections/related_work.tex:10

**Claim context in the report:** Recent methods differ in how they turn that example into a query mask. SegIC learns to use foundation-model correspondences as geometric and semantic guidance~\cite{segic}; SINE introduces in-context interaction~\cite{sine}; and VRP-SAM learns a reference-prompt encoder~\cite{vrpsam}. Frozen-model pipelines instead reuse pretrained representations and segmentation priors: PerSAM derives target-specific information from one example~\cite{persam}, and Matcher links general feature matching to promptable segmentation~\cite{matcher}. INSID3 operates directly on DINOv3 features~\cite{insid3}, whereas CG-ICS combines generated concepts, search, and visual exemplars~\cite{cgics}. We compare these last two complete inference systems and then examine their individual processing stages.

**Audit:** ICLR2024 six-author identity verified. Training-free feature matching plus promptable segmentation supported. Dataset citation supports COCO four 20-class held-out folds and LVIS ten92-class folds. Project episode counts remain local evidence, not attributed to Matcher.

### U11 — `insid3` at paper/sections/related_work.tex:10

**Claim context in the report:** Recent methods differ in how they turn that example into a query mask. SegIC learns to use foundation-model correspondences as geometric and semantic guidance~\cite{segic}; SINE introduces in-context interaction~\cite{sine}; and VRP-SAM learns a reference-prompt encoder~\cite{vrpsam}. Frozen-model pipelines instead reuse pretrained representations and segmentation priors: PerSAM derives target-specific information from one example~\cite{persam}, and Matcher links general feature matching to promptable segmentation~\cite{matcher}. INSID3 operates directly on DINOv3 features~\cite{insid3}, whereas CG-ICS combines generated concepts, search, and visual exemplars~\cite{cgics}. We compare these last two complete inference systems and then examine their individual processing stages.

**Audit:** Title, six authors, CVPR 2026 verified. Frozen DINOv3, original-feature grouping, debiased matching and orthogonal projection supported. The report describes the executed implementation: its candidate-coverage factor and thresholded seed differ from paper Eq.13–14. Preserve existing Gaussian-noise/zero-image and CRF distinction. Optional add pages 21638–21648.

### U12 — `cgics` at paper/sections/related_work.tex:10

**Claim context in the report:** Recent methods differ in how they turn that example into a query mask. SegIC learns to use foundation-model correspondences as geometric and semantic guidance~\cite{segic}; SINE introduces in-context interaction~\cite{sine}; and VRP-SAM learns a reference-prompt encoder~\cite{vrpsam}. Frozen-model pipelines instead reuse pretrained representations and segmentation priors: PerSAM derives target-specific information from one example~\cite{persam}, and Matcher links general feature matching to promptable segmentation~\cite{matcher}. INSID3 operates directly on DINOv3 features~\cite{insid3}, whereas CG-ICS combines generated concepts, search, and visual exemplars~\cite{cgics}. We compare these last two complete inference systems and then examine their individual processing stages.

**Audit:** Title, three authors, ECCV 2026, LNCS17074, pp336–352 and DOI verified. Reference robustness claim is explicitly the paper objective. Local author preprint v2 Sections3.2–3.4 confirms reference-IoU × query-presence, tree-search and stitched-image box extraction. General exponents alpha/beta describe project parameterization; default1/1 matches paper. No new measured performance asserted from this citation.

### U13 — `cgics` at paper/sections/related_work.tex:12

**Claim context in the report:** The chosen reference is itself a source of variation. Good Examples studies example selection~\cite{goodexamples}, UNICL-SAM addresses uncertainty in imperfect examples~\cite{unicl}, and CG-ICS explicitly targets reference robustness~\cite{cgics}. Our reference study measures performance when the query is fixed and the supplied demonstration changes. Cross-domain work provides a complementary perspective: RePRI uses transductive optimization~\cite{repri}, while ABCDFSS adapts feature-processing layers at test time~\cite{abcdfss}. Their adaptation mechanisms differ from the frozen inference evaluated here, motivating a separate domain-transfer analysis.

**Audit:** Title, three authors, ECCV 2026, LNCS17074, pp336–352 and DOI verified. Reference robustness claim is explicitly the paper objective. Local author preprint v2 Sections3.2–3.4 confirms reference-IoU × query-presence, tree-search and stitched-image box extraction. General exponents alpha/beta describe project parameterization; default1/1 matches paper. No new measured performance asserted from this citation.

### U14 — `repri` at paper/sections/related_work.tex:12

**Claim context in the report:** The chosen reference is itself a source of variation. Good Examples studies example selection~\cite{goodexamples}, UNICL-SAM addresses uncertainty in imperfect examples~\cite{unicl}, and CG-ICS explicitly targets reference robustness~\cite{cgics}. Our reference study measures performance when the query is fixed and the supplied demonstration changes. Cross-domain work provides a complementary perspective: RePRI uses transductive optimization~\cite{repri}, while ABCDFSS adapts feature-processing layers at test time~\cite{abcdfss}. Their adaptation mechanisms differ from the frozen inference evaluated here, motivating a separate domain-transfer analysis.

**Audit:** CVPR2021 six authors pp13979–13988 verified; spelling/order matches official BibTeX. Transductive inference optimizes support and unlabeled-query terms; report contrasts this optimization with frozen inference correctly.

### U15 — `abcdfss` at paper/sections/related_work.tex:12

**Claim context in the report:** The chosen reference is itself a source of variation. Good Examples studies example selection~\cite{goodexamples}, UNICL-SAM addresses uncertainty in imperfect examples~\cite{unicl}, and CG-ICS explicitly targets reference robustness~\cite{cgics}. Our reference study measures performance when the query is fixed and the supplied demonstration changes. Cross-domain work provides a complementary perspective: RePRI uses transductive optimization~\cite{repri}, while ABCDFSS adapts feature-processing layers at test time~\cite{abcdfss}. Their adaptation mechanisms differ from the frozen inference evaluated here, motivating a separate domain-transfer analysis.

**Audit:** Single author Jonas Herzog, CVPR2024 pp23605–23615 verified. Test-time feature-processing adaptation is accurate: small networks attached to a pretrained feature pyramid are learned using augmented-view consistency.

### U16 — `insid3` at paper/sections/methods.tex:10

**Claim context in the report:** INSID3 uses the semantic correspondence and spatial grouping structure of frozen DINOv3 features~\cite{insid3,dinov3}. Figure~\ref{fig:insid3} separates two feature paths. Original query features provide coherent clusters and within-query similarity. Debiased reference and query features provide cross-image matching, reducing the influence of absolute image position.

**Audit:** Title, six authors, CVPR 2026 verified. Frozen DINOv3, original-feature grouping, debiased matching and orthogonal projection supported. The report describes the executed implementation: its candidate-coverage factor and thresholded seed differ from paper Eq.13–14. Preserve existing Gaussian-noise/zero-image and CRF distinction. Optional add pages 21638–21648.

### U17 — `cgics` at paper/sections/methods.tex:23

**Claim context in the report:** CG-ICS combines concept search with a parallel visual transfer route~\cite{cgics}. A multimodal language model generates candidate concepts from the reference and a highlighted view of the annotated target. Each concept $c$ is scored by reference fidelity and query presence:

**Audit:** Title, three authors, ECCV 2026, LNCS17074, pp336–352 and DOI verified. Reference robustness claim is explicitly the paper objective. Local author preprint v2 Sections3.2–3.4 confirms reference-IoU × query-presence, tree-search and stitched-image box extraction. General exponents alpha/beta describe project parameterization; default1/1 matches paper. No new measured performance asserted from this citation.

### U18 — `matcher` at paper/sections/experiments.tex:4

**Claim context in the report:** We evaluate COCO-$20^i$ and LVIS-$92^i$ using the author-prepared class memberships, with four folds of 20 classes and ten folds of 92 classes, respectively~\cite{coco,lvis,matcher}. The standard comparison contains 4,000 COCO and 23,000 LVIS episodes per system. A reference cohort varies the demonstration for fixed query targets. A separate component cohort contains 400 COCO and 1,000 LVIS query targets, with up to three references each (1,200 and 2,757 episodes per configuration). All methods within a comparison receive matched inputs.

**Audit:** ICLR2024 six-author identity verified. Training-free feature matching plus promptable segmentation supported. Dataset citation supports COCO four 20-class held-out folds and LVIS ten92-class folds. Project episode counts remain local evidence, not attributed to Matcher.

## Proposed exact changes

- Optional metadata: `insid3` → `pages = {21638--21648}`.
- Optional metadata: `segic` → `volume = {15096}`.
- Recommended — paper/sections/methods.tex:
  - Existing: The final aggregation combines each cluster's cross-image alignment, original-space similarity to the seed, and candidate coverage.
  - Suggested: In the evaluated implementation, final aggregation combines each cluster's cross-image alignment, original-space similarity to the seed, and candidate coverage.
  - Reason: Paper and code differ; retain executed formula but mark implementation scope.
- Optional — paper/sections/related_work.tex:
  - Existing: SegIC learns to use foundation-model correspondences as geometric and semantic guidance
  - Suggested: SegIC turns foundation-model correspondences into geometric, visual, and meta instructions
  - Reason: Align with original three instruction categories. Existing broad description is defensible.

## Access and evidence boundaries

CG-ICS method verification used the already-retrieved author PDF `report_revision_20261116/method_sources/cgics_original.pdf` (SHA256 `f8478bdb95d8b938af4e3fe108c11b88e865e8bdaa806f027007e486c665be61`), identified on its first page as arXiv:2606.28149v2, 16 July 2026. Its score formula and visual-transfer route were read directly; publisher metadata was freshly verified online. Current direct arXiv retrieval failed, so it is not described as a newly downloaded paper.

INSID3 paper Section3 uses cross/intra aggregation and an explicit seed union. The executed local upstream file `batch2_20261116/upstream-insid3/models/insid3.py` includes forward filtering, candidate-coverage weighting and thresholded selection. Thus the report should identify its detailed aggregation as the evaluated implementation. The existing appendix distinction for the positional input and CRF must remain.

Some direct CVF fetches were403; official indexed primary pages provided metadata and abstracts. Matcher PDF direct-open was oversized; the indexed official PDF supplied the exact dataset paragraph. No third-party paper summaries were used as scientific support.

Detailed source-reference IDs and all citation contexts are in `bib_ics.json`.
