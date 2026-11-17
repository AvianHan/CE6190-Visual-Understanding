# Data, statistics, and adaptive-inference citation audit

Audit date: 2026-11-17 (SGT). Ten references checked against original papers, publisher records, and official dataset guidance. All active citation occurrences were inspected; unused `old_*` sections were excluded. No report source was modified.

## Findings

- No metadata corrections are required for these ten entries.
- The ISIC + HAM10000 paired citation follows the official ISIC 2018 attribution guidance. HAM10000 alone is not a source for Task 1 segmentation masks.
- Separate original COCO/LVIS dataset credit from the Matcher evaluation-split citation for clearer attribution.
- The Agreement Bound and cost/IoU identities are mathematically valid with their stated scope; none proves empirical AGS effectiveness.

## coco

Verified: Lin et al.; Microsoft COCO: Common Objects in Context; ECCV 2014, LNCS 8693, pp. 740–755; DOI 10.1007/978-3-319-10602-1_48.

**Citation use:** Supports the underlying COCO dataset, not the COCO-20^i split by itself. The adjacent Matcher citation supports the one-shot fold protocol. Existing composite citation is supported but separating source and protocol citations would make attribution precise.

**Action:** Optional citation placement clarification only.

**Checked sources:**
- https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48 (retrieval: `turn100view0`)

**All active locations:**
- `paper/sections/experiments.tex:4` — We evaluate COCO-$20^i$ and LVIS-$92^i$ using the author-prepared class memberships, with four folds of 20 classes and ten folds of 92 classes, respectively~\cite{coco,lvis,matcher}. The standard comparison contains 4,000 COCO and 23,000 LVIS episodes per system. A reference cohort varies the demonstration for fixed query targets. A separate component cohort contains 400 COCO and 1,000 LVIS query targets, with up to three references each (1,200 and 2,757 episodes per configuration). All methods within a comparison receive matched inputs.

## lvis

Verified: Gupta, Dollár, Girshick; LVIS: A Dataset for Large Vocabulary Instance Segmentation; CVPR 2019, pp. 5356–5364.

**Citation use:** Supports the underlying LVIS segmentation dataset. The 920 retained classes / ten 92-class folds come from Matcher, not the original LVIS article. Matcher paper section 4.1 confirms at least two images per class and 2,300 episodes per fold.

**Action:** Optional citation placement clarification only.

**Checked sources:**
- https://openaccess.thecvf.com/content_CVPR_2019/html/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.html (retrieval: `turn107search1`)
- https://proceedings.iclr.cc/paper_files/paper/2024/file/4df9a5e6bad9e64ebcea453e031142bb-Paper-Conference.pdf (retrieval: `turn122search26`)

**All active locations:**
- `paper/sections/experiments.tex:4` — We evaluate COCO-$20^i$ and LVIS-$92^i$ using the author-prepared class memberships, with four folds of 20 classes and ten folds of 92 classes, respectively~\cite{coco,lvis,matcher}. The standard comparison contains 4,000 COCO and 23,000 LVIS episodes per system. A reference cohort varies the demonstration for fixed query targets. A separate component cohort contains 400 COCO and 1,000 LVIS query targets, with up to three references each (1,200 and 2,757 episodes per configuration). All methods within a comparison receive matched inputs.

## suim

Verified: Islam, Edge, Xiao, Luo, Mehtaz, Morse, Enan, Sattar; Semantic Segmentation of Underwater Imagery: Dataset and Benchmark; IROS 2020, pp. 1769–1776; DOI 10.1109/IROS45743.2020.9340821.

**Citation use:** Supports underwater semantic segmentation images and masks. The report correctly cites it as the transfer dataset; the custom merged pool, excluded malformed pairs, 1,000-query sampling, and seven foreground classes are implementation choices rather than the original benchmark protocol.

**Action:** Retain. Do not convert the proposed transfer study into a claim of standard SUIM benchmark reproduction.

**Checked sources:**
- https://arxiv.org/abs/2004.01241 (retrieval: `turn100view2`)
- https://ieeexplore.ieee.org/abstract/document/9340821/ (retrieval: `turn107search4`)
- https://experts.umn.edu/en/publications/semantic-segmentation-of-underwater-imagery-dataset-and-benchmark-2/ (retrieval: `turn110view4`)

**All active locations:**
- `paper/sections/extension.tex:26` — The second batch evaluates four connected parts. A fixed-candidate-pool ablation compares ranking by $s_rs_q$, $s_r$, and $s_q$ using the same candidate evidence and final visual prompts. A fresh-query comparison tests the original full system, the single-concept route, and the one-initial-candidate setting. Domain transfer uses SUIM underwater images~\cite{suim} and public ISIC~2018 Task~1 lesion images and masks~\cite{isic,ham10000}. Finally, the agreement gate is compared with original full search, staged always-search, an initial-score gate, and random routing.

## isic

Verified: Codella et al.; Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC); arXiv:1902.03368 submitted 2019. Current year=2019 is correct for the cited preprint; 2018 is the challenge year.

**Citation use:** Supports ISIC 2018 lesion segmentation Task 1 and the public challenge dataset. Official data source distinguishes Task 1 images/masks from Task 3 diagnostic labels. Report already states a reorganized Task 1 training-image evaluation rather than official held-out challenge test.

**Action:** Retain 2019 bibliography year and paired HAM10000 acknowledgment; do not relabel dataset as HAM10000 segmentation or patient-disjoint clinical evaluation.

**Checked sources:**
- https://arxiv.org/abs/1902.03368 (retrieval: `turn100view3`)
- https://challenge.isic-archive.com/data/ (retrieval: `turn107view0`)

**All active locations:**
- `paper/sections/extension.tex:26` — The second batch evaluates four connected parts. A fixed-candidate-pool ablation compares ranking by $s_rs_q$, $s_r$, and $s_q$ using the same candidate evidence and final visual prompts. A fresh-query comparison tests the original full system, the single-concept route, and the one-initial-candidate setting. Domain transfer uses SUIM underwater images~\cite{suim} and public ISIC~2018 Task~1 lesion images and masks~\cite{isic,ham10000}. Finally, the agreement gate is compared with original full search, staged always-search, an initial-score gate, and random routing.

## ham10000

Verified: Tschandl, Rosendahl, Kittler; The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions; Scientific Data 5, article 180161 (2018); DOI 10.1038/sdata.2018.161.

**Citation use:** HAM10000 is primarily a diagnostic-classification dataset, so it alone would not substantiate the Task 1 segmentation-mask claim. However, the official ISIC 2018 dataset citation instructions explicitly request both Codella and Tschandl. Keeping this paired citation follows the dataset attribution guidance; it is not padding and does not assert that 10,015 HAM10000 records are the Task 1 mask pool.

**Action:** Retain the existing paired citation; record this distinction in audit, no extra defensive paragraph needed in the paper.

**Checked sources:**
- https://www.nature.com/articles/sdata2018161 (retrieval: `turn107search2`)
- https://challenge.isic-archive.com/data/ (retrieval: `turn107view0`)

**All active locations:**
- `paper/sections/extension.tex:26` — The second batch evaluates four connected parts. A fixed-candidate-pool ablation compares ranking by $s_rs_q$, $s_r$, and $s_q$ using the same candidate evidence and final visual prompts. A fresh-query comparison tests the original full system, the single-concept route, and the one-initial-candidate setting. Domain transfer uses SUIM underwater images~\cite{suim} and public ISIC~2018 Task~1 lesion images and masks~\cite{isic,ham10000}. Finally, the agreement gate is compared with original full search, staged always-search, an initial-score gate, and random routing.

## branchynet

Verified: Teerapittayanon, McDanel, Kung; BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks; ICPR 2016, pp. 2464–2469; DOI 10.1109/ICPR.2016.7900006. IEEE online addition in 2017 does not change conference year 2016.

**Citation use:** Supports the broad concept of input-dependent early exit / computation allocation. The report applies that principle to CG-ICS search; it does not claim BranchyNet uses the proposed mask-agreement gate or proves AGS improves accuracy.

**Action:** Retain as conceptual antecedent.

**Checked sources:**
- https://ieeexplore.ieee.org/abstract/document/7900006/authors (retrieval: `turn105search3`)
- https://dash.harvard.edu/entities/publication/73120379-30e9-6bd4-e053-0100007fdf3b (retrieval: `turn105search0`)

**All active locations:**
- `paper/sections/extension.tex:3` — The component results motivate allocating search according to the input. Visual prompting reduces the observed marginal benefit of concept search, while the remaining search benefit varies across datasets and queries. We propose Agreement-Gated Search (AGS), which uses consistency between the initial text-only and visual-only predictions to decide whether to search further. AGS applies input-dependent computation allocation~\cite{branchynet,adaptiveinference} to the concept-search stage of CG-ICS.

## adaptiveinference

Verified: Bolukbasi, Wang, Dekel, Saligrama; Adaptive Neural Networks for Efficient Inference; ICML 2017, PMLR 70, pp. 527–536.

**Citation use:** Supports choosing network components / networks depending on the input and computational cost. Properly used for the concept of input-dependent allocation; AGS is a new application hypothesis, not a theorem from this paper.

**Action:** Retain.

**Checked sources:**
- https://proceedings.mlr.press/v70/bolukbasi17a.html (retrieval: `turn103view2`)

**All active locations:**
- `paper/sections/extension.tex:3` — The component results motivate allocating search according to the input. Visual prompting reduces the observed marginal benefit of concept search, while the remaining search benefit varies across datasets and queries. We propose Agreement-Gated Search (AGS), which uses consistency between the initial text-only and visual-only predictions to decide whether to search further. AGS applies input-dependent computation allocation~\cite{branchynet,adaptiveinference} to the concept-search stage of CG-ICS.

## bootstrap

Verified: Efron; Bootstrap Methods: Another Look at the Jackknife; Annals of Statistics 7(1), 1–26 (1979); DOI 10.1214/aos/1176344552.

**Citation use:** Foundational support for bootstrap resampling and estimation of sampling distributions. Efron alone does not specifically justify treating repeated references as a cluster; the paired Field–Welsh citation supplies that distinction. Replicate counts and exploratory intervals are project design choices, not claimed to be prescribed by the cited article.

**Action:** Retain; do not claim guaranteed nominal coverage from the citation.

**Checked sources:**
- https://blogs.helsinki.fi/bk-club/files/2012/05/Efron_Bootstrap_AS1979.pdf (retrieval: `turn113view0`)
- https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full (retrieval: `turn110view0`)

**All active locations:**
- `paper/sections/experiments.tex:23` — Confidence intervals resample whole query-image clusters, keeping their concepts and references together~\cite{bootstrap,clusterbootstrap}. We report exploratory 95\% intervals using 2,000 bootstrap replicates for main/component contrasts and 1,000 for parameter/area analyses. Reference sensitivity is the standard deviation across distinct references after averaging repeated executions of identical inputs. Failed predictions retain missing accuracy; paired contrasts use common successful inputs.

## clusterbootstrap

Verified: C. A. Field and A. H. Welsh; Bootstrapping Clustered Data; JRSS Series B 69(3), 369–390 (2007); DOI 10.1111/j.1467-9868.2007.00593.x.

**Citation use:** The original article, pp. 370–371, describes resampling entire clusters with replacement, preserving the dependent observations. That supports the report’s query-image cluster resampling with concepts/references held together. Its formal consistency results rely on model conditions; the report correctly says exploratory intervals and does not assert all dependencies between shared reference images vanish.

**Action:** Retain. An image-cluster CI is not a guarantee of fully independent data or complete finite-sample coverage.

**Checked sources:**
- https://rss.onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-9868.2007.00593.x (retrieval: `turn113search0`)
- https://bemlar.ism.ac.jp/zhuang/Refs/Refs/field2007jrssb.pdf (retrieval: `turn110view2`)
- https://academic.oup.com/jrsssb/article-abstract/69/3/369/7109361?login=true (retrieval: `turn113search2`)

**All active locations:**
- `paper/sections/experiments.tex:23` — Confidence intervals resample whole query-image clusters, keeping their concepts and references together~\cite{bootstrap,clusterbootstrap}. We report exploratory 95\% intervals using 2,000 bootstrap replicates for main/component contrasts and 1,000 for parameter/area analyses. Reference sensitivity is the standard deviation across distinct references after averaging repeated executions of identical inputs. Failed predictions retain missing accuracy; paired contrasts use common successful inputs.

## kosub2019

Verified: Sven Kosub; A Note on the Triangle Inequality for the Jaccard Distance; Pattern Recognition Letters 120, 36–38 (2019); DOI 10.1016/j.patrec.2018.12.007. The arXiv preprint dates to 2016 but the journal citation year 2019 is correct.

**Citation use:** Supports Jaccard distance as a metric on finite sets. Applying reverse triangle inequality yields |J(Mt,G)-J(Mv,G)| <= 1-J(Mt,Mv), exactly as written. The bound constrains only the difference of the two probe accuracies; the report’s identical-wrong-mask counterexample and exclusion of the joint-prompt output are correct.

**Action:** Retain. Do not present this bound as proof that high agreement entails high absolute IoU or that AGS improves speed/accuracy.

**Checked sources:**
- https://www.sciencedirect.com/science/article/pii/S0167865518309188 (retrieval: `turn105search1`)
- https://arxiv.org/abs/1612.02696 (retrieval: `turn110view3`)
- https://arxiv.org/pdf/1612.02696 (retrieval: `turn113view1`)

**All active locations:**
- `paper/sections/mathematical_appendix.tex:7` — $d(X,Y)=1-J(X,Y)$ be Jaccard distance~\cite{kosub2019}.

## Proposed attribution edit

File: `paper/sections/experiments.tex`

```latex
We evaluate COCO-$20^i$ and LVIS-$92^i$, derived from COCO~\cite{coco} and LVIS~\cite{lvis}. We use the author-prepared class memberships, with four folds of 20 classes and ten folds of 92 classes, respectively~\cite{matcher}.
```

Distinguishes original dataset credit from the later ICS evaluation split; numerical protocol is unchanged.

## Mathematical audit

### Agreement bound

**VALID**. Jaccard distance is metric, reverse triangle inequality follows by applying triangle inequality in both directions. Nonempty-mask domain is sufficient. Agreement only bounds accuracy difference, not absolute accuracy or final fused prediction.

### IoU aggregation identity

**VALID**. For positive union sizes, sum(TP)/sum(U)=mean(UJ)/mean(U); subtract mean(J), expand population covariance, and use U=G+FP. It is descriptive weighting algebra, not a causal decomposition.

### Gate cost and accuracy decomposition

**VALID UNDER THE STATED SHARED-PREFIX/COUNTERFACTUAL BRANCH DEFINITIONS**. Subtract Cp+Cs from Cp+A*Ce+(1-A)*Cs; likewise select early vs search final IoU. E[A D]>E[H] is the exact expected-speed condition under these definitions, not measured AGS evidence. Any comparison with independently different random candidate streams needs matching or integration over randomness.

### Search–visual interaction

**VALID DESCRIPTIVE FACTORIAL CONTRAST**. The difference-in-differences expression is symmetric: it measures how search changes visual-prompt benefit and how visual prompts change search benefit. A negative cohort average establishes diminishing joint gains in that paired cohort, not a universal per-query or cross-domain causal law.

## Access transparency

- Direct CVF LVIS open returned 403; publisher-indexed search returned the official CVF metadata and abstract.
- Nature DOI redirected to inaccessible identity endpoint; official Nature search indexed the full metadata and abstract.
- OUP direct page blocked by CDN; official Wiley/OUP indexed pages plus the original article PDF hosted at ISM were used.
- Project Euclid direct page returned only iframe; original Efron article PDF hosted by University of Helsinki was read, with its bibliographic cover and first article page.
- ScienceDirect direct open returned 403; publisher indexed metadata and author arXiv full text were both checked.
- No bibliography metadata errors were found in these ten entries. No main bib or TeX files edited.
