# PRISM-V2: Class-Conditional, Interaction-Aware Progressive Structural Pruning Beyond Cross-Layer Bidirectional Measurement

**Research Manuscript — Implementation-Driven Study**

*Based on the PRISM-V2 ResNet-56 / CIFAR-10 prototype and experimental artifacts.*

## Abstract

Structured channel pruning remains a primary route to reducing convolutional network parameters and floating-point operations while retaining task accuracy. Cross-layer Bidirectional Measurement (CLBM) evaluates channels through weight-based synthesis and distribution scores derived from significant kernels, then allocates layer-wise pruning rates from batch-normalization statistics. While this design captures cross-layer feature propagation in the weight domain, it does not explicitly model class-conditional predictive sensitivity or synergistic interactions among candidate channels, and it typically applies pruning in a single shot before distillation-based recovery. This paper presents PRISM-V2, a progressive structural pruning framework implemented for CIFAR-10 ResNet-56 with option-A residual shortcuts. PRISM-V2 ranks channels with a composite keep score that combines a first-order activation–gradient importance proxy, activation energy, a class-conditional sensitivity term (mean and worst-class responses), and an activation-redundancy penalty. A budgeted pair-interaction probe then temporarily masks low-scoring channel pairs and raises keep scores when loss increases are superadditive. Pruning proceeds over multiple rounds with physical channel removal, knowledge distillation plus a relational feature loss, and validation-gated acceptance with rollback. On the reported run, the dense teacher reaches 92.96% test accuracy with 853,018 parameters and 250,971,392 FLOPs (2 operations per multiply– accumulate). The accepted PRISM-compressed model retains 391,568 parameters (54.10% reduction; 2.18×) and 112,438,404 FLOPs (55.20% reduction; 2.23×) at 92.10% test accuracy (−0.86 percentage points). Round 3 was rejected by the rollback rule, illustrating recoverability-aware stopping. The study does not claim superiority over published CLBM numbers under unmatched protocols; instead it documents a methodologically complementary pruning policy and the evidence available from a single controlled ResNet-56 experiment.

**Index Terms:** Channel pruning, class-conditional importance, CIFAR-10, interaction-aware selection, knowledge distillation,

progressive compression, ResNet-56, structured pruning.

## 1. Introduction

Deep residual networks achieve strong image-classification accuracy, yet their parameter count and multiply–accumulate cost remain barriers for latency- and memory-constrained deployment. In the experimental setting of this paper, a full CIFAR ResNet-56 already contains 853,018 trainable parameters and approximately 251.0 million FLOPs under a two-operations-per-MAC counting convention. Structured channel pruning removes entire feature maps so that the resulting sparse-free architecture maps directly onto dense convolution kernels, avoiding the irregular sparsity patterns of unstructured weight pruning.

A recurring difficulty is deciding which channels are expendable. Magnitude and geometric median criteria [1], [2], feature-map rank [3], and batch-normalization scaling [4] provide useful but incomplete proxies. Recent cross-layer methods argue that a channel’s value depends both on how it is synthesized in the current layer and how it is consumed downstream. CLBM [5] formalizes this idea with significant-kernel L1 statistics, a bidirectional product of synthesis and distribution scores, adaptive BN-based pruning rates, and teacher–student knowledge distillation.

Despite these advances, three methodological gaps remain relevant to predictive conservation under aggressive pruning. First, weight-norm criteria do not observe class-conditional loss sensitivity: a channel that is critical for a difficult class may appear modest in kernel L1 while carrying high predictive value for that class. Second, independent per-channel ranking does not test whether two low-scoring channels are collectively necessary; removing both can produce superadditive damage. Third, one-shot pruning followed by lengthy fine-tuning provides limited intermediate control when a proposed compression ratio exceeds recoverability for a given teacher and data split.

This paper studies PRISM-V2, an implemented progressive pruning pipeline that addresses those gaps with data-dependent scoring, budgeted interaction probes, and validation-gated round acceptance. The experimental evidence is restricted to one ResNet-56 / CIFAR-10 run with seed 42. Within that scope, PRISM-V2 compresses the network by more than half in parameters and FLOPs while losing less than one percentage point of test accuracy, and it rejects a third pruning round that fails the configured recovery tolerance.

Contributions. (1) A composite channel keep score that unifies first-order activation–gradient importance, activation energy, class-conditional mean/worst-class sensitivity, and an activation-correlation redundancy penalty, as coded in the PRISM-V2 notebook. (2) A bounded pair-interaction protection step that masks channel pairs on a calibration minibatch and adjusts keep scores under positive superadditive loss. (3) A progressive structural compression loop with physical channel inheritance on option-A ResNet-56 shortcuts, relational knowledge distillation, and rollback when post-recovery validation accuracy falls outside tolerance. (4) A fully artifact-backed quantitative report for the released run, including per-round acceptance history, resource metrics, class-wise test diagnostics, and an explicit non-claim regarding unmatched CLBM leaderboard comparisons.

## 2. Related Work

Neural network compression spans quantization, low-rank factorization, knowledge distillation, and pruning [6], [7].

Among pruning methods, unstructured schemes remove individual weights and can achieve high sparsity, but hardware acceleration often requires structured removal of filters, channels, or blocks [8]. Filter L1 pruning [1] and soft filter pruning [9] prioritize small-norm filters. Filter Pruning via Geometric Median (FPGM) [2] removes filters near the geometric median rather than relying solely on magnitude. HRank [3] ranks filters by feature-map rank. Network Slimming [4] uses BN scaling factors as importance. Similarity- and clustering-based methods identify redundant filters through pairwise geometry [10].

Gradient- and loss-aware criteria estimate the effect of removing units through Taylor expansions or empirical masking [11]. Molchanov et al. popularized first-order approximations that combine activations and gradients to estimate the change in loss induced by removing a feature map—an idea PRISM-V2 adapts in a class-aggregated form. Correlation- and redundancy-aware pruning penalizes channels whose activations are linearly dependent, reducing the chance of keeping near-duplicates. Knowledge distillation [7], [12] is widely used after pruning so that a compact student inherits dark knowledge from the dense teacher; relational distillation further matches pairwise feature similarities within a minibatch, which is useful when student and teacher feature widths differ after channel removal.

Progressive and recovery-aware pruning schedules appear in soft pruning and iterative magnitude methods, where sparsity increases gradually to stabilize optimization [9]. PRISM-V2 adopts a related philosophy at the structuralarchitecture level: each round commits to a new width configuration only if validation accuracy remains within a tolerance after a fixed recovery budget. This differs from CLBM’s one-shot prune-then-finetune pipeline and from purely heuristic iterative schedules that lack an explicit rollback test.

CLBM [5] is the primary reference method for this study. It selects significant kernels via layer-wise gamma-like L1 thresholds, computes synthesis score SG and distribution score SR from significant-kernel norms, forms SV = SG · SR, allocates layer pruning rates from BN γ and β, and fine-tunes with soft-target distillation. Empirically, Lian et al. report strong ResNet-56 / CIFAR-10 trade-offs in their evaluation protocol (94.22% → 94.36% with 0.32 M parameters and 44.41 M FLOPs under their counting convention). The present work does not re-run that protocol inside the released results folder; published CLBM numbers are cited only as literature context.

PRISM-V2 is complementary rather than a drop-in replacement of CLBM’s bidirectional weight metric. It inherits the high-level goal of structured compression with distillation recovery, but replaces weight-centric bidirectional scoring with class-conditional data-dependent importance, interaction probes, and progressive acceptance control.

## 3. Challenges and Motivation

### 3.1 Fundamental Challenge

Let a convolutional stage produce C channels. Structured pruning seeks a subset S ⊂ {1,…,C} with |S| ≪ C such that the induced subnetwork f_S preserves validation risk of the dense network f. Exact combinatorial search over subsets is intractable, so practical methods assign scalar importance scores and retain top-ranked channels. The quality of that scalarization determines whether compression remains recoverable.

### 3.2 Where CLBM Leaves Residual Risk

CLBM’s bidirectional score is informative about weight-space propagation: channels with weak significant-kernel synthesis or weak downstream distribution receive low SV and are preferentially removed [5]. However, SV is computed from kernel L1 norms (after significant-kernel masking) and does not condition on class labels or on the empirical loss landscape of a calibration stream. Consequently, two channels with similar SV may differ substantially in their contribution to class-conditional cross-entropy. Furthermore, ranking channels independently cannot detect pairs (i, j) for which ΔL({i,j}) > ΔL({i}) + ΔL({j}), i.e., synergistic predictive value. Finally, a single global prune rate allocation followed by one fine-tuning stage offers limited opportunity to reject an over-aggressive architecture once recovery fails.

### 3.3 Research Hypothesis

Hypothesis. Augmenting structured pruning with (i) class-conditional first-order importance, (ii) explicit redundancy penalization from activation correlations, (iii) budgeted pairinteraction protection, and (iv) progressive validation-gated acceptance yields compressed ResNet-56 models that retain competitive accuracy at substantial parameter/FLOP reduction, while providing an automatic stop when further pruning exceeds recoverability under a fixed recovery budget. This hypothesis motivates PRISM-V2’s design. It does not assert that every CLBM limitation is solved, nor that PRISM-V2 dominates CLBM on unmatched published tables.

*Fig. 1. Conceptual contrast between CLBM’s weight-centric bidirectional scoring and PRISM-V2’s class-conditional, interaction-aware progressive policy.*

## 4. Proposed Methodology

### 4.1 Overall Framework

PRISM-V2 proceeds as follows. A dense CIFAR ResNet56 teacher is obtained (trained or loaded). For each pruning round r = 1…R: (1) score stage-output and mid-block channels on a calibration loader; (2) apply interaction protection on a capped module set; (3) select retained channel IDs toward round-specific retention ratios of the original widths; (4) instantiate a physically smaller network and inherit matching weights; (5) recover with distillation; (6) accept the student if validation accuracy remains within `rollback_tolerance_pp` of the previously accepted model, otherwise revert. The CIFAR10 test set is reserved for a single final evaluation.

*Fig. 2. PRISM-V2 progressive compression pipeline used in the ResNet-56 experiment.*

### 4.2 Baseline Architecture

The backbone is a CIFAR ResNet-56 with stage widths 16/32/64, nine basic blocks per stage, bias-free convolutions followed by BatchNorm, and option-A residual shortcuts. Because pruning changes channel cardinalities, shortcuts are implemented as IndexedOptionA modules that copy surviving channels into the residual path using original-architecture channel IDs, with central padding semantics preserved. Midblock channels (outputs of the first 3×3 convolution in each BasicBlock) and stage-output channels are pruned under separate retention schedules. Weight inheritance copies convolution and BN slices by ID so that retained filters initialize from their dense counterparts without reindexing ambiguity.

### 4.3 Original CLBM Mechanism (Foundation)

For completeness, we summarize CLBM’s scoring [5]. Let ||K^l\_{n,m}||\_1 be the L1 norm of kernel (n,m) in layer l. After estimating the mean E_l and variance V_l of kernel norms, CLBM forms a layer-adaptive significant-kernel threshold T_l = E_l − c √V_l, and a binary mask D^l\_{n,m}=1[||K^l\_{n,m}||\_1 ≥ T_l]. Synthesis importance for output channel n aggregates significant kernel norms in the current filter, SG^l_n = Σ_m ||K^l\_{n,m}||*\_1 · D^l\_*{n,m}, while distribution importance for the corresponding input channel aggregates significant norms in the next-layer group SR^{l}\_n. The joint score is SV^l_n = SG^l_n · SR^l_n (last layer uses SG only). Layer-wise rates are then allocated from BN γ/β statistics, and pruning is executed once before KD finetuning. PRISM-V2 does not reimplement SV inside the released V2 notebook; the equations above anchor the research gap rather than claiming an internal CLBM ablation in the V2 artifacts.

Mapping to the gap. Because SG and SR are functions of weight norms, they are invariant to which class currently drives the loss and to whether two low-SV channels are substitutes or complements under masking. PRISM-V2 therefore changes the information sources used for ranking (calibration labels, gradients, activations, masked-loss probes) while keeping the outer goal—structured width reduction with teacher-guided recovery—unchanged.

### 4.4 Proposed Scoring Modification

Let a_b,c,h,w denote channel-c activations for sample b in a hooked feature map, and let g_b,c,h,w = ∂L/∂a_b,c,h,w for cross-entropy L. PRISM-V2 accumulates a per-image Taylorstyle proxy by summing |a ⊙ g| over spatial locations. Averaging over the calibration stream yields a global gradient– activation vector g. An activation-energy vector p is obtained from mean squared activations. Centered, column-normalized pooled activation vectors produce a Gram matrix of absolute correlations; redundancy for channel c is the maximum offdiagonal absolute correlation involving c.

Class-conditional statistics are essential to the modification. For each class k, the implementation accumulates the sum of per-image proxies and divides by the class count, producing a matrix cls ∈ R^{K×C}. With observed classes K_obs, class_imp_c = 0.5 · mean\_{k∈K_obs}(cls\_{k,c}) + 0.5 · max\_{k∈K_obs}(cls\_{k,c}). The worst-class term reduces the risk of discarding features used primarily by a subset of categories. Let N(·) denote min– max normalization to [0,1] within a module. With the configured weights (gradient 0.60, redundancy 0.20, class sensitivity 0.35), the keep score is score = 0.60·N(g) + 0.40·N(p) + 0.35·N(class_imp) − 0.20·N(redundancy). Hooks are attached to the last block of each stage (stage outputs) and to each block’s bn1 module (mid channels). Calibration uses eight training batches by default (8 in the reported config).

### 4.5 Pair-Interaction Protection

Independent ranking can still propose simultaneous deletion of synergistic channels. PRISM-V2 therefore ranks modules by their current minimum keep score, retains the lowest 8 modules, and within each module considers the 3 lowestscoring channels. For candidates i and j, temporary forward hooks zero the corresponding feature maps and measure cross- entropy deltas relative to an unmasked baseline loss L0 on one calibration minibatch: interaction(i,j) = ΔL(i,j) − ΔL(i) − ΔL(j). If interaction(i,j) > 0, the member with the larger current keep score is protected by adding 0.30 · interaction(i,j) / (|L0|+ε) to its score. Masking is used only for candidate evaluation; accepted networks are rebuilt by deleting channels structurally. Across the reported run, 72 pair evaluations were logged, of which 31 exhibited positive interaction.

*Fig. 3. Sign of pair-interaction probes recorded over three pruning rounds.*

### 4.6 Pruning Procedure

PRISM-V2 performs structured pruning: entire channels are removed from convolutions, BatchNorm layers, and the classifier input. Retention targets are expressed as fractions of the original stage widths (16/32/64), not of the current compressed widths, with floors of 8 stage channels and 8 mid channels. Configured schedules are stage ratios (0.85, 0.7, 0.58) and mid ratios (0.8, 0.65, 0.52) over 3 rounds. Within each module, channels are sorted by descending keep score (ties broken by ID) and the top-n IDs are retained. Algorithmic sketch (one round). Input: accepted student f, teacher t, calibration loader C_cal, validation loader C_val, round index r. (1) Compute scores ← score_channels(f, C_cal). (2) scores ← interaction_protection(f, scores, C_cal). (3) selection ← target_selection(f.selection, scores, r). (4) f′ ← inherit_weights(f, CIFARResNet56(selection)). (5) f′ ← fit_KD(f′, t; recovery_epochs). (6) If Acc(f′, C_val) ≥ Acc(f, C_val) − τ then accept f ← f′, else reject and keep f. Output after R rounds: accepted student evaluated once on the heldout test set.

For the reported schedule, Round 1 targets 85%/80% of original stage/mid widths, Round 2 targets 70%/65%, and Round 3 targets 58%/52%. The accepted student corresponds to the Round-2 targets because Round 3 was rejected. Exact surviving channel IDs are stored in `final_fp32.pt` under the selection field for exact reconstruction.

### 4.7 Fine-Tuning and Knowledge Distillation

After each structural proposal, the student is trained for 40 epochs with SGD (Nesterov), momentum 0.9, weight decay 0.0001, recovery learning rate 0.025, and cosine annealing. The loss combines hard-label cross-entropy, KL distillation against the frozen teacher at temperature 5.0 with weight 0.7, and a relational term of weight 0.05 matching pairwise cosinesimilarity matrices of student and teacher pooled features. Acceptance requires val_acc ≥ max(min_accuracy_before_accept, accepted_val − `rollback_tolerance_pp`), with `rollback_tolerance_pp` = 0.50.

### 4.8 Computational Complexity

Scoring cost is one forward–backward pass over the calibration batches with feature hooks. Interaction probing evaluates O(M · P choose 2) masked forwards for M modules and P candidates per round—intentionally capped. These costs affect the pruning-time pipeline only. Inference cost of the accepted student equals that of the structurally smaller ResNet56; no runtime decoder or dynamic mask is required. Energy is not claimed: the implementation records `energy_mj_per_inference` as unavailable without instrumented power sampling.

## 5. Experimental Setup

### 5.1 Dataset

CIFAR-10 provides 50,000 training and 10,000 test 32×32 RGB images across 10 classes. A deterministic 10% validation split (seed 42) yields 45,000 training and 5,000 validation images. Training uses RandomCrop(32, padding=4) and RandomHorizontalFlip; validation/test use normalization only with mean (0.4914, 0.4822, 0.4465) and std (0.2470, 0.2435, 0.2616). The test set is unused until final metrics.

### 5.2 Model Architectures

All experiments use the CIFAR ResNet-56 described in Section 4.2. The dense teacher in the reported log was loaded from a baseline checkpoint and reached 92.94% validation accuracy before pruning. No additional architectures are present in the released results folder for this V2 run.

### 5.3 Implementation Details

Framework: PyTorch 2.11.0+cu128 on CUDA (from `run.log`). Batch size 128, seed 42, smoke_test=False. Teacher training recipe in the notebook specifies 200 epochs at learning rate 0.1; the logged run loaded an existing baseline checkpoint rather than retraining from scratch in that session. Recovery uses 40 epochs per round at learning rate 0.025. FLOPs are counted with forward hooks using two FLOPs per convolutional MAC and two FLOPs per dense multiply–add. Model size is the serialized FP32 `state_dict` size in megabytes. Latency uses 100 timed single-image inferences after warmup.

### 5.4 Evaluation Metrics

Primary metrics: Top-1 test accuracy (%), parameter count, FLOPs, serialized size (MB), and macro-averaged precision/recall/F1. Derived quantities: Parameter Reduction (%) = 100 · (P_orig − P_comp) / P_orig, FLOPs Reduction (%) = 100 · (F_orig − F_comp) / F_orig, Compression Ratio = P_orig / P_comp (analogously for FLOPs). These formulas reproduce `summary.json` exactly for the reported artifacts (54.0962% parameters; 55.1987% FLOPs).

## 6. Experimental Results and Discussion

### 6.1 Baseline Performance

The dense ResNet-56 teacher attains 92.96% Top-1 test accuracy, macro-F1 0.9295, 853,018 parameters, 250,971,392 FLOPs, and a 3.374 MB FP32 `state_dict`. Mean single-image latency on the measurement device is 8.142 ms. These figures establish the reference point for compression.

### 6.2 Performance Across Pruning Rounds

| Round | Accepted | Validation before FT (%) | Validation after FT (%) | Previous accepted validation (%) | Parameters | FLOPs |
|---:|:---:|---:|---:|---:|---:|---:|
| 1 | Yes | 28.64 | 92.78 | 92.94 | 578,207 | 173.3M |
| 2 | Yes | 32.64 | 92.30 | 92.78 | 391,568 | 112.4M |
| 3 | No | 27.66 | 91.60 | 92.30 | 257,771 | 75.0M |

*Table I. Progressive pruning history. Round 3 fails the 0.50 pp rollback tolerance (91.60% < 92.30% − 0.50%) and is rejected.*

Round 1 reduces the network to 578,207 parameters and recovers to 92.78% validation accuracy (accepted). Round 2 reaches 391,568 parameters and 92.30% validation accuracy (accepted). Round 3 proposes 257,771 parameters; after recovery, validation accuracy is 91.60%, which is 0.70 pp below the previous accepted model and outside the 0.50 pp tolerance, so the proposal is discarded. The final deployed student is therefore the Round-2 network. Immediate postprune validation accuracies before recovery are low (≈28–33%), underscoring that structural deletion requires distillation-based healing before acceptance decisions.

*Fig. 4. Validation recovery and resource trajectory across pruning rounds (green = accepted, red = rejected).*

### 6.3 Final Test Comparison

| Metric | Original ResNet-56 | PRISM-compressed |
|---|---:|---:|
| Test accuracy (%) | 92.96 | 92.10 |
| Test loss | 0.3591 | 0.4601 |
| Macro precision | 0.9296 | 0.9211 |
| Macro recall | 0.9296 | 0.9210 |
| Macro F1 | 0.9295 | 0.9210 |
| Parameters | 853,018 | 391,568 |
| FLOPs (2 ops/MAC) | 250,971,392 | 112,438,404 |
| Size (MB) | 3.374 | 1.599 |
| Mean latency (ms) | 8.142 | 8.194 |
| Parameter reduction | — | 54.10% |
| FLOPs reduction | — | 55.20% |
| Parameter compression | 1.00× | 2.18× |
| FLOPs reduction factor | 1.00× | 2.23× |
| Accuracy change (pp) | — | −0.86 |

*Table II. Final held-out test comparison from `comparison.csv` and `summary.json`.*

*Fig. 5. Side-by-side test accuracy, parameter count, and FLOPs for the dense and PRISM-compressed models (artifact from the experimental run).*

The compressed model loses 0.86 percentage points of test accuracy while removing 54.10% of parameters and 55.20% of FLOPs. Serialized model size falls from 3.374 MB to 1.599 MB. Mean latency remains essentially unchanged on the measurement device (8.142 ms → 8.194 ms), indicating that theoretical FLOP reduction does not automatically translate into wall-clock speedup under the tested single-image regime; latency claims beyond this device-local measurement are not made.

### 6.4 Comparison with CLBM

A fair numerical contest with CLBM requires identical preprocessing, residual-shortcut convention, FLOPs counting, teacher checkpoint, recovery budget, and seeds. The released PRISM-V2 results directory does not contain a completed matched CLBM run. Published CLBM ResNet-56 / CIFAR-10 figures (94.22% → 94.36%, 0.32 M parameters, 44.41 M FLOPs) [5] use a different baseline accuracy and a FLOPs convention that is not commensurate with the 2-ops/MAC counter used here (dense FLOPs 250.97 M in this run). Therefore this paper does not claim that PRISM-V2 outperforms CLBM. Methodologically, PRISM-V2 targets class-conditional and interaction effects that CLBM’s weightbased SV score does not encode, and it adds progressive rollback that CLBM’s one-shot pipeline does not provide. Whether those additions improve the Pareto frontier under matched budgets remains an open experimental question flagged in Section 7 and in the Remaining Issues report.

### 6.5 Parameter and Computational Efficiency

Compression ratios of 2.18× (parameters) and 2.23× (FLOPs) place the accepted student near the moderate retention regime that CLBM’s own ResNet-56 analysis associates with stable accuracy (CLBM reports an operating point near \~35% FLOPs retained under their protocol). PRISM-V2’s rejected Round 3 would have moved further toward aggressive compression (257,771 parameters; 75.03 M FLOPs) but failed recoverability under the 40-epoch budget and 0.50 pp tolerance. This behavior is evidence for the hypothesis that validation-gated progressive pruning can prevent irreversible over-pruning even when scoring continues to propose smaller architectures.

### 6.6 Class-Wise Behavior

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Airplane | 0.92 | 0.94 | 0.93 | 1,000 |
| Automobile | 0.97 | 0.96 | 0.96 | 1,000 |
| Bird | 0.89 | 0.88 | 0.89 | 1,000 |
| Cat | 0.83 | 0.84 | 0.84 | 1,000 |
| Deer | 0.91 | 0.93 | 0.92 | 1,000 |
| Dog | 0.89 | 0.85 | 0.87 | 1,000 |
| Frog | 0.93 | 0.96 | 0.95 | 1,000 |
| Horse | 0.96 | 0.94 | 0.95 | 1,000 |
| Ship | 0.95 | 0.95 | 0.95 | 1,000 |
| Truck | 0.95 | 0.96 | 0.95 | 1,000 |

*Table III. Per-class test metrics of the final PRISM-compressed ResNet-56 (from `classification_report.txt`).*

The weakest class under the compressed model is cat (precision 0.83, recall 0.84, F1 0.84); automobile remains strong (0.97 / 0.96 / 0.96). These outcomes are descriptive of the single reported run. They are consistent with the motivation for a worst-class sensitivity term, but they do not by themselves prove that the class-conditional component caused the observed per-class profile—an ablation would be required.

*Fig. 6. Confusion matrix of the final PRISM-compressed model on the CIFAR-10 test set.*

### 6.7 Ablation Status

The released artifacts do not include ablations that disable `class_sensitivity_weight`, `interaction_weight`, `relation_weight`, or rollback independently. Interaction logs show that 31/72 probed pairs were superadditive, which supports the existence of the phenomenon the protection step targets, but not the marginal accuracy contribution of protection. Claims about component necessity are therefore marked theoretically motivated / partially supported rather than experimentally validated in the Claim-to-Evidence mapping.

### 6.8 Interpretation

Observed behavior aligns with the research hypothesis at the system level: progressive, recoverability-aware pruning produced a compact model with <1 pp test-accuracy loss and automatically stopped before a non-recoverable round. A plausible explanation is that class-conditional scoring and interaction protection preserve predictive channels that pure weight-norm ranking might undervalue, while rollback prevents committing to an over-pruned topology under a fixed recovery budget. Alternative explanations—favorable seed, strong teacher, or distillation hyperparameters—cannot be ruled out without multi-seed and matched-baseline controls.

The training dynamics inside each recovery phase are also informative. In Round 1, validation accuracy climbs from the mid-80%s in early epochs to 92.78% by epoch 40; Round 2 similarly recovers from the mid-80%s to 92.30%. Round 3 plateaus near 91.6% best validation accuracy despite continued training-loss decrease, suggesting a capacity or topology limit rather than simple under-optimization under the same 40- epoch, lr=0.025 recipe. This pattern supports treating acceptance as an empirical recoverability test rather than assuming that longer fine-tuning would always close the gap.

Relative to the dense teacher, the compressed model’s test loss rises from 0.359 to 0.460 while Top-1 falls by 0.86 pp. The larger relative increase in loss versus accuracy indicates softened confidence even when argmax decisions remain mostly intact—an expected side effect of capacity reduction that distillation only partially offsets. Macro-F1 tracks accuracy closely (0.9295 → 0.9210), so the degradation is not concentrated in a single pathological class despite the weaker cat category.

### 6.9 Limitations

Limitations of the present evidence include: single seed and single architecture; no matched CLBM comparison in the results folder; capped pairwise (not higher-order) interaction search; no per-candidate short recovery probes; device-local latency without energy instrumentation; FP32-only pipeline without quantization; and FLOPs conventions that differ from some published CIFAR tables. The notebook itself cautions that accuracy gains are hypotheses rather than guarantees and that the final model can be less accurate than CLBM under some protocols.

## 7. Conclusion and Future Work

CLBM advanced structured pruning by coupling significant-kernel synthesis and distribution measurements with BN-adaptive rates and distillation. PRISM-V2 addresses complementary risks: class-conditional predictive sensitivity, synergistic channel pairs, and irreversible one-shot over-pruning. On ResNet-56 / CIFAR-10, the reported implementation retains 92.10% test accuracy at 54.10% / 55.20% parameter/FLOP reduction after accepting two progressive rounds and rejecting a third. Future work should (i) ablate class-sensitivity and interaction terms, (ii) execute matched-budget CLBM comparisons under identical FLOPs counting, (iii) repeat seeds for mean±std reporting, and (iv) instrument energy if deployment claims are required.

## References

[1] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, “Pruning filters for efficient ConvNets,” in Proc. ICLR, 2017.

[2] Y. He, P. Liu, Z. Wang, Z. Hu, and Y. Yang, “Filter pruning via geometric median for deep convolutional neural networks acceleration,” in Proc. IEEE/CVF CVPR, 2019, pp. 4340–4349.

[3] M. Lin et al., “HRank: Filter pruning using high-rank feature map,” in Proc. IEEE/CVF CVPR, 2020, pp. 1529–1538.

[4] Z. Liu, J. Li, Z. Shen, G. Huang, S. Yan, and C. Zhang, “Learning efficient convolutional networks through network slimming,” in Proc. IEEE ICCV, 2017, pp. 2736–2744.

[5] S. Lian, J. Cai, Y. Zhao, S. Poslad, and J. Pei, “A cross-layer bidirectional measurement pruning for model compression guided by feature propagation,” Information Sciences, vol. 740, Art. no. 123216, 2026, doi: 10.1016/j.ins.2026.123216.

[6] S. Han, H. Mao, and W. J. Dally, “Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding,” in Proc. ICLR, 2016.

[7] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015.

[8] W. Wen, C. Wu, Y. Wang, Y. Chen, and H. Li, “Learning structured sparsity in deep neural networks,” in Proc. NeurIPS, 2016.

[9] Y. He, G. Kang, X. Dong, Y. Fu, and Y. Yang, “Soft filter pruning for accelerating deep convolutional neural networks,” in Proc. IJCAI, 2018.

[10] J. Pei, Z. Huang, and J. Zhu, “Pruning rate-controlled filter order– information structure similarity graph clustering for DCNN structure optimization methods,” Multimedia Tools and Applications, vol. 83, pp. 78407–78431, 2024.

[11] P. Molchanov, S. Tyree, T. Karras, T. Aila, and J. Kautz, “Pruning convolutional neural networks for resource efficient inference,” in Proc. ICLR, 2017.

[12] J. Gou, B. Yu, S. J. Maybank, and D. Tao, “Knowledge distillation: A survey,” Int. J. Comput. Vis., vol. 129, pp. 1789–1819, 2021.

[13] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE CVPR, 2016, pp. 770–778.

[14] A. Krizhevsky, “Learning multiple layers of features from tiny images,” Technical Report, Univ. of Toronto, 2009.

[15] S. Ahmed et al., “DeepCompress-ViT: Rethinking model compression to enhance efficiency of vision transformers at the edge,” in Proc. IEEE/CVF CVPR, 2025. (Organizational/style reference only; methodology not adopted.)

[16] PRISM-V2 implementation notebook, “PRISM_V2_ClassConditional_Interaction_ResNet56_CIFAR10.ipynb,” and run artifacts under results (55)/prism_runs/prism_v2_resnet56_c10/ (`comparison.csv`, `summary.json`, `round_history.jsonl`, `classification_report.txt`, `pair_interactions.jsonl`, `run.log`, `config.json`).

