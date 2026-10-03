# Data for the final MATH-AI paper

From the release root, run:

```sh
python reproduce_results.py
```

Python 3.9 or newer is sufficient. The script uses only the standard library, verifies packaged source checksums, and writes derived files to `results/generated/`. It does not run training or model evaluation.

## Contents and provenance

- `reports/*/summary.json`: selected archived report rows for all 24 independent runs in the final paper, plus the separate denser evaluation of seed-303 linear IPW. Selection metadata and original project-relative report paths are retained. Duplicate comparison runs are omitted. The observer report contains only its two cheap-RLOO runs; unused selector diagnostics are omitted.
- `diagnostics/*.json`: 120 updates each for seed-303 linear IPW and group-scaled IPW. Each row preserves audit indices, bought labels, cheap labels at those indices, and the recorded gradient norm. These are field-selected snapshots of the original audit sidecars and Trainer state.
- `reference_tables/*.tex`: exact snapshots of the five tables included by the final manuscript, used for numerical comparison.
- `SOURCE_MANIFEST.json`: checksums of every packaged source file and checksums/relative paths of the original sources. Selection and path sanitization mean a packaged report can intentionally have a different checksum from its original.
- `generated/`: derived outputs, all replaceable by rerunning the command above.

The main trajectory set contains 24 unique runs and 98 checkpoint rows: the six-arm seed-101 pilot, nine original long runs, two pre-audit continuations, one additional seed-202 cheap-RLOO continuation, two observer continuations, and four native cheap-GRPO continuations. Denser evaluation reuses an existing trained policy and is stored separately.

## Recomputed outputs

| Output | Check |
| --- | --- |
| `short_results.{csv,tex}` | Six rows match the paper's pilot table. |
| `density_scaling.{csv,tex}` | Both seed-303 feedback-density rows match. |
| `preaudit_results.{csv,tex}` | Both pre-audit rows match. |
| `traj_results.{csv,tex}` | All 24 trajectory rows match, including the four native cheap-GRPO runs and the two observers. |
| `costs_cr.{csv,tex}` | All 22 ledger rows match. Observers are excluded as in the paper. |
| `costs_exact.csv` | Unrounded completion tokens, Trainer tokens and training seconds. |
| `trajectories.csv` | All 98 original-panel checkpoints, in long format. |
| `endpoints_and_labels.csv` | Endpoints and separate policy, evaluation, diagnostic and shadow-predictor label counts. |
| `paired_question_comparisons.csv` | All eight paired-question contrasts and SEs quoted in the main text. |
| `denser_evaluation.csv` | The separate seed-303 evaluation, including updates 60 and 100. |
| `seed303_audit_diagnostics.csv` / `seed303_audit_summary.json` | Audit-window counts, 45 versus 275 purchased false positives, and gradient-norm associations. |
| `long_accuracy.csv` / `long_accuracy.svg` | The same six series as the paper's accuracy figure; the standalone SVG uses independent styling. |
| `verification.json` | The checks performed and their scope. |

Question-level accuracy and wrong-trigger vectors in the standard report snapshots have 128 entries, each the mean of two responses. Their means are checked against reported aggregates. Paired SEs use the sample standard deviation of question-level differences divided by `sqrt(128)`; training seeds are analyzed separately. Observer report snapshots contain aggregate trajectories only and are not used for paired SEs.

## Interpretation

The historical field `fp_occupancy` is the paper's wrong-trigger share: incorrect responses containing case-sensitive text `python`, divided by all responses. Training uses a token selector, so this field is not a conditional verifier false-positive rate.

Native cheap GRPO uses `3/4 F(c)` with group size four and ran on H100 SXM; earlier runs used H100 NVL. Its starting measured accuracies differ. No exact same-hardware `F(c)` control is implied.

Observer policy updates use zero trusted labels; each observer separately bought 102 labels for its shadow predictor. Training diagnostics retain truth for every training response. These counts refer to roles, not a claim of statistically independent or additional unique labels. Evaluation counts include the repeated initial control. The denser evaluation adds 2,048 labels beyond that policy's original 1,536.

The reconstruction starts from saved report summaries and selected logs; it does not regrade response text. The release does not include the starting model weights or recover an unrecorded historical upstream model revision, and this command is not a claim of exact training reproducibility.
