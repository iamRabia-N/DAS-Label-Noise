# DAS-Label-Noise

Code and results for the paper *Base-learner disagreement and the robustness of stacking ensembles to class-conditional label noise*.

The study measures why stacking degrades more slowly than bagging, boosting, and voting when training labels carry class-conditional noise. The measured mechanism converts into the disagreement-augmented stacking method (DAS). The method is tested on 10 datasets, three injected noise models, and the real human annotation errors of CIFAR-10N. The evidence base is 4,435 controlled measurements. Fixed random seeds control every experiment. This code produced every result in the paper.

## Pipeline

The repository contains 39 scripts named s01 to s18 in execution order. Each script is the complete code of one experiment run. `s02_noise_injection.py` is the module most scripts import. The `Outputs` directory contains every result file the paper cites.

Stage 1 covers the mechanism study on CoverType and EuroSAT.

| Script | Task |
|:---|:---|
| `s01` | Prepare the CoverType and EuroSAT splits |
| `s02` | Noise injection module for symmetric and cyclic asymmetric noise |
| `s03a`, `s04b` | Clean-data grid search over 36 Random Forest and 72 XGBoost configurations |
| `s03b` to `s03e` | Train Random Forest, XGBoost, voting, and stacking on CoverType across the noise grid |
| `s04a` | Extract frozen ResNet18 features for EuroSAT |
| `s04c` to `s04f2` | Train the four ensembles on the EuroSAT features |
| `s05` | Aggregate the main grid and run the Friedman and pairwise Wilcoxon tests |
| `s06` | Bootstrap confidence intervals for the NSI and the relative drops, with effect sizes |
| `s07` | Pair-flip noise experiment in both domains |
| `s08` | Equal-sample-size control on the exact 100,000-instance stacking subset |
| `s09` | Base-learner agreement with Cohen's kappa against the stacking-voting gap |

Stage 2 covers the replication grid, the baselines, and DAS.

| Script | Task |
|:---|:---|
| `s10` | Prepare the six tabular datasets with fixed preprocessing rules |
| `s11` | Extract ResNet18 features for Fashion-MNIST and CIFAR-10 and import the CIFAR-10N labels |
| `s12a` to `s12e` | Tune the eight new datasets and merge the results |
| `s13a` to `s13e` | Cache the out-of-fold and validation probabilities of the three base learners per dataset, condition, and fold |
| `s14a`, `s14b` | Evaluate six combiner-level methods from the identical cache with the DAS deltas and the kappa-gap correlation |
| `s15a`, `s15b` | Run the confident learning baseline with cleanlab and the GCE robust-loss XGBoost baseline from the same cache |
| `s16a`, `s16b` | Two-configuration DAS ablation on the mechanism-study data |
| `s16c` | Four-configuration per-statistic ablation |
| `s17` | Final aggregation of eight methods on 415 units with Friedman, Nemenyi, average ranks, and baseline Wilcoxon tests |
| `s18` | Result figures Fig2 to Fig4 from the released CSVs. Fig1 is a drawn diagram with no source script |

## Protocol

- The base seed is 42. The noise pattern of a unit is seeded by `42 + 100 * fold + 1000 * noise_level`, so every method within a fold receives the identical noisy training set.
- Every experiment uses five-fold stratified cross-validation. Hyperparameters come from a clean-data search performed once per dataset and are never re-tuned on noisy labels.
- Noise enters training labels only. Validation labels are never modified.
- On CoverType, stacking and its equal-size control use the same fixed stratified 100,000-instance subset, drawn once with seed 42.
- Cached probabilities are stored as float16. Every method within a unit reads the identical cache, so no comparison is biased by the storage precision.

## Execution

The scripts target the Kaggle environment and read inputs from `/kaggle/input` and write to `/kaggle/working`. For any other environment, change the path constants at the top of each file. Scripts locate their inputs by filename search, so the outputs of the producing step must be present in the input tree. Feature extraction and XGBoost training use an NVIDIA T4. Aggregation and analysis scripts run on CPU. Long scripts checkpoint their progress and resume from an attached copy of their own output, so a rerun never repeats completed units. The dependency order is `s01` through `s09` for stage 1 and `s10` through `s18` for stage 2.

Required packages are numpy, pandas, scipy, scikit-learn, xgboost, torch, torchvision, cleanlab, ucimlrepo, matplotlib, seaborn, psutil, and Pillow.

## Data sources

| Dataset | Loaded by | Source |
|:---|:---|:---|
| Forest CoverType | s01 | https://archive.ics.uci.edu/dataset/31/covertype |
| EuroSAT | s01 | https://github.com/phelber/EuroSAT |
| Dry Bean, Pen Digits, Statlog Shuttle, Statlog Satellite, ISOLET | s10 | UCI Machine Learning Repository through `ucimlrepo` |
| HAR | s10 | OpenML |
| Fashion-MNIST, CIFAR-10 | s11 | torchvision |
| CIFAR-10N human labels | s11 | http://noisylabels.com |

## Run records

The tuning runs for the eight new datasets hit the 12-hour session limit. `s12c` and `s12d` are the continuation runs and `s12e` merges the results. Values recovered from the interrupted runs are marked inside the JSON files themselves in `source` and `note` fields, with nothing recomputed or altered. `s13a` to `s13e` are five runs of one template on five dataset groups, split to fit the session limit, and each produced a disjoint set of probability files. CIFAR-10N labels are matched to the torchvision CIFAR-10 index, and `s11` checks the match against the CIFAR-10N clean labels before any use.
