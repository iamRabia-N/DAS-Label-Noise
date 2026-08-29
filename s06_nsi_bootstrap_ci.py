import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from itertools import combinations

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MASTER_CSV = Path('/kaggle/working/aggregated_results/master_all_results.csv')
OUTPUT_DIR = Path('/kaggle/working/robustness_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path('/kaggle/working/paper_figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MASTER_CSV)

N_BOOTSTRAP = 1000
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
ensembles_order = ['RF', 'XGBoost', 'Voting', 'Stacking']

bootstrap_records = []
for (ensemble, domain, noise_type), group in df.groupby(['ensemble', 'domain', 'noise_type']):
    fold_matrix = np.zeros((5, len(NOISE_LEVELS)))
    for j, nl in enumerate(NOISE_LEVELS):
        for i, fold in enumerate(sorted(group['fold'].unique())):
            row = group[(group['noise_level'] == nl) & (group['fold'] == fold)]
            fold_matrix[i, j] = row['f1_macro'].values[0] if len(row) == 1 else np.nan

    if np.isnan(fold_matrix).any():
        continue

    mean_f1 = fold_matrix.mean(axis=0)
    point_slope = np.polyfit(NOISE_LEVELS, mean_f1, 1)[0]

    rng = np.random.RandomState(RANDOM_SEED)
    boot_slopes = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sampled_matrix = fold_matrix[rng.choice(5, size=5, replace=True), :]
        boot_slopes[b] = np.polyfit(NOISE_LEVELS, sampled_matrix.mean(axis=0), 1)[0]

    rel_drops = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sampled_matrix = fold_matrix[rng.choice(5, size=5, replace=True), :]
        boot_mean = sampled_matrix.mean(axis=0)
        rel_drops[b] = (boot_mean[0] - boot_mean[-1]) / boot_mean[0] * 100

    point_rel_drop = (mean_f1[0] - mean_f1[-1]) / mean_f1[0] * 100

    bootstrap_records.append({
        'ensemble': ensemble,
        'domain': domain,
        'noise_type': noise_type,
        'nsi_slope': round(point_slope, 4),
        'nsi_ci_low': round(np.percentile(boot_slopes, 2.5), 4),
        'nsi_ci_high': round(np.percentile(boot_slopes, 97.5), 4),
        'rel_drop_pct': round(point_rel_drop, 2),
        'rel_drop_ci_low': round(np.percentile(rel_drops, 2.5), 2),
        'rel_drop_ci_high': round(np.percentile(rel_drops, 97.5), 2),
        'f1_clean': round(mean_f1[0], 4),
        'f1_at_40': round(mean_f1[-1], 4),
    })

boot_df = pd.DataFrame(bootstrap_records).sort_values(['domain', 'noise_type', 'ensemble'])
boot_df.to_csv(OUTPUT_DIR / 'nsi_with_bootstrap_ci.csv', index=False)

ratio_records = []
for ensemble in ensembles_order:
    for noise_type in ['symmetric', 'asymmetric']:
        tab = boot_df[(boot_df['ensemble'] == ensemble) & (boot_df['domain'] == 'Tabular') &
                      (boot_df['noise_type'] == noise_type)]
        img = boot_df[(boot_df['ensemble'] == ensemble) & (boot_df['domain'] == 'Image') &
                      (boot_df['noise_type'] == noise_type)]
        if len(tab) == 1 and len(img) == 1 and img['rel_drop_pct'].values[0] > 0.1:
            ratio_records.append({
                'ensemble': ensemble,
                'noise_type': noise_type,
                'tab_rel_drop': tab['rel_drop_pct'].values[0],
                'img_rel_drop': img['rel_drop_pct'].values[0],
                'ratio': round(tab['rel_drop_pct'].values[0] / img['rel_drop_pct'].values[0], 2),
            })

ratio_df = pd.DataFrame(ratio_records)
ratio_df.to_csv(OUTPUT_DIR / 'cross_domain_ratios.csv', index=False)

ensemble_pairs = list(combinations(ensembles_order, 2))
effect_records = []
for domain in ['Tabular', 'Image']:
    for noise_type in ['symmetric', 'asymmetric']:
        for ens1, ens2 in ensemble_pairs:
            scores1 = df[(df['ensemble'] == ens1) & (df['domain'] == domain) &
                         (df['noise_type'] == noise_type)]['f1_macro'].values
            scores2 = df[(df['ensemble'] == ens2) & (df['domain'] == domain) &
                         (df['noise_type'] == noise_type)]['f1_macro'].values
            if len(scores1) != len(scores2) or len(scores1) == 0:
                continue
            try:
                stat, p_value = stats.wilcoxon(scores1, scores2)
            except Exception:
                stat, p_value = np.nan, np.nan
            mean_diff = scores1.mean() - scores2.mean()
            diff = scores1 - scores2
            nonzero = diff[diff != 0]
            if len(nonzero) > 0:
                ranks = stats.rankdata(np.abs(nonzero))
                W_pos = ranks[nonzero > 0].sum()
                W_neg = ranks[nonzero < 0].sum()
                rank_biserial = (W_pos - W_neg) / (W_pos + W_neg) if (W_pos + W_neg) > 0 else 0
            else:
                rank_biserial = 0
            n_greater = np.sum(scores1[:, None] > scores2[None, :])
            n_less = np.sum(scores1[:, None] < scores2[None, :])
            cliffs_delta = (n_greater - n_less) / (len(scores1) * len(scores2))
            abs_r = abs(rank_biserial)
            if abs_r < 0.1:
                magnitude = 'negligible'
            elif abs_r < 0.3:
                magnitude = 'small'
            elif abs_r < 0.5:
                magnitude = 'medium'
            else:
                magnitude = 'large'
            effect_records.append({
                'domain': domain,
                'noise_type': noise_type,
                'ensemble_1': ens1,
                'ensemble_2': ens2,
                'mean_diff': round(mean_diff, 4),
                'wilcoxon_p': round(p_value, 6) if not np.isnan(p_value) else None,
                'rank_biserial_r': round(rank_biserial, 3),
                'cliffs_delta': round(cliffs_delta, 3),
                'effect_magnitude': magnitude,
                'better': ens1 if mean_diff > 0 else ens2,
            })

effect_df = pd.DataFrame(effect_records)
effect_df.to_csv(OUTPUT_DIR / 'wilcoxon_with_effect_sizes.csv', index=False)

latex_lines = [
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"\textbf{Domain} & \textbf{Method} & \textbf{Noise type} & \textbf{NSI [95\% CI]} & \textbf{Rel.\ drop \% [95\% CI]} \\",
    r"\midrule",
]
for domain in ['Tabular', 'Image']:
    for ensemble in ensembles_order:
        for noise_type in ['symmetric', 'asymmetric']:
            r = boot_df[(boot_df['ensemble'] == ensemble) & (boot_df['domain'] == domain) &
                        (boot_df['noise_type'] == noise_type)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            nsi_str = f"${r['nsi_slope']:+.3f}$ $[{r['nsi_ci_low']:+.3f}, {r['nsi_ci_high']:+.3f}]$"
            rd_str = f"${r['rel_drop_pct']:.1f}$ $[{r['rel_drop_ci_low']:.1f}, {r['rel_drop_ci_high']:.1f}]$"
            latex_lines.append(f"{domain} & {ensemble} & {noise_type} & {nsi_str} & {rd_str} \\\\")
    latex_lines.append(r"\midrule")
latex_lines.append(r"\bottomrule")
latex_lines.append(r"\end{tabular}")
with open(OUTPUT_DIR / 'nsi_latex_table.tex', 'w') as f:
    f.write('\n'.join(latex_lines))

# FIGURE 4: NSI ranking with bootstrap CIs
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
COLOR_TABULAR = '#5E81AC'
COLOR_IMAGE = '#D08770'
panel_titles = {'symmetric': '(a) Symmetric noise', 'asymmetric': '(b) Cyclic asymmetric noise'}

for ax_idx, noise_type in enumerate(['symmetric', 'asymmetric']):
    ax = axes[ax_idx]
    sub = boot_df[boot_df['noise_type'] == noise_type].copy()
    sub['label'] = sub['ensemble'] + ' (' + sub['domain'] + ')'
    sub = sub.sort_values('nsi_slope', ascending=True).reset_index(drop=True)
    y_positions = np.arange(len(sub))
    bar_colors = [COLOR_TABULAR if d == 'Tabular' else COLOR_IMAGE for d in sub['domain'].values]
    ax.barh(y_positions, sub['nsi_slope'].values, color=bar_colors,
            edgecolor='white', linewidth=0.8, height=0.65, zorder=2)
    for k, val in enumerate(sub['nsi_slope'].values):
        ax.text(val - 0.025, k, f'{val:.3f}', va='center', ha='right',
                fontsize=10, color='#222222')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sub['label'].values, fontsize=10.5)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.set_title(panel_titles[noise_type], loc='left', fontsize=12, pad=10)
    ax.set_xlabel('NSI slope', fontsize=11, labelpad=8)
    ax.axvline(x=0, color='#333333', linewidth=0.8, zorder=1)
    ax.set_xlim([-1.0, 0.05])
    ax.set_xticks(np.arange(-1.0, 0.06, 0.2))
    ax.tick_params(axis='x', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(facecolor=COLOR_TABULAR, edgecolor='white', label='Tabular'),
    mpatches.Patch(facecolor=COLOR_IMAGE, edgecolor='white', label='Image'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=11, columnspacing=2.5, handletextpad=0.6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.14, wspace=0.35)
plt.savefig(FIG_DIR / 'fig4_nsi_ranking.pdf', format='pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig4_nsi_ranking.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(FIG_DIR / 'Fig4.tif', format='tiff', dpi=600, bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)
