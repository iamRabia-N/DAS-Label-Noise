import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from itertools import combinations

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path('/kaggle/working/aggregated_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path('/kaggle/working/paper_figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_INPUT = Path('/kaggle/input/datasets/rabianaz22')

result_paths = {
    'RF_Tabular': BASE_INPUT / '3b-tabular-rf-3rd-paper/rf_results/rf_results_full.csv',
    'XGBoost_Tabular': BASE_INPUT / '3c-tabular-xgb-3rd-paper/xgb_results/xgb_results_full.csv',
    'Voting_Tabular': BASE_INPUT / '3d-tabular-voting-3rd-paper/voting_results/voting_results_full.csv',
    'Stacking_Tabular': BASE_INPUT / '3e-tabular-stacking-3rd-paper/stacking_results/stacking_results_full.csv',
    'RF_Image': BASE_INPUT / '4c-image-rf-3rd-paper/rf_image_results/rf_image_results_full.csv',
    'XGBoost_Image': BASE_INPUT / '4d-image-xgb-3rd-paper/xgb_image_results/xgb_image_results_full.csv',
    'Voting_Image': BASE_INPUT / '4e-image-voting-3rd-paper/voting_image_results/voting_image_results_full.csv',
    'Stacking_Image_Sym': BASE_INPUT / '4f-parta-image-stacking-symmetric-3rd-paper/stacking_symmetric_results/stacking_symmetric_results_full.csv',
    'Stacking_Image_Asym': BASE_INPUT / '4f-partb-image-stacking-asymmetric-3rd-paper/stacking_asymmetric_results/stacking_asymmetric_results_full.csv',
}

loaded_data = {}
for name, path in result_paths.items():
    assert path.exists(), f"Missing: {path}"
    loaded_data[name] = pd.read_csv(path)

stacking_img_combined = pd.concat(
    [loaded_data['Stacking_Image_Sym'], loaded_data['Stacking_Image_Asym']],
    ignore_index=True
)
loaded_data['Stacking_Image'] = stacking_img_combined
del loaded_data['Stacking_Image_Sym']
del loaded_data['Stacking_Image_Asym']

master_dfs = []
for name, df in loaded_data.items():
    df = df.copy()
    parts = name.split('_')
    df['ensemble'] = parts[0]
    df['domain'] = parts[1]
    master_dfs.append(df)

master_df = pd.concat(master_dfs, ignore_index=True)
master_df.to_csv(OUTPUT_DIR / 'master_all_results.csv', index=False)

PALETTE_ENSEMBLES = {
    'RF': '#0072B2',
    'XGBoost': '#E69F00',
    'Voting': '#009E73',
    'Stacking': '#CC79A7',
}

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'lines.markeredgewidth': 0.8,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'grid.color': '#cccccc',
    'figure.dpi': 100,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'none',
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc',
    'legend.facecolor': 'white',
})

summary = master_df.groupby(
    ['ensemble', 'domain', 'noise_type', 'noise_level']
).agg(
    f1_macro_mean=('f1_macro', 'mean'),
    f1_macro_std=('f1_macro', 'std'),
    accuracy_mean=('accuracy', 'mean'),
    accuracy_std=('accuracy', 'std'),
    n_folds=('fold', 'count')
).reset_index()
summary = summary.sort_values(['domain', 'ensemble', 'noise_type', 'noise_level']).reset_index(drop=True)
summary.to_csv(OUTPUT_DIR / 'master_summary.csv', index=False)

nsi_records = []
for (ensemble, domain, noise_type), group in summary.groupby(['ensemble', 'domain', 'noise_type']):
    group_sorted = group.sort_values('noise_level')
    f1_clean = group_sorted[group_sorted['noise_level'] == 0.0]['f1_macro_mean'].values[0]
    f1_max_noise = group_sorted[group_sorted['noise_level'] == 0.4]['f1_macro_mean'].values[0]
    noise_levels = group_sorted['noise_level'].values
    f1_values = group_sorted['f1_macro_mean'].values
    slope = np.polyfit(noise_levels, f1_values, 1)[0]
    total_drop = f1_clean - f1_max_noise
    nsi_records.append({
        'ensemble': ensemble,
        'domain': domain,
        'noise_type': noise_type,
        'f1_clean': round(f1_clean, 4),
        'f1_at_40_noise': round(f1_max_noise, 4),
        'absolute_drop': round(total_drop, 4),
        'relative_drop_pct': round((total_drop / f1_clean) * 100, 2),
        'nsi_slope': round(slope, 4),
    })

nsi_df = pd.DataFrame(nsi_records).sort_values(['domain', 'noise_type', 'ensemble']).reset_index(drop=True)
nsi_df.to_csv(OUTPUT_DIR / 'master_nsi_table.csv', index=False)

ensembles_order = ['RF', 'XGBoost', 'Voting', 'Stacking']
domains_order = ['Tabular', 'Image']
noise_types_order = ['symmetric', 'asymmetric']

# FIGURE 1: degradation curves
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharey='row', sharex='col')
markers = {'RF': 'o', 'XGBoost': 's', 'Voting': '^', 'Stacking': 'D'}
panel_labels = [['(a)', '(b)'], ['(c)', '(d)']]

for i, domain in enumerate(domains_order):
    for j, noise_type in enumerate(noise_types_order):
        ax = axes[i, j]
        for ensemble in ensembles_order:
            subset = summary[
                (summary['ensemble'] == ensemble) &
                (summary['domain'] == domain) &
                (summary['noise_type'] == noise_type)
            ].sort_values('noise_level')
            if len(subset) == 0:
                continue
            levels_pct = subset['noise_level'].values * 100
            f1_mean = subset['f1_macro_mean'].values
            f1_std = subset['f1_macro_std'].values
            color = PALETTE_ENSEMBLES[ensemble]
            ax.fill_between(levels_pct, f1_mean - f1_std, f1_mean + f1_std,
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(levels_pct, f1_mean, marker=markers[ensemble], linestyle='-',
                    linewidth=1.6, markersize=6.5, markeredgewidth=0.7,
                    markeredgecolor='white', color=color, label=ensemble)
        domain_label = 'Forest CoverType' if domain == 'Tabular' else 'EuroSAT'
        noise_label = 'symmetric noise' if noise_type == 'symmetric' else 'asymmetric noise'
        ax.set_title(f'{panel_labels[i][j]} {domain_label}, {noise_label}',
                     fontsize=10, fontweight='normal', loc='left', pad=6)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_ylim([0.4, 1.0])
        ax.set_yticks(np.arange(0.4, 1.01, 0.1))
        if i == 1:
            ax.set_xlabel('Noise level (%)')
        if j == 0:
            ax.set_ylabel('F1 macro')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)

handles = [plt.Line2D([0], [0], marker=markers[e], color=PALETTE_ENSEMBLES[e],
                      linestyle='-', markersize=7, markeredgewidth=0.7,
                      markeredgecolor='white', linewidth=1.6, label=e)
           for e in ensembles_order]
fig.legend(handles, ensembles_order, loc='lower center', ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=10, columnspacing=2.0, handletextpad=0.6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.10)
plt.savefig(FIG_DIR / 'fig1_main_degradation_curves.pdf', format='pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig1_main_degradation_curves.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(FIG_DIR / 'Fig1.tif', format='tiff', dpi=600, bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# FIGURE 2: NSI heatmap
fig, ax = plt.subplots(figsize=(9, 4.8))
row_order = [(d, e) for d in ['Tabular', 'Image'] for e in ensembles_order]
col_order = ['symmetric', 'asymmetric']
matrix = np.zeros((len(row_order), len(col_order)))
for i, (d, e) in enumerate(row_order):
    for j, n in enumerate(col_order):
        v = nsi_df[(nsi_df['domain'] == d) & (nsi_df['ensemble'] == e) &
                   (nsi_df['noise_type'] == n)]['nsi_slope'].values
        matrix[i, j] = v[0] if len(v) > 0 else np.nan

im = ax.imshow(matrix, cmap=plt.cm.Blues_r, aspect='auto', vmin=-0.85, vmax=0.0)
ax.set_yticks(np.arange(len(row_order)))
ax.set_yticklabels([f'{e} ({d})' for d, e in row_order], fontsize=11)
ax.set_xticks(np.arange(len(col_order)))
ax.set_xticklabels(['Symmetric', 'Cyclic asymmetric'], fontsize=12)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')
mid_value = -0.4
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if np.isnan(val):
            continue
        txt_color = 'white' if val < mid_value else '#1a1a1a'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=12, color=txt_color)
ax.tick_params(top=False, bottom=False, left=False, right=False)
for spine in ax.spines.values():
    spine.set_edgecolor('#888888')
    spine.set_linewidth(0.6)
cbar = fig.colorbar(im, ax=ax, shrink=0.9, aspect=18, pad=0.04)
cbar.set_label('NSI slope', fontsize=11, labelpad=8)
cbar.ax.tick_params(labelsize=10)
cbar.outline.set_edgecolor('#888888')
cbar.outline.set_linewidth(0.5)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_nsi_heatmap.pdf', format='pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig2_nsi_heatmap.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(FIG_DIR / 'Fig2.tif', format='tiff', dpi=600, bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# FIGURE 3: clean vs 40% comparison bars
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
BAR_COLORS = {'clean': '#3B4252', 'symmetric': '#5E81AC', 'asymmetric': '#BF616A'}
panel_titles = {'Tabular': '(a) Forest CoverType', 'Image': '(b) EuroSAT'}

for ax_idx, domain in enumerate(['Tabular', 'Image']):
    ax = axes[ax_idx]
    conditions = []
    for ensemble in ensembles_order:
        def get_f1(nt, nl):
            v = summary[(summary['ensemble'] == ensemble) & (summary['domain'] == domain) &
                        (summary['noise_type'] == nt) & (summary['noise_level'] == nl)
                        ]['f1_macro_mean'].values
            return v[0] if len(v) > 0 else 0
        conditions.append({'Ensemble': ensemble, 'clean': get_f1('symmetric', 0.0),
                           'symmetric': get_f1('symmetric', 0.4),
                           'asymmetric': get_f1('asymmetric', 0.4)})
    df_plot = pd.DataFrame(conditions)
    x = np.arange(len(ensembles_order)) * 1.8
    width = 0.50
    bars_clean = ax.bar(x - width, df_plot['clean'], width, color=BAR_COLORS['clean'],
                        edgecolor='white', linewidth=0.8, zorder=2)
    bars_sym = ax.bar(x, df_plot['symmetric'], width, color=BAR_COLORS['symmetric'],
                      edgecolor='white', linewidth=0.8, zorder=2)
    bars_asym = ax.bar(x + width, df_plot['asymmetric'], width, color=BAR_COLORS['asymmetric'],
                       edgecolor='white', linewidth=0.8, zorder=2)
    for bars in [bars_clean, bars_sym, bars_asym]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=10, color='#222222')
    ax.set_title(panel_titles[domain], loc='left', fontsize=12, pad=10)
    ax.set_xlabel('Ensemble method', labelpad=8, fontsize=11)
    if ax_idx == 0:
        ax.set_ylabel('F1 macro', labelpad=8, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(ensembles_order, fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylim([0.5, 1.04])
    ax.set_yticks(np.arange(0.5, 1.01, 0.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

legend_handles = [
    mpatches.Patch(facecolor=BAR_COLORS['clean'], edgecolor='white', label='Clean (0% noise)'),
    mpatches.Patch(facecolor=BAR_COLORS['symmetric'], edgecolor='white', label='Symmetric (40%)'),
    mpatches.Patch(facecolor=BAR_COLORS['asymmetric'], edgecolor='white', label='Cyclic asymmetric (40%)'),
]
fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.02), fontsize=11, columnspacing=2.5, handletextpad=0.6)
plt.tight_layout()
plt.subplots_adjust(bottom=0.16)
plt.savefig(FIG_DIR / 'fig3_critical_noise_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig3_critical_noise_comparison.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(FIG_DIR / 'Fig3.tif', format='tiff', dpi=600, bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# Friedman tests
friedman_results = []
for domain in ['Tabular', 'Image']:
    for noise_type in ['symmetric', 'asymmetric']:
        for noise_level in [0.0, 0.1, 0.2, 0.3, 0.4]:
            data_for_test = []
            for ensemble in ensembles_order:
                fold_scores = master_df[
                    (master_df['ensemble'] == ensemble) &
                    (master_df['domain'] == domain) &
                    (master_df['noise_type'] == noise_type) &
                    (master_df['noise_level'] == noise_level)
                ]['f1_macro'].values
                if len(fold_scores) == 5:
                    data_for_test.append(fold_scores)
            if len(data_for_test) == 4:
                stat, p_value = stats.friedmanchisquare(*data_for_test)
                friedman_results.append({
                    'domain': domain,
                    'noise_type': noise_type,
                    'noise_level': noise_level,
                    'friedman_statistic': round(stat, 4),
                    'p_value': round(p_value, 6),
                    'significant_at_0.05': p_value < 0.05
                })

friedman_df = pd.DataFrame(friedman_results)
friedman_df.to_csv(OUTPUT_DIR / 'friedman_tests.csv', index=False)

# Pairwise Wilcoxon tests
wilcoxon_results = []
ensemble_pairs = list(combinations(ensembles_order, 2))
for domain in ['Tabular', 'Image']:
    for noise_type in ['symmetric', 'asymmetric']:
        for ens1, ens2 in ensemble_pairs:
            scores1 = master_df[(master_df['ensemble'] == ens1) & (master_df['domain'] == domain) &
                                (master_df['noise_type'] == noise_type)]['f1_macro'].values
            scores2 = master_df[(master_df['ensemble'] == ens2) & (master_df['domain'] == domain) &
                                (master_df['noise_type'] == noise_type)]['f1_macro'].values
            if len(scores1) == len(scores2) and len(scores1) > 0:
                try:
                    stat, p_value = stats.wilcoxon(scores1, scores2)
                    mean_diff = scores1.mean() - scores2.mean()
                    wilcoxon_results.append({
                        'domain': domain,
                        'noise_type': noise_type,
                        'ensemble_1': ens1,
                        'ensemble_2': ens2,
                        'mean_diff': round(mean_diff, 4),
                        'wilcoxon_statistic': round(stat, 4),
                        'p_value': round(p_value, 6),
                        'better': ens1 if mean_diff > 0 else ens2,
                        'significant_at_0.05': p_value < 0.05
                    })
                except Exception:
                    pass

wilcoxon_df = pd.DataFrame(wilcoxon_results)
wilcoxon_df.to_csv(OUTPUT_DIR / 'wilcoxon_pairwise_tests.csv', index=False)

# Paper-ready table
paper_table_rows = []
for domain in ['Tabular', 'Image']:
    for ensemble in ensembles_order:
        row = {'Domain': domain, 'Ensemble': ensemble}
        for noise_type in ['symmetric', 'asymmetric']:
            for nl in [0.0, 0.2, 0.4]:
                key = f"{noise_type[:3].upper()}{int(nl*100)}"
                subset = summary[
                    (summary['ensemble'] == ensemble) & (summary['domain'] == domain) &
                    (summary['noise_type'] == noise_type) & (summary['noise_level'] == nl)
                ]
                if len(subset) > 0:
                    row[key] = f"{subset['f1_macro_mean'].values[0]:.3f}\u00b1{subset['f1_macro_std'].values[0]:.3f}"
        for noise_type in ['symmetric', 'asymmetric']:
            nsi_row = nsi_df[(nsi_df['ensemble'] == ensemble) & (nsi_df['domain'] == domain) &
                             (nsi_df['noise_type'] == noise_type)]
            if len(nsi_row) > 0:
                row[f'NSI_{noise_type[:3]}'] = f"{nsi_row['nsi_slope'].values[0]:.3f}"
        paper_table_rows.append(row)

paper_table = pd.DataFrame(paper_table_rows)
paper_table.to_csv(OUTPUT_DIR / 'paper_ready_table.csv', index=False)
