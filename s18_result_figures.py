import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

OUTPUT_DIR = Path('/kaggle/working/manuscript_figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_file(fname):
    hits = sorted(Path('/kaggle/input').rglob(fname))
    assert hits, f"{fname} not found - attach the Outputs data"
    return hits[0]


plt.rcParams['font.size'] = 9
plt.rcParams['pdf.fonttype'] = 42

SOURCES = {
    ('Tabular', 'Random Forest'): ['rf_summary.csv'],
    ('Tabular', 'XGBoost'): ['xgb_summary.csv'],
    ('Tabular', 'Voting'): ['voting_summary.csv'],
    ('Tabular', 'Stacking'): ['stacking_summary.csv'],
    ('Image', 'Random Forest'): ['rf_image_summary.csv'],
    ('Image', 'XGBoost'): ['xgb_image_summary.csv'],
    ('Image', 'Voting'): ['voting_image_summary.csv'],
    ('Image', 'Stacking'): ['stacking_symmetric_summary.csv', 'stacking_asymmetric_summary.csv'],
}
data = {}
for (rep, method), fnames in SOURCES.items():
    df = pd.concat([pd.read_csv(find_file(f)) for f in fnames], ignore_index=True)
    assert 'f1_macro_mean' in df.columns and 'noise_type' in df.columns and 'noise_level' in df.columns
    data[(rep, method)] = df

METHOD_ORDER = ['Random Forest', 'XGBoost', 'Voting', 'Stacking']
CMAPS = {'Random Forest': plt.cm.Blues, 'XGBoost': plt.cm.Oranges, 'Voting': plt.cm.Greens, 'Stacking': plt.cm.Purples}
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey='row')
panels = [('Tabular', 'symmetric', axes[0, 0], '(a) Tabular, symmetric'),
          ('Tabular', 'asymmetric', axes[0, 1], '(b) Tabular, cyclic asymmetric'),
          ('Image', 'symmetric', axes[1, 0], '(c) Image, symmetric'),
          ('Image', 'asymmetric', axes[1, 1], '(d) Image, cyclic asymmetric')]
bar_w = 0.16
for rep, nt, ax, title in panels:
    for mi, method in enumerate(METHOD_ORDER):
        df = data[(rep, method)]
        sub = df[df['noise_type'] == nt].drop_duplicates(subset='noise_level').set_index('noise_level')
        shades = CMAPS[method](np.linspace(0.35, 0.95, len(LEVELS)))
        for li, lv in enumerate(LEVELS):
            if lv in sub.index:
                val = float(sub.loc[lv, 'f1_macro_mean'])
            else:
                clean = df[df['noise_level'] == 0.0]
                assert len(clean), f"no clean row for {rep} {method}"
                val = float(clean['f1_macro_mean'].iloc[0])
            ax.bar(mi + (li - 2) * bar_w, val, width=bar_w, color=shades[li], edgecolor='black', linewidth=0.3)
    ax.set_xticks(range(len(METHOD_ORDER)))
    ax.set_xticklabels(METHOD_ORDER, fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(title, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
axes[0, 0].set_ylabel('Macro F1')
axes[1, 0].set_ylabel('Macro F1')
legend_handles = [Patch(facecolor=plt.cm.Greys(0.35 + 0.15 * i), edgecolor='black', label=f'{int(lv * 100)}%') for i, lv in enumerate(LEVELS)]
axes[0, 1].legend(handles=legend_handles, title='Noise level', fontsize=7, title_fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig2.pdf', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'Fig2.png', dpi=300, bbox_inches='tight')
plt.close()

kg = pd.read_csv(find_file('kappa_vs_gap_dedup.csv'))
assert len(kg) == 20, f"expected 20 dedup conditions, got {len(kg)}"
fig, ax = plt.subplots(figsize=(7, 5))
styles = {'Tabular': dict(color='steelblue', marker='o'), 'Image': dict(color='darkorange', marker='s')}
for dom, g in kg.groupby('domain'):
    g = g.sort_values('kappa_mean')
    ax.plot(g['kappa_mean'], g['gap'], linestyle=':', linewidth=1, color=styles[dom]['color'], alpha=0.7)
    ax.scatter(g['kappa_mean'], g['gap'], s=45, label=dom, color=styles[dom]['color'], marker=styles[dom]['marker'], edgecolor='black', linewidth=0.4, zorder=3)
ax.axhline(0, color='grey', linewidth=0.8)
ax.set_xlabel("Mean pairwise Cohen's kappa")
ax.set_ylabel('Stacking minus voting macro F1')
ax.legend(title='Representation')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig3.pdf', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'Fig3.png', dpi=300, bbox_inches='tight')
plt.close()

ranks = pd.read_csv(find_file('avg_ranks_by_noise_type.csv'))
METHODS = ['RF', 'XGB', 'ET', 'Voting', 'PlainStack', 'DAS', 'CL_XGB', 'GCE_XGB']
LABELS = {'RF': 'Random Forest', 'XGB': 'XGBoost', 'ET': 'Extra Trees', 'Voting': 'Voting', 'PlainStack': 'Plain stacking', 'DAS': 'DAS', 'CL_XGB': 'Confident learning', 'GCE_XGB': 'GCE XGBoost'}
structured = ranks[ranks['noise_type'] == 'structured_combined'][METHODS].iloc[0]
allc = ranks[ranks['noise_type'] == 'all_common_conditions'][METHODS].iloc[0]
order = list(structured.sort_values().index)
x = np.arange(len(order))
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(x - 0.2, [structured[m] for m in order], width=0.4, color='crimson', edgecolor='black', linewidth=0.4, label='Structured noise (40 blocks)')
ax.bar(x + 0.2, [allc[m] for m in order], width=0.4, color='steelblue', edgecolor='black', linewidth=0.4, label='All conditions (80 blocks)')
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in order], fontsize=8, rotation=20, ha='right')
ax.set_ylabel('Average rank (lower is better)')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'Fig4.pdf', bbox_inches='tight')
plt.savefig(OUTPUT_DIR / 'Fig4.png', dpi=300, bbox_inches='tight')
plt.close()
