import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path('/kaggle/working/final_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path('/kaggle/working/figures_final')
FIG_DIR.mkdir(parents=True, exist_ok=True)

def find_file(fname):
    hits = sorted(Path('/kaggle/input').rglob(fname))
    assert hits, f"{fname} not found - attach the required input"
    return hits[0]

gate_tab = pd.read_csv(find_file('gate_results_full.csv'))
gate_img = pd.read_csv(find_file('gate_results_full_image.csv'))
cl = pd.read_csv(find_file('cleanlab_results_full.csv'))
gce = pd.read_csv(find_file('robustloss_results_full.csv'))
assert len(gate_tab) == 300 and len(gate_img) == 115, "gate row counts wrong"
assert len(cl) == 415 and len(gce) == 415, "baseline row counts wrong - attach the FINAL 415-unit outputs"

GATE_METHODS = {'f1_RF': 'RF', 'f1_XGB': 'XGB', 'f1_ET': 'ET', 'f1_Voting': 'Voting', 'f1_PlainStack': 'PlainStack', 'f1_DAS': 'DAS'}
long_rows = []
for g in [gate_tab, gate_img]:
    m = g.melt(id_vars=['dataset', 'condition', 'fold'], value_vars=list(GATE_METHODS.keys()), var_name='mcol', value_name='f1')
    m['method'] = m['mcol'].map(GATE_METHODS)
    long_rows.append(m[['dataset', 'condition', 'fold', 'method', 'f1']])
clm = cl.rename(columns={'f1_CL_XGB': 'f1'})[['dataset', 'condition', 'fold', 'f1']].copy()
clm['method'] = 'CL_XGB'
gcm = gce.rename(columns={'f1_GCE_XGB': 'f1'})[['dataset', 'condition', 'fold', 'f1']].copy()
gcm['method'] = 'GCE_XGB'
master = pd.concat(long_rows + [clm, gcm], ignore_index=True)
master['noise_type'] = master['condition'].str.rsplit('_', n=1).str[0]
master['noise_level'] = master['condition'].str.rsplit('_', n=1).str[1].astype(int)

METHODS = ['RF', 'XGB', 'ET', 'Voting', 'PlainStack', 'DAS', 'CL_XGB', 'GCE_XGB']
units = master.groupby(['dataset', 'condition', 'fold'])['method'].nunique()
assert (units == 8).all(), f"units missing methods: {units[units != 8]}"
assert len(master) == 415 * 8, f"expected {415*8} rows, got {len(master)}"
master.to_csv(OUTPUT_DIR / 'final_master_per_fold.csv', index=False)

summary = master.groupby(['dataset', 'condition', 'method'])['f1'].mean().reset_index()
wide = summary.pivot_table(index=['dataset', 'condition'], columns='method', values='f1')[METHODS].round(4).reset_index()
wide.to_csv(OUTPUT_DIR / 'final_summary_by_condition.csv', index=False)

DATASETS = sorted(master['dataset'].unique())
COMMON_CONDS = ['clean_0', 'symmetric_10', 'symmetric_20', 'symmetric_30', 'symmetric_40', 'asymmetric_10', 'asymmetric_20', 'asymmetric_30', 'asymmetric_40', 'pairflip_40']
for c in COMMON_CONDS:
    have = set(wide[wide['condition'] == c]['dataset'])
    assert have == set(DATASETS), f"{c}: missing datasets {set(DATASETS)-have}"

Q_NEMENYI = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
k = len(METHODS)
N = len(DATASETS)
CD_per_condition = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6.0 * N))
CD_structured = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6.0 * 40))
CD_all = Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6.0 * 80))

fr_rows = []
rank_frames = []
for cond in COMMON_CONDS:
    piv = summary[summary['condition'] == cond].pivot_table(index='dataset', columns='method', values='f1')[METHODS]
    stat, p = friedmanchisquare(*[piv[m].values for m in METHODS])
    ranks = piv.rank(axis=1, ascending=False)
    ar = ranks.mean()
    fr_rows.append({'condition': cond, 'friedman_chi2': round(float(stat), 3), 'p_value': round(float(p), 6), **{f'rank_{m}': round(float(ar[m]), 2) for m in METHODS}})
    r = ranks.reset_index().melt(id_vars='dataset', var_name='method', value_name='rank')
    r['condition'] = cond
    rank_frames.append(r)
frdf = pd.DataFrame(fr_rows)
frdf.to_csv(OUTPUT_DIR / 'friedman_by_condition.csv', index=False)

allranks = pd.concat(rank_frames, ignore_index=True)
allranks['noise_type'] = allranks['condition'].str.rsplit('_', n=1).str[0]
nt_rows = []
for nt in ['clean', 'symmetric', 'asymmetric', 'pairflip']:
    sub = allranks[allranks['noise_type'] == nt]
    ar = sub.groupby('method')['rank'].mean()
    nt_rows.append({'noise_type': nt, 'n_blocks': sub['dataset'].nunique() * sub['condition'].nunique(), **{m: round(float(ar[m]), 2) for m in METHODS}})
structured = allranks[allranks['noise_type'].isin(['asymmetric', 'pairflip'])]
ar = structured.groupby('method')['rank'].mean()
nt_rows.append({'noise_type': 'structured_combined', 'n_blocks': len(structured) // len(METHODS), **{m: round(float(ar[m]), 2) for m in METHODS}})
ar = allranks.groupby('method')['rank'].mean()
nt_rows.append({'noise_type': 'all_common_conditions', 'n_blocks': len(allranks) // len(METHODS), **{m: round(float(ar[m]), 2) for m in METHODS}})
ntdf = pd.DataFrame(nt_rows)
ntdf.to_csv(OUTPUT_DIR / 'avg_ranks_by_noise_type.csv', index=False)

drop_rows = []
for key in DATASETS:
    sub = wide[wide['dataset'] == key].set_index('condition')
    clean = sub.loc['clean_0']
    for cond in ['symmetric_40', 'asymmetric_40', 'pairflip_40']:
        for m in METHODS:
            drop_rows.append({'dataset': key, 'condition': cond, 'method': m, 'f1_clean': round(float(clean[m]), 4), 'f1_noisy': round(float(sub.loc[cond, m]), 4), 'rel_drop_pct': round(float((clean[m] - sub.loc[cond, m]) / clean[m] * 100), 2)})
drops = pd.DataFrame(drop_rows)
drops.to_csv(OUTPUT_DIR / 'relative_drops_all_methods.csv', index=False)

c10n = wide[wide['dataset'] == 'cifar10'].set_index('condition')
c10n_rows = []
for cond in ['c10n_aggre_0', 'c10n_random1_0', 'c10n_worse_0']:
    for m in METHODS:
        c10n_rows.append({'condition': cond, 'method': m, 'f1': round(float(c10n.loc[cond, m]), 4), 'rel_drop_vs_clean_pct': round(float((c10n.loc['clean_0', m] - c10n.loc[cond, m]) / c10n.loc['clean_0', m] * 100), 2)})
c10ndf = pd.DataFrame(c10n_rows)
c10ndf.to_csv(OUTPUT_DIR / 'cifar10n_verdict_all_methods.csv', index=False)

pf = master.pivot_table(index=['dataset', 'condition', 'fold'], columns='method', values='f1').reset_index()
pf['noise_type'] = pf['condition'].str.rsplit('_', n=1).str[0]
wb_rows = []
for nt in ['symmetric', 'asymmetric', 'pairflip']:
    sub = pf[(pf['noise_type'] == nt)]
    for a, b in [('DAS', 'CL_XGB'), ('DAS', 'GCE_XGB'), ('DAS', 'XGB'), ('PlainStack', 'CL_XGB'), ('PlainStack', 'GCE_XGB'), ('DAS', 'PlainStack')]:
        d = (sub[a] - sub[b]).values
        try:
            stat, p = wilcoxon(d)
        except ValueError:
            stat, p = np.nan, np.nan
        wb_rows.append({'noise_type': nt, 'comparison': f'{a} vs {b}', 'n': len(d), 'mean_delta': round(float(d.mean()), 4), 'wilcoxon_p': round(float(p), 6) if not np.isnan(p) else None})
wbdf = pd.DataFrame(wb_rows)
wbdf.to_csv(OUTPUT_DIR / 'wilcoxon_vs_baselines.csv', index=False)

plt.rcParams['font.size'] = 10
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, nt in zip(axes, ['symmetric', 'asymmetric']):
    sub = summary[summary['condition'].isin([f'{nt}_{l}' for l in [10, 20, 30, 40]] + ['clean_0'])].copy()
    sub['level'] = sub['condition'].apply(lambda c: 0 if c == 'clean_0' else int(c.rsplit('_', 1)[1]))
    curve = sub.groupby(['level', 'method'])['f1'].mean().reset_index()
    for m in METHODS:
        cm = curve[curve['method'] == m].sort_values('level')
        ax.plot(cm['level'], cm['f1'], marker='o', label=m)
    ax.set_xlabel('Noise level (%)')
    ax.set_ylabel('Mean F1 macro over 8 datasets')
    ax.set_title(f'{nt.capitalize()} noise')
    ax.grid(alpha=0.3)
axes[1].legend(loc='lower left', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig18a_degradation_curves.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
plot_nt = ntdf[ntdf['noise_type'].isin(['symmetric', 'asymmetric', 'pairflip'])].set_index('noise_type')[METHODS]
plot_nt.T.plot(kind='bar', ax=ax)
ax.set_ylabel('Average rank (lower = better)')
ax.axhline((len(METHODS) + 1) / 2, color='grey', linestyle='--', linewidth=1)
ax.set_title(f'Average ranks by noise type (Nemenyi CD: {CD_structured:.2f} for 40 structured blocks, {CD_all:.2f} for all 80 blocks)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig18b_avg_ranks.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
hm = drops[drops['condition'] == 'asymmetric_40'].pivot_table(index='dataset', columns='method', values='rel_drop_pct')[METHODS]
sns.heatmap(hm, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Relative F1 drop (%)'})
ax.set_title('Relative F1 drop at 40% cyclic asymmetric noise')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig18c_asym40_drops_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
c10n_piv = c10ndf.pivot_table(index='method', columns='condition', values='rel_drop_vs_clean_pct').loc[METHODS]
c10n_piv.plot(kind='bar', ax=ax)
ax.set_ylabel('Relative F1 drop vs clean (%)')
ax.set_title('CIFAR-10N real human noise: degradation by method')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig18d_cifar10n.png', dpi=300, bbox_inches='tight')
plt.close()
