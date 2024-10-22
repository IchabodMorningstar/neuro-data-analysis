from scipy.stats import ttest_ind, kruskal, f_oneway
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
from pandas import IndexSlice as idx
from scipy.stats import ttest_rel

def stat_tests(inputs, **kwargs):
    # cells = inputs['cells']
    w_counts = inputs['win_cts_by_win']
    num_positions = w_counts.index.get_level_values('position').nunique()

    def safe_kruskal(*args):
        try: return kruskal(*args)
        except: return [np.nan, np.nan]

    kruskal_p = w_counts.groupby(['cell', 'win']).apply(lambda x: safe_kruskal(*x)[1])
    kruskal_p.name = 'kruskal_p'

    anova_p = w_counts.groupby(['cell', 'win']).apply(lambda x: f_oneway(*x)[1])
    anova_p.name = 'anova_p'

    def ttests(x):
        results = []
        for p in range(len(x)):
            if kwargs.get('opposite', False):
                others = np.concatenate([x.iloc[(p + num_positions / 2) % num_positions]])
                if np.mean(x.iloc[p]) < others.mean():
                    results.append(999)
                    continue
            else:
                others = np.concatenate([x.iloc[i] for i in range(len(x)) if i != p])
            results.append(ttest_ind(x.iloc[p], others, equal_var=True).pvalue)

        return [999 if np.isnan(r) else r for r in results]

    def ttests_for_cell(w_counts, cell):
        return w_counts.loc[([cell],),].groupby(['cell', 'win']).apply(ttests)

    # tstats = Parallel(n_jobs=8)([delayed(ttests_for_cell)(w_counts, cell) for cell in w_counts.index.levels[0]])
    tstats = [ttests_for_cell(w_counts, cell) for cell in w_counts.index.levels[0]]
    tstats = pd.concat(tstats)
    tstats.name = 't_p'

    stats_by_cell_win = pd.DataFrame(tstats)
    t_by_cell_win_pos = stats_by_cell_win.explode('t_p')
    t_by_cell_win_pos['position'] = t_by_cell_win_pos.groupby(['cell', 'win']).cumcount()
    t_by_cell_win_pos = t_by_cell_win_pos.set_index('position', append=True)
    t_by_cell_win_pos.t_p = t_by_cell_win_pos.t_p.astype(float)

    wcgb = w_counts.groupby(['cell', 'win'])
    stdevs = pd.DataFrame({'stdev': wcgb.apply(lambda x: np.std(np.hstack(x))), 'grandmean': wcgb.apply(lambda x: np.mean(np.hstack(x)))})
    mean = w_counts.apply(lambda x: np.mean(x))
    means = pd.DataFrame({'cts': w_counts, 'mean': mean}).reset_index().set_index(['cell', 'win']).join(stdevs)
    means['stdevs'] = (means['mean'] - means.grandmean) / means.stdev
    means = means.drop(['cts', 'mean', 'stdev', 'grandmean'], axis=1).reset_index().set_index(['cell', 'win', 'position'])
    t_by_cell_win_pos = t_by_cell_win_pos.join(means)

    stats_by_cell_win['min_t_p'] = stats_by_cell_win.t_p.apply(lambda x: np.min(x))
    stats_by_cell_win['min_t_p_pos'] = stats_by_cell_win.t_p.apply(lambda x: np.argmin(x))
    stats_by_cell_win = stats_by_cell_win.join(anova_p)
    stats_by_cell_win = stats_by_cell_win.join(kruskal_p)

    return {'t_by_cell_win_pos': t_by_cell_win_pos, 'stats_by_cell_win': stats_by_cell_win}

def top_n_by_cell_win_t(inputs, **kwargs):
    return top_n_by_x(
        lambda stats_by_cell_win, t_by_cell_win_pos, n: t_by_cell_win_pos.groupby(['position']).tstat.apply(lambda x: x.nlargest(n)).index.droplevel([0, 3]),
        inputs, **kwargs)

def top_n_by_cell_t(inputs, **kwargs):
    return top_n_by_x(
        lambda stats_by_cell_win, t_by_cell_win_pos, n: t_by_cell_win_pos.groupby(['position']).tstat.apply(lambda x: x.nlargest(n)).index.droplevel([0, 2, 3]),
        inputs, **kwargs)

def top_n_by_cell_kruskal(inputs, **kwargs):
    return top_n_by_x(
        lambda stats_by_cell_win, t_by_cell_win_pos, n: stats_by_cell_win.groupby('min_tstat_pos').kruskal_p.apply(lambda x: x.nsmallest(n)).index.droplevel([0, 2]),
        inputs, **kwargs)

def top_n_by_cell_win_kruskal(inputs, **kwargs):
    return top_n_by_x(
        lambda stats_by_cell_win, t_by_cell_win_pos, n: stats_by_cell_win.groupby('min_tstat_pos').kruskal_p.apply(lambda x: x.nsmallest(n)).index.droplevel([0]),
        inputs, **kwargs)

def top_n_by_cell_anova(inputs, **kwargs):
    return top_n_by_x(
        lambda stats_by_cell_win, t_by_cell_win_pos, n: stats_by_cell_win.groupby('min_tstat_pos').anova_p.apply(lambda x: x.nsmallest(n)).index.droplevel([0, 2]),
        inputs, **kwargs)

def top_n_by_cell_win_anova(inputs, **kwargs):
    return top_n_by_x(
        lambda stats_by_cell_win, t_by_cell_win_pos, n: stats_by_cell_win.groupby('min_tstat_pos').anova_p.apply(lambda x: x.nsmallest(n)).index.droplevel([0]),
        inputs, **kwargs)

def top_n_across_pos_by_cell_anova(inputs, **kwargs):
    return top_n_by_x(
        lambda stats_by_cell_win, t_by_cell_win_pos, n: stats_by_cell_win.groupby('min_tstat_pos').anova_p.apply(lambda x: x.nsmallest(n)).index.droplevel([0, 2]),
        inputs, **kwargs)

def top_n_by_x(idx_func, inputs, **kwargs):
    n = kwargs['n']
    t_by_cell_win_pos = inputs['t_by_cell_win_pos']
    stats_by_cell_win = inputs['stats_by_cell_win']
    win_cts_by_win = inputs['win_cts_by_win']
    win_cts = inputs['win_cts']
    tidx = idx_func(stats_by_cell_win, t_by_cell_win_pos, n)
    win_cts_by_win = pd.DataFrame(win_cts_by_win).reset_index().set_index(['cell', 'win'])
    win_cts_by_win = win_cts_by_win.loc[tidx].set_index('position', append=True).win_cts
    win_cts = win_cts.reset_index().set_index(['cell', 'win'])
    win_cts = win_cts.loc[tidx].reset_index().set_index(['cell', 'position', 'trial', 'win'])
    win_cts_by_trial = pd.DataFrame(win_cts.groupby(['cell', 'position', 'trial']).win_cts.apply(lambda x: x.tolist()))
    idx = win_cts_by_win.index.get_level_values('cell').unique()
    return {'win_cts_by_win': win_cts_by_win, 'win_cts': win_cts, 'win_cts_by_trial': win_cts_by_trial, 'idx': idx}
    # return {'win_cts_by_win': win_cts_by_win.loc[idx[tidx.get_level_values('cell'), :, tidx.get_level_values('win')]],
    #         'win_cts': win_cts.loc[idx[tidx.get_level_values('cell'), :, :, tidx.get_level_values('win')]]}


# def top_n(inputs, **kwargs):
#     """
#     Arguments:
#         by_pos  - if True, return top n for each position, otherwise return top n across all positions
#         by_win  - if True, return top cell/windows, otherwise return top cells (based on min/max stat)
#     """
#     n = kwargs['n']
#     by_pos = kwargs['by_pos']
#     stat = kwargs['stat']
#     by_win = kwargs['by_win']

#     gby = ['cell']
#     if by_win: gby.append('win')

#     if stat == 't':
#         if by_pos: gby.append('position')
#         t_by_cell_win_pos = inputs['t_by_cell_win_pos'].groupby(gby).tstat.max()

#         gby.remove('cell')
#         if len(gby) == 0: idx = t_by_cell_win_pos.nlargest(n).index
#         else: idx = inputs['t_by_cell_win_pos'].groupby(gby).tstat.apply(lambda x: x.nlargest(n)).index

#         return {'idx': idx}
#     else:
#         stats_by_cell_win = inputs['stats_by_cell_win']
#         if by_pos: gby.append('min_tstat_pos')
#         stats_by_cell_win = stats_by_cell_win.groupby(gby)
#         if stat == 'anova':
#             stats_by_cell_win = stats_by_cell_win.anova_p.min()
#         elif stat == 'kruskal':
#             stats_by_cell_win = stats_by_cell_win.kruskal_p.min()

#         return {}
    
def sig_stat(inputs, **kwargs):
    stats_by_cell_win = inputs['stats_by_cell_win']
    keep_win = kwargs.get('keep_win', False)
    if 'win' in kwargs:
        win = kwargs['win']
        indx = stats_by_cell_win.loc[idx[:,win],][kwargs['stat']] < kwargs.get('alpha', 0.05)
        return {'idx': indx[indx].index.droplevel(1).unique()} if not keep_win else {'idx': indx[indx].index}
    else:
        indx = stats_by_cell_win.anova_p < kwargs.get('alpha', 0.05)
        if not keep_win: indx = indx.groupby('cell').agg('any')
        return {'idx': indx[indx].index}

# def sig_t(inputs, **kwargs):
#     t_by_cell_win_pos = inputs['t_by_cell_win_pos']
#     win = kwargs.get('win', 0)
#     indx = t_by_cell_win_pos.loc[idx[:,win],].t_p < kwargs.get('alpha', 0.05)
#     print(indx)
#     return {'idx': indx[indx].index.droplevel(1)}

def t_btw_wins(inputs, **kwargs):
    wcbwdf = inputs['win_cts_by_win']
    ttestpv = wcbwdf.groupby(['cell', 'position']).agg(lambda x: ttest_rel(*x)[1]).rename('ttestpv')
    test = ttestpv.groupby('cell').agg('min') < kwargs.get('alpha', 0.05)
    test = test[test]
    return {'idx': test.index, 'ttestpv': ttestpv}

def min_stdevs(inputs, **kwargs):
    t_by_cell_win_pos = inputs['t_by_cell_win_pos']
    indx = t_by_cell_win_pos.stdevs >= kwargs['min']
    # if not kwargs.get('keep_win', False):
    #     indx = indx[indx].index.droplevel(1).unique()
    # else:
    #     indx = indx[indx].index
    # if not kwargs.get('keep_position', False):
    #     indx = indx.droplevel(indx.names.index('position')).unique()
    return {'idx': indx[indx].index}

def reduce_index(inputs, **kwargs):
    levels_to_keep = kwargs['levels']
    idx = inputs['idx']
    return {'idx': idx.droplevel([i for i in idx.names if i not in levels_to_keep])}