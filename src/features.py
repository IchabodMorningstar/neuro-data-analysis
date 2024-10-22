import pandas as pd
import numpy as np

colors = ['#FF0000', '#FFA500', '#FFFF00', '#008000', '#00FFFF', '#0000FF', '#800080', '#FF00FF']

def gt_n_trials(inputs, **kwargs):
    n = kwargs['n']
    df = inputs['raw_df']
    cells = inputs['cells']
    num_trials = df.groupby(['cell', 'position']).ts.transform('count')
    min_trials = num_trials.groupby('cell').agg('min')
    min_trials.name = 'min_trials'
    cells = cells.join(min_trials)
    cells = cells[cells['min_trials'] >= n]
    # return {'raw_df': df[(num_trials >= n).groupby(['cell']).transform('all')].copy(), 'cells': cells}
    return {'raw_df': df.loc[cells.index], 'cells': cells.copy()}

def _wins(win_start, win_end, win_size, win_stride):
    return [
        [win_start + i * win_stride, win_start + i * win_stride + win_size]
        for i in range((win_end - win_size - win_start) // win_stride + 1)
    ]

# def _ts_win_cts(ts, win_start, win_end, win_size, win_stride, calc_rate):
#     assert (win_end - win_size - win_start) % win_stride == 0

#     wins = [
#         [win_start + i * win_stride, win_start + i * win_stride + win_size]
#         for i in range((win_end - win_size - win_start) // win_stride + 1)
#     ]
#     return wins, [
#         ts[(ts * 1000 >= win[0]) & (ts * 1000 < win[1])].shape[0] / (win_size / 1000 if calc_rate else 1)
#         for win in wins
#     ]

def _ts_win_cts(ts, wins, calc_rate):
    return [
        ts[
            (ts * 1000 >= win_start)
            & (ts * 1000 < win_end)
        ].shape[0] / ((win_end - win_start) / 1000 if calc_rate else 1)
        for win_start, win_end in wins
    ]

def win_cts_by_trial(inputs, **kwargs):
    df = inputs['raw_df'].copy()
    calc_rate = kwargs.get('calc_rate', False)
    if 'wins' in kwargs:
        wins = kwargs['wins']
    else:
        wins = _wins(*[kwargs[k] for k in ['win_start', 'win_end', 'win_size', 'win_stride']])

    df['win_cts'] = df['ts'].apply(lambda x: _ts_win_cts(x, wins, calc_rate))
    df = df.drop(columns=['ts'])
    return {'win_cts_by_trial': df, 'wins': wins}

    # win_start, win_end, win_size, win_stride = [kwargs[k] for k in ['win_start', 'win_end', 'win_size', 'win_stride']]

    # df['win_cts'] = df['ts'].apply(lambda x: _ts_win_cts(x, win_start, win_end, win_size, win_stride, calc_rate))
    # df = df.drop(columns=['ts'])

    # return {'win_cts_by_trial': df}

def _ts_win_isi(ts, wins):
    isi_d = np.diff(ts)
    isi_t = ts[:-1] + isi_d / 2
    return [
        isi_d[(isi_t * 1000 >= win_start) & (isi_t * 1000 < win_end)].tolist()
        for win_start, win_end in wins
    ]

def win_isi_by_trial(inputs, **kwargs):
    df = inputs['raw_df'].copy()
    wins = _wins(*[kwargs[k] for k in ['win_start', 'win_end', 'win_size', 'win_stride']]) if 'wins' not in kwargs else kwargs['wins']
    df['win_isi'] = df['ts'].apply(lambda x: _ts_win_isi(x, wins))
    df = df.drop(columns=['ts'])
    return {'win_isi_by_trial': df}

def win_cts_by_win(inputs):
    df = inputs['win_cts_by_trial']
    # cells = df.groupby('cell')[['monkey', 'area']].agg('first')
    # min_trials = df.groupby(['cell', 'position']).win_cts.transform('count').groupby('cell').agg('min')
    # min_trials.name = 'min_trials'
    # cells = cells.join(min_trials)
    wdf = df.explode('win_cts')
    wdf['win'] = wdf.groupby(['cell', 'position', 'trial']).cumcount()
    wdf = wdf.set_index('win', append=True)
    return {'win_cts': wdf, 'win_cts_by_win': wdf.groupby(['cell', 'position', 'win']).win_cts.apply(lambda x: x.tolist())}

def win_isi_by_win(inputs):
    df = inputs['win_isi_by_trial']
    wdf = df.explode('win_isi')
    wdf['win'] = wdf.groupby(['cell', 'position', 'trial']).cumcount()
    wdf = wdf.set_index('win', append=True)
    return {'win_isi': wdf, 'win_isi_by_win': wdf.groupby(['cell', 'position', 'win']).win_isi.apply(lambda x: x.tolist())}

# def young(inputs):
#     return _subset(inputs, lambda x: x.monkey == 'YOUNG')

# def pfc(inputs):
#     return _subset(inputs, lambda x: x.area == 'PFC')

def subset(inputs, **kwargs):
    cells = inputs['cells']
    df = inputs['raw_df'].join(cells)
    if (kwargs['key'] == 'cell'):
        df = df.iloc[df.index.get_level_values('cell') == kwargs['value']]
        return {'raw_df': df.drop(columns=['monkey', 'area', 'subregion', 'phase', 'min_trials'], errors='ignore'),
                'cells': cells[cells.index == kwargs['value']]}
    else:
        df = df[df[kwargs['key']] == kwargs['value']]
        return {'raw_df': df.drop(columns=['monkey', 'area', 'subregion', 'phase', 'min_trials'], errors='ignore'), 
                'cells': cells[cells[kwargs['key']] == kwargs['value']]}

def exclude(inputs, **kwargs):
    cells = inputs['cells']
    df = inputs['raw_df'].join(cells)
    df = df[df[kwargs['key']] != kwargs['value']]
    return {'raw_df': df.drop(columns=['monkey', 'area']), 'cells': cells[cells[kwargs['key']] != kwargs['value']]}

def _select_by_idx(df, idx):
    orig_idx_cols = df.index.names
    idx = idx.copy().droplevel([c for c in idx.names if c not in orig_idx_cols])
    df = df.reset_index().set_index(idx.names)
    return df.loc[idx].reset_index().set_index(orig_idx_cols)

def select(inputs, **kwargs):
    if 'wins' in kwargs:
        print(inputs['win_cts_by_win'])
        indx = inputs['win_cts_by_win'].loc[:, :, kwargs['wins']].index
    else:
        indx = inputs['idx']
    raw_df = inputs['raw_df']
    cell_indx = indx.get_level_values('cell').unique()
    outputs = {'raw_df': _select_by_idx(raw_df, indx), 'cells': inputs['cells'].loc[cell_indx]}

    if 'win_cts' in inputs:
        win_cts = outputs['win_cts'] = _select_by_idx(inputs['win_cts'], indx)
        outputs['win_cts_by_trial'] = pd.DataFrame(win_cts.groupby(['cell', 'position', 'trial']).win_cts.apply(lambda x: x.tolist()))
        outputs['win_cts_by_win'] = win_cts.groupby(['cell', 'position', 'win']).win_cts.apply(lambda x: x.tolist())

    if 'stats_by_cell_win' in inputs:
        stats_by_cell_win_idx_cols = inputs['stats_by_cell_win'].index.names
        indx_wo_pos = indx.copy()
        if 'position' in indx_wo_pos.names:
            indx_wo_pos = indx_wo_pos.droplevel('position')
        outputs['stats_by_cell_win'] = inputs['stats_by_cell_win'].reset_index().set_index(indx_wo_pos.names).loc[indx_wo_pos].reset_index().set_index(stats_by_cell_win_idx_cols)

    if 't_by_cell_win_pos' in inputs:
        if indx.names == ['cell', 'position']:
            outputs['t_by_cell_win_pos'] = inputs['t_by_cell_win_pos'].reset_index().set_index(['cell', 'position']).loc[indx].reset_index().set_index(['cell', 'win', 'position'])
        else:
            outputs['t_by_cell_win_pos'] = _select_by_idx(inputs['t_by_cell_win_pos'], indx)

    return outputs