from joblib import Parallel, delayed
import numpy as np
import torch
import pandas as pd

def tvt_split(inputs, **kwargs):
    num_splits = kwargs['num_splits']
    num_val = kwargs['num_val']
    num_test = kwargs['num_test']
    win_df = inputs['win_cts_by_trial']

    if win_df.empty:
        raise ValueError('No data to split')
    
    dfs = {'train': [], 'val': [], 'test': []}

    def split(x):
        s = np.zeros(len(x), dtype=int)
        i = np.random.choice(len(s), num_val + num_test, replace=False)
        s[i[:num_val]] = 1
        s[i[num_val:num_val + num_test]] = 2
        return s

    wc = win_df.groupby(['cell', 'position']).win_cts
    splits = Parallel(n_jobs=8)([delayed(wc.transform)(split) for _ in range(num_splits)])

    for sp in splits:
        sp.name = 'split'
        df = pd.DataFrame(sp).join(win_df)
        dfs['train'].append(df[df.split == 0].drop(columns=['split']))
        dfs['val'].append(df[df.split == 1].drop(columns=['split']))
        dfs['test'].append(df[df.split == 2].drop(columns=['split']))

    return {'tvt_dfs': dfs}

def gen_samples(inputs, **kwargs):
    num_samples_per_pos = {'train': kwargs['num_train'], 
                           'val': kwargs['num_val'], 
                           'test': kwargs['num_test']}
    tvts = inputs['tvt_dfs']
    cells = inputs['cells']

    def _gen_samples(madf, type):
        madf = madf.join(cells)

        t = {}
        for i in range(num_samples_per_pos[type]):
            s = madf.groupby(['cell', 'position']).win_cts.agg(lambda x: x.sample(1))
            df = pd.DataFrame(s)

            cat = df.groupby('position').apply(lambda x: np.concatenate(x.win_cts.array))
            num_positions = len(cat)
            if i == 0: t = torch.empty(num_positions * num_samples_per_pos[type], len(cat[0]) + 1)
            t[i*num_positions:(i+1)*num_positions,:-1] = torch.tensor(np.stack(cat.array))
            t[i*num_positions:(i+1)*num_positions, -1] = torch.tensor(cat.index)
        return t

    return {type: Parallel(n_jobs=8)([delayed(_gen_samples)(s, type) for s in splits]) for type, splits in tvts.items()}
    # return {type: {i: _gen_samples(s, type) for i, s in enumerate(splits)} for type, splits in tvts.items()}