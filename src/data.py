from scipy.io import loadmat
import numpy as np
import pandas as pd
from step import Step

CUE_ON_T = 1.0
SAMPLE_ON_T = SACCADE_ON_TIME = 3.0
MNM_END_TIME = 3.0
SUBREGIONS = {'Mid-Dorsal': 'MD', 'Posterior-Ventral': 'PV', 'Anterior-Dorsal': 'AD', 'Posterior-Dorsal': 'PD', 'Anterior-Ventral': 'AV'}


def _mat_to_df(dg, monkey, area, num_stimuli=1, numeric_cell=False, 
               SAMPLE_ON_T=SAMPLE_ON_T, SACCADE_ON_TIME=SACCADE_ON_TIME, MNM_END_TIME=MNM_END_TIME):
    rows = []
    cells = []
    for c in range(dg.shape[0]):
        c_num = str(c + 1).zfill(3)
        cell_name = f'{monkey}-{area}-{c_num}'
        for p in range(dg.shape[1]):
            feature_names = dg[c][p][0].dtype.names
            for t in range(dg[c, p].shape[1]):
                cue_on_t = dg[c, p][0, t]['Cue_onT'][0][0]
                ts = dg[c, p][0, t]['TS'].flatten()
                if num_stimuli == 1:
                    ts = ts - cue_on_t + CUE_ON_T
                    ts = np.array([t for t in ts if t >= 0 and t <= SACCADE_ON_TIME + 1.0], dtype=np.float32)
                    if (len(ts) > 0):
                        rows.append([cell_name, p, t, ts]) #, cue_on_t])
                else:
                    sample_on_t = dg[c, p][0, t]['Sample_onT'][0][0]
                    ts2 = np.array([SAMPLE_ON_T + t - sample_on_t for t in ts if t > sample_on_t and t <= sample_on_t + MNM_END_TIME], dtype=np.float32)
                    ts = np.array([t - cue_on_t + CUE_ON_T for t in ts if t >= cue_on_t - CUE_ON_T and t <= cue_on_t + SAMPLE_ON_T - CUE_ON_T], dtype=np.float32)
                    ts = np.concatenate((ts, ts2))
                    try:
                        ismatch = dg[c, p][0, t]['IsMatch'][0][0]
                        if len(ts) > 0:
                            rows.append([c + 1, p, t, ts, ismatch])
                    except: pass

    df = pd.DataFrame(rows, columns=['cell', 'position', 'trial', 'ts'] + ([] if num_stimuli == 1 else ['ismatch']))
    return df

def _get_cells(df, style='monkey-area'):
    cells = df.cell.unique()
    cells = pd.DataFrame(cells, columns=['cell'])
    cells['monkey'] = cells.cell.apply(lambda x: x.split('-')[0])
    cells['area'] = cells.cell.apply(lambda x: x.split('-')[1])
    if 'subregion' in style:
        cells['subregion'] = cells.cell.apply(lambda x: x.split('-')[2])
    if 'phase' in style:
        cells['phase'] = cells.cell.apply(lambda x: x.split('-')[3])
    cells = cells.set_index('cell')
    return cells

def load_odr(_):
    mat = loadmat('../data/raw/ODR/ODR_young_PFC.mat')
    df = _mat_to_df(mat['data_group'], 'YOUNG', 'PFC')

    mat = loadmat('../data/raw/ODR/ODR_adult_PFC_PPC.mat')
    for key in [key for key in mat.keys() if '__' not in key]:
        monkey = key.split('_')[0]
        area = key.split('_')[1]
        new_df = _mat_to_df(mat[key], monkey, area)
        df = pd.concat([df, new_df])

    cells = _get_cells(df)
    df = df.drop(columns=['monkey', 'area'], errors='ignore')

    return {'raw_df': df.set_index(['cell', 'position', 'trial']), 'cells': cells}

def load_mnm_spatial(_):
    return load_mnm('spatial', 0)

def load_mnm_feature(_):
    return load_mnm('feature', 1)

def load_mnm(data_type, monkey_info_col):
    ad_mat = loadmat(f'../data/raw/MNM_original_data/all_{data_type}_data.mat')[f'all_{data_type}_data']
    ai_mat = loadmat(f'../data/raw/MNM_original_data/all_{data_type}_info.mat')[f'all_{data_type}_info']

    ad_mat = ad_mat[:, :8]
    ai_mat = ai_mat[:, [monkey_info_col, 3, 4]]
    extract = np.vectorize(lambda x: x[0])
    ai_mat = extract(ai_mat)
    
    ai_df = pd.DataFrame(ai_mat, columns=['monkey', 'phase', 'subregion'])
    ai_df.monkey = ai_df.monkey.apply(lambda x: x[0:3].upper())
    ai_df.subregion = ai_df.subregion.apply(lambda x: SUBREGIONS[x].upper())
    ai_df

    ad_df = _mat_to_df(ad_mat, '<REPLACE>', 'PFC', num_stimuli=2, numeric_cell=True)

    df = pd.merge(ad_df, ai_df, left_on='cell', right_index=True)
    df['cell'] = df.apply(lambda r: '-'.join([r.monkey, 'PFC', str(r.subregion), r.phase, str(r.cell)]), axis=1)
    cells = _get_cells(df, style='monkey-area-subregion-phase')
    df = df.drop(columns=['monkey', 'subregion', 'phase'])

    return {'raw_df': df.set_index(['cell', 'position', 'trial']), 'cells': cells}

def load_msng(_):
    md = loadmat('../data/raw/MSNG_data/msng_data.mat')['msng_data']
    mi = loadmat('../data/raw/MSNG_data/msng_info2.mat')['msng_info']

    mi = mi[:,[1,2]]
    extract = np.vectorize(lambda x: x[0])
    mi = extract(mi)
    SRNUM_TO_AREA = {'46': 'PFC', '8': 'PFC'}

    md = md[:, 0:16]

    mi_df = pd.DataFrame(mi, columns=['subregion', 'monkey'])
    mi_df.monkey = mi_df.monkey.apply(lambda x: x[0:3].upper())
    mi_df['area'] = mi_df.subregion.apply(lambda x: SRNUM_TO_AREA.get(x, 'PPC'))
    mi_df.index = mi_df.index + 1

    # md_df = load_data(md, 'MSNG', 'ADULT', '<REPLACE>', '<REPLACE>', num_stimuli=2, numeric_cell=True)
    md_df = _mat_to_df(md, '<REPLACE>', '<REPLACE>', num_stimuli=2, numeric_cell=True, 
                       MNM_END_TIME=1.5, SAMPLE_ON_T=4.5)
    # md_df = md_df.drop(['monkey', 'area', 'subregion'], axis=1)

    df = pd.merge(md_df, mi_df, left_on='cell', right_index=True)
    df['cell'] = df.apply(lambda r: '-'.join([r.monkey, r.area, str(r.subregion), str(r.cell)]), axis=1)
    cells = _get_cells(df, style='monkey-area-subregion')
    df = df.drop(columns=['monkey', 'subregion', 'area'])
    return {'raw_df': df.set_index(['cell', 'position', 'trial']), 'cells': cells, 'msng_data': md, 'msng_info': mi}