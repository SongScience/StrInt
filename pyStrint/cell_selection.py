import pandas as pd
import numpy as np
import time
import logging
from scipy.sparse import csr_matrix
try:
    import torch
except Exception:
    torch = None

try:
    import numba as nb
except Exception:
    nb = None

from loess.loess_1d import loess_1d
REALMIN = np.finfo(float).tiny
from . import optimizers
from . import utils


# TODO del after test
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f'{func.__name__}\t{end - start} seconds')
        return result
    return wrapper


def randomize(mat_orig, seed = 1111):
    rng = np.random.RandomState(seed)
    mat = mat_orig.copy()
    values = mat.values.astype(float, copy=False)
    floor_vals = np.floor(values)
    frac = values - floor_vals
    mask = (values != 0) & (frac != 0)
    if np.any(mask):
        rand = rng.rand(*values.shape)
        new_vals = np.where(rand < frac, floor_vals + 1, floor_vals)
        values[mask] = new_vals[mask]
    row_sums = values.sum(axis=1)
    zero_rows = row_sums == 0
    if np.any(zero_rows):
        arg_max = np.argmax(mat_orig.values[zero_rows], axis=1)
        row_indices = np.where(zero_rows)[0]
        values[row_indices, arg_max] = 1
    return mat



def randomization(weight,spot_cell_num, seed = 1111):
    weight_threshold = 0.001
    if not utils.check_weight_sum_to_one(weight):
        # not sum as one
        weight = pd.DataFrame(weight).div(np.sum(weight, axis = 1), axis = 0)
    # eliminating small num
    weight[weight < weight_threshold] = 0
    # estimated cell number per spot (can be fractional)
    # num = weight * spot_cell_num
    num = pd.DataFrame(spot_cell_num.reshape(spot_cell_num.shape[0],1) * weight)
    # randomize to obtain integer cell-type number per spot
    num = randomize(num,seed)
    # num.to_csv(path + 'cell_type_num_per_spot.csv', index = True, header= True, sep = ',')
    return num


def estimate_cell_number(st_exp, mean_cell_numbers):
    # transpose because cytospace has cell as columns
    st_data = st_exp.T
    # Read data
    expressions = st_data.values.astype(float)
    # Data normalization
    expressions_tpm_log = normalize_data(expressions)
    # Set up fitting problem
    RNA_reads = np.sum(expressions_tpm_log, axis=0, dtype=float)
    mean_RNA_reads = np.mean(RNA_reads)
    min_RNA_reads = np.min(RNA_reads)
    min_cell_numbers = 1 if min_RNA_reads > 0 else 0
    fit_parameters = np.polyfit(np.array([min_RNA_reads, mean_RNA_reads]),
                                np.array([min_cell_numbers, mean_cell_numbers]), 1)
    polynomial = np.poly1d(fit_parameters)
    # cell_number_to_node_assignment = polynomial(RNA_reads).astype(int)
    cell_number_to_node_assignment = np.round(polynomial(RNA_reads)).astype(int)
    return cell_number_to_node_assignment


def normalize_data(data):
    data = np.nan_to_num(data).astype(float)
    data *= 10**6 / np.sum(data, axis=0, dtype=float)
    np.log2(data + 1, out=data)
    np.nan_to_num(data, copy=False)
    return data


def half_life_prob(t,T=10):
    '''
    # When one cell has been picked for T times, 
    # its prob to be picked again decreases by half.
    # T default as 10
    '''
    return (1/2)**(t/T)

def id_to_idx(trans_id_idx,cell_id):
    return list(trans_id_idx.loc[cell_id][0])


if nb is not None:
    @nb.njit(cache=True)
    def _rowwise_corr_with_vector_numba(mat, vec):
        n_rows, n_cols = mat.shape
        v_mean = 0.0
        for j in range(n_cols):
            v_mean += vec[j]
        v_mean /= n_cols
        v_var = 0.0
        for j in range(n_cols):
            diff = vec[j] - v_mean
            v_var += diff * diff
        v_std = (v_var / n_cols) ** 0.5
        out = np.zeros(n_rows, dtype=np.float32)
        if v_std == 0:
            return out
        for i in range(n_rows):
            row_mean = 0.0
            for j in range(n_cols):
                row_mean += mat[i, j]
            row_mean /= n_cols
            row_var = 0.0
            dot = 0.0
            for j in range(n_cols):
                a = mat[i, j] - row_mean
                b = vec[j] - v_mean
                dot += a * b
                row_var += a * a
            row_std = (row_var / n_cols) ** 0.5
            denom = row_std * v_std * n_cols
            if denom != 0:
                out[i] = dot / denom
        return out

    @nb.njit(cache=True)
    def _interface_corr_candidates_numba(candi_exp_sum, nn_spot_vals, a_ss_vals, lig_idx, rec_idx):
        n_cand = candi_exp_sum.shape[0]
        n_nn = nn_spot_vals.shape[0]
        n_pairs = lig_idx.shape[0]
        out = np.zeros(n_cand, dtype=np.float32)
        if n_nn == 0 or a_ss_vals.size == 0 or n_pairs == 0:
            return out
        for i in range(n_nn):
            # precompute a_ss mean/std
            a_mean = 0.0
            for j in range(n_pairs):
                a_mean += a_ss_vals[i, j]
            a_mean /= n_pairs
            a_var = 0.0
            for j in range(n_pairs):
                diff = a_ss_vals[i, j] - a_mean
                a_var += diff * diff
            a_std = (a_var / n_pairs) ** 0.5
            if a_std == 0:
                continue
            for c in range(n_cand):
                row_mean = 0.0
                for j in range(n_pairs):
                    L1 = nn_spot_vals[i, lig_idx[j]]
                    R1 = nn_spot_vals[i, rec_idx[j]]
                    L2 = candi_exp_sum[c, lig_idx[j]]
                    R2 = candi_exp_sum[c, rec_idx[j]]
                    row_mean += L1 * R2 + R1 * L2
                row_mean /= n_pairs
                row_var = 0.0
                dot = 0.0
                for j in range(n_pairs):
                    L1 = nn_spot_vals[i, lig_idx[j]]
                    R1 = nn_spot_vals[i, rec_idx[j]]
                    L2 = candi_exp_sum[c, lig_idx[j]]
                    R2 = candi_exp_sum[c, rec_idx[j]]
                    val = L1 * R2 + R1 * L2
                    a = val - row_mean
                    b = a_ss_vals[i, j] - a_mean
                    dot += a * b
                    row_var += a * a
                row_std = (row_var / n_pairs) ** 0.5
                denom = row_std * a_std * n_pairs
                if denom != 0:
                    out[c] += dot / denom
        out /= n_nn
        return out


def _rowwise_corr_with_vector(mat, vec):
    mat = np.asarray(mat, dtype=np.float32)
    vec = np.asarray(vec, dtype=np.float32)
    if nb is not None:
        return _rowwise_corr_with_vector_numba(mat, vec)
    v = vec - np.mean(vec)
    v_std = np.std(v)
    if v_std == 0:
        return np.zeros(mat.shape[0], dtype=np.float32)
    m = mat - np.mean(mat, axis=1, keepdims=True)
    m_std = np.std(m, axis=1)
    denom = m_std * v_std * mat.shape[1]
    denom[denom == 0] = 1.0
    return (m @ v) / denom


def _rowwise_corr_with_vector_torch(mat_t, vec_t):
    vec_c = vec_t - vec_t.mean()
    v_norm = torch.linalg.norm(vec_c)
    if v_norm.item() == 0:
        return torch.zeros((mat_t.shape[0],), device=mat_t.device, dtype=mat_t.dtype)
    mat_c = mat_t - mat_t.mean(dim=1, keepdim=True)
    m_norm = torch.linalg.norm(mat_c, dim=1).clamp_min(1e-12)
    return (mat_c @ vec_c) / (m_norm * v_norm)


def _interface_corr_candidates(candi_exp_sum, nn_spot_vals, a_ss_vals, lig_idx, rec_idx):
    if nn_spot_vals.size == 0 or a_ss_vals.size == 0:
        return np.zeros(candi_exp_sum.shape[0], dtype=np.float32)
    if nb is not None:
        return _interface_corr_candidates_numba(candi_exp_sum, nn_spot_vals, a_ss_vals, lig_idx, rec_idx)
    candi_L2 = candi_exp_sum[:, lig_idx]
    candi_R2 = candi_exp_sum[:, rec_idx]
    corrs = []
    for i in range(nn_spot_vals.shape[0]):
        L1 = nn_spot_vals[i, lig_idx]
        R1 = nn_spot_vals[i, rec_idx]
        aff = L1 * candi_R2 + R1 * candi_L2
        corrs.append(_rowwise_corr_with_vector(aff, a_ss_vals[i]))
    return np.mean(np.vstack(corrs), axis=0)


@timeit
def feature_sort(exp, degree = 2, span = 0.3):
    # 1. input cell x gene
    # exp: gene - row, cell - column
    exp = exp.T
    # 2. calculate mean and var for each gene
    var = np.array(np.log10(exp.var(axis=1) + REALMIN))
    mean = np.array(np.log10(exp.mean(axis=1) + REALMIN))
    # 3. fit model 
    xout, yout, wout = loess_1d(mean, var, frac = span, degree = degree, rotate=False)
    # 4. calculate standaridized value
    exp_center = exp.apply(lambda x: x - np.mean(x), axis=1)
    Z = exp_center.div(yout, axis=0)
    # 5. clipp value by sqrt(N)
    upper_bound = np.sqrt(exp.shape[1])
    Z[Z>upper_bound] = upper_bound
    # 6. sort
    reg_var = pd.DataFrame(Z.var(axis=1))
    sort_reg_var = reg_var.sort_values(by = 0, ascending=False)
    return sort_reg_var


def lr_shared_top_k_gene(sort_reg_var, lr_df, k = 3000, keep_lr_per = 0.8):
    # shared lr genes
    genes = sort_reg_var.index.tolist()
    lr_share_genes = list(set(lr_df[0]).union(set(lr_df[1])).intersection(set(genes)))
    # keep top lr genes
    lr_var = sort_reg_var.loc[lr_share_genes]
    take_num = int(len(lr_var) * keep_lr_per)
    p = "{:.0%}".format(keep_lr_per)
    a = lr_var.sort_values(by = 0, ascending=False).iloc[0:take_num].index.tolist()
    # combine with top k feature genes
    feature_genes = sort_reg_var.iloc[0:k].index.tolist()
    lr_feature_genes = list(set(feature_genes + a))
    return lr_feature_genes


def norm_center(data):
    #first sum to one, then centered
    df = pd.DataFrame(data)
    a = df.apply(lambda x: (x)/np.sum(x) , axis=1)
    return a.apply(lambda x: (x - np.mean(x)) , axis=1)


@timeit
def init_solution(cell_type_num, spot_idx, csr_st_exp, csr_sc_exp, meta_df, trans_id_idx, T_HALF,
                  init_method='residual', init_copy_spot_frac=0.0, init_copy_min_corr=None,
                  init_copy_require_improve=True):
    spot_i = -1
    picked_index = {}
    correlations = []
    agg_exp_by_spot = []
    sc_index = np.array(meta_df.index)
    meta_df = np.array(meta_df)
    sc_exp_values = csr_sc_exp.toarray()
    st_exp_values = csr_st_exp.toarray()
    picked_time = pd.Series(np.zeros(len(sc_index)), index = sc_index, dtype=float)
    for spot_name in spot_idx:
        spot_i += 1
        prob = half_life_prob(t = picked_time.values, T = T_HALF)
        Es = st_exp_values[spot_i]
        # Use correlation (not dot product) so "topk" really means highest correlation.
        cor_st_sc = _rowwise_corr_with_vector(sc_exp_values, Es)
        adj_cor = cor_st_sc * prob
        w_i = cell_type_num.loc[spot_name]
        est_type = w_i[w_i != 0]
        picked_index[spot_name] = []
        if init_method == 'greedy':
            sum_exp = np.zeros(sc_exp_values.shape[1], dtype=np.float32)
            for cell_type, count in est_type.items():
                take_n = int(count)
                if take_n <= 0:
                    continue
                candi_idx = np.where(meta_df == cell_type)[0]
                if candi_idx.size == 0:
                    continue
                for _ in range(take_n):
                    candi_exp = sc_exp_values[candi_idx]
                    candi_sum = candi_exp + sum_exp
                    candi_cor = _rowwise_corr_with_vector(candi_sum, Es)
                    scores = candi_cor * prob[candi_idx]
                    best_pos = int(np.argmax(scores))
                    best_idx = candi_idx[best_pos]
                    best_id = sc_index[best_idx]
                    picked_time.loc[best_id] += 1
                    picked_index[spot_name].append(best_id)
                    sum_exp = candi_sum[best_pos]
                    candi_idx = np.delete(candi_idx, best_pos)
                    if candi_idx.size == 0:
                        break
            agg_exp = sum_exp
        elif init_method == 'residual':
            sum_exp = np.zeros(sc_exp_values.shape[1], dtype=np.float32)
            total_cells = int(est_type.sum())
            for cell_type, count in est_type.items():
                take_n = int(count)
                if take_n <= 0:
                    continue
                candi_idx = np.where(meta_df == cell_type)[0]
                if candi_idx.size == 0:
                    continue
                for _ in range(take_n):
                    candi_exp = sc_exp_values[candi_idx]
                    target = Es * total_cells - sum_exp
                    candi_cor = _rowwise_corr_with_vector(candi_exp, target)
                    scores = candi_cor * prob[candi_idx]
                    best_pos = int(np.argmax(scores))
                    best_idx = candi_idx[best_pos]
                    best_id = sc_index[best_idx]
                    picked_time.loc[best_id] += 1
                    picked_index[spot_name].append(best_id)
                    sum_exp = sum_exp + sc_exp_values[best_idx]
                    candi_idx = np.delete(candi_idx, best_pos)
                    if candi_idx.size == 0:
                        break
            agg_exp = sum_exp
        else:
            for cell_type, count in est_type.items():
                take_n = int(count)
                if take_n <= 0:
                    continue
                candi_idx = np.where(meta_df == cell_type)[0]
                if candi_idx.size == 0:
                    continue
                scores = adj_cor[candi_idx]
                top_pos = np.argsort(scores)[::-1][:take_n]
                selected_idx = candi_idx[top_pos]
                selected_cell_id = list(sc_index[selected_idx])
                picked_time.loc[selected_cell_id] += 1
                picked_index[spot_name].extend(selected_cell_id)
            if picked_index[spot_name]:
                candi_idx = id_to_idx(trans_id_idx, picked_index[spot_name])
                agg_exp = sc_exp_values[candi_idx].sum(axis=0)
            else:
                agg_exp = np.zeros(sc_exp_values.shape[1], dtype=np.float32)
        if np.std(Es) == 0 or np.std(agg_exp) == 0:
            cor = 0.0
        else:
            cor = np.corrcoef(Es, np.array(agg_exp))[0, 1]
        correlations.append(cor)
        agg_exp_by_spot.append(agg_exp)
        # break
    if init_copy_spot_frac and init_copy_spot_frac > 0:
        n_copy = max(1, int(len(spot_idx) * float(init_copy_spot_frac)))
        spot_corr = pd.Series(correlations, index=spot_idx)
        target_spots = spot_corr.sort_values().index[:n_copy].tolist()
        for spot_name in target_spots:
            target_i = spot_idx.index(spot_name)
            Es = st_exp_values[target_i]
            best_corr = -np.inf
            best_donor = None
            for donor_i, donor_spot in enumerate(spot_idx):
                if donor_i == target_i:
                    continue
                donor_agg = agg_exp_by_spot[donor_i]
                if np.std(Es) == 0 or np.std(donor_agg) == 0:
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(Es, donor_agg)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_donor = donor_spot
            if best_donor is None:
                continue
            if init_copy_min_corr is not None and best_corr < float(init_copy_min_corr):
                continue
            if init_copy_require_improve and best_corr <= correlations[target_i]:
                continue
            prev_ids = picked_index.get(spot_name, [])
            for cell_id in prev_ids:
                picked_time.loc[cell_id] -= 1
            picked_index[spot_name] = list(picked_index.get(best_donor, []))
            if picked_index[spot_name]:
                picked_time.loc[picked_index[spot_name]] += 1
                candi_idx = id_to_idx(trans_id_idx, picked_index[spot_name])
                agg_exp = sc_exp_values[candi_idx].sum(axis=0)
            else:
                agg_exp = np.zeros(sc_exp_values.shape[1], dtype=np.float32)
            agg_exp_by_spot[target_i] = agg_exp
            correlations[target_i] = best_corr
    print(f'\t Init solution: max - {np.max(correlations):.4f}, \
    mean - {np.mean(correlations):.4f}, \
    min - {np.min(correlations):.4f}')
    picked_time = picked_time.to_frame(name='count')
    return picked_index, correlations, picked_time


def eva_metric(arr1, arr2, metric = 'correlation'):
    # df1, df2: two array with same length
    # return the correlation between two dataframes
    if metric in ('correlation', 'spot_cor'):
        val = np.corrcoef(arr1, arr2)[0,1]
    
    if metric == 'rmse':
        val = -1 * np.sqrt(np.mean((arr1 - arr2)**2))
    return val


@timeit
def reselect_cell(st_exp, spots_nn_lst, st_aff_profile_df, 
                  sc_exp, csr_sc_exp, sc_meta, trans_id_idx,
                  sum_sc_agg_exp, sc_agg_aff_profile_df, 
                  init_sc_df, init_picked_time, lr_df, p = 0.1,repeat_penalty = 10, metric = 'correlation', st_aff_profile_map=None,
                  use_vectorized=False, lr_indices=None, spot_cor_sc_exp=None, spot_cor_st_exp=None, sum_sc_agg_exp_exp=None,
                  spot_subset=None, swap_method='correlation', use_gpu=False, candidate_topk=0):
    '''
    Reselect cells from sc exp data for higher exp and interface correlation
    p: weight of interface correlation
    repeat_penalty: penalty for repeated selection of each cell, if set as 10, the prob of being selected will decrease by half after 10 times.

    No repeat
    Runtime: 20s for each spot with 10 cells; 2s each cell.

    '''
    tp_idx_dict = get_tp_idx_dict(sc_meta)
    exp_sc_values = None
    exp_st_values = None
    exp_sum_sc_agg_values = None
    if use_vectorized:
        sc_exp_values = sc_exp.to_numpy(dtype=np.float32, copy=False)
        st_exp_values = st_exp.to_numpy(dtype=np.float32, copy=False)
        sum_sc_agg_values = sum_sc_agg_exp.to_numpy(dtype=np.float32, copy=False)
        if metric == 'spot_cor' and spot_cor_sc_exp is not None and spot_cor_st_exp is not None:
            exp_sc_values = spot_cor_sc_exp.to_numpy(dtype=np.float32, copy=False)
            exp_st_values = spot_cor_st_exp.to_numpy(dtype=np.float32, copy=False)
            if sum_sc_agg_exp_exp is not None:
                exp_sum_sc_agg_values = sum_sc_agg_exp_exp.to_numpy(dtype=np.float32, copy=False)
        spot_to_idx = {s: i for i, s in enumerate(st_exp.index)}
        cell_ids = sc_exp.index.to_numpy()
        cell_id_to_pos = {cid: i for i, cid in enumerate(cell_ids)}
        sc_meta_aligned = sc_meta.loc[sc_exp.index]
        cell_types = sc_meta_aligned['celltype'].to_numpy()
        picked_counts = init_picked_time.loc[sc_exp.index, 'count'].to_numpy(dtype=np.float32, copy=True)
        tp_pos_dict = {k: np.array([cell_id_to_pos[c] for c in v if c in cell_id_to_pos], dtype=np.int64)
                       for k, v in tp_idx_dict.items()}
        if lr_indices is None:
            lr_df_align, lig_idx, rec_idx = optimizers.prepare_lr_indices(lr_df, sc_exp.columns)
        else:
            lr_df_align, lig_idx, rec_idx = lr_indices
    new_spot_cell_dict = {}
    spot_idx_lst = list(st_exp.index)
    spot_subset_set = set(spot_subset) if spot_subset is not None else None
    result = pd.DataFrame()
    picked_time = init_picked_time.copy()
    gene_num = st_exp.shape[1]
    for spot in spot_idx_lst:
    # for spot in ['11x49']:
        '''
        s_: spot exp of st_exp or sc_agg
        _i: indice numerical index of cell_id or spot
        '''
        ########## ST ##########
        # Transform to numerical spot id, subset from csr_matrix
        # 
        s_exp = st_exp.loc[spot]
        if exp_st_values is not None:
            exp_s_exp = exp_st_values[spot_to_idx[spot]]
            exp_std = np.std(exp_s_exp)
            if exp_std == 0:
                exp_std = 1.0
            norm_s_exp = exp_s_exp / exp_std
        else:
            exp_s_exp = s_exp.values
            norm_s_exp = s_exp/np.std(s_exp)
        ########## SC ########## 
        if sum_sc_agg_exp_exp is not None and exp_sum_sc_agg_values is not None:
            exp_sc_agg_sum = sum_sc_agg_exp_exp.loc[spot].values
        else:
            exp_sc_agg_sum = sum_sc_agg_exp.loc[spot].values
        exp_sc_std = np.std(exp_sc_agg_sum)
        if exp_sc_std == 0:
            exp_sc_std = 1.0
        norm_s_sc_agg_sum = exp_sc_agg_sum/exp_sc_std
        # generate baseline corr
        if metric == 'spot_cor':
            if np.std(exp_s_exp) == 0 or np.std(exp_sc_agg_sum) == 0:
                max_exp_cor = 0.0
            else:
                max_exp_cor = float(np.corrcoef(exp_s_exp, exp_sc_agg_sum)[0, 1])
        else:
            max_exp_cor = eva_metric(norm_s_exp, norm_s_sc_agg_sum)
        # print(f'Baseline cor of spot {spot} is {max_exp_cor}')
        ###### Interface ########
        nn_spot = spots_nn_lst[spot]
        a_ss = pd.DataFrame()
        a_cc = pd.DataFrame()
        if p != 0 and nn_spot:
            if st_aff_profile_map is not None and spot in st_aff_profile_map:
                # Filter nn_spot to only include spots that exist in the map
                spot_df = st_aff_profile_map[spot]
                if isinstance(spot_df.index, pd.MultiIndex):
                    nn_spot_filtered = [s for s in nn_spot if s in spot_df.index.get_level_values(1)]
                    a_ss = spot_df.loc[(spot, nn_spot_filtered), :] if nn_spot_filtered else pd.DataFrame()
                else:
                    nn_spot_filtered = [s for s in nn_spot if s in spot_df.index]
                    a_ss = spot_df.loc[nn_spot_filtered] if nn_spot_filtered else pd.DataFrame()
            else:
                a_ss = st_aff_profile_df.loc[(spot,nn_spot),:] if nn_spot else pd.DataFrame()
            # replace the no-aff cells
            sum_a_ss = a_ss.sum(axis = 1) if not a_ss.empty else pd.Series()
            sum_a_ss = sum_a_ss[sum_a_ss!=0]
            nn_spot = sum_a_ss.index.get_level_values(1).tolist()
            if nn_spot:
                if st_aff_profile_map is not None and spot in st_aff_profile_map:
                    spot_df = st_aff_profile_map[spot]
                    if isinstance(spot_df.index, pd.MultiIndex):
                        nn_spot_filtered = [s for s in nn_spot if s in spot_df.index.get_level_values(1)]
                        a_ss = spot_df.loc[(spot, nn_spot_filtered), :] if nn_spot_filtered else pd.DataFrame()
                    else:
                        nn_spot_filtered = [s for s in nn_spot if s in spot_df.index]
                        a_ss = spot_df.loc[nn_spot_filtered] if nn_spot_filtered else pd.DataFrame()
                else:
                    a_ss = st_aff_profile_df.loc[(spot,nn_spot),:] if nn_spot else pd.DataFrame()
                a_cc = sc_agg_aff_profile_df.loc[(spot,nn_spot),:]
        spot_cell_lst = init_sc_df[init_sc_df['spot'] == spot]['sc_id'].tolist()
        # print(f'orig spot_cell_lst {spot_cell_lst}')
        if spot_subset_set is not None and spot not in spot_subset_set:
            exp_cor = max_exp_cor
            max_aff_cor = 0
            max_mix_cor = max_exp_cor
            mix_corr = max_exp_cor
            inter_cor = 0
        elif p == 0 or nn_spot == [] or a_cc.empty or a_cc.sum().sum() == 0:
            if use_vectorized:
                picked_counts, spot_cell_lst, exp_cor = cellReplaceByExp_fast(
                    spot,
                    spot_cell_lst,
                    sc_exp_values,
                    cell_types,
                    tp_pos_dict,
                    st_exp_values,
                    picked_counts,
                    repeat_penalty,
                    cell_id_to_pos,
                    spot_to_idx,
                    cell_ids,
                    metric=metric,
                    exp_sc_values=exp_sc_values,
                    exp_st_values=exp_st_values,
                    swap_method=swap_method,
                    use_gpu=use_gpu,
                    candidate_topk=candidate_topk,
                )
            else:
                picked_time, spot_cell_lst, exp_cor = expSwap_SPROUT(spot_cell_lst, csr_sc_exp, sc_meta, trans_id_idx, tp_idx_dict,
                            s_exp, picked_time, gene_num, repeat_penalty, metric = metric)
            max_aff_cor = 0
            max_mix_cor = max_exp_cor 
            mix_corr = exp_cor
            inter_cor = 0
        else:
            max_aff_cor, max_mix_cor = cal_baseline_aff(a_ss, a_cc, max_exp_cor,p = p, metric=metric)
        # for each cell in spot
            if use_vectorized:
                picked_counts, spot_cell_lst, exp_cor, inter_cor, mix_corr = cellReplaceByBoth_fast(
                    spot,
                    spot_cell_lst,
                    sc_exp_values,
                    cell_types,
                    tp_pos_dict,
                    sum_sc_agg_values,
                    st_exp_values,
                    nn_spot,
                    a_ss,
                    picked_counts,
                    p,
                    repeat_penalty,
                    cell_id_to_pos,
                    spot_to_idx,
                    lig_idx,
                    rec_idx,
                    cell_ids,
                    metric=metric,
                    exp_sc_values=exp_sc_values,
                    exp_st_values=exp_st_values,
                    swap_method=swap_method,
                )
            else:
                picked_time, spot_cell_lst, exp_cor, inter_cor, mix_corr = cellReplaceByBoth(spot,spot_cell_lst, sc_exp, sc_meta, tp_idx_dict, 
                                                                                            sum_sc_agg_exp,s_exp, nn_spot, a_ss, 
                                                                                            lr_df, picked_time, p, repeat_penalty)

        tmp = pd.DataFrame(spot_cell_lst,columns = ['sc_id'])
        tmp['spot'] = spot
        # print(tmp)
        tmp['exp_cor_before'] = max_exp_cor
        tmp['interface_cor_before'] = max_aff_cor
        tmp['mix_cor_before'] = max_mix_cor
        tmp['exp_cor_after'] = exp_cor
        tmp['interface_cor_after']= inter_cor
        tmp['mix_cor_after'] = mix_corr
        result = pd.concat((result,tmp))
        new_spot_cell_dict[spot] = spot_cell_lst
        # if spot == '11x49':
        #     break
    result['celltype'] = sc_meta.loc[result['sc_id']]['celltype'].values
    if use_vectorized:
        picked_time.loc[sc_exp.index, 'count'] = picked_counts
    result.index = range(len(result))
    result.index = result.index.map(str)
    correlations = result['exp_cor_after']
    if metric == 'spot_cor':
        spot_corr = result.groupby('spot', sort=False)['exp_cor_after'].mean()
        print(f'\t Swapped solution: max - {np.max(spot_corr):.4f}, \
    mean - {np.mean(spot_corr):.4f}, \
    min - {np.min(spot_corr):.4f}')
    else:
        print(f'\t Swapped solution: max - {np.max(correlations):.4f}, \
    mean - {np.mean(correlations):.4f}, \
    min - {np.min(correlations):.4f}')
    return result, picked_time


def get_sum_sc_agg(sc_exp,sc_agg_meta,st_exp):
    sc_agg_exp = sc_exp.loc[sc_agg_meta['sc_id']]
    sc_agg_exp['spot'] = sc_agg_meta['spot'].values
    sum_sc_agg_exp = sc_agg_exp.groupby('spot').sum()
    sum_sc_agg_exp = sum_sc_agg_exp.loc[st_exp.index]
    return sum_sc_agg_exp


@timeit
def cal_sc_candi_aff_profile(s_exp, candi_exp, lr_df):
    '''
    s_exp (1): summed sc_agg exp shape(1,genes)
    candi_exp (2): exp of candidate cells + remain cells shape(candidates,genes)
    aff_profile = L1*R2 + L2*R1
    '''
    st_L1 = s_exp[lr_df[0]]
    st_R1 = s_exp[lr_df[1]]
    st_L2 = candi_exp[lr_df[0]]
    st_R2 = candi_exp[lr_df[1]]
    #print(st_R2)
    #st_LR_df1 = pd.concat([st_L1 * st_R1.values[i] for i in range(st_R1.shape[0])], keys=st_R1.index.tolist())
    st_LR_df1 = st_R2 * st_L1.values
    #print(st_LR_df1)
    #st_LR_df2 = pd.concat([st_L2 * st_R2.values[i] for i in range(st_R2.shape[0])], keys=st_R2.index.tolist())
    st_LR_df2 = st_L2 * st_R1.values
    #print(st_LR_df2)
    sc_agg_aff_profile_df = st_LR_df1.values + st_LR_df2
    return sc_agg_aff_profile_df


@timeit
def get_tp_idx_dict(sc_meta):
    '''
    generate dict with celltype as key, corresponding cell_id list as values
    '''
    tp_idx_dict = {}
    for tp in sc_meta.celltype.unique():
        # get indices where "tp" equals current value
        indices = sc_meta.index[sc_meta['celltype'] == tp].tolist()
        # add key-value pair to dictionary
        tp_idx_dict[tp] = indices
    return tp_idx_dict


@timeit
def dict2df(spot_cell_dict,st_exp,sc_exp,sc_meta):
    new_picked_df = pd.DataFrame()
    for key, value in spot_cell_dict.items():
        tmp = pd.DataFrame(value)
        tmp[1] = key
        corr = np.corrcoef(sc_exp.loc[value].sum(),st_exp.loc[key])[0,1]
        tmp['corr'] = corr
        new_picked_df = pd.concat((new_picked_df,tmp))
    new_picked_df = new_picked_df.reset_index()
    del new_picked_df['index']
    new_picked_df.index = new_picked_df.index.map(str)
    new_picked_df['celltype'] = sc_meta.loc[new_picked_df[0]]['celltype'].values
    new_picked_df.columns = ['sc_id','spot','corr','celltype']
    new_picked_df['spot'] = new_picked_df['spot'].astype('str')
    return new_picked_df


@timeit
def cal_baseline_aff(a_ss, a_cc, max_exp_cor, p, metric = 'correlation'):  
    corr = np.diag(np.corrcoef(a_ss, a_cc)[:a_ss.shape[0], a_ss.shape[0]:])
    max_aff_cor =  np.nan_to_num(corr).mean()
    max_mix_cor = max_exp_cor*(1-p) + max_aff_cor*p
    return max_aff_cor, max_mix_cor


@timeit
def cal_interface_candi_cor(spot,nn_spot, a_ss, sum_sc_agg_exp, candi_exp_sum, lr_df):
    interface_candi_cor = pd.DataFrame()
    for nn_s in nn_spot:
        a_sn = a_ss.loc[(spot, nn_s)]
        nn_s_agg_exp = sum_sc_agg_exp.loc[nn_s]
        candi_aff_profile = cal_sc_candi_aff_profile(nn_s_agg_exp, candi_exp_sum, lr_df)
        interface_candi_tmp = candi_aff_profile.T.corrwith(a_sn)
        interface_candi_tmp = pd.DataFrame(interface_candi_tmp, columns=[nn_s])
        interface_candi_cor = pd.concat((interface_candi_cor, interface_candi_tmp), axis=1)
    interface_candi_cor['mean'] = interface_candi_cor.mean(axis=1)
    return interface_candi_cor


@timeit
def cellReplaceByBoth(spot,spot_cell_lst, sc_exp, sc_meta, tp_idx_dict, sum_sc_agg_exp,
                        s_exp, nn_spot, a_ss,  lr_df, picked_time,
                        p,repeat_penalty):
    '''
    Default mode, replace cell by highest exp and affinity correlation
    '''
    for i in range(len(spot_cell_lst)):
        cell = spot_cell_lst[i]
        # print(cell)
        spot_cell_lst.remove(cell)
        # print(spot_cell_lst)
        # calculate remain agg exp
        spot_remain_mat = sc_exp.loc[spot_cell_lst]
        remain_exp = np.sum(spot_remain_mat)
        # get candidate cells from the same type
        removed_type = sc_meta.loc[cell]['celltype']
        candi_cell_id = tp_idx_dict[removed_type]
        # print('candi_cell_id:', candi_cell_id)
        candi_exp = sc_exp.loc[candi_cell_id]
        # calculate replaced agg for each candidates
        candi_exp_sum = candi_exp + remain_exp
        # [exp cor]
        exp_candi_cor = candi_exp_sum.T.corrwith(s_exp)
        # print('candi_exp_sum:', candi_exp_sum)
        # print('candi_exp_sum_sum:',candi_exp_sum.sum().sum())
        # print('exp_candi_cor',exp_candi_cor)
        # [interface cor]
        # interface cor with the nn spot of target spot
        # (spot,nn_spot, a_ss, sum_sc_agg_exp, candi_exp_sum, lr_df)
        interface_candi_cor = cal_interface_candi_cor(spot,nn_spot, a_ss, sum_sc_agg_exp, candi_exp_sum, lr_df)
        # TODO debug
        # picked_time['prob'] = 1
        prob = half_life_prob(picked_time['count'].values,repeat_penalty)
        picked_time['prob'] = prob
        # TODO debug
        cor_df = interface_candi_cor.loc[exp_candi_cor.index,'mean']*p + (1-p)*exp_candi_cor
        adj_cor_df = picked_time.loc[cor_df.index,'prob'] * cor_df
        max_idx = adj_cor_df.idxmax()
        mix_corr = adj_cor_df.loc[adj_cor_df.idxmax()]
        spot_cell_lst.insert(0,max_idx)
        exp_cor = exp_candi_cor.loc[max_idx]
        inter_cor = interface_candi_cor['mean'].loc[max_idx]
        # update cell picked time
        picked_time.loc[cell,'count'] -= 1
        picked_time.loc[max_idx,'count'] += 1
        # print(f'  Change cell {cell} exp_cor is {exp_cor}; inter_cor is {inter_cor}; mix cor of {spot} is {mix_corr}')
        # break
    return picked_time, spot_cell_lst, exp_cor, inter_cor, mix_corr


@timeit
def cellReplaceByBoth_fast(spot, spot_cell_lst, sc_exp_values, cell_types, tp_pos_dict, sum_sc_agg_values,
                           st_exp_values, nn_spot, a_ss, picked_counts, p, repeat_penalty,
                           cell_id_to_pos, spot_to_idx, lig_idx, rec_idx, cell_ids,
                           metric='correlation', exp_sc_values=None, exp_st_values=None,
                           swap_method='correlation'):
    spot_idx = spot_to_idx.get(spot)
    if spot_idx is None:
        return picked_counts, spot_cell_lst, 0, 0, 0
    if metric == 'spot_cor' and exp_sc_values is not None and exp_st_values is not None:
        exp_sc = exp_sc_values
        exp_st = exp_st_values
    else:
        exp_sc = sc_exp_values
        exp_st = st_exp_values
    s_exp = exp_st[spot_idx]
    spot_cells_pos = np.array([cell_id_to_pos[c] for c in spot_cell_lst if c in cell_id_to_pos], dtype=np.int64)
    if spot_cells_pos.size == 0:
        return picked_counts, spot_cell_lst, 0, 0, 0
    total_exp_exp = exp_sc[spot_cells_pos].sum(axis=0)
    total_exp_iface = sc_exp_values[spot_cells_pos].sum(axis=0)
    curr_cor = None
    if metric == 'spot_cor':
        curr_cor = _rowwise_corr_with_vector(total_exp_exp.reshape(1, -1), s_exp)[0]

    nn_idx = [spot_to_idx.get(s, -1) for s in nn_spot]
    nn_idx = [i for i in nn_idx if i >= 0]
    if nn_idx:
        nn_spot_vals = sum_sc_agg_values[nn_idx]
        a_ss_vals = a_ss.to_numpy(dtype=np.float32, copy=False)
    else:
        nn_spot_vals = np.zeros((0, sc_exp_values.shape[1]), dtype=np.float32)
        a_ss_vals = np.zeros((0, len(lig_idx)), dtype=np.float32)

    exp_cor = 0
    inter_cor = 0
    mix_corr = 0
    spot_cells = list(spot_cell_lst)
    spot_cells_pos = np.array([cell_id_to_pos[c] for c in spot_cells if c in cell_id_to_pos], dtype=np.int64)
    spot_cell_types = cell_types[spot_cells_pos]
    unique_types = set(spot_cell_types)
    candi_pos_map = {}
    for tp in unique_types:
        candi_pos = tp_pos_dict.get(tp)
        if candi_pos is None or candi_pos.size == 0:
            continue
        candi_pos_map[tp] = candi_pos

    total_cells = len(spot_cells)
    for cell, removed_type in zip(spot_cells, spot_cell_types):
        spot_cell_lst.remove(cell)
        cell_pos = cell_id_to_pos.get(cell)
        if cell_pos is None:
            continue
        candi_pos = candi_pos_map.get(removed_type)
        if candi_pos is None:
            continue
        remain_exp = total_exp_exp - exp_sc[cell_pos]
        candi_exp = exp_sc[candi_pos]
        candi_exp_sum = candi_exp + remain_exp
        if swap_method == 'residual':
            target = s_exp * total_cells - remain_exp
            exp_candi_score = _rowwise_corr_with_vector(candi_exp, target)
            candi_sum_cor = _rowwise_corr_with_vector(candi_exp_sum, s_exp)
        else:
            exp_candi_score = _rowwise_corr_with_vector(candi_exp_sum, s_exp)
            candi_sum_cor = exp_candi_score

        remain_exp_iface = total_exp_iface - sc_exp_values[cell_pos]
        candi_exp_sum_iface = sc_exp_values[candi_pos] + remain_exp_iface
        interface_candi_cor = _interface_corr_candidates(candi_exp_sum_iface, nn_spot_vals, a_ss_vals, lig_idx, rec_idx)

        prob = half_life_prob(picked_counts, repeat_penalty)
        cor_df = interface_candi_cor * p + (1 - p) * exp_candi_score
        adj_cor = prob[candi_pos] * cor_df
        max_idx = int(np.argmax(adj_cor))

        selected_pos = candi_pos[max_idx]
        selected_cell = cell_ids[selected_pos]
        candidate_cor = float(candi_sum_cor[max_idx])
        accept_swap = True
        if metric == 'spot_cor' and curr_cor is not None:
            accept_swap = candidate_cor > curr_cor
        if accept_swap:
            spot_cell_lst.insert(0, selected_cell)
            exp_cor = candidate_cor
            inter_cor = float(interface_candi_cor[max_idx])
            mix_corr = float(adj_cor[max_idx])
            total_exp_exp = candi_exp_sum[max_idx]
            total_exp_iface = candi_exp_sum_iface[max_idx]
            if curr_cor is not None:
                curr_cor = candidate_cor
            picked_counts[cell_pos] -= 1
            picked_counts[selected_pos] += 1
        else:
            spot_cell_lst.insert(0, cell)
    if metric == 'spot_cor' and curr_cor is not None:
        exp_cor = float(curr_cor)
    return picked_counts, spot_cell_lst, exp_cor, inter_cor, mix_corr


@timeit
def cellReplaceByExp(spot_cell_lst, sc_exp, sc_meta, tp_idx_dict,
                        s_exp, picked_time,
                        repeat_penalty):
    '''
    Default mode, replace cell by highest exp and affinity correlation
    Legacy implementation is kept for reference but disabled.
    '''
    if False:
        for i in range(len(spot_cell_lst)):
            cell = spot_cell_lst[i]
            # print(cell)
            spot_cell_lst.remove(cell)
            print(spot_cell_lst)
            # calculate remain agg exp
            spot_remain_mat = sc_exp.loc[spot_cell_lst]
            remain_exp = np.sum(spot_remain_mat)
            # get candidate cells from the same type
            removed_type = sc_meta.loc[cell]['celltype']
            candi_cell_id = tp_idx_dict[removed_type]
            print('candi_cell_id', candi_cell_id)
            candi_exp = sc_exp.loc[candi_cell_id]
            # calculate replaced agg for each candidates
            candi_exp_sum = candi_exp + remain_exp
            # [exp cor]
            exp_candi_cor = candi_exp_sum.T.corrwith(s_exp)
            print('adj_cor', exp_candi_cor)
            prob = half_life_prob(picked_time['count'].values,repeat_penalty)
            picked_time['prob'] = prob
            cor_df = exp_candi_cor
            adj_cor_df = picked_time.loc[cor_df.index,'prob'] * cor_df
            max_idx = adj_cor_df.idxmax()
            spot_cell_lst.insert(0,max_idx)
            exp_cor = exp_candi_cor.loc[max_idx]
            picked_time.loc[cell,'count'] -= 1
            picked_time.loc[max_idx,'count'] += 1
            # break
        return picked_time, spot_cell_lst, exp_cor
    raise NotImplementedError("Use cellReplaceByExp_fast or expSwap_SPROUT instead.")


@timeit
def cellReplaceByExp_fast(spot, spot_cell_lst, sc_exp_values, cell_types, tp_pos_dict,
                          st_exp_values, picked_counts, repeat_penalty,
                          cell_id_to_pos, spot_to_idx, cell_ids, metric='correlation',
                          exp_sc_values=None, exp_st_values=None,
                          swap_method='correlation', use_gpu=False, candidate_topk=0):
    spot_idx = spot_to_idx.get(spot)
    if spot_idx is None:
        return picked_counts, spot_cell_lst, 0
    if metric == 'spot_cor' and exp_sc_values is not None and exp_st_values is not None:
        exp_sc = exp_sc_values
        exp_st = exp_st_values
    else:
        exp_sc = sc_exp_values
        exp_st = st_exp_values
    s_exp = exp_st[spot_idx]
    s_std = np.std(s_exp)
    if s_std == 0:
        return picked_counts, spot_cell_lst, 0
    norm_Es = s_exp / s_std

    spot_cells_pos = np.array([cell_id_to_pos[c] for c in spot_cell_lst if c in cell_id_to_pos], dtype=np.int64)
    if spot_cells_pos.size == 0:
        return picked_counts, spot_cell_lst, 0
    total_exp = exp_sc[spot_cells_pos].sum(axis=0)
    curr_cor = None
    if metric == 'spot_cor':
        curr_cor = _rowwise_corr_with_vector(total_exp.reshape(1, -1), s_exp)[0]

    exp_cor = 0
    spot_cells = list(spot_cell_lst)
    spot_cell_types = cell_types[spot_cells_pos]
    use_gpu_now = bool(
        use_gpu and metric == 'spot_cor' and torch is not None and torch.cuda.is_available()
    )
    if use_gpu_now:
        device = torch.device("cuda")
        exp_sc_t = torch.as_tensor(exp_sc, dtype=torch.float32, device=device)
        s_exp_t = torch.as_tensor(s_exp, dtype=torch.float32, device=device)
        total_exp_t = exp_sc_t[torch.as_tensor(spot_cells_pos, dtype=torch.long, device=device)].sum(dim=0)

    candi_pos_map = {}
    for tp in set(spot_cell_types):
        candi_pos = tp_pos_dict.get(tp)
        if candi_pos is None or candi_pos.size == 0:
            continue
        candi_pos_map[tp] = candi_pos

    total_cells = len(spot_cells)
    for cell, removed_type in zip(spot_cells, spot_cell_types):
        spot_cell_lst.remove(cell)
        cell_pos = cell_id_to_pos.get(cell)
        if cell_pos is None:
            continue
        candi_pos = candi_pos_map.get(removed_type)
        if candi_pos is None:
            continue
        if candidate_topk and candidate_topk > 0 and candi_pos.size > candidate_topk:
            if use_gpu_now:
                candi_idx_t = torch.as_tensor(candi_pos, dtype=torch.long, device=device)
                candi_exp_t = exp_sc_t[candi_idx_t]
                score_t = _rowwise_corr_with_vector_torch(candi_exp_t, s_exp_t)
                k = int(min(candidate_topk, candi_pos.size))
                top_local = torch.topk(score_t, k=k, largest=True).indices.detach().cpu().numpy()
                candi_pos = candi_pos[top_local]
            else:
                coarse = _rowwise_corr_with_vector(exp_sc[candi_pos], s_exp)
                k = int(min(candidate_topk, candi_pos.size))
                top_local = np.argpartition(coarse, -k)[-k:]
                candi_pos = candi_pos[top_local]

        remain_exp = total_exp - exp_sc[cell_pos]
        candi_exp = exp_sc[candi_pos]
        candi_exp_sum = candi_exp + remain_exp

        candi_std = np.std(candi_exp_sum, axis=1)
        candi_std[candi_std == 0] = 1.0
        norm_candi_sum = candi_exp_sum / candi_std[:, None]

        if metric == 'rmse':
            exp_candi_score = -np.sqrt(np.mean((norm_candi_sum - norm_Es) ** 2, axis=1))
            candi_sum_cor = exp_candi_score
        elif metric == 'spot_cor':
            if swap_method == 'residual':
                target = s_exp * total_cells - remain_exp
                if use_gpu_now:
                    candi_idx_t = torch.as_tensor(candi_pos, dtype=torch.long, device=device)
                    candi_exp_t = exp_sc_t[candi_idx_t]
                    target_t = s_exp_t * total_cells - total_exp_t + exp_sc_t[cell_pos]
                    exp_candi_score_t = _rowwise_corr_with_vector_torch(candi_exp_t, target_t)
                    prob_t = torch.as_tensor(half_life_prob(picked_counts, repeat_penalty), dtype=torch.float32, device=device)
                    adj_t = prob_t[candi_idx_t] * exp_candi_score_t
                    max_idx = int(torch.argmax(adj_t).item())
                    selected_pos = candi_pos[max_idx]
                    selected_cell = cell_ids[selected_pos]
                    candi_sum_cor = _rowwise_corr_with_vector(candi_exp_sum, s_exp)
                    candidate_cor = float(candi_sum_cor[max_idx])
                else:
                    exp_candi_score = _rowwise_corr_with_vector(candi_exp, target)
                    candi_sum_cor = _rowwise_corr_with_vector(candi_exp_sum, s_exp)
            else:
                if use_gpu_now:
                    candi_idx_t = torch.as_tensor(candi_pos, dtype=torch.long, device=device)
                    candi_exp_sum_t = exp_sc_t[candi_idx_t] + total_exp_t - exp_sc_t[cell_pos]
                    exp_candi_score_t = _rowwise_corr_with_vector_torch(candi_exp_sum_t, s_exp_t)
                    prob_t = torch.as_tensor(half_life_prob(picked_counts, repeat_penalty), dtype=torch.float32, device=device)
                    adj_t = prob_t[candi_idx_t] * exp_candi_score_t
                    max_idx = int(torch.argmax(adj_t).item())
                    selected_pos = candi_pos[max_idx]
                    selected_cell = cell_ids[selected_pos]
                    candidate_cor = float(exp_candi_score_t[max_idx].item())
                    candi_sum_cor = None
                else:
                    exp_candi_score = _rowwise_corr_with_vector(candi_exp_sum, s_exp)
                    candi_sum_cor = exp_candi_score
        else:
            if swap_method == 'residual':
                target = s_exp * total_cells - remain_exp
                exp_candi_score = _rowwise_corr_with_vector(candi_exp, target)
                candi_sum_cor = (norm_candi_sum @ norm_Es) / norm_Es.shape[0]
            else:
                exp_candi_score = (norm_candi_sum @ norm_Es) / norm_Es.shape[0]
                candi_sum_cor = exp_candi_score

        if not (use_gpu_now and metric == 'spot_cor'):
            prob = half_life_prob(picked_counts, repeat_penalty)
            adj_cor = prob[candi_pos] * exp_candi_score
            max_idx = int(np.argmax(adj_cor))
            selected_pos = candi_pos[max_idx]
            selected_cell = cell_ids[selected_pos]
            candidate_cor = float(candi_sum_cor[max_idx])
        accept_swap = True
        if metric == 'spot_cor' and curr_cor is not None:
            accept_swap = candidate_cor > curr_cor
        if accept_swap:
            spot_cell_lst.insert(0, selected_cell)
            exp_cor = candidate_cor
            total_exp = candi_exp_sum[max_idx]
            if use_gpu_now:
                total_exp_t = exp_sc_t[selected_pos] + total_exp_t - exp_sc_t[cell_pos]
            if curr_cor is not None:
                curr_cor = candidate_cor
            picked_counts[cell_pos] -= 1
            picked_counts[selected_pos] += 1
        else:
            spot_cell_lst.insert(0, cell)
    if metric == 'spot_cor' and curr_cor is not None:
        exp_cor = float(curr_cor)
    return picked_counts, spot_cell_lst, exp_cor




def expSwap_SPROUT(spot_cell_lst, s_sc_exp, sc_meta, trans_id_idx, tp_idx_dict,
                        s_exp, after_picked_time, gene_num,
                        repeat_penalty, metric = 'correlation'):
    '''
    s_exp: csr_matrix of spot exp
    s_sc_exp: csr_matrix of norm_sc_exp
    trans_id_idx: df of number index and cell_id 
    gene_num = len(lr_hvg_genes)
    '''
    max_cor_rep = -999
    max_cor = 999
    norm_Es = csr_matrix(s_exp/np.std(s_exp))
    for i in range(len(spot_cell_lst)):
        cell_i = spot_cell_lst[i]
        spot_cell_lst.remove(cell_i)
        # print('i', i, spot_cell_lst)
        spot_cell_idx = id_to_idx(trans_id_idx, spot_cell_lst)
        # print(spot_cell_idx)
        spot_remain_mat = s_sc_exp[spot_cell_idx]
        remain_exp = np.array(np.sum(spot_remain_mat,axis = 0))
        removed_type = sc_meta.loc[cell_i]['celltype']
        candi_cell_id = list(tp_idx_dict[removed_type])
        # print('candi_cell_id', candi_cell_id)
        candi_idx = id_to_idx(trans_id_idx, candi_cell_id)    
        candi_exp = s_sc_exp[candi_idx]
        candi_sum = candi_exp + remain_exp
        # print('candi_sum', candi_sum)
        norm_candi_sum = csr_matrix(candi_sum/np.std(candi_sum,axis = 1))
        # TODO rmse add
        # candi_cor_list = np.dot(norm_Es, norm_candi_sum.T)/gene_num
        if metric == 'correlation':
            candi_cor_list = np.dot(norm_Es, norm_candi_sum.T)/gene_num
        elif metric == 'rmse':
            candi_cor_list = csr_matrix(-1*(np.sqrt(np.mean((norm_candi_sum.toarray() - norm_Es.toarray())**2, axis=1))))
        # print('candi_cor_list', candi_cor_list.toarray())
        ### 
        prob = half_life_prob(after_picked_time['count'].values, repeat_penalty)
        after_picked_time['prob'] = prob
        adj_cor = candi_cor_list.multiply(prob[candi_idx]).toarray()
        # print('adj_cor', adj_cor)
        candi_max_cor_idx = np.argsort(adj_cor[0])[-1:][0]
        # print('adj_cor', adj_cor)
        # print('candi_max_cor_idx', candi_max_cor_idx)
        swaped_idx = candi_idx[candi_max_cor_idx]
        swaped_id = candi_cell_id[candi_max_cor_idx]
        ###        
        new_agg = remain_exp + s_sc_exp[swaped_idx]
        # max_cor = np.corrcoef(new_agg, s_exp)[0][1]
        # TODO rmse add
        # print('new_agg', np.array(new_agg), np.array(new_agg).shape)
        # print('s_exp', s_exp.values, s_exp.values.shape)
        max_cor = eva_metric(np.array(new_agg), s_exp.values, metric = 'correlation')
        # max_cor = eva_metric(np.array(new_agg), s_exp.values, metric = 'rmse')
        # print(i, ":", max_cor)
        tmp_cell_id = spot_cell_lst.copy()
        # print(f'max_cor is {max_cor}; max_rep is {max_cor_rep}')
        if max_cor > max_cor_rep:
            max_cor_rep = max_cor
            #print(f'insert {swaped_id} to {tmp_cell_id}')
            tmp_cell_id.insert(0,swaped_id) 
            after_picked_time.loc[swaped_id] += 1
            after_picked_time.loc[cell_i] -= 1
        else:
            #print(f'insert {cell_i} back to {tmp_cell_id}')
            tmp_cell_id.insert(0,cell_i)
        spot_cell_lst = tmp_cell_id
    return after_picked_time, spot_cell_lst, max_cor_rep
