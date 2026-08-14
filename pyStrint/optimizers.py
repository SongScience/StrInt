import pandas as pd
import numpy as np
from scipy.special import digamma
from scipy.spatial import distance_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from scipy import sparse
import scanpy as sc
from sklearn.metrics import mean_squared_error
import time
import logging
from multiprocessing import Pool

import umap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None

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


def pear(D,D_re):
    tmp = np.corrcoef(D.flatten(order='C'), D_re.flatten(order='C'))
    return tmp[0,1] 



def loss_adj(loss1,loss2,loss3,loss4,loss5, eps=1e-12):
    denom2 = loss2 if loss2 != 0 else eps
    denom3 = loss3 if loss3 != 0 else eps
    denom4 = loss4 if loss4 != 0 else eps
    denom5 = loss5 if loss5 != 0 else eps
    adj2 = loss1/denom2
    adj3 = loss1/denom3
    adj4 = loss1/denom4
    adj5 = loss1/denom5
    return adj2,adj3,adj4,adj5

# @timeit
def cal_term1_old(alter_sc_exp,sc_meta,st_exp):
    '''
    1. First term, towards spot expression
    '''  
    # 1.1 Aggregate exp of chosen cells for each spot 
    alter_sc_exp['spot'] = sc_meta['spot']
    sc_spot_sum = alter_sc_exp.groupby('spot').sum()
    del alter_sc_exp['spot']
    sc_spot_sum = sc_spot_sum.loc[st_exp.index]
    # 1.2 equalize sc_spot_sum by dividing cell number each spot
    cell_n_spot = sc_meta.groupby('spot').count().loc[st_exp.index]
    div_sc_spot = sc_spot_sum.div(cell_n_spot.iloc[:,0].values, axis=0)
    # 1.2 Calculate gradient
    term1_df = 2 * (div_sc_spot - st_exp)
    # 1.3 Broadcast gradient for each cell
    term1_df = term1_df.loc[sc_meta['spot']]
    term1_df.index = alter_sc_exp.index
    loss_1 = mean_squared_error(st_exp, div_sc_spot)
    return term1_df,loss_1


# @timeit   
def cal_term1(sc_exp,sc_meta,st_exp,hvg,W_HVG,weight=None):
    '''
    1. First term, towards spot expression
    Optimized with vectorized aggregation to avoid DataFrame groupby overhead.
    '''
    # add for merfish
    alter_sc_exp = sc_exp[st_exp.columns]
    exp_values = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    st_values = st_exp.to_numpy(dtype=np.float32, copy=False)
    n_spots = st_exp.shape[0]
    n_genes = st_exp.shape[1]

    # Align spot codes to st_exp index order for fast aggregation
    spot_codes = pd.Categorical(sc_meta['spot'], categories=st_exp.index, ordered=True).codes
    valid = spot_codes >= 0

    # 1.1 Aggregate exp of chosen cells for each spot (vectorized)
    sc_spot_sum = np.zeros((n_spots, n_genes), dtype=exp_values.dtype)
    cell_n_spot = np.zeros(n_spots, dtype=np.int64)
    np.add.at(sc_spot_sum, spot_codes[valid], exp_values[valid])
    np.add.at(cell_n_spot, spot_codes[valid], 1)

    # 1.2 equalize sc_spot_sum by dividing cell number each spot
    cell_n_spot[cell_n_spot == 0] = 1
    div_sc_spot = sc_spot_sum / cell_n_spot[:, None]

    # 1.3 Calculate gradient
    term1_vals = 2 * (div_sc_spot - st_values)

    # v5 add weight on hvg
    if weight is None:
        weight_vals = np.ones((n_spots, n_genes), dtype=term1_vals.dtype)
        hvg = list(hvg)
        if hvg:
            hvg_idx = st_exp.columns.get_indexer(hvg)
            hvg_idx = hvg_idx[hvg_idx >= 0]
            if hvg_idx.size > 0:
                weight_vals[:, hvg_idx] = W_HVG
    else:
        weight_vals = weight[st_exp.columns].to_numpy(dtype=np.float32, copy=False)
    term1_vals *= weight_vals

    term1_df = pd.DataFrame(term1_vals, index=st_exp.index, columns=st_exp.columns)

    # 1.4 Broadcast gradient for each cell
    term1_df = term1_df.loc[sc_meta['spot']]
    term1_df.index = alter_sc_exp.index
    loss_1 = mean_squared_error(st_exp,div_sc_spot)
    # add for merfish
    term1_df = complete_other_genes(sc_exp, term1_df)
    return term1_df,loss_1


def closed_form_term1_update(alter_sc_exp, sc_meta, st_exp, blend=1.0):
    """
    Closed-form projection for term1: set per-spot mean equal to st_exp.
    Uses a convex blend to avoid overwriting other terms completely.
    """
    if blend is None:
        blend = 1.0
    blend = float(blend)
    if blend <= 0:
        return alter_sc_exp
    if blend > 1:
        blend = 1.0
    st_cols = st_exp.columns
    values = alter_sc_exp[st_cols].to_numpy(copy=True)
    st_values = st_exp.to_numpy(dtype=values.dtype, copy=False)
    spot_codes = pd.Categorical(sc_meta['spot'], categories=st_exp.index, ordered=True).codes
    valid = spot_codes >= 0
    if np.any(valid):
        targets = st_values[spot_codes[valid]]
        if blend >= 1.0:
            values[valid] = targets
        else:
            values[valid] = values[valid] * (1.0 - blend) + targets * blend
        alter_sc_exp.loc[:, st_cols] = values
    return alter_sc_exp


def project_spot_sum_by_sc_ref(alter_sc_exp, sc_meta, st_exp, sc_ref, blend=1.0, eps=1e-8):
    """
    Project per-spot expression so each gene sums to the spot expression,
    distributing values across cells proportional to smurf/sc_ref profiles.
    """
    if blend is None:
        blend = 1.0
    blend = float(blend)
    if blend <= 0:
        return alter_sc_exp
    if blend > 1:
        blend = 1.0
    st_cols = st_exp.columns
    vals = alter_sc_exp[st_cols].to_numpy(copy=True)
    st_vals = st_exp.to_numpy(dtype=vals.dtype, copy=False)
    if hasattr(sc_ref, "to_numpy"):
        ref_vals = sc_ref[st_cols].to_numpy(dtype=vals.dtype, copy=False)
    else:
        ref_vals = np.asarray(sc_ref, dtype=vals.dtype)
    if ref_vals.shape[1] != vals.shape[1]:
        ref_vals = ref_vals[:, :vals.shape[1]]
    spot_codes = pd.Categorical(sc_meta['spot'], categories=st_exp.index, ordered=True).codes
    proj = np.zeros_like(vals)
    for spot_idx in range(st_vals.shape[0]):
        cell_idx = np.where(spot_codes == spot_idx)[0]
        if cell_idx.size == 0:
            continue
        ref_sub = ref_vals[cell_idx]
        denom = ref_sub.sum(axis=0)
        weights = np.zeros_like(ref_sub)
        nonzero = denom > float(eps)
        if np.any(nonzero):
            weights[:, nonzero] = ref_sub[:, nonzero] / denom[nonzero]
        if np.any(~nonzero):
            weights[:, ~nonzero] = 1.0 / float(cell_idx.size)
        proj[cell_idx] = weights * st_vals[spot_idx]
    if blend >= 1.0:
        vals = proj
    else:
        vals = vals * (1.0 - blend) + proj * blend
    alter_sc_exp.loc[:, st_cols] = vals
    return alter_sc_exp


def spot_sum_projection_vals(alter_sc_exp, sc_meta, st_exp, sc_ref, eps=1e-8):
    """
    Compute projected per-spot expression values for st_exp genes only.
    Returns an array shaped (n_cells, n_st_genes).
    """
    st_cols = st_exp.columns
    st_vals = st_exp.to_numpy(dtype=np.float32, copy=False)
    if hasattr(sc_ref, "to_numpy"):
        ref_vals = sc_ref[st_cols].to_numpy(dtype=np.float32, copy=False)
    else:
        ref_vals = np.asarray(sc_ref, dtype=np.float32)
    if ref_vals.shape[1] != st_vals.shape[1]:
        ref_vals = ref_vals[:, :st_vals.shape[1]]
    spot_codes = pd.Categorical(sc_meta['spot'], categories=st_exp.index, ordered=True).codes
    proj = np.zeros((alter_sc_exp.shape[0], st_vals.shape[1]), dtype=np.float32)
    for spot_idx in range(st_vals.shape[0]):
        cell_idx = np.where(spot_codes == spot_idx)[0]
        if cell_idx.size == 0:
            continue
        ref_sub = ref_vals[cell_idx]
        denom = ref_sub.sum(axis=0)
        weights = np.zeros_like(ref_sub)
        nonzero = denom > float(eps)
        if np.any(nonzero):
            weights[:, nonzero] = ref_sub[:, nonzero] / denom[nonzero]
        if np.any(~nonzero):
            weights[:, ~nonzero] = 1.0 / float(cell_idx.size)
        proj[cell_idx] = weights * st_vals[spot_idx]
    return proj


# @timeit
def cal_term2(alter_sc_exp,sc_distribution):
    '''
    2. Second term, towards sc cell-type specific expression
    '''
    sc_dist_vals = sc_distribution.to_numpy(dtype=np.float32, copy=False) if hasattr(sc_distribution, "to_numpy") else np.asarray(sc_distribution, dtype=np.float32)
    alter_vals = alter_sc_exp.to_numpy(dtype=np.float32, copy=False) if hasattr(alter_sc_exp, "to_numpy") else np.asarray(alter_sc_exp, dtype=np.float32)
    term2 = 2.0 * (alter_vals - sc_dist_vals)
    term2_df = pd.DataFrame(term2,index = alter_sc_exp.index,columns=alter_sc_exp.columns, dtype=np.float32)
    loss_2 = mean_squared_error(sc_dist_vals, alter_vals)
    return term2_df,loss_2     


def cal_term2_values(alter_sc_exp, sc_distribution):
    """Compute term2 values and loss without constructing a DataFrame."""
    sc_dist_vals = sc_distribution.to_numpy(dtype=np.float32, copy=False) if hasattr(sc_distribution, "to_numpy") else np.asarray(sc_distribution, dtype=np.float32)
    alter_vals = alter_sc_exp.to_numpy(dtype=np.float32, copy=False) if hasattr(alter_sc_exp, "to_numpy") else np.asarray(alter_sc_exp, dtype=np.float32)
    term2_vals = 2.0 * (alter_vals - sc_dist_vals)
    loss_2 = mean_squared_error(sc_dist_vals, alter_vals)
    return term2_vals, loss_2


def _generalized_kl_loss_grad(target, pred, eps=1e-8):
    target_vals = np.asarray(target, dtype=np.float32)
    pred_vals = np.asarray(pred, dtype=np.float32)
    pred_safe = pred_vals + float(eps)
    target_safe = target_vals + float(eps)
    loss = np.mean(target_safe * (np.log(target_safe) - np.log(pred_safe)) + pred_safe - target_safe)
    grad = 1.0 - (target_safe / pred_safe)
    return grad, float(loss)


def cal_term1_kl(sc_exp, sc_meta, st_exp, hvg, W_HVG, weight=None, eps=1e-8):
    """
    KL/CE-form term1 (generalized KL / I-divergence) toward spot expression.
    Uses MSE solution as initialization and applies generalized KL gradient.
    """
    alter_sc_exp = sc_exp[st_exp.columns]
    exp_values = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    st_values = st_exp.to_numpy(dtype=np.float32, copy=False)
    n_spots = st_exp.shape[0]
    n_genes = st_exp.shape[1]

    spot_codes = pd.Categorical(sc_meta['spot'], categories=st_exp.index, ordered=True).codes
    valid = spot_codes >= 0

    sc_spot_sum = np.zeros((n_spots, n_genes), dtype=exp_values.dtype)
    cell_n_spot = np.zeros(n_spots, dtype=np.int64)
    np.add.at(sc_spot_sum, spot_codes[valid], exp_values[valid])
    np.add.at(cell_n_spot, spot_codes[valid], 1)
    cell_n_spot[cell_n_spot == 0] = 1
    div_sc_spot = sc_spot_sum / cell_n_spot[:, None]

    grad_spot, loss_1 = _generalized_kl_loss_grad(st_values, div_sc_spot, eps=eps)

    if weight is None:
        weight_vals = np.ones((n_spots, n_genes), dtype=exp_values.dtype)
        hvg = list(hvg)
        if hvg:
            hvg_idx = st_exp.columns.get_indexer(hvg)
            hvg_idx = hvg_idx[hvg_idx >= 0]
            if hvg_idx.size > 0:
                weight_vals[:, hvg_idx] = W_HVG
    else:
        weight_vals = weight[st_exp.columns].to_numpy(dtype=np.float32, copy=False)
    grad_spot = grad_spot * weight_vals

    spot_codes_safe = spot_codes.copy()
    spot_codes_safe[~valid] = 0
    term1_cells = grad_spot[spot_codes_safe]
    if not valid.all():
        term1_cells[~valid] = 0
    term1_df = pd.DataFrame(term1_cells, index=alter_sc_exp.index, columns=alter_sc_exp.columns)
    term1_df = complete_other_genes(sc_exp, term1_df)
    return term1_df, loss_1


def cal_term2_kl(alter_sc_exp, sc_distribution, eps=1e-8):
    """
    KL/CE-form term2 (generalized KL / I-divergence) toward sc distribution.
    """
    sc_dist_vals = sc_distribution.to_numpy(dtype=np.float32, copy=False) if hasattr(sc_distribution, "to_numpy") else np.asarray(sc_distribution, dtype=np.float32)
    alter_vals = alter_sc_exp.to_numpy(dtype=np.float32, copy=False) if hasattr(alter_sc_exp, "to_numpy") else np.asarray(alter_sc_exp, dtype=np.float32)
    term2_vals, loss_2 = _generalized_kl_loss_grad(sc_dist_vals, alter_vals, eps=eps)
    term2_df = pd.DataFrame(term2_vals, index=alter_sc_exp.index, columns=alter_sc_exp.columns, dtype=np.float32)
    return term2_df, loss_2


# @timeit
def findSpotKNN_old(st_coord, st_tp): 
    coordinates = st_coord.values
    if st_tp != 'slide-seq':
        k = 6
    else:
        k = 6
    kdtree = KDTree(coordinates)
    distances, indices = kdtree.query(coordinates, k+1)
    knn_dict = {}
    spots_id = st_coord.index.tolist()
    for i, nearest_indices in enumerate(indices):
        point = nearest_indices[0]
        knn = nearest_indices[1:].tolist()
        knn_dict[spots_id[point]] = [spots_id[i] for i in knn]
    return knn_dict


def findSpotKNN(st_coord, st_tp):
    # TODO
    # write on 1107 to replace findSpotNeighbor in the future
    # no need for slide-seq exceptions
    # for sc usage, moderate k? == further research
    thred = 5
    total_sum = 0
    coordinates = st_coord.values
    if st_tp != 'slide-seq':
        k = 6
    else:
        k = 6
    kdtree = KDTree(coordinates)
    distances, indices = kdtree.query(coordinates, k=k + 1)
    # Autocalculate the outlier threshold based on the distances
    threshold = np.percentile(distances[:, 1:], thred)
    indices = pd.DataFrame(indices,index = st_coord.index)
    distances = pd.DataFrame(distances,index = st_coord.index)
    # print(threshold)
    knn_dict = {}
    spots_id = st_coord.index.tolist()
    for key in spots_id:
        nearest_neighbors = indices.loc[key, 1:]
        nearest_distances = distances.loc[key, 1:]
        # keep nn within threshold
        filtered_neighbors = nearest_neighbors[nearest_distances <= threshold]
        str_idx = [spots_id[index] for index in filtered_neighbors]
        knn_dict[key] = str_idx
        total_sum += len(str_idx)
    print(f'By setting k as {k}, each spot has average {total_sum/st_coord.shape[0]:.2f} neighbors.')
    return knn_dict


# # @timeit
def findSpotNeighbor(st_coord,st_tp):
    # old
    all_x = np.sort(list(set(st_coord.iloc[:,0])))
    delta_x = all_x[1] - all_x[0]

    if st_tp == 'visium':
        n_thred = 2*delta_x
        print(f'visium format, setting threshold as {n_thred}')
    else:
        n_thred = delta_x + 0.001

    st_dist = pd.DataFrame(distance_matrix(st_coord,st_coord),columns = st_coord.index, index = st_coord.index)
    st_dist[(st_dist < n_thred)&(st_dist >0)] = 1
    st_dist[st_dist != 1] = 0
    return st_dist


# # @timeit
def findCellKNN(st_coord,st_tp,sc_meta,sc_coord,k, spot_codes_all=None, spot_levels=None, centroid_arr=None): 
    '''
    st_tp = 'visium'
    k = 2
    sc_coord = obj_spex.sc_coord
    '''
    if st_tp == 'slide-seq':
        return findCellKNN_slide(sc_meta, sc_coord)

    sc_index = sc_meta.index.to_numpy()
    spot_labels = sc_meta['spot'].to_numpy()
    if spot_codes_all is None or spot_levels is None:
        spot_codes_all, spot_levels = pd.factorize(spot_labels)
    n_spots = len(spot_levels)
    sc_knn = {key: [] for key in sc_index}

    # 1. Find neighboring cells from adjacent spot using spot coordinates
    _, sc_coord_st = sc_prep(st_coord, sc_meta)
    sc_coord_st_arr = sc_coord_st[['st_x', 'st_y']].to_numpy(dtype=np.float32, copy=False)
    all_x = np.sort(np.unique(sc_coord_st_arr[:, 0]))
    if len(all_x) > 1:
        delta_x = all_x[1] - all_x[0]
    else:
        delta_x = 0
    if st_tp == 'visium':
        n_thred = 2 * delta_x
    else:
        n_thred = delta_x + 0.001

    tree = KDTree(sc_coord_st_arr)
    neighbor_idx_list = tree.query_ball_point(sc_coord_st_arr, r=n_thred)

    # 2. Build centroid coordinates for cell-wise threshold
    sc_coord_arr = sc_coord.to_numpy(dtype=np.float32, copy=False) if isinstance(sc_coord, pd.DataFrame) else np.asarray(sc_coord, dtype=np.float32)
    if centroid_arr is None:
        sc_coord_df = pd.DataFrame(sc_coord_arr, columns=['x', 'y'], index=sc_index)
        sc_coord_df['spot'] = spot_labels
        sc_centroid = sc_coord_df.groupby('spot', sort=False)[['x', 'y']].mean()
        _, sc_centroid_cells = sc_prep(sc_centroid, sc_meta)
        centroid_arr = sc_centroid_cells[['st_x', 'st_y']].to_numpy(dtype=np.float32, copy=False)

    # 3. For each cell, pick k nearest neighbors per neighbor spot
    for i, cell_id in enumerate(sc_index):
        neighbors = neighbor_idx_list[i]
        if not neighbors:
            continue
        neighbors = [n for n in neighbors if n != i]
        if not neighbors:
            continue
        neighbors = np.asarray(neighbors, dtype=np.int64)
        diffs = sc_coord_arr[neighbors] - sc_coord_arr[i]
        dist_real = np.sqrt((diffs ** 2).sum(axis=1))
        diffs_c = centroid_arr[neighbors] - centroid_arr[i]
        dist_centroid = np.sqrt((diffs_c ** 2).sum(axis=1))
        keep = dist_centroid >= (dist_real - 1e-8)
        if not np.any(keep):
            continue
        neighbors = neighbors[keep]
        dist_real = dist_real[keep]
        spot_codes = spot_codes_all[neighbors]
        # Sort by spot code then distance, then pick first k per spot
        order = np.lexsort((dist_real, spot_codes))
        counts = np.zeros(n_spots, dtype=np.int32)
        for idx in order:
            scode = spot_codes[idx]
            if counts[scode] < k:
                sc_knn[cell_id].append(sc_index[neighbors[idx]])
                counts[scode] += 1

    # Fallback: if no neighbors were found, use global kNN in sc_coord
    if not any(sc_knn.values()):
        k_use = max(int(k), 1)
        tree = KDTree(sc_coord_arr)
        distances, indices = tree.query(sc_coord_arr, k=k_use + 1)
        for i, cell_id in enumerate(sc_index):
            neigh_idx = indices[i][1:] if indices.ndim > 1 else []
            sc_knn[cell_id] = [sc_index[j] for j in neigh_idx if j != i]

    # remove no neighbor cells
    empty_keys = [k for k, v in sc_knn.items() if not v]
    for key in empty_keys:
        del sc_knn[key]

    # Fallback: if too few cells have neighbors, use global kNN
    min_keep = max(5, int(0.1 * len(sc_index)))
    if len(sc_knn) < min_keep:
        k_use = max(int(k), 1)
        tree = KDTree(sc_coord_arr)
        distances, indices = tree.query(sc_coord_arr, k=k_use + 1)
        sc_knn = {}
        for i, cell_id in enumerate(sc_index):
            neigh_idx = indices[i][1:] if indices.ndim > 1 else []
            sc_knn[cell_id] = [sc_index[j] for j in neigh_idx if j != i]
    return sc_knn


# @timeit
def findCellKNN_slide(sc_meta,sc_coord):
    # k=4+1 include self
    k = 7
    # drop 95 out of 100
    thred = 95
    sum = 0
    sc_knn = {}
    for key in sc_meta.index.tolist():
        sc_knn[key] = []
    idx_lst = sc_coord.index
    kdtree = KDTree(sc_coord)
    distances, indices = kdtree.query(sc_coord, k=k)
    threshold = np.percentile(distances[:, 1:], thred)
    # print(threshold)
    indices = pd.DataFrame(indices,index = sc_meta.index)
    distances = pd.DataFrame(distances,index = sc_meta.index)
    for key in sc_meta.index.tolist():
        # remove self
        nearest_neighbors = indices.loc[key, 1:]
        nearest_distances = distances.loc[key, 1:]
        # keep nn within threshold
        filtered_neighbors = nearest_neighbors[nearest_distances <= threshold]
        str_idx = [idx_lst[index] for index in filtered_neighbors]
        sc_knn[key] = str_idx
        sum += len(str_idx)
    print(f'Running slide-seq data, each cell has average {sum/sc_meta.shape[0]} neighbor.')
    return sc_knn


# @timeit
def apply_nsmallest(x, k, nn_dict):
    x = x.dropna(axis=1, how='all')
    for cell in x.columns:
        #print(cell,x[cell].nsmallest(k).index.tolist())
        if cell != 'spot':
            nn_dict[cell].extend(x[cell].nsmallest(k).index.tolist())
    return nn_dict


# @timeit
def complete_other_genes(alter_sc_exp, term_LR_df):
    '''
    Complete non-LR genes as zero for term3,4
    '''
    base_vals = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    term_df = pd.DataFrame(
        np.zeros(base_vals.shape, dtype=base_vals.dtype),
        columns=alter_sc_exp.columns,
        index=alter_sc_exp.index,
    )
    term_df.update(term_LR_df)
    return term_df


# @timeit
def cal_term3(alter_sc_exp,sc_knn,aff,sc_dist,rl_agg, ind=None, n_cp=None):
    # v3 added norm_aff and norm rl_cp to regulize the values
    # v5 Updated the calculation of term3 with [ind], accelerated.
    # v5 Scale both data to a fixed range every time
    MIN = 0
    MAX = 100
    if isinstance(aff, pd.DataFrame):
        norm_aff = np.sqrt(aff.astype(np.float32, copy=False) / 2.0)
    else:
        norm_aff = np.sqrt(np.asarray(aff, dtype=np.float32) / 2.0)
    term3_LR = pd.DataFrame()
    sc_dist_re = sc_dist.copy()
    if isinstance(sc_dist_re, pd.DataFrame):
        sc_dist_re = sc_dist_re.astype(np.float32)
    mask = sc_dist_re != 0
    sc_dist_re[mask] = 1 / sc_dist_re[mask]
    # ind: the neighboring indicator matrix
    # Row: cell; Col: the neighbor of this cell
    if ind is None or n_cp is None:
        ind = pd.DataFrame(False, columns=norm_aff.columns, index=norm_aff.index)
        for idx, cp in sc_knn.items():
            ind.loc[idx, cp] = True
        # n_cp: the neighboring cells of each cell (row-wise summation)
        n_cp = ind.sum(axis = 1)
    cp_aff_df = norm_aff[ind]
    cp_dist_df = sc_dist_re[ind]
    if isinstance(cp_aff_df, pd.DataFrame):
        cp_aff_df = cp_aff_df.astype(np.float32)
    if isinstance(cp_dist_df, pd.DataFrame):
        cp_dist_df = cp_dist_df.astype(np.float32)
    cp_aff_adj = utils.scale_global_MIN_MAX(cp_aff_df,MIN,MAX)
    cp_dist_adj = utils.scale_global_MIN_MAX(cp_dist_df,MIN,MAX)
    tmp1 = cp_aff_adj - cp_dist_adj
    tmp2 = tmp1.fillna(0)
    term3_LR = 2*rl_agg.dot(tmp2.T)/n_cp
    # fillna(0) because if a cell has no neighbor, /n_cp cause divide by zero error; generates NA.
    term3_LR = term3_LR.fillna(0)
    # print('\t filled na')
    # Calculating the loss; Normlize by total neighbor count
    loss = np.sum(tmp2**2).sum()
    loss /= n_cp.sum()
    # v4 simplify
    term3_df = complete_other_genes(alter_sc_exp, term3_LR.T)
    return term3_df,loss


def cal_term3_sparse(
    alter_sc_exp,
    sc_knn,
    aff,
    sc_dist,
    rl_agg,
    ind=None,
    n_cp=None,
    row_idx=None,
    col_idx=None,
    cp_dist_adj=None,
    n_cells=None,
):
    MIN = 0
    MAX = 100

    if ind is None or n_cp is None:
        ind, n_cp = build_knn_indicator_sparse(sc_knn, alter_sc_exp.index)
    if row_idx is None or col_idx is None:
        row_idx, col_idx = ind.nonzero()

    if sparse.issparse(aff):
        if n_cells is None:
            n_cells = aff.shape[0]
        cp_aff_sparse = np.sqrt(aff[row_idx, col_idx].A1.astype(np.float32, copy=False) / 2.0)
    elif isinstance(aff, pd.DataFrame):
        norm_aff = np.sqrt(aff.to_numpy(dtype=np.float32, copy=False) / 2.0)
        if n_cells is None:
            n_cells = norm_aff.shape[0]
        cp_aff_sparse = norm_aff[row_idx, col_idx]
    else:
        norm_aff = np.sqrt(np.asarray(aff, dtype=np.float32) / 2.0)
        if n_cells is None:
            n_cells = norm_aff.shape[0]
        cp_aff_sparse = norm_aff[row_idx, col_idx]

    cp_aff_adj = _scale_minmax_array(cp_aff_sparse, MIN, MAX)

    if cp_dist_adj is None:
        if sparse.issparse(sc_dist):
            sc_dist_re = sc_dist.tocsr(copy=True)
            if sc_dist_re.nnz:
                sc_dist_re.data = 1.0 / sc_dist_re.data.astype(np.float32, copy=False)
        elif isinstance(sc_dist, pd.DataFrame):
            sc_dist_re = sc_dist.to_numpy(dtype=np.float32, copy=False)
            mask = sc_dist_re != 0
            sc_dist_re = sc_dist_re.copy()
            sc_dist_re[mask] = 1.0 / sc_dist_re[mask]
        else:
            sc_dist_re = np.asarray(sc_dist, dtype=np.float32)
            mask = sc_dist_re != 0
            sc_dist_re = sc_dist_re.copy()
            sc_dist_re[mask] = 1.0 / sc_dist_re[mask]

        if sparse.issparse(sc_dist_re):
            cp_dist_sparse = sc_dist_re[row_idx, col_idx].A1
        else:
            cp_dist_sparse = sc_dist_re[row_idx, col_idx]
        cp_dist_adj = _scale_minmax_array(cp_dist_sparse, MIN, MAX)

    tmp2_sparse = cp_aff_adj - cp_dist_adj

    rl_vals = rl_agg.to_numpy(dtype=np.float32, copy=False)
    if nb is not None and row_idx.size > 0:
        term3_vals = _accumulate_term3_edges_numba(
            rl_vals, row_idx.astype(np.int64), col_idx.astype(np.int64), tmp2_sparse.astype(np.float32), n_cells
        )
    else:
        tmp2_mat = sparse.csr_matrix((tmp2_sparse, (row_idx, col_idx)), shape=(n_cells, n_cells))
        term3_vals = rl_vals.dot(tmp2_mat.T)
    n_cp_safe = np.where(n_cp == 0, 1, n_cp).astype(np.float32)
    term3_vals = 2.0 * term3_vals / n_cp_safe
    term3_LR = pd.DataFrame(term3_vals, index=rl_agg.index, columns=rl_agg.columns)
    term3_LR = term3_LR.fillna(0)

    total_ncp = float(n_cp.sum())
    if total_ncp > 0:
        loss = float(np.sum(tmp2_sparse ** 2) / total_ncp)
    else:
        loss = 0.0
    term3_df = complete_other_genes(alter_sc_exp, term3_LR.T)
    return term3_df, loss


def build_knn_indicator_df(sc_knn, index):
    ind = pd.DataFrame(False, columns=index, index=index)
    for idx, cp in sc_knn.items():
        ind.loc[idx, cp] = True
    n_cp = ind.sum(axis=1)
    return ind, n_cp


def build_knn_indicator_sparse(sc_knn, index):
    idx_map = {k: i for i, k in enumerate(index)}
    rows = []
    cols = []
    for idx, cp in sc_knn.items():
        i = idx_map.get(idx)
        if i is None:
            continue
        for c in cp:
            j = idx_map.get(c)
            if j is None:
                continue
            rows.append(i)
            cols.append(j)
    if rows:
        data = np.ones(len(rows), dtype=np.float32)
        ind = sparse.csr_matrix((data, (rows, cols)), shape=(len(index), len(index)))
    else:
        ind = sparse.csr_matrix((len(index), len(index)), dtype=np.float32)
    n_cp = np.asarray(ind.sum(axis=1)).ravel()
    return ind, n_cp


def _scale_minmax_array(values, min_val, max_val):
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        return np.zeros_like(values, dtype=np.float32)
    scaled = (values - vmin) / (vmax - vmin)
    scaled = scaled * (max_val - min_val) - min_val
    return scaled.astype(np.float32, copy=False)


# @timeit
def cal_aff_profile(exp, lr_df):
    lr_df_align = lr_df[lr_df[0].isin(exp.columns) & lr_df[1].isin(exp.columns)].copy()
    st_L = exp[lr_df_align[0]]
    st_R = exp[lr_df_align[1]]

    # Vectorized construction of spot-pair affinity profiles
    L_vals = st_L.to_numpy(dtype=np.float32)
    R_vals = st_R.to_numpy(dtype=np.float32)
    # out[i, j, p] = R[i, p] * L[j, p] + L[i, p] * R[j, p]
    out = (R_vals[:, None, :] * L_vals[None, :, :]) + (L_vals[:, None, :] * R_vals[None, :, :])
    out_2d = out.reshape(st_L.shape[0] * st_L.shape[0], st_L.shape[1])

    index = pd.MultiIndex.from_product([st_R.index, st_L.index])
    st_aff_profile_df = pd.DataFrame(out_2d, index=index, columns=st_L.columns)
    return st_aff_profile_df


def cal_aff_profile_sparse(exp, lr_df, spots_nn_lst):
    lr_df_align = lr_df[lr_df[0].isin(exp.columns) & lr_df[1].isin(exp.columns)].copy()
    if lr_df_align.empty:
        return pd.DataFrame()
    st_L = exp[lr_df_align[0]]
    st_R = exp[lr_df_align[1]]
    L_vals = st_L.to_numpy(dtype=np.float32, copy=False)
    R_vals = st_R.to_numpy(dtype=np.float32, copy=False)

    spots = list(exp.index)
    spot_to_idx = {s: i for i, s in enumerate(spots)}
    rows = []
    idx = []

    for spot, neighbors in spots_nn_lst.items():
        if spot not in spot_to_idx:
            continue
        i = spot_to_idx[spot]
        # include self
        nn_list = [spot] + [n for n in neighbors if n != spot]
        for nn in nn_list:
            j = spot_to_idx.get(nn)
            if j is None:
                continue
            row = R_vals[i] * L_vals[j] + L_vals[i] * R_vals[j]
            rows.append(row)
            idx.append((spot, nn))

    if not rows:
        return pd.DataFrame()

    index = pd.MultiIndex.from_tuples(idx)
    out_2d = np.vstack(rows)
    return pd.DataFrame(out_2d, index=index, columns=st_L.columns)


def build_spot_aff_profile_map(st_aff_profile_df):
    if st_aff_profile_df is None or st_aff_profile_df.empty:
        return {}
    return {spot: df for spot, df in st_aff_profile_df.groupby(level=0)}


# @timeit
def cal_sc_aff_profile(cell, cell_n, exp, lr_df):
    lr_df_align = lr_df[lr_df[0].isin(exp.columns) & lr_df[1].isin(exp.columns)].copy()
    st_L1 = exp.loc[cell,lr_df_align[0]]
    st_R1 = exp.loc[cell_n,lr_df_align[1]]
    st_L2 = exp.loc[cell_n,lr_df_align[0]]
    st_R2 = exp.loc[cell,lr_df_align[1]]
    #print(st_R2)
    #st_LR_df1 = pd.concat([st_L1 * st_R1.values[i] for i in range(st_R1.shape[0])], keys=st_R1.index.tolist())
    st_LR_df1 = st_R1 * st_L1.values
    #print(st_LR_df1)
    #st_LR_df2 = pd.concat([st_L2 * st_R2.values[i] for i in range(st_R2.shape[0])], keys=st_R2.index.tolist())
    st_LR_df2 = st_L2 * st_R2.values
    #print(st_LR_df2)
    st_aff_profile_df = st_LR_df1.values + st_LR_df2
    return st_aff_profile_df


def prepare_lr_indices(lr_df, genes):
    lr_df_align = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)].copy()
    gene_index = {gene: i for i, gene in enumerate(genes)}
    lig_idx = lr_df_align[0].map(gene_index).to_numpy(dtype=np.int32, copy=False)
    rec_idx = lr_df_align[1].map(gene_index).to_numpy(dtype=np.int32, copy=False)
    return lr_df_align, lig_idx, rec_idx


def cal_sc_aff_profile_fast(exp_values, lig_idx, rec_idx, cell_pos, nn_pos):
    exp_values = np.asarray(exp_values, dtype=np.float32)
    lig_cell = exp_values[cell_pos][:, lig_idx]
    rec_cell = exp_values[cell_pos][:, rec_idx]
    lig_nn = exp_values[nn_pos][:, lig_idx]
    rec_nn = exp_values[nn_pos][:, rec_idx]
    return lig_cell * rec_nn + lig_nn * rec_cell


def knn_distance_matrix_dense(sc_coord, sc_knn, index):
    coord = sc_coord.to_numpy(dtype=np.float32, copy=False) if isinstance(sc_coord, pd.DataFrame) else np.asarray(sc_coord, dtype=np.float32)
    idx_map = {k: i for i, k in enumerate(index)}
    n = len(index)
    dist = np.zeros((n, n), dtype=np.float32)
    for cell_id, nn_list in sc_knn.items():
        if cell_id not in idx_map:
            continue
        i = idx_map[cell_id]
        if not nn_list:
            continue
        nn_idx = [idx_map[nn] for nn in nn_list if nn in idx_map]
        if not nn_idx:
            continue
        diffs = coord[nn_idx] - coord[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        dist[i, nn_idx] = d
    return pd.DataFrame(dist, index=index, columns=index)


def knn_distance_matrix_sparse(sc_coord, sc_knn, index):
    coord = sc_coord.to_numpy(dtype=np.float32, copy=False) if isinstance(sc_coord, pd.DataFrame) else np.asarray(sc_coord, dtype=np.float32)
    idx_map = {k: i for i, k in enumerate(index)}
    rows = []
    cols = []
    data = []
    for cell_id, nn_list in sc_knn.items():
        i = idx_map.get(cell_id)
        if i is None or not nn_list:
            continue
        nn_idx = [idx_map[nn] for nn in nn_list if nn in idx_map]
        if not nn_idx:
            continue
        diffs = coord[nn_idx] - coord[i]
        d = np.sqrt((diffs ** 2).sum(axis=1))
        rows.extend([i] * len(nn_idx))
        cols.extend(nn_idx)
        data.extend(d.tolist())
    if not rows:
        return sparse.csr_matrix((len(index), len(index)), dtype=np.float32)
    return sparse.csr_matrix((np.asarray(data, dtype=np.float32), (rows, cols)), shape=(len(index), len(index)))


if nb is not None:
    @nb.njit(cache=True)
    def _spot_mean_abs_numba(values, spot_codes, n_spots):
        sums = np.zeros(n_spots, dtype=np.float64)
        counts = np.zeros(n_spots, dtype=np.int64)
        n_rows = values.shape[0]
        n_cols = values.shape[1]
        for i in range(n_rows):
            s = spot_codes[i]
            total = 0.0
            for j in range(n_cols):
                v = values[i, j]
                if v < 0:
                    v = -v
                total += v
            sums[s] += total / n_cols
            counts[s] += 1
        out = np.zeros(n_spots, dtype=np.float64)
        for i in range(n_spots):
            if counts[i] > 0:
                out[i] = sums[i] / counts[i]
        return out


def spot_mean_abs(values, spot_codes, n_spots):
    values = np.asarray(values, dtype=np.float32)
    spot_codes = np.asarray(spot_codes)
    if nb is not None:
        return _spot_mean_abs_numba(values, spot_codes, n_spots)
    sums = np.zeros(n_spots, dtype=np.float64)
    counts = np.zeros(n_spots, dtype=np.int64)
    row_mean = np.mean(np.abs(values), axis=1)
    np.add.at(sums, spot_codes, row_mean)
    np.add.at(counts, spot_codes, 1)
    out = np.zeros(n_spots, dtype=np.float64)
    mask = counts > 0
    out[mask] = sums[mask] / counts[mask]
    return out


def apply_spot_cell(x):
    return x.index.tolist()


# @timeit
def multiply_spots(df,res_tmp):
    spot_lst = df.index.get_level_values('spot').tolist()
    return df.multiply(res_tmp.loc[spot_lst].values,axis = 1)


def _groupby_rows_sum(values, row_codes, n_groups):
    out = np.zeros((n_groups, values.shape[1]), dtype=values.dtype)
    np.add.at(out, row_codes, values)
    return out


def _groupby_cols_mean(values, col_codes, n_groups):
    counts = np.bincount(col_codes, minlength=n_groups).astype(np.float32)
    counts[counts == 0] = 1.0
    out_T = np.zeros((n_groups, values.shape[0]), dtype=values.dtype)
    np.add.at(out_T, col_codes, values.T)
    out = (out_T / counts[:, None]).T
    return out


def prepare_spot_pair_lookup(st_exp, st_aff_profile_df, max_spot_neighbors=None):
    spots = list(st_exp.index)
    spot_to_idx = {s: i for i, s in enumerate(spots)}
    n_spots = len(spots)
    pair_to_row = np.full(n_spots * n_spots, -1, dtype=np.int32)
    spot_neighbors = [None for _ in range(n_spots)]
    if isinstance(max_spot_neighbors, str) and max_spot_neighbors.lower() == 'auto':
        if st_aff_profile_df is not None and not st_aff_profile_df.empty:
            counts = st_aff_profile_df.index.get_level_values(0).value_counts()
            if len(counts) > 0:
                med = int(np.median(counts.values))
                max_spot_neighbors = int(np.clip(med, 3, 20))
            else:
                max_spot_neighbors = None
        else:
            max_spot_neighbors = None
    if st_aff_profile_df is not None and not st_aff_profile_df.empty:
        idx0 = st_aff_profile_df.index.get_level_values(0)
        idx1 = st_aff_profile_df.index.get_level_values(1)
        spot0 = np.array([spot_to_idx.get(s, -1) for s in idx0], dtype=np.int64)
        spot1 = np.array([spot_to_idx.get(s, -1) for s in idx1], dtype=np.int64)
        valid = (spot0 >= 0) & (spot1 >= 0)
        if np.any(valid):
            pair_id = spot0[valid] * n_spots + spot1[valid]
            row_idx = np.nonzero(valid)[0]
            pair_to_row[pair_id] = row_idx.astype(np.int32, copy=False)
    st_aff_values = (
        st_aff_profile_df.to_numpy(dtype=np.float32, copy=False)
        if st_aff_profile_df is not None and not st_aff_profile_df.empty
        else np.zeros((0, 0), dtype=np.float32)
    )
    if max_spot_neighbors is not None and st_aff_profile_df is not None and not st_aff_profile_df.empty:
        max_spot_neighbors = int(max_spot_neighbors)
        if max_spot_neighbors > 0:
            scores = np.mean(np.abs(st_aff_values), axis=1) if st_aff_values.size else np.zeros(0, dtype=np.float32)
            for s_idx in range(n_spots):
                rows = np.where((spot0 == s_idx) & valid)[0]
                if rows.size == 0:
                    spot_neighbors[s_idx] = np.array([], dtype=np.int64)
                    continue
                row_scores = scores[rows]
                if rows.size > max_spot_neighbors:
                    top_idx = np.argpartition(row_scores, -max_spot_neighbors)[-max_spot_neighbors:]
                    rows = rows[top_idx]
                neigh_spots = spot1[rows]
                spot_neighbors[s_idx] = np.unique(neigh_spots.astype(np.int64))
    return spots, spot_to_idx, pair_to_row, st_aff_values, spot_neighbors


def prepare_knn_arrays(knn_df, cell_index, spot_to_idx):
    cell_ids = knn_df['cell_idx'].to_numpy()
    nn_ids = knn_df['nn_cell_idx'].to_numpy()
    nn_spots = knn_df['spot'].to_numpy()
    cell_pos = np.array([cell_index.get(c, -1) for c in cell_ids], dtype=np.int64)
    nn_pos = np.array([cell_index.get(c, -1) for c in nn_ids], dtype=np.int64)
    nn_spot_idx = np.array([spot_to_idx.get(s, -1) for s in nn_spots], dtype=np.int64)
    valid = (cell_pos >= 0) & (nn_pos >= 0) & (nn_spot_idx >= 0)
    return cell_pos, nn_pos, nn_spot_idx, valid


if nb is not None:
    @nb.njit(cache=True)
    def _groupby_rows_sum_numba(values, row_codes, n_groups):
        out = np.zeros((n_groups, values.shape[1]), dtype=values.dtype)
        for i in range(values.shape[0]):
            g = row_codes[i]
            out[g, :] += values[i, :]
        return out

    @nb.njit(cache=True)
    def _groupby_cols_mean_numba(values, col_codes, n_groups):
        counts = np.zeros(n_groups, dtype=np.float32)
        out_T = np.zeros((n_groups, values.shape[0]), dtype=values.dtype)
        for j in range(values.shape[1]):
            g = col_codes[j]
            out_T[g, :] += values[:, j]
            counts[g] += 1.0
        for g in range(n_groups):
            if counts[g] == 0:
                counts[g] = 1.0
            out_T[g, :] /= counts[g]
        return out_T.T

    @nb.njit(cache=True)
    def _accumulate_term3_edges_numba(rl_vals, row_idx, col_idx, weights, n_cells):
        n_genes = rl_vals.shape[0]
        out = np.zeros((n_genes, n_cells), dtype=np.float32)
        for e in range(row_idx.shape[0]):
            i = row_idx[e]
            j = col_idx[e]
            w = weights[e]
            for g in range(n_genes):
                out[g, i] += rl_vals[g, j] * w
        return out


def cal_term4_vectorized(
    st_exp,
    sc_knn,
    st_aff_profile_df,
    sc_exp,
    sc_meta,
    spot_cell_dict,
    lr_df,
    lr_indices=None,
    knn_df=None,
    spot_knn_idx=None,
    spot_filter=None,
    cell_index=None,
    spot_to_idx=None,
    pair_to_row=None,
    st_aff_values=None,
    knn_arrays=None,
    spot_neighbors=None,
    exp_values=None,
):
    """
    Array-only term4 calculation with per-spot loop to bound memory.
    Avoids pandas groupby/apply and uses NumPy accumulations.
    """
    alter_sc_exp = sc_exp[st_exp.columns]
    if exp_values is None:
        exp_values = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    if cell_index is None:
        cell_index = {k: i for i, k in enumerate(alter_sc_exp.index)}

    if lr_indices is None:
        lr_df_align, lig_idx, rec_idx = prepare_lr_indices(lr_df, alter_sc_exp.columns)
    else:
        lr_df_align, lig_idx, rec_idx = lr_indices

    n_pairs = len(lig_idx)
    if n_pairs == 0:
        return complete_other_genes(sc_exp, pd.DataFrame()), 0.0

    if spot_to_idx is None or pair_to_row is None or st_aff_values is None:
        _, spot_to_idx, pair_to_row, st_aff_values, spot_neighbors = prepare_spot_pair_lookup(
            st_exp, st_aff_profile_df, None
        )
    n_spots = len(spot_to_idx)

    if knn_df is None:
        knn_df = pd.DataFrame(sc_knn.items(), columns=['cell_idx', 'nn_cell_idx']).explode('nn_cell_idx')
        nn_cell_idx = knn_df['nn_cell_idx'].tolist()
        knn_df['spot'] = sc_meta.loc[nn_cell_idx, 'spot'].values

    if knn_arrays is None:
        cell_pos_arr, nn_pos_arr, nn_spot_idx_arr, valid = prepare_knn_arrays(knn_df, cell_index, spot_to_idx)
    else:
        cell_pos_arr, nn_pos_arr, nn_spot_idx_arr, valid = knn_arrays

    if not np.all(valid):
        cell_pos_arr = cell_pos_arr[valid]
        nn_pos_arr = nn_pos_arr[valid]
        nn_spot_idx_arr = nn_spot_idx_arr[valid]

    if cell_pos_arr.size == 0:
        return complete_other_genes(sc_exp, pd.DataFrame()), 0.0

    if spot_filter is None:
        spot_iter = st_exp.index.tolist()
    else:
        spot_iter = [s for s in spot_filter if s in st_exp.index]

    # Prepare sparse pair->gene maps for aggregation
    from scipy.sparse import csr_matrix
    n_genes = exp_values.shape[1]
    valid_lig = lig_idx >= 0
    valid_rec = rec_idx >= 0
    lig_map = csr_matrix(
        (np.ones(valid_lig.sum(), dtype=np.float32),
         (np.arange(n_pairs)[valid_lig], lig_idx[valid_lig])),
        shape=(n_pairs, n_genes), dtype=np.float32
    )
    rec_map = csr_matrix(
        (np.ones(valid_rec.sum(), dtype=np.float32),
         (np.arange(n_pairs)[valid_rec], rec_idx[valid_rec])),
        shape=(n_pairs, n_genes), dtype=np.float32
    )
    gene_counts = np.bincount(lig_idx, minlength=n_genes).astype(np.float32)
    gene_counts += np.bincount(rec_idx, minlength=n_genes).astype(np.float32)
    gene_counts[gene_counts == 0] = 1.0

    cell_grad = np.zeros((exp_values.shape[0], n_genes), dtype=np.float32)
    loss_4 = 0.0
    n_knn = 0

    for spot in spot_iter:
        spot_cells = spot_cell_dict.get(spot)
        if not spot_cells:
            continue
        if spot_knn_idx is not None:
            idxs = spot_knn_idx.get(spot)
            if idxs is None or len(idxs) == 0:
                continue
            cell_pos = cell_pos_arr[idxs]
            nn_pos = nn_pos_arr[idxs]
            nn_spot_idx = nn_spot_idx_arr[idxs]
        else:
            cell_pos = np.array([cell_index.get(c, -1) for c in spot_cells], dtype=np.int64)
            cell_pos = cell_pos[cell_pos >= 0]
            if cell_pos.size == 0:
                continue
            mask = np.isin(cell_pos_arr, cell_pos)
            if not np.any(mask):
                continue
            cell_pos = cell_pos_arr[mask]
            nn_pos = nn_pos_arr[mask]
            nn_spot_idx = nn_spot_idx_arr[mask]

        if nn_pos.size == 0:
            continue

        n_knn += int(nn_pos.size)

        # 1) a_cc: mean affinity per neighbor spot
        tmp_acc = cal_sc_aff_profile_fast(exp_values, lig_idx, rec_idx, cell_pos, nn_pos)
        if spot_neighbors is not None:
            spot_idx = spot_to_idx.get(spot, -1)
            if spot_idx >= 0 and spot_neighbors[spot_idx] is not None:
                allowed = spot_neighbors[spot_idx]
                if allowed.size > 0:
                    keep_mask = np.isin(nn_spot_idx, allowed)
                    if not np.any(keep_mask):
                        continue
                    cell_pos = cell_pos[keep_mask]
                    nn_pos = nn_pos[keep_mask]
                    nn_spot_idx = nn_spot_idx[keep_mask]
        spot_levels, spot_codes = np.unique(nn_spot_idx, return_inverse=True)
        acc_sums = np.zeros((spot_levels.size, tmp_acc.shape[1]), dtype=np.float32)
        np.add.at(acc_sums, spot_codes, tmp_acc)
        acc_counts = np.bincount(spot_codes, minlength=spot_levels.size).astype(np.float32)
        acc_counts[acc_counts == 0] = 1.0
        a_cc_vals = acc_sums / acc_counts[:, None]

        # 2) a_ss lookup
        spot_idx = spot_to_idx.get(spot, -1)
        if spot_idx < 0:
            continue
        pair_id = spot_idx * n_spots + spot_levels
        row_idx = pair_to_row[pair_id]
        valid_rows = row_idx >= 0
        a_ss_vals = np.zeros_like(a_cc_vals, dtype=np.float32)
        if np.any(valid_rows):
            a_ss_vals[valid_rows] = st_aff_values[row_idx[valid_rows]]

        a_cc_modi = np.sqrt(a_cc_vals / 2.0)
        a_ss_modi = np.sqrt(a_ss_vals / 2.0)
        res_tmp_vals = a_cc_modi - a_ss_modi
        loss_4 += float(np.sum(res_tmp_vals[valid_rows] ** 2))

        # 3) sum_r / sum_l by (cell, nn_spot)
        group2_id = cell_pos.astype(np.int64) * n_spots + nn_spot_idx.astype(np.int64)
        group2_levels, group2_inv = np.unique(group2_id, return_inverse=True)
        sums_r = _groupby_rows_sum(exp_values[nn_pos][:, rec_idx], group2_inv, len(group2_levels))
        sums_l = _groupby_rows_sum(exp_values[nn_pos][:, lig_idx], group2_inv, len(group2_levels))
        counts2 = np.bincount(group2_inv, minlength=len(group2_levels)).astype(np.float32)
        counts2[counts2 == 0] = 1.0
        mean_r = sums_r / counts2[:, None]
        mean_l = sums_l / counts2[:, None]

        # 4) res for each (cell, nn_spot) group
        group2_nn_spot = group2_levels % n_spots
        pos_map = np.full(n_spots, -1, dtype=np.int64)
        pos_map[spot_levels] = np.arange(spot_levels.size)
        res_idx = pos_map[group2_nn_spot]
        res_group = np.zeros_like(mean_r, dtype=np.float32)
        valid_res = res_idx >= 0
        if np.any(valid_res):
            res_group[valid_res] = res_tmp_vals[res_idx[valid_res]]

        contrib_r = mean_r * res_group
        contrib_l = mean_l * res_group

        # 5) aggregate to genes then to cells
        gene_r = contrib_r @ rec_map
        gene_l = contrib_l @ lig_map
        if hasattr(gene_r, 'toarray'):
            gene_r = gene_r.toarray()
        if hasattr(gene_l, 'toarray'):
            gene_l = gene_l.toarray()

        cell_pos_group2 = (group2_levels // n_spots).astype(np.int64)
        cell_grad += _groupby_rows_sum(gene_r + gene_l, cell_pos_group2, exp_values.shape[0])

    if n_knn > 0:
        loss_4 /= float(n_knn)

    cell_grad = cell_grad / gene_counts[None, :]
    term4_df = pd.DataFrame(cell_grad, index=alter_sc_exp.index, columns=alter_sc_exp.columns)
    return term4_df, loss_4


def cal_term4_sparse(
    st_exp,
    sc_knn,
    st_aff_profile_df,
    sc_exp,
    sc_meta,
    spot_cell_dict,
    lr_df,
    lr_indices=None,
    knn_df=None,
    spot_knn_idx=None,
    spot_filter=None,
    cell_index=None,
    spot_to_idx=None,
    pair_to_row=None,
    st_aff_values=None,
    knn_arrays=None,
    spot_neighbors=None,
    exp_values=None,
):
    """
    Sparse term4: only compute gradients for LR genes, then expand to full genes with zeros.
    """
    alter_sc_exp = sc_exp[st_exp.columns]
    if exp_values is None:
        exp_values = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    if cell_index is None:
        cell_index = {k: i for i, k in enumerate(alter_sc_exp.index)}

    if lr_indices is None:
        lr_df_align, lig_idx, rec_idx = prepare_lr_indices(lr_df, alter_sc_exp.columns)
    else:
        lr_df_align, lig_idx, rec_idx = lr_indices

    n_pairs = len(lig_idx)
    if n_pairs == 0:
        term4_vals = np.zeros_like(exp_values)
        term4_df = pd.DataFrame(term4_vals, index=alter_sc_exp.index, columns=alter_sc_exp.columns)
        return term4_df, 0.0

    if spot_to_idx is None or pair_to_row is None or st_aff_values is None:
        _, spot_to_idx, pair_to_row, st_aff_values, spot_neighbors = prepare_spot_pair_lookup(
            st_exp, st_aff_profile_df, None
        )
    n_spots = len(spot_to_idx)

    if knn_df is None:
        knn_df = pd.DataFrame(sc_knn.items(), columns=['cell_idx', 'nn_cell_idx']).explode('nn_cell_idx')
        nn_cell_idx = knn_df['nn_cell_idx'].tolist()
        knn_df['spot'] = sc_meta.loc[nn_cell_idx, 'spot'].values

    if knn_arrays is None:
        cell_pos_arr, nn_pos_arr, nn_spot_idx_arr, valid = prepare_knn_arrays(knn_df, cell_index, spot_to_idx)
    else:
        cell_pos_arr, nn_pos_arr, nn_spot_idx_arr, valid = knn_arrays

    if not np.all(valid):
        cell_pos_arr = cell_pos_arr[valid]
        nn_pos_arr = nn_pos_arr[valid]
        nn_spot_idx_arr = nn_spot_idx_arr[valid]

    if cell_pos_arr.size == 0:
        term4_vals = np.zeros_like(exp_values)
        term4_df = pd.DataFrame(term4_vals, index=alter_sc_exp.index, columns=alter_sc_exp.columns)
        return term4_df, 0.0

    if spot_filter is None:
        spot_iter = st_exp.index.tolist()
    else:
        spot_iter = [s for s in spot_filter if s in st_exp.index]

    # Build LR gene index map to shrink computation
    lr_gene_idx = np.unique(np.concatenate([lig_idx[lig_idx >= 0], rec_idx[rec_idx >= 0]])).astype(np.int64)
    if lr_gene_idx.size == 0:
        term4_vals = np.zeros_like(exp_values)
        term4_df = pd.DataFrame(term4_vals, index=alter_sc_exp.index, columns=alter_sc_exp.columns)
        return term4_df, 0.0
    lr_pos = np.full(exp_values.shape[1], -1, dtype=np.int64)
    lr_pos[lr_gene_idx] = np.arange(lr_gene_idx.size, dtype=np.int64)
    lig_lr = lr_pos[lig_idx]
    rec_lr = lr_pos[rec_idx]

    from scipy.sparse import csr_matrix
    valid_lig = lig_lr >= 0
    valid_rec = rec_lr >= 0
    lig_map = csr_matrix(
        (np.ones(valid_lig.sum(), dtype=np.float32),
         (np.arange(n_pairs)[valid_lig], lig_lr[valid_lig])),
        shape=(n_pairs, lr_gene_idx.size), dtype=np.float32
    )
    rec_map = csr_matrix(
        (np.ones(valid_rec.sum(), dtype=np.float32),
         (np.arange(n_pairs)[valid_rec], rec_lr[valid_rec])),
        shape=(n_pairs, lr_gene_idx.size), dtype=np.float32
    )
    gene_counts = np.bincount(lig_lr[valid_lig], minlength=lr_gene_idx.size).astype(np.float32)
    gene_counts += np.bincount(rec_lr[valid_rec], minlength=lr_gene_idx.size).astype(np.float32)
    gene_counts[gene_counts == 0] = 1.0

    cell_grad_lr = np.zeros((exp_values.shape[0], lr_gene_idx.size), dtype=np.float32)
    loss_4 = 0.0
    n_knn = 0

    for spot in spot_iter:
        spot_cells = spot_cell_dict.get(spot)
        if not spot_cells:
            continue
        if spot_knn_idx is not None:
            idxs = spot_knn_idx.get(spot)
            if idxs is None or len(idxs) == 0:
                continue
            cell_pos = cell_pos_arr[idxs]
            nn_pos = nn_pos_arr[idxs]
            nn_spot_idx = nn_spot_idx_arr[idxs]
        else:
            cell_pos = np.array([cell_index.get(c, -1) for c in spot_cells], dtype=np.int64)
            cell_pos = cell_pos[cell_pos >= 0]
            if cell_pos.size == 0:
                continue
            mask = np.isin(cell_pos_arr, cell_pos)
            if not np.any(mask):
                continue
            cell_pos = cell_pos_arr[mask]
            nn_pos = nn_pos_arr[mask]
            nn_spot_idx = nn_spot_idx_arr[mask]

        if nn_pos.size == 0:
            continue

        n_knn += int(nn_pos.size)

        if spot_neighbors is not None:
            spot_idx = spot_to_idx.get(spot, -1)
            if spot_idx >= 0 and spot_neighbors[spot_idx] is not None:
                allowed = spot_neighbors[spot_idx]
                if allowed.size > 0:
                    keep_mask = np.isin(nn_spot_idx, allowed)
                    if not np.any(keep_mask):
                        continue
                    cell_pos = cell_pos[keep_mask]
                    nn_pos = nn_pos[keep_mask]
                    nn_spot_idx = nn_spot_idx[keep_mask]

        # 1) a_cc: mean affinity per neighbor spot
        tmp_acc = cal_sc_aff_profile_fast(exp_values, lig_idx, rec_idx, cell_pos, nn_pos)
        spot_levels, spot_codes = np.unique(nn_spot_idx, return_inverse=True)
        acc_sums = np.zeros((spot_levels.size, tmp_acc.shape[1]), dtype=np.float32)
        np.add.at(acc_sums, spot_codes, tmp_acc)
        acc_counts = np.bincount(spot_codes, minlength=spot_levels.size).astype(np.float32)
        acc_counts[acc_counts == 0] = 1.0
        a_cc_vals = acc_sums / acc_counts[:, None]

        # 2) a_ss lookup
        spot_idx = spot_to_idx.get(spot, -1)
        if spot_idx < 0:
            continue
        pair_id = spot_idx * n_spots + spot_levels
        row_idx = pair_to_row[pair_id]
        valid_rows = row_idx >= 0
        a_ss_vals = np.zeros_like(a_cc_vals, dtype=np.float32)
        if np.any(valid_rows):
            a_ss_vals[valid_rows] = st_aff_values[row_idx[valid_rows]]

        a_cc_modi = np.sqrt(a_cc_vals / 2.0)
        a_ss_modi = np.sqrt(a_ss_vals / 2.0)
        res_tmp_vals = a_cc_modi - a_ss_modi
        loss_4 += float(np.sum(res_tmp_vals[valid_rows] ** 2))

        # 3) sum_r / sum_l by (cell, nn_spot)
        group2_id = cell_pos.astype(np.int64) * n_spots + nn_spot_idx.astype(np.int64)
        group2_levels, group2_inv = np.unique(group2_id, return_inverse=True)
        sums_r = _groupby_rows_sum(exp_values[nn_pos][:, rec_idx], group2_inv, len(group2_levels))
        sums_l = _groupby_rows_sum(exp_values[nn_pos][:, lig_idx], group2_inv, len(group2_levels))
        counts2 = np.bincount(group2_inv, minlength=len(group2_levels)).astype(np.float32)
        counts2[counts2 == 0] = 1.0
        mean_r = sums_r / counts2[:, None]
        mean_l = sums_l / counts2[:, None]

        # 4) res for each (cell, nn_spot) group
        group2_nn_spot = group2_levels % n_spots
        pos_map = np.full(n_spots, -1, dtype=np.int64)
        pos_map[spot_levels] = np.arange(spot_levels.size)
        res_idx = pos_map[group2_nn_spot]
        res_group = np.zeros_like(mean_r, dtype=np.float32)
        valid_res = res_idx >= 0
        if np.any(valid_res):
            res_group[valid_res] = res_tmp_vals[res_idx[valid_res]]

        contrib_r = mean_r * res_group
        contrib_l = mean_l * res_group

        # 5) aggregate to LR genes then to cells
        gene_r = contrib_r @ rec_map
        gene_l = contrib_l @ lig_map
        if hasattr(gene_r, 'toarray'):
            gene_r = gene_r.toarray()
        if hasattr(gene_l, 'toarray'):
            gene_l = gene_l.toarray()

        cell_pos_group2 = (group2_levels // n_spots).astype(np.int64)
        cell_grad_lr += _groupby_rows_sum(gene_r + gene_l, cell_pos_group2, exp_values.shape[0])

    if n_knn > 0:
        loss_4 /= float(n_knn)

    cell_grad_lr = cell_grad_lr / gene_counts[None, :]
    term4_vals = np.zeros_like(exp_values)
    term4_vals[:, lr_gene_idx] = cell_grad_lr
    term4_df = pd.DataFrame(term4_vals, index=alter_sc_exp.index, columns=alter_sc_exp.columns)
    return term4_df, loss_4


# @timeit
def calSumNNRL(exp, spot_knn_df, cell_neigbors, gene_lst, idx_map=None, gene_idx=None):
    '''
    Calculating the multiplier in term 4
    sum_{c \in s, c' \in s', c' \in N(c),g' = R(g)} e_{c'g'}
    '''
    if nb is not None:
        exp_values = exp.to_numpy(dtype=np.float32, copy=False)
        if gene_idx is None:
            gene_idx = exp.columns.get_indexer(gene_lst)
            gene_idx = gene_idx[gene_idx >= 0]
        if gene_idx.size == 0:
            return pd.DataFrame(columns=list(gene_lst))

        exp_sub = exp_values[:, gene_idx]
        cell_idx = spot_knn_df['cell_idx'].to_numpy()
        spot_vals = spot_knn_df['spot'].to_numpy()
        nn_ids = np.asarray(cell_neigbors)

        cell_levels, cell_codes = np.unique(cell_idx, return_inverse=True)
        spot_levels, spot_codes = np.unique(spot_vals, return_inverse=True)
        n_cells = len(cell_levels)

        group_id = spot_codes * n_cells + cell_codes
        unique_groups, group_inv = np.unique(group_id, return_inverse=True)
        n_groups = unique_groups.size

        if idx_map is None:
            idx_map = {k: i for i, k in enumerate(exp.index)}
        try:
            nn_pos = np.array([idx_map[x] for x in nn_ids], dtype=np.int64)
        except KeyError:
            nn_pos = np.array([idx_map.get(x, -1) for x in nn_ids], dtype=np.int64)
            valid = nn_pos >= 0
            nn_pos = nn_pos[valid]
            group_inv = group_inv[valid]

        sums, counts = _accumulate_group_sums(exp_sub, nn_pos, group_inv, n_groups)
        counts_safe = np.where(counts == 0, 1, counts).astype(np.float32)
        means = sums / counts_safe[:, None]

        spot_idx = unique_groups // n_cells
        cell_idx = unique_groups % n_cells
        index = pd.MultiIndex.from_arrays(
            [spot_levels[spot_idx], cell_levels[cell_idx]], names=['spot', 'cell_idx']
        )
        return pd.DataFrame(means, index=index, columns=exp.columns[gene_idx])

    tmp_sum_r = exp.loc[cell_neigbors, gene_lst]
    tmp_sum_r['spot'] = spot_knn_df['spot'].values
    tmp_sum_r['cell_idx'] = spot_knn_df['cell_idx'].values
    # v3 sum => mean
    sum_ncg = tmp_sum_r.groupby(['spot', 'cell_idx']).mean()
    return sum_ncg


if nb is not None:
    @nb.njit(cache=True)
    def _accumulate_group_sums(exp_sub, nn_pos, group_inv, n_groups):
        n_genes = exp_sub.shape[1]
        sums = np.zeros((n_groups, n_genes), dtype=np.float32)
        counts = np.zeros(n_groups, dtype=np.int64)
        for i in range(nn_pos.shape[0]):
            g = group_inv[i]
            row = exp_sub[nn_pos[i]]
            for j in range(n_genes):
                sums[g, j] += row[j]
            counts[g] += 1
        return sums, counts


# @timeit
def cal_term4(st_exp,sc_knn,st_aff_profile_df,sc_exp,sc_meta,spot_cell_dict,lr_df, lr_indices=None, knn_df=None, spot_filter=None, spot_knn_idx=None, cell_index=None):
    ''' 
    st_exp = obj_spex.st_exp
    sc_knn = obj_spex.sc_knn 
    st_aff_profile_df = obj_spex.st_aff_profile_df
    alter_sc_exp = obj_spex.alter_sc_exp
    sc_meta = obj_spex.sc_agg_meta
    spot_cell_dict = obj_spex.spot_cell_dict
    lr_df = obj_spex.lr_df
    '''
    alter_sc_exp = sc_exp[st_exp.columns]
    exp_values = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    if cell_index is None:
        cell_index = {k: i for i, k in enumerate(alter_sc_exp.index)}
    if lr_indices is None:
        lr_df_align, lig_idx, rec_idx = prepare_lr_indices(lr_df, alter_sc_exp.columns)
    else:
        lr_df_align, lig_idx, rec_idx = lr_indices
    lig_gene_idx = lig_idx
    rec_gene_idx = rec_idx
    # generate knn_df: cell_idx	-> nn_cell_idx
    if knn_df is None:
        knn_df = pd.DataFrame(sc_knn.items(), columns=['cell_idx', 'nn_cell_idx'])
        knn_df = knn_df.explode('nn_cell_idx')
    nn_cell_idx = knn_df['nn_cell_idx'].tolist()
    df = sc_meta.loc[nn_cell_idx].copy()
    knn_df['spot'] = df['spot'].values
    term4_list = []
    loss_4 = 0
    n_knn = 0
    if spot_filter is None:
        spot_iter = st_exp.index
    else:
        spot_iter = [s for s in spot_filter if s in st_exp.index]
    for spot in spot_iter:
        spot_cells = spot_cell_dict[spot]
        # 1. find knn id and its affiliated spot
        if spot_knn_idx is not None:
            idxs = spot_knn_idx.get(spot)
            if idxs is None or len(idxs) == 0:
                continue
            spot_knn_df = knn_df.iloc[idxs]
        else:
            spot_knn_df = knn_df[knn_df['cell_idx'].isin(spot_cells)]
        cell_idx = spot_knn_df['cell_idx']
        cell_nn_idx = spot_knn_df['nn_cell_idx']
        n_knn += len(cell_nn_idx)
        # some spot has no nn for any cell in it.
        if cell_nn_idx.tolist():
            # 2. calculate acc
            cell_pos = np.array([cell_index[c] for c in cell_idx])
            nn_pos = np.array([cell_index[c] for c in cell_nn_idx])
            tmp_acc = cal_sc_aff_profile_fast(exp_values, lig_idx, rec_idx, cell_pos, nn_pos)
            spot_vals = spot_knn_df['spot'].values
            spot_levels, spot_codes = np.unique(spot_vals, return_inverse=True)
            n_spots = len(spot_levels)
            acc_sums = np.zeros((n_spots, tmp_acc.shape[1]), dtype=np.float32)
            np.add.at(acc_sums, spot_codes, tmp_acc)
            acc_counts = np.bincount(spot_codes, minlength=n_spots).astype(np.float32)
            acc_counts[acc_counts == 0] = 1.0
            a_cc_vals = acc_sums / acc_counts[:, None]
            # 3. calculate ass
            a_ss = st_aff_profile_df.loc[(spot, spot_levels.tolist()), :]
            a_cc_modi = np.sqrt(a_cc_vals / 2)
            a_ss_modi = np.sqrt(a_ss.to_numpy(dtype=np.float32, copy=False) / 2)
            res_tmp_vals = a_cc_modi - a_ss_modi
            loss_tmp = float(np.sum(res_tmp_vals ** 2))
            loss_4 += loss_tmp
            if np.isnan(loss_tmp):
                print(f'{spot}')
                a_cc = pd.DataFrame(a_cc_vals, index=spot_levels, columns=lr_df_align.index)
                print(f'acc{a_cc}')
                print(f'ass{a_ss}')
                print(f'a_cc_modi{a_cc_modi}')
                print(f'a_ss_modi{a_ss_modi}')
                res_tmp = pd.DataFrame(res_tmp_vals, index=spot_levels, columns=lr_df_align.index)
                print(f'res_tmp{res_tmp}')
            # 4. calculate multiplier
            sum_r = calSumNNRL(alter_sc_exp, spot_knn_df, cell_nn_idx, lr_df_align[1], idx_map=cell_index, gene_idx=rec_gene_idx)
            sum_l = calSumNNRL(alter_sc_exp, spot_knn_df, cell_nn_idx, lr_df_align[0], idx_map=cell_index, gene_idx=lig_gene_idx)
            spot_idx_r = sum_r.index.get_level_values('spot')
            spot_idx_l = sum_l.index.get_level_values('spot')
            spot_indexer = pd.Index(spot_levels)
            pos_r = spot_indexer.get_indexer(spot_idx_r)
            pos_l = spot_indexer.get_indexer(spot_idx_l)
            res_tmp_r = res_tmp_vals[pos_r]
            res_tmp_l = res_tmp_vals[pos_l]
            res_L_vals = sum_r.to_numpy(dtype=np.float32, copy=False) * res_tmp_r
            res_R_vals = sum_l.to_numpy(dtype=np.float32, copy=False) * res_tmp_l
            res_LR_vals = np.concatenate([res_L_vals, res_R_vals], axis=1)
            res_LR_cols = np.concatenate([sum_r.columns.to_numpy(), sum_l.columns.to_numpy()])
            cell_levels, cell_codes = np.unique(sum_r.index.get_level_values('cell_idx'), return_inverse=True)
            if nb is not None:
                summed_vals = _groupby_rows_sum_numba(res_LR_vals, cell_codes, len(cell_levels))
            else:
                summed_vals = _groupby_rows_sum(res_LR_vals, cell_codes, len(cell_levels))
            col_names = res_LR_cols
            col_levels, col_codes = np.unique(col_names, return_inverse=True)
            if nb is not None:
                mean_vals = _groupby_cols_mean_numba(summed_vals, col_codes, len(col_levels))
            else:
                mean_vals = _groupby_cols_mean(summed_vals, col_codes, len(col_levels))
            res = pd.DataFrame(mean_vals, index=cell_levels, columns=col_levels)
            term4_list.append(res)
        #break
    if n_knn > 0:
        loss_4 /= n_knn
    # if np.isnan(loss_4):
    #     print('nananananana')
    term4_LR = pd.concat(term4_list, axis=0) if term4_list else pd.DataFrame()
    term4_df = complete_other_genes(sc_exp, term4_LR)
    return term4_df, loss_4


_TERM4_PARALLEL_CONTEXT = {}


def _init_term4_parallel_context(alter_sc_exp, sc_meta, st_aff_profile_df, spot_cell_dict,
                                 knn_df, lr_df_align, lig_idx, rec_idx, cell_index, exp_values, idx_map,
                                 spot_knn_idx=None):
    global _TERM4_PARALLEL_CONTEXT
    _TERM4_PARALLEL_CONTEXT = {
        'alter_sc_exp': alter_sc_exp,
        'sc_meta': sc_meta,
        'st_aff_profile_df': st_aff_profile_df,
        'spot_cell_dict': spot_cell_dict,
        'knn_df': knn_df,
        'spot_knn_idx': spot_knn_idx,
        'lr_df_align': lr_df_align,
        'lig_idx': lig_idx,
        'rec_idx': rec_idx,
        'cell_index': cell_index,
        'exp_values': exp_values,
        'idx_map': idx_map,
        'lig_gene_idx': lig_idx,
        'rec_gene_idx': rec_idx,
    }


def _process_spot_term4(spot):
    ctx = _TERM4_PARALLEL_CONTEXT
    spot_cells = ctx['spot_cell_dict'].get(spot)
    if not spot_cells:
        return pd.DataFrame(), 0.0, 0

    knn_df = ctx['knn_df']
    if ctx.get('spot_knn_idx') is not None:
        idxs = ctx['spot_knn_idx'].get(spot)
        if idxs is None or len(idxs) == 0:
            return pd.DataFrame(), 0.0, 0
        spot_knn_df = knn_df.iloc[idxs]
    else:
        spot_knn_df = knn_df[knn_df['cell_idx'].isin(spot_cells)]
    if spot_knn_df.empty:
        return pd.DataFrame(), 0.0, 0

    cell_idx = spot_knn_df['cell_idx']
    cell_nn_idx = spot_knn_df['nn_cell_idx']
    n_knn = len(cell_nn_idx)
    if n_knn == 0:
        return pd.DataFrame(), 0.0, 0

    cell_pos = np.array([ctx['cell_index'][c] for c in cell_idx])
    nn_pos = np.array([ctx['cell_index'][c] for c in cell_nn_idx])
    tmp_acc = cal_sc_aff_profile_fast(ctx['exp_values'], ctx['lig_idx'], ctx['rec_idx'], cell_pos, nn_pos)
    spot_vals = spot_knn_df['spot'].to_numpy()
    spot_levels, spot_codes = np.unique(spot_vals, return_inverse=True)
    n_spots = spot_levels.size
    acc_sums = np.zeros((n_spots, tmp_acc.shape[1]), dtype=np.float32)
    np.add.at(acc_sums, spot_codes, tmp_acc)
    acc_counts = np.bincount(spot_codes, minlength=n_spots).astype(np.float32)
    acc_counts[acc_counts == 0] = 1.0
    a_cc_vals = acc_sums / acc_counts[:, None]
    a_ss = ctx['st_aff_profile_df'].loc[(spot, spot_levels.tolist()), :]
    a_cc_modi = np.sqrt(a_cc_vals / 2)
    a_ss_modi = np.sqrt(a_ss.to_numpy(dtype=np.float32, copy=False) / 2)
    res_tmp_vals = a_cc_modi - a_ss_modi
    loss_tmp = float(np.sum(res_tmp_vals ** 2))

    sum_r = calSumNNRL(
        ctx['alter_sc_exp'], spot_knn_df, cell_nn_idx, ctx['lr_df_align'][1],
        idx_map=ctx['idx_map'], gene_idx=ctx['rec_gene_idx']
    )
    sum_l = calSumNNRL(
        ctx['alter_sc_exp'], spot_knn_df, cell_nn_idx, ctx['lr_df_align'][0],
        idx_map=ctx['idx_map'], gene_idx=ctx['lig_gene_idx']
    )
    spot_idx_r = sum_r.index.get_level_values('spot')
    spot_idx_l = sum_l.index.get_level_values('spot')
    spot_indexer = pd.Index(spot_levels)
    pos_r = spot_indexer.get_indexer(spot_idx_r)
    pos_l = spot_indexer.get_indexer(spot_idx_l)
    res_tmp_r = res_tmp_vals[pos_r]
    res_tmp_l = res_tmp_vals[pos_l]
    res_L_vals = sum_r.to_numpy(dtype=np.float32, copy=False) * res_tmp_r
    res_R_vals = sum_l.to_numpy(dtype=np.float32, copy=False) * res_tmp_l
    res_LR_vals = np.concatenate([res_L_vals, res_R_vals], axis=1)
    res_LR_cols = np.concatenate([sum_r.columns.to_numpy(), sum_l.columns.to_numpy()])
    cell_levels, cell_codes = np.unique(sum_r.index.get_level_values('cell_idx'), return_inverse=True)
    if nb is not None:
        summed_vals = _groupby_rows_sum_numba(res_LR_vals, cell_codes, len(cell_levels))
    else:
        summed_vals = _groupby_rows_sum(res_LR_vals, cell_codes, len(cell_levels))
    col_names = res_LR_cols
    col_levels, col_codes = np.unique(col_names, return_inverse=True)
    if nb is not None:
        mean_vals = _groupby_cols_mean_numba(summed_vals, col_codes, len(col_levels))
    else:
        mean_vals = _groupby_cols_mean(summed_vals, col_codes, len(col_levels))
    res = pd.DataFrame(mean_vals, index=cell_levels, columns=col_levels)

    return res, loss_tmp, n_knn


def cal_term4_parallel(st_exp, sc_knn, st_aff_profile_df, sc_exp, sc_meta, spot_cell_dict,
                       lr_df, lr_indices=None, knn_df=None, spot_filter=None, n_cores=4, cell_index=None,
                       spot_knn_idx=None):
    alter_sc_exp = sc_exp[st_exp.columns]
    exp_values = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    if cell_index is None:
        cell_index = {k: i for i, k in enumerate(alter_sc_exp.index)}
    if lr_indices is None:
        lr_df_align, lig_idx, rec_idx = prepare_lr_indices(lr_df, alter_sc_exp.columns)
    else:
        lr_df_align, lig_idx, rec_idx = lr_indices
    if knn_df is None:
        knn_df = pd.DataFrame(sc_knn.items(), columns=['cell_idx', 'nn_cell_idx'])
        knn_df = knn_df.explode('nn_cell_idx')
        nn_cell_idx = knn_df['nn_cell_idx'].tolist()
        df = sc_meta.loc[nn_cell_idx].copy()
        knn_df['spot'] = df['spot'].values

    n_cores = max(1, int(n_cores))
    if n_cores <= 1:
        return cal_term4(
            st_exp, sc_knn, st_aff_profile_df, sc_exp, sc_meta, spot_cell_dict,
            lr_df, lr_indices=lr_indices, knn_df=knn_df, spot_filter=spot_filter
        )

    if spot_filter is None:
        spot_iter = st_exp.index.tolist()
    else:
        spot_iter = [s for s in spot_filter if s in st_exp.index]

    with Pool(processes=n_cores, initializer=_init_term4_parallel_context,
              initargs=(alter_sc_exp, sc_meta, st_aff_profile_df, spot_cell_dict,
                        knn_df, lr_df_align, lig_idx, rec_idx, cell_index, exp_values, cell_index,
                        spot_knn_idx)) as pool:
        results = pool.map(_process_spot_term4, spot_iter)

    term4_list = [r[0] for r in results if not r[0].empty]
    total_loss = sum(r[1] for r in results)
    total_n_knn = sum(r[2] for r in results)
    loss_4 = total_loss / total_n_knn if total_n_knn > 0 else 0.0
    term4_LR = pd.concat(term4_list, axis=0) if term4_list else pd.DataFrame()
    term4_df = complete_other_genes(sc_exp, term4_LR)
    return term4_df, loss_4


# @timeit
def cal_term5(alter_sc_exp):
    values = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    term5_df = pd.DataFrame(values * 2.0, index=alter_sc_exp.index, columns=alter_sc_exp.columns)
    loss5 = float(np.mean(values ** 2))
    return term5_df, loss5


def cal_term5_values(alter_sc_exp):
    """Compute term5 values and loss without constructing a DataFrame."""
    values = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
    term5_vals = values * 2.0
    loss5 = float(np.mean(values ** 2))
    return term5_vals, loss5

#### first edition ####
def prepare_lr_agg_indices(lr_df, genes):
    """Precompute LR aggregation indices for fast repeated computation."""
    genes = list(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    lr_filt = lr_df[lr_df[0].isin(gene_to_idx) & lr_df[1].isin(gene_to_idx)]
    if lr_filt.empty:
        return None
    
    lig_genes = lr_filt[0].values
    rec_genes = lr_filt[1].values
    lig_idx = np.array([gene_to_idx[g] for g in lig_genes], dtype=np.int64)
    rec_idx = np.array([gene_to_idx[g] for g in rec_genes], dtype=np.int64)
    
    # For R aggregation: group rec_idx by lig_genes, output indexed by lig
    unique_lig, lig_inv = np.unique(lig_genes, return_inverse=True)
    # For L aggregation: group lig_idx by rec_genes, output indexed by rec
    unique_rec, rec_inv = np.unique(rec_genes, return_inverse=True)
    
    # Combined unique output genes
    all_out_genes = np.unique(np.concatenate([unique_lig, unique_rec]))
    out_gene_to_idx = {g: i for i, g in enumerate(all_out_genes)}
    
    # Build mapping: for each output gene, list of (source_gene_idx, count) for mean
    # R contribution: output[lig] += mean(rec values grouped by lig)
    # L contribution: output[rec] += mean(lig values grouped by rec)
    
    return {
        'lig_idx': lig_idx,
        'rec_idx': rec_idx,
        'lig_inv': lig_inv,
        'rec_inv': rec_inv,
        'n_lig_groups': len(unique_lig),
        'n_rec_groups': len(unique_rec),
        'unique_lig': unique_lig,
        'unique_rec': unique_rec,
        'all_out_genes': all_out_genes,
        'out_gene_to_idx': out_gene_to_idx,
    }


def generate_LR_agg_fast(alter_sc_exp, lr_agg_idx):
    """Fast LR aggregation using precomputed indices."""
    if lr_agg_idx is None:
        return pd.DataFrame(index=[], columns=alter_sc_exp.index)
    
    exp_vals = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)  # cells x genes
    n_cells = exp_vals.shape[0]
    
    lig_idx = lr_agg_idx['lig_idx']
    rec_idx = lr_agg_idx['rec_idx']
    lig_inv = lr_agg_idx['lig_inv']
    rec_inv = lr_agg_idx['rec_inv']
    n_lig = lr_agg_idx['n_lig_groups']
    n_rec = lr_agg_idx['n_rec_groups']
    unique_lig = lr_agg_idx['unique_lig']
    unique_rec = lr_agg_idx['unique_rec']
    all_out = lr_agg_idx['all_out_genes']
    out_map = lr_agg_idx['out_gene_to_idx']
    
    # R aggregation: for each lig group, mean of rec expression
    rec_vals = exp_vals[:, rec_idx]  # cells x n_pairs
    r_sums = np.zeros((n_cells, n_lig), dtype=np.float32)
    r_counts = np.zeros(n_lig, dtype=np.int64)
    np.add.at(r_sums.T, lig_inv, rec_vals.T)
    np.add.at(r_counts, lig_inv, 1)
    r_counts[r_counts == 0] = 1
    r_means = r_sums / r_counts  # cells x n_lig
    
    # L aggregation: for each rec group, mean of lig expression
    lig_vals = exp_vals[:, lig_idx]  # cells x n_pairs
    l_sums = np.zeros((n_cells, n_rec), dtype=np.float32)
    l_counts = np.zeros(n_rec, dtype=np.int64)
    np.add.at(l_sums.T, rec_inv, lig_vals.T)
    np.add.at(l_counts, rec_inv, 1)
    l_counts[l_counts == 0] = 1
    l_means = l_sums / l_counts  # cells x n_rec
    
    # Combine into output matrix indexed by all_out_genes
    n_out = len(all_out)
    result = np.zeros((n_out, n_cells), dtype=np.float32)
    
    for i, g in enumerate(unique_lig):
        out_i = out_map[g]
        result[out_i] += r_means[:, i]
    for i, g in enumerate(unique_rec):
        out_i = out_map[g]
        result[out_i] += l_means[:, i]
    
    return pd.DataFrame(result, index=all_out, columns=alter_sc_exp.index)


# @timeit
def generate_LR_agg(alter_sc_exp,lr_df):
    ''' L(g1): g1 as Receptor
        R(g1): g1 as Ligand
        L(g1) \ne R(g1)
        Since they were calculated by diff gene sum
    '''
    alter_sc_exp = alter_sc_exp.astype(np.float32, copy=False)
    # V3: summed expression of pair RL => mean
    # V4: sum L(g1), R(g1) together
    # keep lr gene pairs exist in sc_exp
    genes = alter_sc_exp.columns.tolist()
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # summation of paired Receptor genes for each Ligand (row) in every cell (col).
    r_agg = alter_sc_exp[lr_df[1]].T
    r_agg['L'] = lr_df[0].values
    # v3
    # r_agg = r_agg.groupby('L').sum()
    r_agg = r_agg.groupby('L').mean()
    # summation of paired Ligand genes for each Receptor (row) in every cell (col).
    # v4 bug: lr_df[1] to lr_df[0],
    # should select L gene exp, agg by R
    l_agg = alter_sc_exp[lr_df[0]].T
    l_agg['R'] = lr_df[1].values
    # l_agg = l_agg.groupby('R').sum()
    l_agg = l_agg.groupby('R').mean()
    
    rl_agg = pd.concat([r_agg,l_agg])
    # v4 added
    rl_agg = rl_agg.groupby(level=0).sum()
    rl_agg.columns = alter_sc_exp.index
    return rl_agg


# @timeit
def chunk_cal_aff(adata, sc_dis_mat, lr_df):
    genes = list(adata.columns)
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    gene_index =dict(zip(genes, range(len(genes))))
    index = lr_df.replace({0: gene_index, 1:gene_index}).astype(int)
    ligandindex = index[0].reset_index()[0]
    receptorindex = index[1].reset_index()[1]
    scores = index[2].reset_index()[2]
    Atotake = ligandindex
    Btotake = receptorindex
    allscores = scores
    idx_data = csr_matrix(adata).T
    for i in range(len(ligandindex)):
        if ligandindex[i] != receptorindex[i]:
            Atotake = Atotake.append(pd.Series(receptorindex[i]),ignore_index=True)
            Btotake = Btotake.append(pd.Series(ligandindex[i]),ignore_index=True)
            allscores = allscores.append(pd.Series(scores[i]),ignore_index=True)
    A = idx_data[Atotake.tolist()]
    B = idx_data[Btotake.tolist()]
    full_A = np.dot(csr_matrix(np.diag(allscores)), A).T  
    chunk_size = 20
    cells = list(range(adata.shape[0]))
    affinitymat = np.array([[]]).reshape(0,adata.shape[0])
    affinitymat = csr_matrix(affinitymat)
    #s = time.time()

    for process_i in range(chunk_size):
        #a = time.time() 
        cell_chunk = list(np.array_split(cells, chunk_size)[process_i])
        chunk_A = full_A[cell_chunk]
        chunk_aff = np.dot(chunk_A, B)
        chunk_dis_mat = sc_dis_mat[cell_chunk]
        sparse_A = chunk_dis_mat.multiply(chunk_aff)
        #print(chunk_aff.sum())
        affinitymat = sparse.vstack([affinitymat, sparse_A])
        #b = time.time()
        #print(f'{process_i} done, cost {(b - a):.2f}s.')
    return affinitymat


# @timeit
def sc_prep(st_coord, sc_meta):
    picked_sc_meta = sc_meta.copy()
    # broadcast_st_adj_sc
    st_coord = st_coord.loc[picked_sc_meta['spot'].unique()]
    idx_lst = st_coord.index.tolist()
    idx_dict = {k: v for v, k in enumerate(idx_lst)}
    picked_sc_meta['indice'] = picked_sc_meta['spot'].map(idx_dict)
    coord_dict_x = {v: k for v, k in enumerate(list(st_coord['x']))}
    coord_dict_y = {v: k for v, k in enumerate(list(st_coord['y']))}
    picked_sc_meta['st_x'] = picked_sc_meta['indice'].map(coord_dict_x)
    picked_sc_meta['st_y'] = picked_sc_meta['indice'].map(coord_dict_y)
    picked_sc_meta = picked_sc_meta.sort_values(by = 'indice')
    # dist calculation
    sc_coord = picked_sc_meta[['st_x','st_y']]
    return st_coord, sc_coord


# @timeit
def sc_adj_cal(st_coord, picked_sc_meta, chunk_size=12, use_kdtree=False, compute_ans=True):
    alpha = 0
    st_coord, sc_coord = sc_prep(st_coord, picked_sc_meta)
    # alpha = 0 for visium data
    all_x = np.sort(list(set(st_coord.iloc[:, 0])))
    unit_len = all_x[1] - all_x[0]
    r = 2 * unit_len + alpha

    indicator = lil_matrix((len(sc_coord), len(sc_coord)))
    ans = {} if compute_ans else None

    sc_arr = np.array(sc_coord)
    if use_kdtree:
        tree = KDTree(sc_arr)
        neighbors = tree.query_ball_point(sc_arr, r=r)
        for i, neigh in enumerate(neighbors):
            indicator[i, neigh] = 1
        if compute_ans:
            n_last_row = 0
            for process_i in range(chunk_size):
                X = np.array_split(sc_arr, chunk_size)[process_i]
                chunk = distance_matrix(X, sc_arr)
                ans[process_i] = chunk
                n_last_row += chunk.shape[0]
    else:
        n_last_row = 0
        for process_i in range(chunk_size):
            X = np.array_split(sc_arr, chunk_size)[process_i]
            chunk = distance_matrix(X, sc_arr)
            if compute_ans:
                ans[process_i] = chunk
            neigh = [np.flatnonzero(d < r) for d in chunk]
            for i in range(len(neigh)):
                indicator[i + n_last_row, neigh[i]] = 1
            n_last_row += chunk.shape[0]
    return st_coord, indicator, ans


def coord_eva(coord, ans, chunk_size = 12):
    coord = np.array(coord)
    cor_all = 0
    for process_i in range(chunk_size):
        X = np.array_split(coord, chunk_size)[process_i]
        Y = coord
        chunk = distance_matrix(X,Y)
        cor = pear(ans[process_i], chunk)
        # print(cor)
        cor_all += cor
    # print(f'Avearge shape correlation is: {cor_all/chunk_size}')
    return cor_all/chunk_size


# @timeit
def embedding(sparse_A, ans, path, left_range = 0, right_range = 30, steps = 30, dim = 2, verbose = False):
    aff = np.array(sparse_A, dtype = 'f')
    mask1 = (aff < 9e-300) & (aff >= 0)
    aff[mask1]=0.1
    np.fill_diagonal(aff,0)
    mask = aff != 0
    aff[mask] = 1 /aff[mask]
    #D = csr_matrix(aff) too less neighbor will occur
    del mask
    if ans is None:
        n_neighbors = int(np.round((left_range + 1) * 15))
        n_neighbors = max(2, n_neighbors)
        coord = umap.UMAP(n_components=dim, metric="precomputed", n_neighbors=n_neighbors, random_state=103).fit_transform(aff)
        return coord, 0.0

    max_shape = 0
    if verbose:
    # save all reconstructed result
        for i in range(int(left_range),int(right_range)):
            for j in range(steps):
                coord = umap.UMAP(n_components=dim, metric = "precomputed", n_neighbors=int(np.round((i+1)*15)), random_state = 100*j+3).fit_transform(aff)
                cor = coord_eva(coord, ans, chunk_size = 12)
                pd.DataFrame(coord).to_csv(path + str(i) + '_' + str(j) + '.csv',index = False, header= False, sep = ',')
                # print(f'neighbor_{(i+1)*15}, random_{j} cor: {cor}')
                if cor > max_shape:
                    max_shape = cor
                    best_in_shape = coord
        pd.DataFrame(best_in_shape).to_csv(path + 'coord_best.csv',index = False, header= False, sep = ',')
        # print(f'max shape cor is {max_shape}')
    else:
    # only output the best reconstructed result
        for i in range(int(left_range),int(right_range)):
            for j in range(steps):
                coord = umap.UMAP(n_components=dim, metric = "precomputed", n_neighbors=(i+1)*15, random_state = 100*j+3).fit_transform(aff)
                cor = coord_eva(coord, ans, chunk_size = 12)
                if cor > max_shape:
                    max_shape = cor
                    best_in_shape = coord
    #print('Reached a correlation in shape at:', max_shape)
    return best_in_shape, max_shape


def refine_embedding_local(sparse_A, ans, init_coord, n_neighbors, n_epochs=30, n_runs=3, jitter=1e-3, seed=103):
    """
    Local refinement of coordinates to improve shape correlation.
    Uses UMAP with init coordinates and small jitter, keeps best correlation.
    """
    if ans is None:
        return init_coord, 0.0
    init_coord = np.asarray(init_coord)
    sparse_A = np.asarray(sparse_A)
    dim = init_coord.shape[1]
    best_coord = init_coord
    best_shape = coord_eva(init_coord, ans, chunk_size=12)
    n_neighbors = max(2, int(n_neighbors))
    for i in range(int(n_runs)):
        rng = np.random.RandomState(seed + i)
        noise = rng.normal(scale=float(jitter), size=init_coord.shape)
        coord = umap.UMAP(
            n_components=dim,
            metric="precomputed",
            n_neighbors=n_neighbors,
            random_state=seed + i,
            init=init_coord + noise,
            n_epochs=int(n_epochs),
        ).fit_transform(sparse_A)
        cor = coord_eva(coord, ans, chunk_size=12)
        if cor > best_shape:
            best_shape = cor
            best_coord = coord
    return best_coord, best_shape


# @timeit
def calculate_affinity_mat(lr_df, data):
    '''
    This function calculate the affinity matrix from TPM and LR pairs.
    '''
    # fetch the ligands' and receptors' indexes in the TPM matrix 
    # data.shape = gene * cell
    genes = data.index.tolist()
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # replace Gene ID to the index of each gene in data matrix #
    gene_index =dict(zip(genes, range(len(genes))))
    index = lr_df.replace({0: gene_index, 1:gene_index}).astype(int)

    ligandindex = index[0].reset_index()[0]
    receptorindex = index[1].reset_index()[1]
    scores = index[2].reset_index()[2]
    
    Atotake = ligandindex
    Btotake = receptorindex
    allscores = scores
    idx_data = data.reset_index()
    del idx_data[idx_data.columns[0]]
    
    for i in range(len(ligandindex)):
        if ligandindex[i] != receptorindex[i]:
            Atotake = Atotake.append(pd.Series(receptorindex[i]),ignore_index=True)
            Btotake = Btotake.append(pd.Series(ligandindex[i]),ignore_index=True)
            allscores = allscores.append(pd.Series(scores[i]),ignore_index=True)

    A = idx_data.loc[Atotake.tolist()]
    B = idx_data.loc[Btotake.tolist()]

    affinitymat = np.dot(np.dot(np.diag(allscores), A).T , B)
    
    return affinitymat


def prepare_affinity_indices(lr_df, genes):
    genes = list(genes)
    lr_df_align = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    if lr_df_align.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    gene_index = {gene: i for i, gene in enumerate(genes)}
    lig_idx = lr_df_align[0].map(gene_index).to_numpy(dtype=np.int64, copy=False)
    rec_idx = lr_df_align[1].map(gene_index).to_numpy(dtype=np.int64, copy=False)
    if lr_df_align.shape[1] >= 3:
        scores = lr_df_align.iloc[:, 2].to_numpy(dtype=np.float32, copy=False)
    else:
        scores = np.ones(lig_idx.shape[0], dtype=np.float32)
    mask = lig_idx != rec_idx
    if np.any(mask):
        lig_idx_extra = rec_idx[mask]
        rec_idx_extra = lig_idx[mask]
        lig_idx = np.concatenate([lig_idx, lig_idx_extra])
        rec_idx = np.concatenate([rec_idx, rec_idx_extra])
        scores = np.concatenate([scores, scores[mask]])
    return lig_idx, rec_idx, scores


def calculate_affinity_mat_fast(data_values, affinity_idx):
    lig_idx, rec_idx, scores = affinity_idx
    if lig_idx.size == 0:
        n_cells = data_values.shape[1]
        return np.zeros((n_cells, n_cells), dtype=np.float32)
    A = data_values[lig_idx]
    B = data_values[rec_idx]
    affinitymat = (scores[:, None] * A).T @ B
    return affinitymat


if nb is not None:
    @nb.njit(cache=True)
    def _affinity_edges_numba(lig_vals, rec_vals, scores, row_idx, col_idx):
        n_edges = row_idx.shape[0]
        n_pairs = scores.shape[0]
        out = np.zeros(n_edges, dtype=np.float32)
        for e in range(n_edges):
            i = row_idx[e]
            j = col_idx[e]
            acc = 0.0
            for p in range(n_pairs):
                acc += scores[p] * lig_vals[p, i] * rec_vals[p, j]
            out[e] = acc
        return out


def calculate_affinity_knn_sparse(data_values, affinity_idx, row_idx, col_idx, n_cells):
    lig_idx, rec_idx, scores = affinity_idx
    if lig_idx.size == 0 or row_idx.size == 0:
        return sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
    A = data_values[lig_idx]
    B = data_values[rec_idx]
    # Compute affinity only for KNN edges: sum_p scores[p] * A[p, i] * B[p, j]
    if nb is not None and row_idx.size > 0:
        edge_vals = _affinity_edges_numba(A, B, scores.astype(np.float32, copy=False), row_idx.astype(np.int64), col_idx.astype(np.int64))
    else:
        A_cols = A[:, row_idx]
        B_cols = B[:, col_idx]
        edge_vals = np.sum((scores[:, None] * A_cols) * B_cols, axis=0).astype(np.float32, copy=False)
    return sparse.csr_matrix((edge_vals, (row_idx, col_idx)), shape=(n_cells, n_cells))


# @timeit
def aff_embedding(alter_sc_exp,st_coord,sc_meta,lr_df,save_path, left_range = 1, right_range = 2, steps = 1, dim = 2,verbose = False,
                  fast_adj = False, compute_shape = True, chunk_size = 12):
    # 3.1 prep initial embedding that term3 requires
    ordered_st_coord, sc_dis_mat, ans = sc_adj_cal(
        st_coord, sc_meta, chunk_size = chunk_size, use_kdtree = fast_adj, compute_ans = compute_shape
    )
    ########################print(f'Start affinity calculation...') 
    sparse_A = chunk_cal_aff(alter_sc_exp, sc_dis_mat, lr_df)
    sparse_A[sparse_A!=0] = sparse_A[sparse_A!=0] - 0.1
    sparse_A = sparse_A + np.ones(sparse_A.shape) * 0.1
    np.fill_diagonal(sparse_A,1)
    #########################print(f'End affinity calculation.')
    #print(f'Start embedding...')
    coord, max_shape = embedding(sparse_A, ans, save_path, left_range, right_range, steps, dim, verbose = verbose)
    #print(f'End embedding.')
    return coord, max_shape, ordered_st_coord,sparse_A,ans


# @timeit
def get_hvg(adata):
    p_adata = sc.pp.normalize_total(adata, target_sum=1e4,copy = True)
    sc.pp.log1p(p_adata)
    sc.pp.highly_variable_genes(p_adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    # sc.pl.highly_variable_genes(p_adata)
    p_adata = p_adata[:, p_adata.var.highly_variable]
    # adata.layers["log"] = p_adataUMAP
    return set(p_adata.var_names)


def center_shift_embedding(sc_coord, sc_meta_orig, max_dist):
    # added in v6
    '''
    shift cells belongs to each spot by their centroid and spot coordinates 
    sc_meta must have st_x and st_y
    sc_meta_orig = obj_spex.sc_meta
    sc_coord = obj_spex.sc_coord
    max_dist = 1
    '''
    ##### tailored for each spot #####
    sc_meta = sc_meta_orig.copy()
    sc_meta[['spex_UMAP1','spex_UMAP2']] = sc_coord
    umap_core = sc_meta.groupby('spot').mean()[['spex_UMAP1','spex_UMAP2']]
    idx_lst = umap_core.index.tolist()
    idx_dict = {k: v for v, k in enumerate(idx_lst)}
    coord_dict_x = {v: k for v, k in enumerate(list(umap_core['spex_UMAP1']))}
    coord_dict_y = {v: k for v, k in enumerate(list(umap_core['spex_UMAP2']))}
    sc_meta['indice'] = sc_meta['spot'].map(idx_dict)
    sc_meta['core1'] = sc_meta['indice'].map(coord_dict_x)
    sc_meta['core2'] = sc_meta['indice'].map(coord_dict_y)
    # calculating the unit length for the gap between two spot
    x_coors = np.sort(list(set(sc_meta['st_x'])))
    unit_len = x_coors[1] - x_coors[0]
    spot_space = unit_len/2
    # calculating the scale factor
    core_dist = pd.DataFrame(distance_matrix(sc_coord,umap_core))
    core_dist['spot'] = sc_meta['spot'].values
    max_center_dist = pd.DataFrame(np.diag(core_dist.groupby('spot').max()))
    scale_factor = list(spot_space/(max_center_dist[max_center_dist!=0])[0])
    scale_factor_dict = {v: k for v, k in enumerate(scale_factor)}
    sc_meta['centering_scale_factor'] = sc_meta['indice'].map(scale_factor_dict)
    #print(sc_meta['centering_scale_factor'].head(5))
    # center shift and scale
    tmp = sc_meta[['spex_UMAP1','spex_UMAP2']] - sc_meta[['core1','core2']].values
    tmp1 = tmp*(max_dist * sc_meta[['centering_scale_factor','centering_scale_factor']].values)
    sc_meta[['adj_spex_UMAP1','adj_spex_UMAP2']] = tmp1 + sc_meta[['st_x','st_y']].values
    for idx,row in sc_meta.iterrows():
        # add v9
        # if spot only have one cell, the scale factor would be nan
        if (row['core1'] - row['spex_UMAP1'] < 0.000001) and (row['core2'] - row['spex_UMAP2'] < 0.000001):
            sc_meta.loc[idx,'adj_spex_UMAP1'] = row['st_x']
            sc_meta.loc[idx,'adj_spex_UMAP2'] = row['st_y']
            sc_meta.loc[idx,'centering_scale_factor'] = 0
    return sc_meta
