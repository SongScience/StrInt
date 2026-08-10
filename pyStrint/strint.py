from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from . import optimizers
from . import utils
from . import cell_selection
from . import preprocess as pp

import time
import logging

import pandas as pd
import numpy as np
import os
import warnings
import json
from datetime import datetime

# TODO del after test
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f'{func.__name__}\t{end - start} seconds')
        return result
    return wrapper


class strInt:
    '''

    '''
    def __init__(self, save_path = None, st_adata = None, weight = None, 
                 sc_ref = None, sc_adata = None, cell_type_key = 'celltype', lr_df = None, 
                 st_tp = 'visium', species = 'Human'
                 ):
        self.save_path = save_path +'/'
        self.st_adata = st_adata
        self.weight = weight
        self.sc_ref = sc_ref
        self.sc_adata = sc_adata
        self.cell_type_key = cell_type_key
        self.lr_df = lr_df
        self.st_tp = st_tp
        self.species = species
        # cache for performance
        self._term1_weight = None
        self._term1_weight_cols = None
        self._term1_weight_W_HVG = None
        self._sc_coord_cache = None
        self._sc_dist_cache = None
        self._sc_knn_cache = None
        self._term3_cache = None
        self._term4_cache = None
        self._loss3_cache = None
        self._loss4_cache = None
        self._lr_idx_cache = None
        self._lr_idx_cols = None
        self._knn_df_cache = None
        self._active_spots = None
        self._spot_codes_cache = None
        self._spot_codes_spots = None
        self._knn_ind_cache = None
        self._knn_ncp_cache = None
        self._knn_ind_sparse_cache = None
        self._knn_ncp_sparse_cache = None
        self._use_sparse_term3 = False
        self.term4_n_cores = 1
        self._spot_knn_idx_cache = None
        self._affinity_idx_cache = None
        self._affinity_idx_cols = None
        self._lr_agg_idx_cache = None
        self._lr_agg_idx_cols = None
        self._use_vectorized_term4 = False
        self._use_sparse_term4 = False
        self._cell_index_cache = None
        self._cell_index_idx = None
        self._spot_codes_all_cache = None
        self._spot_levels_cache = None
        self._centroid_arr_cache = None
        self._st_aff_profile_map = None
        self._spot_to_idx_cache = None
        self._pair_to_row_cache = None
        self._st_aff_values_cache = None
        self._knn_arrays_cache = None
        self._spot_to_idx_keys = None
        self._spot_neighbors_cache = None
        self._spot_neighbors_max = None
        self._spot_neighbors_max_cache = None
        self._knn_edge_row_cache = None
        self._knn_edge_col_cache = None
        self._knn_cp_dist_adj_cache = None
        self._knn_n_cells_cache = None
        self._profile = False
        self._profile_every = 1
        self._profile_iter = -1
        self._freeze_embedding = False
        self._fast_embedding = False
        self._embedding_compute_shape = True
        self._term2_df_cache = None
        self._term2_df_index = None
        self._term2_df_cols = None
        self._term5_df_cache = None
        self._term5_df_index = None
        self._term5_df_cols = None
        self._term1_df_cache = None
        self._term1_df_index = None
        self._term1_df_cols = None
        self._term1_spot_codes = None
        self._term1_valid_mask = None
        self._term1_cell_n_spot = None
        self._term1_sc_spot_sum = None
        self._term1_last_values = None
        self._term1_st_col_idx = None
        self._term1_use_incremental = False
        self._term1_block_size = None
        self._lr_gene_mask = None
        self._non_lr_gene_mask = None
        self._lr_gene_mask_cols = None
        self._aff_embed_cache_key = None
        self._aff_embed_cache_result = None

    def _check_input(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.species not in ['Human', 'Mouse']:
            raise ValueError('Species should be chosen among either Human or Mouse.')
        if self.lr_df is None:
            self.lr_df = pp.load_lr_df(species = self.species,lr_dir = None)

        utils.check_st_tp(self.st_tp)
        self.st_adata, self.sc_adata, self.sc_ref, self.weight = utils.check_index_str(self.st_adata, self.sc_adata, self.sc_ref, self.weight)
        self.st_adata, self.weight = utils.check_spots(self.st_adata, self.weight)
        self.st_coord = utils.check_st_coord(self.st_adata)
        self.lr_df = utils.align_lr_gene(self)
        utils.check_st_sc_pair(self.st_adata, self.sc_adata)
        self.sc_adata, self.sc_ref = utils.check_sc(self.sc_adata, self.sc_ref)
        utils.check_decon_type(self.weight, self.sc_adata, self.cell_type_key)
        print('Parameters checked!')

    def _aff_embedding_cached(self, alter_sc_exp):
        vals = alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
        flat_vals = vals.ravel()
        if flat_vals.size > 2048:
            sample = np.concatenate([flat_vals[:1024], flat_vals[-1024:]])
        else:
            sample = flat_vals
        sample_hash = hash(sample.tobytes())
        key = (
            vals.shape,
            float(vals.sum()),
            float(vals.mean()),
            float(vals.std()),
            sample_hash,
            tuple(self.st_coord.index),
            tuple(self.sc_agg_meta.index),
            self.left_range,
            self.right_range,
            self.steps,
            self.dim,
            bool(self._fast_embedding),
            bool(self._embedding_compute_shape),
        )
        if self._aff_embed_cache_key == key and self._aff_embed_cache_result is not None:
            return self._aff_embed_cache_result
        result = optimizers.aff_embedding(
            alter_sc_exp,
            self.st_coord,
            self.sc_agg_meta,
            self.lr_df,
            self.save_path,
            self.left_range,
            self.right_range,
            self.steps,
            self.dim,
            fast_adj=self._fast_embedding,
            compute_shape=self._embedding_compute_shape,
        )
        self._aff_embed_cache_key = key
        self._aff_embed_cache_result = result
        return result

    @timeit
    def prep(self):
        ######### init ############
        # 1. check input and parameters
        self._check_input()
        # 2. creat obj
        self.st_exp = self.st_adata.to_df()
        self.sc_exp = self.sc_adata.to_df()
        self.sc_meta = self.sc_adata.obs.copy()
        del self.sc_adata
        # 3. generate obj
        self.svg = optimizers.get_hvg(self.st_adata)
        del self.st_adata
        # print('Getting svg genes')
        # 4. remove no neighbor spots
        spots_nn_lst = optimizers.findSpotKNN(self.st_coord,self.st_tp)
        empty_spots = [k for k, v in spots_nn_lst.items() if not v]
        if empty_spots:
            # empty_spots is not empty
            #remove from spots_nn_lst is key have empty value, which means spot have no neighbor
            spots_nn_lst = {k: v for k, v in spots_nn_lst.items() if v}
            #remove empty spots from st_exp
            self.st_exp = self.st_exp.drop(empty_spots)
            self.st_coord = self.st_coord.drop(empty_spots)
            self.weight = self.weight.drop(empty_spots)
        self.spots_nn_lst = spots_nn_lst
        self.st_aff_profile_df = optimizers.cal_aff_profile_sparse(self.st_exp, self.lr_df, self.spots_nn_lst)
        self._st_aff_profile_map = optimizers.build_spot_aff_profile_map(self.st_aff_profile_df)
        self.lr_df_align = self.lr_df[self.lr_df[0].isin(self.st_exp.columns) & self.lr_df[1].isin(self.st_exp.columns)].copy()
        if self.lr_df_align.shape[0] < 10:
            print(f"Warning: only {self.lr_df_align.shape[0]} LR pairs after alignment; term4/loss4 may be near zero.")

    @staticmethod
    def _spotwise_corr(spot_exp_df, agg_exp_df):
        """Vectorized per-spot Pearson correlation with zero-variance guard."""
        a = spot_exp_df.to_numpy(dtype=np.float32, copy=False)
        b = agg_exp_df.to_numpy(dtype=np.float32, copy=False)
        if a.shape != b.shape or a.shape[0] == 0:
            return np.zeros((a.shape[0],), dtype=np.float32)
        a_center = a - a.mean(axis=1, keepdims=True)
        b_center = b - b.mean(axis=1, keepdims=True)
        a_norm = np.sqrt((a_center * a_center).sum(axis=1))
        b_norm = np.sqrt((b_center * b_center).sum(axis=1))
        denom = a_norm * b_norm
        corr = np.zeros((a.shape[0],), dtype=np.float32)
        valid = denom > 0
        if np.any(valid):
            corr[valid] = (a_center[valid] * b_center[valid]).sum(axis=1) / denom[valid]
        return corr

    def _copy_best_spot_profiles(self, result, spot_exp, sc_exp, spot_frac=0.0, min_corr=None,
                                 require_improve=True):
        if not spot_frac or spot_frac <= 0:
            return result
        base_result = result.copy()
        spot_ids = list(spot_exp.index)
        n_copy = max(1, int(len(spot_ids) * float(spot_frac)))
        sum_sc_agg_exp = cell_selection.get_sum_sc_agg(sc_exp, base_result, spot_exp)
        spot_corr_vals = self._spotwise_corr(spot_exp, sum_sc_agg_exp)
        spot_corr = pd.Series(spot_corr_vals, index=spot_ids)
        target_spots = spot_corr.sort_values().index[:n_copy].tolist()
        agg_mat = sum_sc_agg_exp.to_numpy(dtype=np.float32, copy=False)
        replacements = {}
        for spot in target_spots:
            target_i = spot_ids.index(spot)
            target_vec = spot_exp.loc[spot].to_numpy(dtype=np.float32, copy=False)
            donor_corr = cell_selection._rowwise_corr_with_vector(agg_mat, target_vec)
            donor_corr[target_i] = -np.inf
            best_i = int(np.argmax(donor_corr))
            best_corr = float(donor_corr[best_i])
            if min_corr is not None and best_corr < float(min_corr):
                continue
            if require_improve and best_corr <= float(spot_corr.iloc[target_i]):
                continue
            donor_spot = spot_ids[best_i]
            donor_ids = base_result.loc[base_result['spot'] == donor_spot, 'sc_id'].tolist()
            target_mask = base_result['spot'] == spot
            target_count = int(target_mask.sum())
            if target_count == 0:
                continue
            if len(donor_ids) >= target_count:
                new_ids = donor_ids[:target_count]
            else:
                pad = base_result.loc[target_mask, 'sc_id'].tolist()
                new_ids = donor_ids + pad[:max(0, target_count - len(donor_ids))]
            replacements[spot] = new_ids
        for spot, new_ids in replacements.items():
            target_mask = result['spot'] == spot
            if not target_mask.any():
                continue
            result.loc[target_mask, 'sc_id'] = new_ids
            if 'celltype' in result.columns:
                result.loc[target_mask, 'celltype'] = self.sc_meta.loc[new_ids, self.cell_type_key].values
        return result


    @timeit
    def select_cells(self, user_sc_exp = None, user_sc_agg_meta = None, p = 0, mean_num_per_spot = 10, metric = 'correlation', 
                     max_rep = 1, repeat_penalty = 10, seed = 1111, use_vectorized_select = True,
                     use_gpu = False,
                     candidate_topk = 0, early_stop_tol = 0.0, early_stop_patience = 0,
                     init_method = 'topk', focus_low_spots_frac = 0.0, focus_extra_passes = 0,
                     swap_method = 'correlation', init_copy_spot_frac = 0.0,
                     init_copy_min_corr = None, init_copy_require_improve = True,
                     copy_after_swaps = False):
        """
        Select cells for each spot based on the deconvolution result and the spatial expression data.
        Parameters
        ----------
        user_sc_exp : pd.DataFrame, optional, default None
            The spatial expression data for the user-defined cells.
        user_sc_agg_meta : pd.DataFrame, optional, default None
            The cell selection result for the user-defined cells.
        p : float, optional, default 0.1
            The probability of swapping a cell.
        mean_num_per_spot : int, optional, default 10
            The mean number of cells in each spot.
        metric : str, optional, default 'correlation'
            The metric for evaluation of cell selection. 'rmse', 'correlation', or 'spot_cor'.
        max_rep : int, optional, default 3  
            The maximum number of repetitions for cell selection.
        repeat_penalty : int, optional, default 10
            The penalty for repeating the same cell type in the same spot.
        Returns
        -------
        sc_agg_meta : pd.DataFrame
            The cell selection result.
        """
        if user_sc_agg_meta is not None:
            self.sc_agg_meta = user_sc_agg_meta
            self.alter_sc_exp = user_sc_exp
            self.sc_agg_meta.index = self.sc_agg_meta.index.astype(str)
            self.alter_sc_exp.index = self.sc_agg_meta.index
            self.sc_agg_meta['spot'] = self.sc_agg_meta['spot'].astype(str)
            self.alter_sc_exp = self.alter_sc_exp[self.st_exp.columns]
            self.alter_sc_exp.columns.get_level_values(0).name = 'symbol'
            self.alter_sc_exp = pp.scale_sum(self.alter_sc_exp,1e4)
            self.spot_cell_dict = self.sc_agg_meta.groupby('spot').apply(optimizers.apply_spot_cell).to_dict()
            self.lr_df_align = self.lr_df[self.lr_df[0].isin(self.alter_sc_exp.columns) & self.lr_df[1].isin(self.alter_sc_exp.columns)].copy()
            if self.lr_df_align.shape[0] < 10:
                print(f"Warning: only {self.lr_df_align.shape[0]} LR pairs after alignment; term4/loss4 may be near zero.")
            result = self.sc_agg_meta
        else:
            self.repeat_penalty = repeat_penalty
            self.p = p
            if metric == 'spot_cor' and not use_vectorized_select:
                raise ValueError("metric='spot_cor' requires use_vectorized_select=True.")
            if mean_num_per_spot == 0:	
                self.num = self.weight	
                print(f'\t mean_num_per_spot == 0; Using the exact cell number in each spot provided in weight.')
            elif mean_num_per_spot == 1:
                self.num = self.weight.apply(lambda x: x.eq(x.max()).astype(int), axis=1)
                print(f'\t mean_num_per_spot == 1; Using the idxmax celltype for each spot.')
            else:
                print(f'\t Estimating the cell number in each spot by the deconvolution result.')	
                self.weight = utils.check_decon_sum(self.weight)
                # print(self.weight.head(5))
                spot_cell_num = cell_selection.estimate_cell_number(self.st_exp, mean_num_per_spot)
                self.num = cell_selection.randomization(self.weight,spot_cell_num, seed)
            # spot-correlation objective data prep (scaled, non-mito, shared genes)
            self.spot_cor_sc_exp = None
            self.spot_cor_st_exp = None
            self.spot_cor_genes = None
            if metric == 'spot_cor':
                spot_sc_exp, spot_st_exp = pp.data_clean(self.sc_exp, self.st_exp)
                spot_genes = pp.denoise_genes(spot_sc_exp, spot_st_exp, spot_st_exp, self.species)
                spot_sc_exp = spot_sc_exp[spot_genes]
                spot_st_exp = spot_st_exp[spot_genes]
                spot_sc_exp = pp.scale_sum(spot_sc_exp, 1e4)
                spot_st_exp = pp.scale_sum(spot_st_exp, 1e4)
                self.spot_cor_sc_exp = spot_sc_exp.loc[self.sc_exp.index]
                self.spot_cor_st_exp = spot_st_exp.loc[self.st_exp.index]

            # 1. subset and filter
            self.filter_st_exp, self.filter_sc_exp = pp.subset_inter(self.st_exp, self.sc_exp)
            # 2. feature selection
            if metric == 'spot_cor' and self.spot_cor_sc_exp is not None and self.spot_cor_st_exp is not None:
                # Use all spot-cor genes for optimization/evaluation consistency.
                self.spot_cor_genes = list(self.spot_cor_st_exp.columns)
                if len(self.spot_cor_genes) == 0:
                    raise ValueError("No spot-correlation genes available for optimization.")
                self.spot_cor_sc_exp = self.spot_cor_sc_exp.loc[self.sc_exp.index, self.spot_cor_genes]
                self.spot_cor_st_exp = self.spot_cor_st_exp.loc[self.st_exp.index, self.spot_cor_genes]
                self.lr_hvg_genes = self.spot_cor_genes
                print(f'\t SpexMod selects {len(self.lr_hvg_genes)} feature genes (all spot-cor genes).')
            else:
                self.sort_genes = cell_selection.feature_sort(self.filter_sc_exp, degree = 2, span = 0.3)
                self.lr_hvg_genes = cell_selection.lr_shared_top_k_gene(self.sort_genes, self.lr_df, k = 3000, keep_lr_per = 1)
                print(f'\t SpexMod selects {len(self.lr_hvg_genes)} feature genes.')
                # Keep original scaling from the initial spot-cor prep.

            # 3. scale and norm
            self.trans_id_idx = pd.DataFrame(list(range(self.filter_sc_exp.shape[0])), index = self.filter_sc_exp.index)
            self.hvg_st_exp = self.filter_st_exp.loc[:,self.lr_hvg_genes]
            self.hvg_sc_exp = self.filter_sc_exp.loc[:,self.lr_hvg_genes]
            if metric == 'spot_cor' and self.spot_cor_sc_exp is not None and self.spot_cor_st_exp is not None:
                # Use the spot-cor matrices directly to avoid per-gene normalization.
                norm_hvg_st = self.spot_cor_st_exp
                norm_hvg_sc = self.spot_cor_sc_exp
            else:
                norm_hvg_st = cell_selection.norm_center(self.hvg_st_exp)
                norm_hvg_sc = cell_selection.norm_center(self.hvg_sc_exp)
            self.csr_st_exp = csr_matrix(norm_hvg_st)
            self.csr_sc_exp = csr_matrix(norm_hvg_sc)
            # all lr that exp in st
            self.lr_df_align = self.lr_df[self.lr_df[0].isin(self.filter_st_exp.columns) & self.lr_df[1].isin(self.filter_st_exp.columns)].copy()
            if self.lr_df_align.shape[0] < 10:
                print(f"Warning: only {self.lr_df_align.shape[0]} LR pairs after alignment; term4/loss4 may be near zero.")
            # 4. init cell selection
            self.spot_cell_dict, self.init_cor, self.picked_time = cell_selection.init_solution(
                self.num,
                self.filter_st_exp.index.tolist(),
                self.csr_st_exp,
                self.csr_sc_exp,
                self.sc_meta[self.cell_type_key],
                self.trans_id_idx,
                self.repeat_penalty,
                init_method=init_method,
                init_copy_spot_frac=init_copy_spot_frac,
                init_copy_min_corr=init_copy_min_corr,
                init_copy_require_improve=init_copy_require_improve,
            )
            self.init_sc_df = cell_selection.dict2df(self.spot_cell_dict, norm_hvg_st, norm_hvg_sc,self.sc_meta)
            result = self.init_sc_df
            ########################################
            # TODO debug start
            # print('init new')
            # result = pd.read_csv('/data6/wangjingwan/5.Simpute/4.datasets/SCC_c2l_rep50/spex/spexmod_sc_meta.tsv', sep='\t', header=0, index_col=0)
            # # print('New',result.head(5))
            # self.init_sc_df_strint = result

            # a = pd.DataFrame(self.init_sc_df_strint.groupby('sc_id').size())
            # b = dict(zip(a.index, a[0]))
            # init_pick_time = self.sc_meta.copy()
            # init_pick_time['count'] = 0
            # init_pick_time['count'] = init_pick_time.index.map(b)
            # init_pick_time.fillna(0, inplace=True)
            # self.picked_time = pd.DataFrame(init_pick_time['count'])
            # TODO debug end
            ########################################
            # 5. reselect cells
            print('\t Swap selection start...')
            # Track state history for restoration
            self.state_history = []
            self.result = None
            prev_mean_spot_cor = None
            no_improve_rounds = 0
            if self.p == 0:
                for iter_idx in range(max_rep):
                    self.sum_sc_agg_exp = cell_selection.get_sum_sc_agg(norm_hvg_sc, result, norm_hvg_st)
                    self.sc_agg_aff_profile_df = optimizers.cal_aff_profile(self.sum_sc_agg_exp, self.lr_df_align)
                    spot_cor_sum_sc_agg_exp = None
                    if metric == 'spot_cor' and self.spot_cor_sc_exp is not None:
                        spot_cor_sum_sc_agg_exp = cell_selection.get_sum_sc_agg(self.spot_cor_sc_exp, result, self.spot_cor_st_exp)
                    result, self.after_picked_time = cell_selection.reselect_cell(
                        norm_hvg_st, self.spots_nn_lst, self.st_aff_profile_df,
                        norm_hvg_sc, self.csr_sc_exp, self.sc_meta, self.trans_id_idx,
                        self.sum_sc_agg_exp, self.sc_agg_aff_profile_df,
                        result, self.picked_time, self.lr_df_align,
                        p=self.p, repeat_penalty=self.repeat_penalty, metric=metric,
                        st_aff_profile_map=self._st_aff_profile_map,
                        use_vectorized=use_vectorized_select,
                        use_gpu=use_gpu,
                        candidate_topk=candidate_topk,
                        spot_cor_sc_exp=self.spot_cor_sc_exp,
                        spot_cor_st_exp=self.spot_cor_st_exp,
                        sum_sc_agg_exp_exp=spot_cor_sum_sc_agg_exp,
                        swap_method=swap_method)
                    if metric == 'spot_cor' and self.spot_cor_sc_exp is not None and early_stop_patience > 0:
                        spot_cor_sum_sc_agg_exp = cell_selection.get_sum_sc_agg(self.spot_cor_sc_exp, result, self.spot_cor_st_exp)
                        spot_corr_vals = self._spotwise_corr(self.spot_cor_st_exp, spot_cor_sum_sc_agg_exp)
                        mean_spot_cor = float(np.mean(spot_corr_vals))
                        if prev_mean_spot_cor is not None and (mean_spot_cor - prev_mean_spot_cor) <= float(early_stop_tol):
                            no_improve_rounds += 1
                        else:
                            no_improve_rounds = 0
                        prev_mean_spot_cor = mean_spot_cor
                        if no_improve_rounds >= int(early_stop_patience):
                            print(f'\t Early stop at iter {iter_idx+1}: mean spot_cor={mean_spot_cor:.4f}')
                            break
                if focus_low_spots_frac > 0 and focus_extra_passes > 0 and metric == 'spot_cor' and self.spot_cor_sc_exp is not None:
                    for _ in range(focus_extra_passes):
                        spot_cor_sum_sc_agg_exp = cell_selection.get_sum_sc_agg(self.spot_cor_sc_exp, result, self.spot_cor_st_exp)
                        spot_corr_vals = self._spotwise_corr(self.spot_cor_st_exp, spot_cor_sum_sc_agg_exp)
                        spot_corr = pd.Series(spot_corr_vals, index=self.spot_cor_st_exp.index)
                        n_focus = max(1, int(len(spot_corr) * focus_low_spots_frac))
                        focus_spots = spot_corr.sort_values().index[:n_focus].tolist()
                        self.sum_sc_agg_exp = spot_cor_sum_sc_agg_exp
                        self.sc_agg_aff_profile_df = optimizers.cal_aff_profile(self.sum_sc_agg_exp, self.lr_df_align)
                        result, self.after_picked_time = cell_selection.reselect_cell(
                            norm_hvg_st, self.spots_nn_lst, self.st_aff_profile_df,
                            norm_hvg_sc, self.csr_sc_exp, self.sc_meta, self.trans_id_idx,
                            self.sum_sc_agg_exp, self.sc_agg_aff_profile_df,
                            result, self.picked_time, self.lr_df_align,
                            p=self.p, repeat_penalty=self.repeat_penalty, metric=metric,
                            st_aff_profile_map=self._st_aff_profile_map,
                            use_vectorized=use_vectorized_select,
                            use_gpu=use_gpu,
                            candidate_topk=candidate_topk,
                            spot_cor_sc_exp=self.spot_cor_sc_exp,
                            spot_cor_st_exp=self.spot_cor_st_exp,
                            sum_sc_agg_exp_exp=spot_cor_sum_sc_agg_exp,
                            spot_subset=focus_spots,
                            swap_method=swap_method)
                    if copy_after_swaps:
                        result = self._copy_best_spot_profiles(
                            result,
                            spot_exp=self.spot_cor_st_exp if metric == 'spot_cor' else norm_hvg_st,
                            sc_exp=self.spot_cor_sc_exp if metric == 'spot_cor' else norm_hvg_sc,
                            spot_frac=init_copy_spot_frac,
                            min_corr=init_copy_min_corr,
                            require_improve=init_copy_require_improve,
                        )
            else:
                for i in range(max_rep):
                    self.sum_sc_agg_exp = cell_selection.get_sum_sc_agg(self.filter_sc_exp, result, self.filter_st_exp)
                    self.sc_agg_aff_profile_df = optimizers.cal_aff_profile(self.sum_sc_agg_exp, self.lr_df_align)
                    spot_cor_sum_sc_agg_exp = None
                    if metric == 'spot_cor' and self.spot_cor_sc_exp is not None:
                        spot_cor_sum_sc_agg_exp = cell_selection.get_sum_sc_agg(self.spot_cor_sc_exp, result, self.spot_cor_st_exp)
                    result, self.after_picked_time = cell_selection.reselect_cell(
                        self.filter_st_exp, self.spots_nn_lst, self.st_aff_profile_df,
                        self.filter_sc_exp, self.csr_sc_exp, self.sc_meta, self.trans_id_idx,
                        self.sum_sc_agg_exp, self.sc_agg_aff_profile_df,
                        result, self.picked_time, self.lr_df_align,
                        p=self.p, repeat_penalty=self.repeat_penalty, metric=metric,
                        st_aff_profile_map=self._st_aff_profile_map,
                        use_vectorized=use_vectorized_select,
                        use_gpu=use_gpu,
                        candidate_topk=candidate_topk,
                        spot_cor_sc_exp=self.spot_cor_sc_exp,
                        spot_cor_st_exp=self.spot_cor_st_exp,
                        sum_sc_agg_exp_exp=spot_cor_sum_sc_agg_exp,
                        swap_method=swap_method)
                    # Save state at each iteration
                    self.state_history.append(self.sc_exp.loc[result['sc_id']].values.copy())
                if copy_after_swaps:
                    result = self._copy_best_spot_profiles(
                        result,
                        spot_exp=self.spot_cor_st_exp if metric == 'spot_cor' else self.filter_st_exp,
                        sc_exp=self.spot_cor_sc_exp if metric == 'spot_cor' else self.filter_sc_exp,
                        spot_frac=init_copy_spot_frac,
                        min_corr=init_copy_min_corr,
                        require_improve=init_copy_require_improve,
                    )
                # After optimization, build result DataFrame for losses
                if self.state_history:
                    # Ensure alter_sc_exp is initialized before restoring states
                    if not hasattr(self, 'alter_sc_exp') or self.alter_sc_exp is None:
                        self.alter_sc_exp = pd.DataFrame(
                            self.state_history[0],
                            # The same source cell can be selected for different spots.
                            # Keep a unique per-selection instance index so downstream
                            # KNN matrices have unambiguous row and column labels.
                            index=result.index,
                            columns=self.sc_exp.loc[result['sc_id']].columns
                        )
                    # run_gradient() needs the current spot-cell assignment metadata.
                    self.sc_agg_meta = result.copy()
                    losses = []
                    for state in self.state_history:
                        self.alter_sc_exp.values[:] = state
                        self.run_gradient(compute_term3=True, compute_term4=True)
                        losses.append(self.compute_total(normalize_terms=True, term_norm='rms'))
                    self.result = pd.DataFrame({'total': losses})
            # Restore final state
            self.alter_sc_exp = self.sc_exp.loc[result['sc_id']]
            self.alter_sc_exp.index = result.index
            self.sc_agg_meta = result
            self.spot_cell_dict = self.sc_agg_meta.groupby('spot').apply(optimizers.apply_spot_cell).to_dict()
            return result
            ###############################################################################
            ################################## original ###################################
            # if self.p == 0:
            #     self.sum_sc_agg_exp = cell_selection.get_sum_sc_agg(norm_hvg_sc, result, norm_hvg_st)
            #     self.sc_agg_aff_profile_df = optimizers.cal_aff_profile(self.sum_sc_agg_exp, self.lr_df_align)
            #     result, self.after_picked_time = cell_selection.reselect_cell(norm_hvg_st, self.spots_nn_lst, self.st_aff_profile_df, 
            #                 norm_hvg_sc, self.csr_sc_exp, self.sc_meta, self.trans_id_idx,
            #                 self.sum_sc_agg_exp, self.sc_agg_aff_profile_df,
            #                 result, self.picked_time, self.lr_df_align, 
            #                 p = self.p, repeat_penalty = self.repeat_penalty)
            # else:
            #     for i in range(max_rep):
            #         self.sum_sc_agg_exp = cell_selection.get_sum_sc_agg(self.filter_sc_exp,result,self.filter_st_exp)
            #         self.sc_agg_aff_profile_df = optimizers.cal_aff_profile(self.sum_sc_agg_exp, self.lr_df_align)
            #         result, self.after_picked_time = cell_selection.reselect_cell(self.filter_st_exp, self.spots_nn_lst, self.st_aff_profile_df, 
            #                     self.filter_sc_exp, self.csr_sc_exp, self.sc_meta, self.trans_id_idx,
            #                     self.sum_sc_agg_exp, self.sc_agg_aff_profile_df,
            #                     result, self.picked_time, self.lr_df_align, 
            #                     p = self.p, repeat_penalty = self.repeat_penalty)
            ###############################################################################

            # 6. save result
            self.alter_sc_exp = self.sc_exp.loc[result['sc_id']]
            self.alter_sc_exp.index = result.index
            self.sc_agg_meta = result
            self.spot_cell_dict = self.sc_agg_meta.groupby('spot').apply(optimizers.apply_spot_cell).to_dict()
            return result


    @timeit
    def run_gradient(self, compute_term3 = True, compute_term4 = True):
        log_path = getattr(self, "debug_log_path", None)
        try:
            last_align = getattr(self, "_last_sc_ref_align_diag", {}) or {}
            sc_ref_arr = np.asarray(self.sc_ref)
            sc_ref_has_nan = bool(np.isnan(sc_ref_arr).any()) if np.issubdtype(sc_ref_arr.dtype, np.number) else False
            sc_ref_has_inf = bool(np.isinf(sc_ref_arr).any()) if np.issubdtype(sc_ref_arr.dtype, np.number) else False
            sc_ref_broadcast_like = False
            if sc_ref_arr.ndim == 2 and sc_ref_arr.shape[0] > 1 and np.issubdtype(sc_ref_arr.dtype, np.number):
                sc_ref_broadcast_like = bool(np.all(np.isclose(sc_ref_arr, sc_ref_arr[0:1, :], atol=1e-7)))
            payload = {
                "ts": datetime.now().isoformat(timespec='seconds'),
                "event": "run_gradient_entry_diag",
                "entered_gradient": True,
                "mean_fill_happened": bool(last_align.get("used_mean_broadcast", False) or int(last_align.get("nan_rows_filled", 0)) > 0),
                "mean_fill_reason": last_align.get("used_fallback", "unknown"),
                "sc_ref_shape": list(sc_ref_arr.shape) if hasattr(sc_ref_arr, "shape") else None,
                "sc_ref_has_nan": sc_ref_has_nan,
                "sc_ref_has_inf": sc_ref_has_inf,
                "sc_ref_broadcast_like": sc_ref_broadcast_like,
                "alter_sc_exp_has_nan": bool(np.isnan(self.alter_sc_exp.values).any()),
                "alter_sc_exp_has_inf": bool(np.isinf(self.alter_sc_exp.values).any()),
            }
            line = json.dumps(payload, ensure_ascii=False)
            print(line)
            if log_path:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:
            payload = {
                "ts": datetime.now().isoformat(timespec='seconds'),
                "event": "run_gradient_entry_diag",
                "entered_gradient": True,
                "mean_fill_happened": None,
                "mean_fill_reason": "entry_diag_exception",
                "entry_diag_exception": str(e),
            }
            line = json.dumps(payload, ensure_ascii=False)
            print(line)
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass
        t_total_start = time.perf_counter() if self._profile else None
        t1_start = time.perf_counter() if self._profile else None
        t3_start = None
        t3_end = None
        t4_start = None
        t4_end = None
        # 1. First term
        if (
            self._term1_weight is None
            or self._term1_weight_cols != tuple(self.st_exp.columns)
            or self._term1_weight_W_HVG != self.W_HVG
        ):
            hvg = list(self.svg)
            weight = pd.DataFrame(
                np.ones((self.st_exp.shape[0], self.st_exp.shape[1])),
                columns=self.st_exp.columns,
                index=self.st_exp.index,
            )
            if hvg:
                weight[hvg] = self.W_HVG
            self._term1_weight = weight
            self._term1_weight_cols = tuple(self.st_exp.columns)
            self._term1_weight_W_HVG = self.W_HVG
        if self._term1_use_incremental or self._term1_block_size:
            st_cols = self.st_exp.columns
            exp_values = self.alter_sc_exp[st_cols].to_numpy(dtype=np.float32, copy=False)
            st_values = self.st_exp.to_numpy(dtype=np.float32, copy=False)
            n_spots = self.st_exp.shape[0]
            n_genes = self.st_exp.shape[1]
            spot_codes = self._term1_spot_codes
            valid = self._term1_valid_mask
            cell_n_spot = self._term1_cell_n_spot
            if spot_codes is None or valid is None or cell_n_spot is None or cell_n_spot.shape[0] != n_spots:
                spot_codes = pd.Categorical(self.sc_agg_meta['spot'], categories=self.st_exp.index, ordered=True).codes
                valid = spot_codes >= 0
                cell_n_spot = np.bincount(spot_codes[valid], minlength=n_spots).astype(np.int64, copy=False)
                cell_n_spot[cell_n_spot == 0] = 1
                self._term1_spot_codes = spot_codes
                self._term1_valid_mask = valid
                self._term1_cell_n_spot = cell_n_spot
                self._term1_sc_spot_sum = None
                self._term1_last_values = None

            if self._term1_use_incremental and self._term1_sc_spot_sum is not None and self._term1_last_values is not None:
                delta = exp_values - self._term1_last_values
                np.add.at(self._term1_sc_spot_sum, spot_codes[valid], delta[valid])
                self._term1_last_values = exp_values.copy()
            else:
                self._term1_sc_spot_sum = np.zeros((n_spots, n_genes), dtype=exp_values.dtype)
                np.add.at(self._term1_sc_spot_sum, spot_codes[valid], exp_values[valid])
                self._term1_last_values = exp_values.copy()

            if self._term1_st_col_idx is None or self._term1_st_col_idx.shape[0] != n_genes:
                self._term1_st_col_idx = self.alter_sc_exp.columns.get_indexer(st_cols)

            term1_idx = tuple(self.alter_sc_exp.index)
            term1_cols = tuple(self.alter_sc_exp.columns)
            if self._term1_df_cache is None or self._term1_df_index != term1_idx or self._term1_df_cols != term1_cols:
                self._term1_df_cache = pd.DataFrame(
                    np.zeros(self.alter_sc_exp.shape, dtype=np.float32),
                    index=self.alter_sc_exp.index,
                    columns=self.alter_sc_exp.columns,
                )
                self._term1_df_index = term1_idx
                self._term1_df_cols = term1_cols

            term1_vals_full = self._term1_df_cache.values
            # clear stale columns (e.g., genes not in st_exp)
            term1_vals_full[:] = 0
            st_col_idx = self._term1_st_col_idx
            weight_vals = self._term1_weight.to_numpy(dtype=np.float32, copy=False)
            block_size = self._term1_block_size or n_genes
            sum_sq = 0.0
            spot_codes_safe = spot_codes.copy()
            spot_codes_safe[~valid] = 0
            for start in range(0, n_genes, block_size):
                end = min(start + block_size, n_genes)
                div_block = self._term1_sc_spot_sum[:, start:end] / cell_n_spot[:, None]
                diff_block = div_block - st_values[:, start:end]
                # weighted loss to match gradient (d/dX of sum(weight * diff^2))
                sum_sq += float(np.sum(weight_vals[:, start:end] * diff_block * diff_block))
                term1_block = 2.0 * diff_block * weight_vals[:, start:end]
                term1_cells = term1_block[spot_codes_safe]
                if not valid.all():
                    term1_cells[~valid] = 0
                cols = st_col_idx[start:end]
                term1_vals_full[:, cols] = term1_cells
            self.loss1 = sum_sq / float(n_spots * n_genes)
            self.term1_df = self._term1_df_cache
        else:
            self.term1_df,self.loss1 = optimizers.cal_term1(
                self.alter_sc_exp,
                self.sc_agg_meta,
                self.st_exp,
                self.svg,
                self.W_HVG,
                weight=self._term1_weight,
            )
        t1_end = time.perf_counter() if self._profile else None
        # print('First-term calculation done!')

        # 2. Second term
        t2_start = time.perf_counter() if self._profile else None
        term2_vals, self.loss2 = optimizers.cal_term2_values(self.alter_sc_exp, self.sc_ref)
        term2_idx = tuple(self.alter_sc_exp.index)
        term2_cols = tuple(self.alter_sc_exp.columns)
        if self._term2_df_cache is None or self._term2_df_index != term2_idx or self._term2_df_cols != term2_cols:
            self._term2_df_cache = pd.DataFrame(term2_vals, index=self.alter_sc_exp.index, columns=self.alter_sc_exp.columns)
            self._term2_df_index = term2_idx
            self._term2_df_cols = term2_cols
        else:
            self._term2_df_cache.values[:] = term2_vals
        self.term2_df = self._term2_df_cache
        t2_end = time.perf_counter() if self._profile else None
        # print('Second-term calculation done!')

        # 3. Third term, closer cells have larger affinity
        if compute_term3:
            t3_start = time.perf_counter() if self._profile else None
            if not (self.st_tp == 'slide-seq' and hasattr(self, 'sc_knn')):
                # if slide-seq and already have found sc_knn
                # dont do it again
                # Ensure `sc_coord` exists (init_grad normally creates it). If missing, compute
                # affinity embedding on the current `alter_sc_exp` as a light-weight fallback.
                if not hasattr(self, 'sc_coord') or self.sc_coord is None:
                    try:
                        self.sc_coord, _, _, _, _ = self._aff_embedding_cached(self.alter_sc_exp)
                    except Exception:
                        # fallback to zeros if embedding fails
                        self.sc_coord = pd.DataFrame(np.zeros((self.alter_sc_exp.shape[0], 2), dtype=np.float32), index=self.sc_agg_meta.index, columns=['x','y'])
                sc_coord_arr = self.sc_coord.to_numpy() if isinstance(self.sc_coord, pd.DataFrame) else np.asarray(self.sc_coord)
                if self._sc_coord_cache is None or not np.array_equal(self._sc_coord_cache, sc_coord_arr):
                    # 3.2 get c' = N(c)
                    spot_labels = self.sc_agg_meta['spot'].to_numpy()
                    spot_codes_all, spot_levels = pd.factorize(spot_labels)
                    sc_coord_df = pd.DataFrame(sc_coord_arr, columns=['x', 'y'], index=self.sc_agg_meta.index)
                    sc_coord_df['spot'] = spot_labels
                    sc_centroid = sc_coord_df.groupby('spot', sort=False)[['x', 'y']].mean()
                    _, sc_centroid_cells = optimizers.sc_prep(sc_centroid, self.sc_agg_meta)
                    centroid_arr = sc_centroid_cells[['st_x', 'st_y']].to_numpy(dtype=np.float32, copy=False)

                    self._spot_codes_all_cache = spot_codes_all
                    self._spot_levels_cache = spot_levels
                    self._centroid_arr_cache = centroid_arr

                    self.sc_knn = optimizers.findCellKNN(
                        self.st_coord,
                        self.st_tp,
                        self.sc_agg_meta,
                        self.sc_coord,
                        self.K,
                        spot_codes_all=self._spot_codes_all_cache,
                        spot_levels=self._spot_levels_cache,
                        centroid_arr=self._centroid_arr_cache,
                    )
                    utils.check_empty_dict(self.sc_knn)
                    if self._use_sparse_term3:
                        self.sc_dist = optimizers.knn_distance_matrix_sparse(self.sc_coord, self.sc_knn, self.alter_sc_exp.index)
                        self._knn_ind_sparse_cache, self._knn_ncp_sparse_cache = optimizers.build_knn_indicator_sparse(
                            self.sc_knn, self.alter_sc_exp.index
                        )
                        row_idx, col_idx = self._knn_ind_sparse_cache.nonzero()
                        self._knn_edge_row_cache = row_idx
                        self._knn_edge_col_cache = col_idx
                        sc_dist_re = self.sc_dist.tocsr(copy=False)
                        cp_dist_sparse = sc_dist_re[row_idx, col_idx].A1 if sc_dist_re.nnz else np.array([], dtype=np.float32)
                        self._knn_cp_dist_adj_cache = optimizers._scale_minmax_array(cp_dist_sparse, 0, 100)
                        self._knn_n_cells_cache = sc_dist_re.shape[0]
                        self._knn_ind_cache = None
                        self._knn_ncp_cache = None
                    else:
                        self.sc_dist = optimizers.knn_distance_matrix_dense(self.sc_coord, self.sc_knn, self.alter_sc_exp.index)
                        self._knn_ind_cache, self._knn_ncp_cache = optimizers.build_knn_indicator_df(
                            self.sc_knn, self.alter_sc_exp.index
                        )
                        self._knn_ind_sparse_cache = None
                        self._knn_ncp_sparse_cache = None
                        self._knn_edge_row_cache = None
                        self._knn_edge_col_cache = None
                        self._knn_cp_dist_adj_cache = None
                        self._knn_n_cells_cache = None
                    self._sc_coord_cache = sc_coord_arr.copy()
                    self._sc_dist_cache = self.sc_dist
                    self._sc_knn_cache = self.sc_knn
                    self._knn_df_cache = None
                else:
                    self.sc_dist = self._sc_dist_cache
                    self.sc_knn = self._sc_knn_cache
            # 3.3 get the paring genes (g') of gene g for each cells
            if hasattr(self, '_use_fast_lr_agg') and self._use_fast_lr_agg:
                if self._lr_agg_idx_cache is None or self._lr_agg_idx_cols != tuple(self.alter_sc_exp.columns):
                    self._lr_agg_idx_cache = optimizers.prepare_lr_agg_indices(self.lr_df, self.alter_sc_exp.columns)
                    self._lr_agg_idx_cols = tuple(self.alter_sc_exp.columns)
                self.rl_agg = optimizers.generate_LR_agg_fast(self.alter_sc_exp, self._lr_agg_idx_cache)
            else:
                self.rl_agg = optimizers.generate_LR_agg(self.alter_sc_exp,self.lr_df)
            # 3.4 get the affinity
            if self._affinity_idx_cache is None or self._affinity_idx_cols != tuple(self.alter_sc_exp.columns):
                self._affinity_idx_cache = optimizers.prepare_affinity_indices(self.lr_df, self.alter_sc_exp.columns)
                self._affinity_idx_cols = tuple(self.alter_sc_exp.columns)
            data_values = self.alter_sc_exp.T.to_numpy(dtype=np.float32, copy=False)
            if self._use_sparse_term3 and self._knn_ind_sparse_cache is not None:
                row_idx = self._knn_edge_row_cache
                col_idx = self._knn_edge_col_cache
                self.aff = optimizers.calculate_affinity_knn_sparse(
                    data_values,
                    self._affinity_idx_cache,
                    row_idx,
                    col_idx,
                    len(self.sc_agg_meta.index),
                )
            else:
                self.aff = optimizers.calculate_affinity_mat_fast(data_values, self._affinity_idx_cache)
                np.fill_diagonal(self.aff,0)
                self.aff = pd.DataFrame(self.aff, index = self.sc_agg_meta.index, columns=self.sc_agg_meta.index)
            # 3.5 Calculate the derivative
            if self._use_sparse_term3:
                self.term3_df, self.loss3 = optimizers.cal_term3_sparse(
                    self.alter_sc_exp,
                    self.sc_knn,
                    self.aff,
                    self.sc_dist,
                    self.rl_agg,
                    ind=self._knn_ind_sparse_cache,
                    n_cp=self._knn_ncp_sparse_cache,
                    row_idx=self._knn_edge_row_cache,
                    col_idx=self._knn_edge_col_cache,
                    cp_dist_adj=self._knn_cp_dist_adj_cache,
                    n_cells=self._knn_n_cells_cache,
                )
            else:
                self.term3_df, self.loss3 = optimizers.cal_term3(
                    self.alter_sc_exp,
                    self.sc_knn,
                    self.aff,
                    self.sc_dist,
                    self.rl_agg,
                    ind=self._knn_ind_cache,
                    n_cp=self._knn_ncp_cache,
                )
            self._term3_cache = self.term3_df
            self._loss3_cache = self.loss3
            t3_end = time.perf_counter() if self._profile else None
        else:
            self.term3_df = self._term3_cache
            self.loss3 = self._loss3_cache
        # print('Third term calculation done!')

        # 4. Fourth term, towards spot-spot affinity profile
        if compute_term4:
            t4_start = time.perf_counter() if self._profile else None
            if self._lr_idx_cache is None or self._lr_idx_cols != tuple(self.alter_sc_exp.columns):
                self._lr_idx_cache = optimizers.prepare_lr_indices(self.lr_df_align, self.alter_sc_exp.columns)
                self._lr_idx_cols = tuple(self.alter_sc_exp.columns)
            if self._cell_index_cache is None or self._cell_index_idx != tuple(self.alter_sc_exp.index):
                self._cell_index_cache = {k: i for i, k in enumerate(self.alter_sc_exp.index)}
                self._cell_index_idx = tuple(self.alter_sc_exp.index)
                self._knn_arrays_cache = None
            if (
                self._spot_to_idx_cache is None
                or self._spot_to_idx_keys != tuple(self.st_exp.index)
                or self._spot_neighbors_cache is None
                or self._spot_neighbors_max_cache != self._spot_neighbors_max
            ):
                (
                    _,
                    self._spot_to_idx_cache,
                    self._pair_to_row_cache,
                    self._st_aff_values_cache,
                    self._spot_neighbors_cache,
                ) = optimizers.prepare_spot_pair_lookup(
                    self.st_exp, self.st_aff_profile_df, self._spot_neighbors_max
                )
                self._spot_neighbors_max_cache = self._spot_neighbors_max
                self._spot_to_idx_keys = tuple(self.st_exp.index)
            if self._knn_df_cache is None:
                knn_df = pd.DataFrame(self.sc_knn.items(), columns=['cell_idx', 'nn_cell_idx']).explode('nn_cell_idx')
                nn_cell_idx = knn_df['nn_cell_idx'].tolist()
                knn_df['spot'] = self.sc_agg_meta.loc[nn_cell_idx, 'spot'].values
                self._knn_df_cache = knn_df
                cell_idx_to_rows = knn_df.groupby('cell_idx').indices
                spot_knn_idx = {}
                for spot, cells in self.spot_cell_dict.items():
                    rows = []
                    for cell in cells:
                        idxs = cell_idx_to_rows.get(cell)
                        if idxs is not None:
                            rows.append(idxs)
                    if rows:
                        spot_knn_idx[spot] = np.concatenate(rows)
                    else:
                        spot_knn_idx[spot] = np.array([], dtype=np.int64)
                self._spot_knn_idx_cache = spot_knn_idx
                self._knn_arrays_cache = None

            if self._knn_arrays_cache is None:
                self._knn_arrays_cache = optimizers.prepare_knn_arrays(
                    self._knn_df_cache, self._cell_index_cache, self._spot_to_idx_cache
                )
            
            exp_values = self.alter_sc_exp.to_numpy(dtype=np.float32, copy=False)
            # Choose between vectorized and original term4
            if hasattr(self, '_use_sparse_term4') and self._use_sparse_term4:
                self.term4_df, self.loss4 = optimizers.cal_term4_sparse(
                    self.st_exp, self.sc_knn, self.st_aff_profile_df, self.alter_sc_exp,
                    self.sc_agg_meta, self.spot_cell_dict, self.lr_df_align,
                    lr_indices=self._lr_idx_cache,
                    knn_df=self._knn_df_cache,
                    spot_knn_idx=self._spot_knn_idx_cache,
                    spot_filter=self._active_spots,
                    cell_index=self._cell_index_cache,
                    spot_to_idx=self._spot_to_idx_cache,
                    pair_to_row=self._pair_to_row_cache,
                    st_aff_values=self._st_aff_values_cache,
                    knn_arrays=self._knn_arrays_cache,
                    spot_neighbors=self._spot_neighbors_cache,
                    exp_values=exp_values,
                )
            elif hasattr(self, '_use_vectorized_term4') and self._use_vectorized_term4:
                self.term4_df, self.loss4 = optimizers.cal_term4_vectorized(
                    self.st_exp, self.sc_knn, self.st_aff_profile_df, self.alter_sc_exp,
                    self.sc_agg_meta, self.spot_cell_dict, self.lr_df_align,
                    lr_indices=self._lr_idx_cache,
                    knn_df=self._knn_df_cache,
                    spot_knn_idx=self._spot_knn_idx_cache,
                    spot_filter=self._active_spots,
                    cell_index=self._cell_index_cache,
                    spot_to_idx=self._spot_to_idx_cache,
                    pair_to_row=self._pair_to_row_cache,
                    st_aff_values=self._st_aff_values_cache,
                    knn_arrays=self._knn_arrays_cache,
                    spot_neighbors=self._spot_neighbors_cache,
                    exp_values=exp_values,
                )
            elif self.term4_n_cores and int(self.term4_n_cores) > 1:
                self.term4_df, self.loss4 = optimizers.cal_term4_parallel(
                    self.st_exp, self.sc_knn, self.st_aff_profile_df, self.alter_sc_exp,
                    self.sc_agg_meta, self.spot_cell_dict, self.lr_df_align,
                    lr_indices=self._lr_idx_cache,
                    knn_df=self._knn_df_cache,
                    spot_filter=self._active_spots,
                    cell_index=self._cell_index_cache,
                    n_cores=int(self.term4_n_cores),
                    spot_knn_idx=self._spot_knn_idx_cache,
                )
            else:
                self.term4_df, self.loss4 = optimizers.cal_term4(
                    self.st_exp, self.sc_knn, self.st_aff_profile_df, self.alter_sc_exp,
                    self.sc_agg_meta, self.spot_cell_dict, self.lr_df_align,
                    lr_indices=self._lr_idx_cache,
                    knn_df=self._knn_df_cache,
                    spot_filter=self._active_spots,
                    spot_knn_idx=self._spot_knn_idx_cache,
                    cell_index=self._cell_index_cache,
                )
            self._term4_cache = self.term4_df
            self._loss4_cache = self.loss4
            t4_end = time.perf_counter() if self._profile else None
        else:
            self.term4_df = self._term4_cache
            self.loss4 = self._loss4_cache
        # print('Fourth term calculation done!')
        
        # 5. Fifth term, norm2 regulization
        t5_start = time.perf_counter() if self._profile else None
        term5_vals, self.loss5 = optimizers.cal_term5_values(self.alter_sc_exp)
        term5_idx = tuple(self.alter_sc_exp.index)
        term5_cols = tuple(self.alter_sc_exp.columns)
        if self._term5_df_cache is None or self._term5_df_index != term5_idx or self._term5_df_cols != term5_cols:
            self._term5_df_cache = pd.DataFrame(term5_vals, index=self.alter_sc_exp.index, columns=self.alter_sc_exp.columns)
            self._term5_df_index = term5_idx
            self._term5_df_cols = term5_cols
        else:
            self._term5_df_cache.values[:] = term5_vals
        self.term5_df = self._term5_df_cache
        t5_end = time.perf_counter() if self._profile else None

        if self._profile and self._profile_iter >= 0:
            if self._profile_every > 0 and (self._profile_iter % self._profile_every == 0):
                total = (time.perf_counter() - t_total_start) if t_total_start is not None else 0.0
                t1 = (t1_end - t1_start) if t1_end and t1_start else 0.0
                t2 = (t2_end - t2_start) if t2_end and t2_start else 0.0
                t3 = (t3_end - t3_start) if t3_end and t3_start else 0.0
                t4 = (t4_end - t4_start) if t4_end and t4_start else 0.0
                t5 = (t5_end - t5_start) if t5_end and t5_start else 0.0
                print(
                    f"[profile] iter {self._profile_iter} term1 {t1:.3f}s term2 {t2:.3f}s "
                    f"term3 {t3:.3f}s term4 {t4:.3f}s term5 {t5:.3f}s total {total:.3f}s"
                )
        

    @timeit
    def init_grad(self):
        if isinstance(self.init_sc_embed, pd.DataFrame):
            self.sc_coord = utils.check_sc_coord(self.init_sc_embed)
            print('Using user provided init sc_coord.')
        else:
            print('Initialize cell coordinates by affinity embedding...')
            self.sc_coord, max_shape, _, _, _ = self._aff_embedding_cached(self.alter_sc_exp)
            print(f"Initial shape correlation: {max_shape:.2f}")
            # print(f"{'='*50}\n")
        self.run_gradient()
        # v5 calculte the initial loss of each term to balance their force.
        if not getattr(self, "_skip_loss_adj", False):
            adj2,adj3,adj4,adj5 = optimizers.loss_adj(self.loss1,self.loss2,self.loss3,self.loss4,self.loss5)
            self.ALPHA,self.BETA,self.GAMMA,self.DELTA = self.ALPHA*adj2,self.BETA*adj3,self.GAMMA*adj4,self.DELTA*adj5
        self.sc_agg_meta[['UMAP1','UMAP2']] = self.sc_coord
        # print('Hyperparameters adjusted.')


    @timeit
    def gradient_descent(self, p1 = 0.05, p2 = 0.65, p3 = 0.2, p4 = 0.1, 
                         delta = 0.1, eta = 0.0005, 
                        init_sc_embed = False,
                        iteration = 20, k = 2, W_HVG = 2,
                        left_range = 1, right_range = 2, steps = 1, dim = 2,
                        embed_every = 2,
                        embed_schedule = 'every', embed_last_n = 3,
                        final_refine = False, refine_epochs = 30, refine_runs = 3, refine_jitter = 1e-3,
                        term1_use_incremental = True, term1_block_size = 512,
                        normalize_terms = True, term_norm = 'rms', norm_eps = 1e-8,
                        embed_tol = 1e-3, embed_patience = 3,
                        term3_every = 1, term4_every = 1, lr_block_every = 2,
                        convergence_patience = 3, convergence_tol = 1e-4,
                        use_sparse_term3 = True, term4_n_cores = 1,
                        use_float32 = True,
                        optimizer = 'als_each_sgd', adam_beta1 = 0.9, adam_beta2 = 0.999, adam_eps = 1e-8,
                        als_blend = 0.1, als_iters = 5, allow_hard_spot_overwrite = False,
                        grad_max = None,
                        spot_filter = 'none', spot_top_frac = 0.3, spot_top_k = None,
                        max_spot_neighbors = 'auto',
                        use_vectorized_term4 = True, use_sparse_term4 = True, use_fast_lr_agg = True,
                        profile = False, profile_every = 1,
                        fast_embedding = True, embedding_compute_shape = True,
                        freeze_embedding = False,
                        debug_gradients = False, debug_every = 1,
                        skip_loss_adj = False,
                        ce_kl_iters = 0, ce_kl_eta = None, ce_kl_eps = 1e-8,
                        objective_consistent = True, recompute_loss_after_update = None,
                        ce_kl_term1 = True, ce_kl_term2 = True,
                        spot_sum_proj = 'none', spot_sum_proj_blend = 0.0, spot_sum_proj_eps = 1e-8,
                        spot_sum_term_weight = 0.0):
        def _emit_grad_diag(event, **kwargs):
            payload = {"ts": datetime.now().isoformat(timespec='seconds'), "event": event}
            payload.update(kwargs)
            line = json.dumps(payload, ensure_ascii=False)
            print(line)
            log_path = getattr(self, "debug_log_path", None)
            if log_path:
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(line + "\n")
                except Exception:
                    pass

        objective_consistent = bool(objective_consistent)
        if recompute_loss_after_update is None:
            recompute_loss_after_update = objective_consistent
        else:
            recompute_loss_after_update = bool(recompute_loss_after_update)
        self.GAMMA = p1 # interface
        self.FIRST = p2 # st_exp
        self.ALPHA = p3 # sc_ref
        self.BETA = p4 # affinity embedding
        
        self.DELTA = delta # L2
        self.ETA = eta

        self.init_sc_embed = init_sc_embed # initial cell coordinates
        self.iteration = iteration
        self.K = k
        self.W_HVG = W_HVG
        
        # embedding
        self.left_range = left_range
        self.right_range = right_range
        self.steps = steps
        self.dim = dim
        self._use_sparse_term3 = bool(use_sparse_term3)
        self.term4_n_cores = max(1, int(term4_n_cores))
        self._use_vectorized_term4 = bool(use_vectorized_term4)
        self._use_sparse_term4 = bool(use_sparse_term4)
        self._use_fast_lr_agg = bool(use_fast_lr_agg)
        if max_spot_neighbors is None:
            self._spot_neighbors_max = None
        elif isinstance(max_spot_neighbors, str) and max_spot_neighbors.lower() == 'auto':
            self._spot_neighbors_max = 'auto'
        else:
            self._spot_neighbors_max = int(max_spot_neighbors)
        self._profile = bool(profile)
        self._profile_every = max(1, int(profile_every)) if profile_every is not None else 1
        self._fast_embedding = bool(fast_embedding)
        self._embedding_compute_shape = bool(embedding_compute_shape)
        self._freeze_embedding = bool(freeze_embedding)
        self._skip_loss_adj = bool(skip_loss_adj)
        debug_gradients = bool(debug_gradients)
        debug_every = max(1, int(debug_every)) if debug_every is not None else 1
        self._term1_use_incremental = bool(term1_use_incremental)
        ce_kl_iters = 0 if ce_kl_iters is None else int(ce_kl_iters)
        if ce_kl_iters < 0:
            ce_kl_iters = 0
        ce_kl_eta = self.ETA if ce_kl_eta is None else float(ce_kl_eta)
        ce_kl_eps = float(ce_kl_eps) if ce_kl_eps is not None else 1e-8
        ce_kl_term1 = bool(ce_kl_term1)
        ce_kl_term2 = bool(ce_kl_term2)
        spot_sum_proj = 'none' if spot_sum_proj is None else str(spot_sum_proj).lower()
        spot_sum_proj_blend = 0.0 if spot_sum_proj_blend is None else float(spot_sum_proj_blend)
        spot_sum_proj_eps = float(spot_sum_proj_eps) if spot_sum_proj_eps is not None else 1e-8
        spot_sum_term_weight = 0.0 if spot_sum_term_weight is None else float(spot_sum_term_weight)
        if spot_sum_term_weight < 0:
            spot_sum_term_weight = 0.0
        if spot_sum_proj not in ('none', 'sc_ref', 'smurf'):
            warnings.warn(f"Unknown spot_sum_proj='{spot_sum_proj}', disabling projection.")
            spot_sum_proj = 'none'
        self.SPOT_SUM = spot_sum_term_weight
        if term1_block_size is None:
            self._term1_block_size = None
        else:
            self._term1_block_size = max(1, int(term1_block_size))
        # Robust alignment of `sc_ref` to the selected single-cell IDs.
        # Accept pandas DataFrame or numpy array. If rows are missing, try sensible fallbacks
        # (use available rows, map by celltype if possible, or fill missing rows with mean profile).
        align_diag = {
            "mode": "unknown",
            "missing_sc_id_count": 0,
            "missing_sc_id_ratio": 0.0,
            "used_fallback": "none",
            "used_mean_broadcast": False,
            "nan_rows_filled": 0,
            "exception": "",
        }
        if not isinstance(self.sc_ref, np.ndarray):
            align_diag["mode"] = "dataframe_input"
            sc_ids = self.sc_agg_meta['sc_id']
            try:
                # direct 1:1 alignment when possible
                common = sc_ids[sc_ids.isin(self.sc_ref.index)]
                align_diag["missing_sc_id_count"] = int(len(sc_ids) - len(common))
                align_diag["missing_sc_id_ratio"] = float((len(sc_ids) - len(common)) / max(len(sc_ids), 1))
                if len(common) == len(sc_ids):
                    aligned = self.sc_ref.loc[sc_ids]
                    align_diag["used_fallback"] = "sc_id_exact"
                elif len(common) > 0:
                    # partially overlapping: create DataFrame with missing rows filled by mean
                    aligned = pd.DataFrame(index=sc_ids, columns=self.sc_ref.columns, dtype=float)
                    aligned.loc[common] = self.sc_ref.loc[common]
                    mean_row = self.sc_ref.mean(axis=0)
                    missing = sc_ids[~sc_ids.isin(self.sc_ref.index)]
                    aligned.loc[missing] = mean_row.values
                    align_diag["used_fallback"] = "partial_missing_fill_with_mean"
                    align_diag["used_mean_broadcast"] = True
                else:
                    # try mapping by celltype if available
                    if 'celltype' in self.sc_agg_meta.columns and self.sc_ref.index.isin(self.sc_agg_meta['celltype']).any():
                        aligned = pd.DataFrame(index=sc_ids, columns=self.sc_ref.columns, dtype=float)
                        for idx in sc_ids:
                            ct = self.sc_agg_meta.loc[idx, 'celltype'] if 'celltype' in self.sc_agg_meta.columns else None
                            if ct is not None and ct in self.sc_ref.index:
                                aligned.loc[idx] = self.sc_ref.loc[ct]
                            else:
                                aligned.loc[idx] = self.sc_ref.mean(axis=0).values
                        align_diag["used_fallback"] = "celltype_or_mean_per_row"
                        align_diag["used_mean_broadcast"] = True
                    else:
                        # fallback: broadcast mean profile for all selected cells
                        mean_row = self.sc_ref.mean(axis=0)
                        aligned = pd.DataFrame(np.repeat(mean_row.values[None, :], len(sc_ids), axis=0), index=sc_ids, columns=self.sc_ref.columns)
                        align_diag["used_fallback"] = "broadcast_mean_all_rows"
                        align_diag["used_mean_broadcast"] = True
                # convert to numpy float32
                self.sc_ref = np.asarray(aligned, dtype=np.float32)
            except Exception as e:
                mean_row = np.asarray(self.sc_ref.mean(axis=0))
                self.sc_ref = np.repeat(mean_row[None, :], len(sc_ids), axis=0).astype(np.float32)
                align_diag["used_fallback"] = "exception_broadcast_mean_all_rows"
                align_diag["used_mean_broadcast"] = True
                align_diag["exception"] = str(e)
                warnings.warn(f"sc_ref alignment failed; broadcasting mean profile to match selected cells. Error: {e}")
        else:
            align_diag["mode"] = "ndarray_input"
            # sc_ref is already ndarray. If row-count doesn't match selected cells, broadcast mean.
            try:
                if self.sc_ref.shape[0] != self.alter_sc_exp.shape[0]:
                    mean_row = np.nanmean(self.sc_ref, axis=0)
                    self.sc_ref = np.repeat(mean_row[None, :], self.alter_sc_exp.shape[0], axis=0).astype(np.float32)
                    align_diag["used_fallback"] = "ndarray_row_mismatch_broadcast_mean"
                    align_diag["used_mean_broadcast"] = True
                    warnings.warn("sc_ref ndarray row count did not match selected cells; broadcasting mean row to match.")
            except Exception:
                # as last resort, ensure shape matches by broadcasting zeros
                mean_row = np.zeros((self.alter_sc_exp.shape[1],), dtype=np.float32)
                self.sc_ref = np.repeat(mean_row[None, :], self.alter_sc_exp.shape[0], axis=0)
                align_diag["used_fallback"] = "ndarray_exception_broadcast_zeros"
                align_diag["used_mean_broadcast"] = True
                align_diag["exception"] = "ndarray_shape_check_exception"
        if np.isnan(self.sc_ref).any():
            mask = np.isnan(self.sc_ref).any(axis=1)
            align_diag["nan_rows_filled"] = int(mask.sum())
            mean_row = np.nanmean(self.sc_ref, axis=0)
            self.sc_ref[mask, :] = mean_row
            self.sc_ref = np.nan_to_num(self.sc_ref, nan=0.0)
            if align_diag["used_fallback"] == "none":
                align_diag["used_fallback"] = "nan_row_fill_with_sc_ref_mean"
            align_diag["used_mean_broadcast"] = True
        self._last_sc_ref_align_diag = dict(align_diag)
        _emit_grad_diag("run_gradient_sc_ref_alignment", **align_diag, sc_ref_shape=list(self.sc_ref.shape))
        if use_float32:
            if self.alter_sc_exp.values.dtype != np.float32:
                self.alter_sc_exp = self.alter_sc_exp.astype(np.float32)
            if self.st_exp.values.dtype != np.float32:
                self.st_exp = self.st_exp.astype(np.float32)
            if self.sc_ref.dtype != np.float32:
                self.sc_ref = self.sc_ref.astype(np.float32)
        use_spot_sum_term = spot_sum_term_weight > 0.0
        if use_spot_sum_term:
            res_col = [
                'loss1','loss2','loss3','loss4','loss5','loss6','total',
                'loss1_raw','loss2_raw','loss3_raw','loss4_raw','loss5_raw','loss6_raw','total_raw'
            ]
        else:
            res_col = [
                'loss1','loss2','loss3','loss4','loss5','total',
                'loss1_raw','loss2_raw','loss3_raw','loss4_raw','loss5_raw','total_raw'
            ]
        result_rows = []
        result_idx = []
        def _build_loss_row(loss1_eff, loss2_eff, loss3_eff, loss4_eff, loss5_eff, total_eff,
                            loss1_raw, loss2_raw, loss3_raw, loss4_raw, loss5_raw, total_raw,
                            loss6_eff_val = 0.0, loss6_raw_val = 0.0, include_spot_sum = False):
            if use_spot_sum_term:
                return [
                    self.FIRST * loss1_eff,
                    self.ALPHA * loss2_eff,
                    self.BETA * loss3_eff,
                    self.GAMMA * loss4_eff,
                    self.DELTA * loss5_eff,
                    self.SPOT_SUM * loss6_eff_val if include_spot_sum else 0.0,
                    total_eff,
                    self.FIRST * loss1_raw,
                    self.ALPHA * loss2_raw,
                    self.BETA * loss3_raw,
                    self.GAMMA * loss4_raw,
                    self.DELTA * loss5_raw,
                    self.SPOT_SUM * loss6_raw_val if include_spot_sum else 0.0,
                    total_raw,
                ]
            return [
                self.FIRST * loss1_eff,
                self.ALPHA * loss2_eff,
                self.BETA * loss3_eff,
                self.GAMMA * loss4_eff,
                self.DELTA * loss5_eff,
                total_eff,
                self.FIRST * loss1_raw,
                self.ALPHA * loss2_raw,
                self.BETA * loss3_raw,
                self.GAMMA * loss4_raw,
                self.DELTA * loss5_raw,
                total_raw,
            ]
        if self.st_tp == 'slide-seq':
            # cell coord as spot coord
            self.init_sc_embed = self.st_coord.loc[self.sc_agg_meta['spot']]
            self.init_sc_embed.index = self.sc_agg_meta.index
        self.init_grad()
        if normalize_terms:
            if term_norm == 'l2':
                t1 = self.term1_df.values
                t2 = self.term2_df.values
                t3 = self.term3_df.values
                t4 = self.term4_df.values
                t5 = self.term5_df.values
                norm1 = float(np.sqrt((t1 ** 2).sum()))
                norm2 = float(np.sqrt((t2 ** 2).sum()))
                norm3 = float(np.sqrt((t3 ** 2).sum()))
                norm4 = float(np.sqrt((t4 ** 2).sum()))
                norm5 = float(np.sqrt((t5 ** 2).sum()))
            elif term_norm in ('rms', 'l2_mean'):
                t1 = self.term1_df.values
                t2 = self.term2_df.values
                t3 = self.term3_df.values
                t4 = self.term4_df.values
                t5 = self.term5_df.values
                norm1 = float(np.sqrt((t1 ** 2).mean()))
                norm2 = float(np.sqrt((t2 ** 2).mean()))
                norm3 = float(np.sqrt((t3 ** 2).mean()))
                norm4 = float(np.sqrt((t4 ** 2).mean()))
                norm5 = float(np.sqrt((t5 ** 2).mean()))
            else:
                t1 = self.term1_df.values
                t2 = self.term2_df.values
                t3 = self.term3_df.values
                t4 = self.term4_df.values
                t5 = self.term5_df.values
                norm1 = float(np.max(np.abs(t1)))
                norm2 = float(np.max(np.abs(t2)))
                norm3 = float(np.max(np.abs(t3)))
                norm4 = float(np.max(np.abs(t4)))
                norm5 = float(np.max(np.abs(t5)))
            loss1_eff = self.loss1 / max(norm1, norm_eps)
            loss2_eff = self.loss2 / max(norm2, norm_eps)
            loss3_eff = self.loss3 / max(norm3, norm_eps)
            loss4_eff = self.loss4 / max(norm4, norm_eps)
            loss5_eff = self.loss5 / max(norm5, norm_eps)
        else:
            loss1_eff = self.loss1
            loss2_eff = self.loss2
            loss3_eff = self.loss3
            loss4_eff = self.loss4
            loss5_eff = self.loss5
        best_loss = (
            self.FIRST * loss1_eff + self.ALPHA * loss2_eff + self.BETA * loss3_eff
            + self.GAMMA * loss4_eff + self.DELTA * loss5_eff
        )
        print(f"Pre-update loss: {best_loss:.5f}")
        best_state = self.alter_sc_exp.copy()
        ######### init done ############
        embed_every = 1 if embed_every is None or embed_every < 1 else int(embed_every)
        embed_schedule = 'every' if embed_schedule is None else str(embed_schedule).lower()
        embed_last_n = 1 if embed_last_n is None else int(embed_last_n)
        if embed_last_n < 1:
            embed_last_n = 1
        final_refine = bool(final_refine)
        refine_epochs = 30 if refine_epochs is None else int(refine_epochs)
        refine_runs = 3 if refine_runs is None else int(refine_runs)
        refine_jitter = 1e-3 if refine_jitter is None else float(refine_jitter)
        embed_patience = 1 if embed_patience is None or embed_patience < 1 else int(embed_patience)
        embed_tol = 0.0 if embed_tol is None else float(embed_tol)
        term3_every = 1 if term3_every is None or term3_every < 1 else int(term3_every)
        term4_every = 1 if term4_every is None or term4_every < 1 else int(term4_every)
        lr_block_every = 1 if lr_block_every is None or lr_block_every < 1 else int(lr_block_every)
        convergence_patience = 0 if convergence_patience is None else max(0, int(convergence_patience))
        convergence_tol = 0.0 if convergence_tol is None else float(convergence_tol)
        patience_counter = 0
        best_loss_for_stop = best_loss
        embed_no_improve = 0
        last_shape = None
        max_shape = None
        m = None
        v = None
        als_blend = 0.1 if als_blend is None else float(als_blend)
        if als_blend < 0:
            als_blend = 0.0
        if als_blend > 1:
            als_blend = 1.0
        if (not bool(allow_hard_spot_overwrite)) and als_blend >= 1.0:
            warnings.warn(
                "als_blend>=1.0 causes hard per-cell overwrite to spot ST profile; clamping to 0.1. "
                "Set allow_hard_spot_overwrite=True to keep legacy behavior."
            )
            als_blend = 0.1
        als_iters = 0 if als_iters is None else int(als_iters)
        if als_iters < 0:
            als_iters = 0
        if self._spot_codes_cache is None or self._spot_codes_spots is None:
            spot_codes, spot_levels = pd.factorize(self.sc_agg_meta['spot'])
            self._spot_codes_cache = spot_codes
            self._spot_codes_spots = list(spot_levels)
        if self._lr_gene_mask is None or self._lr_gene_mask_cols != tuple(self.alter_sc_exp.columns):
            if hasattr(self, 'lr_df_align') and self.lr_df_align is not None:
                lr_genes = set(self.lr_df_align[0]).union(set(self.lr_df_align[1]))
            else:
                lr_genes = set()
            cols = list(self.alter_sc_exp.columns)
            lr_mask = np.array([c in lr_genes for c in cols], dtype=bool)
            self._lr_gene_mask = lr_mask
            self._non_lr_gene_mask = ~lr_mask
            self._lr_gene_mask_cols = tuple(cols)
        if use_spot_sum_term:
            st_col_idx = self.alter_sc_exp.columns.get_indexer(self.st_exp.columns)
            st_col_idx = st_col_idx[st_col_idx >= 0]
            self._spot_sum_proj_col_idx = st_col_idx

        for ite in range(self.iteration):
            # print(f'-----Start iteration {ite} -----')
            if np.isnan(self.alter_sc_exp.values).any() or np.isinf(self.alter_sc_exp.values).any():
                _emit_grad_diag(
                    "run_gradient_numeric_anomaly",
                    iter=int(ite),
                    where="pre_term_gradient",
                    has_nan=bool(np.isnan(self.alter_sc_exp.values).any()),
                    has_inf=bool(np.isinf(self.alter_sc_exp.values).any()),
                )
            update_lr = (lr_block_every <= 1) or (ite % lr_block_every == 0)
            if (not self._freeze_embedding) and self.st_tp != 'slide-seq' and (embed_no_improve < embed_patience):
                if embed_schedule == 'first_only':
                    do_embed = (ite == 0)
                elif embed_schedule == 'first_last':
                    do_embed = (ite == 0) or (ite >= self.iteration - embed_last_n)
                else:
                    do_embed = (ite % embed_every == 0)
            else:
                do_embed = False
            if do_embed:
                if self._profile:
                    t_embed_start = time.perf_counter()
                self.sc_coord, max_shape, _, _, _ = self._aff_embedding_cached(self.alter_sc_exp)
                if self._profile:
                    t_embed = time.perf_counter() - t_embed_start
                    if self._profile_every > 0 and (ite % self._profile_every == 0):
                        print(f"[profile] iter {ite} embedding {t_embed:.3f}s")
                if last_shape is not None and (max_shape - last_shape) < embed_tol:
                    embed_no_improve += 1
                else:
                    embed_no_improve = 0
                last_shape = max_shape
                if use_spot_sum_term and spot_sum_proj != 'none' and spot_sum_proj_blend > 0:
                    proj_vals = optimizers.spot_sum_projection_vals(
                        self.alter_sc_exp,
                        self.sc_agg_meta,
                        self.st_exp,
                        self.sc_ref,
                        eps=spot_sum_proj_eps,
                    )
                    if spot_sum_proj_blend >= 1.0:
                        target_vals = proj_vals
                    else:
                        curr_vals = self.alter_sc_exp.values[:, self._spot_sum_proj_col_idx]
                        target_vals = curr_vals * (1.0 - spot_sum_proj_blend) + proj_vals * spot_sum_proj_blend
                    self._spot_sum_proj_target = target_vals
            compute_term3 = (ite % term3_every == 0) or (self._term3_cache is None)
            compute_term4 = (ite % term4_every == 0) or (self._term4_cache is None)
            if not update_lr:
                if self._term3_cache is not None:
                    compute_term3 = False
                if self._term4_cache is not None:
                    compute_term4 = False
            elif objective_consistent:
                # LR genes are updated -> term3/4 must be current for a consistent objective
                compute_term3 = True
                compute_term4 = True
            if spot_filter != 'none' and ite > 0:
                scores = optimizers.spot_mean_abs(self.term1_df.values, self._spot_codes_cache, len(self._spot_codes_spots))
                spot_scores = pd.Series(scores, index=self._spot_codes_spots)
                if spot_top_k is not None:
                    active = spot_scores.sort_values(ascending=False).head(int(spot_top_k)).index.tolist()
                else:
                    take_n = max(1, int(len(spot_scores) * float(spot_top_frac)))
                    active = spot_scores.sort_values(ascending=False).head(take_n).index.tolist()
                self._active_spots = active
            else:
                self._active_spots = None
            self._profile_iter = ite
            self.run_gradient(compute_term3 = compute_term3, compute_term4 = compute_term4)
            term6_vals = None
            loss6 = 0.0
            if use_spot_sum_term and getattr(self, '_spot_sum_proj_target', None) is not None:
                target_vals = self._spot_sum_proj_target
                vals = self.alter_sc_exp.values
                term6_vals = np.zeros_like(vals)
                term6_vals[:, self._spot_sum_proj_col_idx] = 2.0 * (vals[:, self._spot_sum_proj_col_idx] - target_vals)
                diff = vals[:, self._spot_sum_proj_col_idx] - target_vals
                loss6 = float(np.mean(diff * diff))
                self.loss6 = loss6
                self.term6_df = pd.DataFrame(term6_vals, index=self.alter_sc_exp.index, columns=self.alter_sc_exp.columns)
            else:
                self.loss6 = 0.0
                self.term6_df = None
            if normalize_terms:
                if term_norm == 'l2':
                    term1_vals = self.term1_df.values
                    term2_vals = self.term2_df.values
                    term3_vals = self.term3_df.values
                    term4_vals = self.term4_df.values
                    term5_vals = self.term5_df.values
                    term6_vals = term6_vals if term6_vals is not None else None
                    norm1 = float(np.sqrt((term1_vals ** 2).sum()))
                    norm2 = float(np.sqrt((term2_vals ** 2).sum()))
                    norm3 = float(np.sqrt((term3_vals ** 2).sum()))
                    norm4 = float(np.sqrt((term4_vals ** 2).sum()))
                    norm5 = float(np.sqrt((term5_vals ** 2).sum()))
                    norm6 = float(np.sqrt((term6_vals ** 2).sum())) if term6_vals is not None else 1.0
                elif term_norm in ('rms', 'l2_mean'):
                    term1_vals = self.term1_df.values
                    term2_vals = self.term2_df.values
                    term3_vals = self.term3_df.values
                    term4_vals = self.term4_df.values
                    term5_vals = self.term5_df.values
                    term6_vals = term6_vals if term6_vals is not None else None
                    norm1 = float(np.sqrt((term1_vals ** 2).mean()))
                    norm2 = float(np.sqrt((term2_vals ** 2).mean()))
                    norm3 = float(np.sqrt((term3_vals ** 2).mean()))
                    norm4 = float(np.sqrt((term4_vals ** 2).mean()))
                    norm5 = float(np.sqrt((term5_vals ** 2).mean()))
                    norm6 = float(np.sqrt((term6_vals ** 2).mean())) if term6_vals is not None else 1.0
                else:
                    term1_vals = self.term1_df.values
                    term2_vals = self.term2_df.values
                    term3_vals = self.term3_df.values
                    term4_vals = self.term4_df.values
                    term5_vals = self.term5_df.values
                    term6_vals = term6_vals if term6_vals is not None else None
                    norm1 = float(np.max(np.abs(term1_vals)))
                    norm2 = float(np.max(np.abs(term2_vals)))
                    norm3 = float(np.max(np.abs(term3_vals)))
                    norm4 = float(np.max(np.abs(term4_vals)))
                    norm5 = float(np.max(np.abs(term5_vals)))
                    norm6 = float(np.max(np.abs(term6_vals))) if term6_vals is not None else 1.0
                term1_vals = term1_vals / max(norm1, norm_eps)
                term2_vals = term2_vals / max(norm2, norm_eps)
                term3_vals = term3_vals / max(norm3, norm_eps)
                term4_vals = term4_vals / max(norm4, norm_eps)
                term5_vals = term5_vals / max(norm5, norm_eps)
                term6_vals = term6_vals / max(norm6, norm_eps) if term6_vals is not None else None
            else:
                term1_vals = self.term1_df.values
                term2_vals = self.term2_df.values
                term3_vals = self.term3_df.values
                term4_vals = self.term4_df.values
                term5_vals = self.term5_df.values
                term6_vals = term6_vals if term6_vals is not None else None
            if debug_gradients and (ite % debug_every == 0):
                n1 = float(np.sqrt((term1_vals ** 2).sum()))
                n2 = float(np.sqrt((term2_vals ** 2).sum()))
                n3 = float(np.sqrt((term3_vals ** 2).sum()))
                n4 = float(np.sqrt((term4_vals ** 2).sum()))
                n5 = float(np.sqrt((term5_vals ** 2).sum()))
                n6 = float(np.sqrt((term6_vals ** 2).sum())) if term6_vals is not None else 0.0
                print(
                    f"[debug] iter {ite} term norms: t1={n1:.3e} t2={n2:.3e} t3={n3:.3e} "
                    f"t4={n4:.3e} t5={n5:.3e} t6={n6:.3e}"
                )
            # TODO revision test added term1 hyperparameter
            # gradient should assemble the derivative of the weighted objective
            # d(loss_total)/dX = FIRST * d(loss1)/dX + ALPHA * d(loss2)/dX + ...
            gradient_vals = (
                self.FIRST * term1_vals
                + self.ALPHA * term2_vals
                + self.BETA * term3_vals
                + self.GAMMA * term4_vals
                + self.DELTA * term5_vals
            )
            if term6_vals is not None:
                gradient_vals = gradient_vals + self.SPOT_SUM * term6_vals
            if debug_gradients and (ite % debug_every == 0):
                gnorm = float(np.sqrt((gradient_vals ** 2).sum()))
                gmax = float(np.max(np.abs(gradient_vals)))
                print(f"[debug] iter {ite} grad norm: {gnorm:.3e}, grad max: {gmax:.3e}")
            if np.isnan(gradient_vals).any() or np.isinf(gradient_vals).any():
                _emit_grad_diag(
                    "run_gradient_numeric_anomaly",
                    iter=int(ite),
                    where="gradient_vals",
                    has_nan=bool(np.isnan(gradient_vals).any()),
                    has_inf=bool(np.isinf(gradient_vals).any()),
                )
            opt_name = optimizer.lower()
            if opt_name == 'als_then_sgd':
                opt_name = 'als' if ite < als_iters else 'sgd'
            elif opt_name == 'als_then_adam':
                opt_name = 'als' if ite < als_iters else 'adam'
            elif opt_name == 'als_each_sgd':
                opt_name = 'als_each_sgd'
            elif opt_name == 'als_each_adam':
                opt_name = 'als_each_adam'
            if (not update_lr) and (self._non_lr_gene_mask is not None) and self._non_lr_gene_mask.any():
                gradient_non_lr = (
                    self.FIRST * term1_vals
                    + self.ALPHA * term2_vals
                    + self.DELTA * term5_vals
                )
                gvals = gradient_non_lr
                if grad_max is not None:
                    gmax = float(np.max(np.abs(gvals[:, self._non_lr_gene_mask])))
                    if gmax > 0:
                        gvals = gvals * (float(grad_max) / gmax)
                vals = self.alter_sc_exp.values
                vals[:, self._non_lr_gene_mask] = vals[:, self._non_lr_gene_mask] - self.ETA * gvals[:, self._non_lr_gene_mask]
            elif opt_name == 'als':
                gradient_other = (
                    + self.ALPHA * term2_vals
                    + self.BETA * term3_vals
                    + self.GAMMA * term4_vals
                    + self.DELTA * term5_vals
                )
                if grad_max is not None:
                    gmax = float(np.max(np.abs(gradient_other)))
                    if gmax > 0:
                        gradient_other = gradient_other * (float(grad_max) / gmax)
                self.alter_sc_exp.values[:] = self.alter_sc_exp.values - self.ETA * gradient_other
                self.alter_sc_exp = optimizers.closed_form_term1_update(
                    self.alter_sc_exp, self.sc_agg_meta, self.st_exp, blend=als_blend
                )
            elif opt_name == 'als_each_sgd' or opt_name == 'als_each_adam':
                gradient_other = (
                    + self.ALPHA * term2_vals
                    + self.BETA * term3_vals
                    + self.GAMMA * term4_vals
                    + self.DELTA * term5_vals
                )
                if grad_max is not None:
                    gmax = float(np.max(np.abs(gradient_other)))
                    if gmax > 0:
                        gradient_other = gradient_other * (float(grad_max) / gmax)
                self.alter_sc_exp.values[:] = self.alter_sc_exp.values - self.ETA * gradient_other
                self.alter_sc_exp = optimizers.closed_form_term1_update(
                    self.alter_sc_exp, self.sc_agg_meta, self.st_exp, blend=als_blend
                )
                if opt_name == 'als_each_adam':
                    if grad_max is not None:
                        gmax = float(np.max(np.abs(gradient_vals)))
                        if gmax > 0:
                            gradient_vals = gradient_vals * (float(grad_max) / gmax)
                    g = gradient_vals
                    if m is None:
                        m = np.zeros_like(g)
                        v = np.zeros_like(g)
                    m = adam_beta1 * m + (1 - adam_beta1) * g
                    v = adam_beta2 * v + (1 - adam_beta2) * (g * g)
                    m_hat = m / (1 - adam_beta1 ** (ite + 1))
                    v_hat = v / (1 - adam_beta2 ** (ite + 1))
                    update = self.ETA * m_hat / (np.sqrt(v_hat) + adam_eps)
                    self.alter_sc_exp.values[:] = self.alter_sc_exp.values - update
                else:
                    if grad_max is not None:
                        gmax = float(np.max(np.abs(gradient_vals)))
                        if gmax > 0:
                            gradient_vals = gradient_vals * (float(grad_max) / gmax)
                    self.alter_sc_exp.values[:] = self.alter_sc_exp.values - self.ETA * gradient_vals
            elif opt_name == 'adam':
                if grad_max is not None:
                    gmax = float(np.max(np.abs(gradient_vals)))
                    if gmax > 0:
                        gradient_vals = gradient_vals * (float(grad_max) / gmax)
                g = gradient_vals
                if m is None:
                    m = np.zeros_like(g)
                    v = np.zeros_like(g)
                m = adam_beta1 * m + (1 - adam_beta1) * g
                v = adam_beta2 * v + (1 - adam_beta2) * (g * g)
                m_hat = m / (1 - adam_beta1 ** (ite + 1))
                v_hat = v / (1 - adam_beta2 ** (ite + 1))
                update = self.ETA * m_hat / (np.sqrt(v_hat) + adam_eps)
                self.alter_sc_exp.values[:] = self.alter_sc_exp.values - update
            else:
                if grad_max is not None:
                    gmax = float(np.max(np.abs(gradient_vals)))
                    if gmax > 0:
                        gradient_vals = gradient_vals * (float(grad_max) / gmax)
                self.alter_sc_exp.values[:] = self.alter_sc_exp.values - self.ETA * gradient_vals
            self.alter_sc_exp[self.alter_sc_exp < 0.0] = 0
            if np.isnan(self.alter_sc_exp.values).any() or np.isinf(self.alter_sc_exp.values).any():
                _emit_grad_diag(
                    "run_gradient_numeric_anomaly",
                    iter=int(ite),
                    where="post_update_alter_sc_exp",
                    has_nan=bool(np.isnan(self.alter_sc_exp.values).any()),
                    has_inf=bool(np.isinf(self.alter_sc_exp.values).any()),
                )
            # Recompute losses on the updated state for consistent reporting/selection
            if recompute_loss_after_update:
                self.run_gradient(compute_term3 = compute_term3, compute_term4 = compute_term4)
            # TODO revision test added term1 hyperparameter
            # print(f'---{ite} self.loss4 {self.loss4} self.GAMMA {self.GAMMA} self.GAMMA*self.loss4 {self.GAMMA*self.loss4}')
            if normalize_terms:
                if recompute_loss_after_update:
                    t1 = self.term1_df.values
                    t2 = self.term2_df.values
                    t3 = self.term3_df.values
                    t4 = self.term4_df.values
                    t5 = self.term5_df.values
                    if term_norm == 'l2':
                        norm1 = float(np.sqrt((t1 ** 2).sum()))
                        norm2 = float(np.sqrt((t2 ** 2).sum()))
                        norm3 = float(np.sqrt((t3 ** 2).sum()))
                        norm4 = float(np.sqrt((t4 ** 2).sum()))
                        norm5 = float(np.sqrt((t5 ** 2).sum()))
                        norm6 = float(np.sqrt((term6_vals ** 2).sum())) if term6_vals is not None else 1.0
                    elif term_norm in ('rms', 'l2_mean'):
                        norm1 = float(np.sqrt((t1 ** 2).mean()))
                        norm2 = float(np.sqrt((t2 ** 2).mean()))
                        norm3 = float(np.sqrt((t3 ** 2).mean()))
                        norm4 = float(np.sqrt((t4 ** 2).mean()))
                        norm5 = float(np.sqrt((t5 ** 2).mean()))
                        norm6 = float(np.sqrt((term6_vals ** 2).mean())) if term6_vals is not None else 1.0
                    else:
                        norm1 = float(np.max(np.abs(t1)))
                        norm2 = float(np.max(np.abs(t2)))
                        norm3 = float(np.max(np.abs(t3)))
                        norm4 = float(np.max(np.abs(t4)))
                        norm5 = float(np.max(np.abs(t5)))
                        norm6 = float(np.max(np.abs(term6_vals))) if term6_vals is not None else 1.0
                loss1_eff = self.loss1 / max(norm1, norm_eps)
                loss2_eff = self.loss2 / max(norm2, norm_eps)
                loss3_eff = self.loss3 / max(norm3, norm_eps)
                loss4_eff = self.loss4 / max(norm4, norm_eps)
                loss5_eff = self.loss5 / max(norm5, norm_eps)
                loss6_eff = loss6 / max(norm6, norm_eps) if term6_vals is not None else 0.0
            else:
                loss1_eff = self.loss1
                loss2_eff = self.loss2
                loss3_eff = self.loss3
                loss4_eff = self.loss4
                loss5_eff = self.loss5
                loss6_eff = loss6 if term6_vals is not None else 0.0
            loss = (
                self.FIRST * loss1_eff + self.ALPHA * loss2_eff + self.BETA * loss3_eff
                + self.GAMMA * loss4_eff + self.DELTA * loss5_eff
            )
            if term6_vals is not None:
                loss = loss + self.SPOT_SUM * loss6_eff
            loss_raw = (
                self.FIRST * self.loss1 + self.ALPHA * self.loss2 + self.BETA * self.loss3
                + self.GAMMA * self.loss4 + self.DELTA * self.loss5
            )
            if term6_vals is not None:
                loss_raw = loss_raw + self.SPOT_SUM * loss6
            if loss < best_loss:
                best_loss = loss
                best_state = self.alter_sc_exp.copy()
            result_rows.append(_build_loss_row(
                loss1_eff, loss2_eff, loss3_eff, loss4_eff, loss5_eff, loss,
                self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, loss_raw,
                loss6_eff_val = loss6_eff, loss6_raw_val = loss6,
                include_spot_sum = term6_vals is not None,
            ))
            result_idx.append(ite)
            if ite == 0:
                print(f"Initial loss: {loss:.5f}")
            if convergence_patience > 0:
                if loss < best_loss_for_stop * (1 - convergence_tol):
                    best_loss_for_stop = loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= convergence_patience:
                        print(f"Early stopping at iteration {ite} (no improvement for {convergence_patience} iterations)")
                        break
            # print(f"{'='*40}")
            # print(f"Iteration {ite}...")
            # print(f"Shape Correlation: {max_shape:.4f}")
            # print(f"Total Loss:{loss:.5f}")

        if ce_kl_iters > 0:
            opt_name = optimizer.lower()
            for _ in range(ce_kl_iters):
                ite = len(result_idx)
                self._profile_iter = ite
                self.run_gradient(compute_term3=True, compute_term4=True)

                if ce_kl_term1:
                    term1_df, loss1 = optimizers.cal_term1_kl(
                        self.alter_sc_exp,
                        self.sc_agg_meta,
                        self.st_exp,
                        self.svg,
                        self.W_HVG,
                        weight=self._term1_weight,
                        eps=ce_kl_eps,
                    )
                else:
                    term1_df, loss1 = self.term1_df, self.loss1

                if ce_kl_term2:
                    term2_df, loss2 = optimizers.cal_term2_kl(
                        self.alter_sc_exp,
                        self.sc_ref,
                        eps=ce_kl_eps,
                    )
                else:
                    term2_df, loss2 = self.term2_df, self.loss2

                term1_vals = term1_df.values
                term2_vals = term2_df.values
                term3_vals = self.term3_df.values
                term4_vals = self.term4_df.values
                term5_vals = self.term5_df.values

                if normalize_terms:
                    if term_norm == 'l2':
                        norm1 = float(np.sqrt((term1_vals ** 2).sum()))
                        norm2 = float(np.sqrt((term2_vals ** 2).sum()))
                        norm3 = float(np.sqrt((term3_vals ** 2).sum()))
                        norm4 = float(np.sqrt((term4_vals ** 2).sum()))
                        norm5 = float(np.sqrt((term5_vals ** 2).sum()))
                    elif term_norm in ('rms', 'l2_mean'):
                        norm1 = float(np.sqrt((term1_vals ** 2).mean()))
                        norm2 = float(np.sqrt((term2_vals ** 2).mean()))
                        norm3 = float(np.sqrt((term3_vals ** 2).mean()))
                        norm4 = float(np.sqrt((term4_vals ** 2).mean()))
                        norm5 = float(np.sqrt((term5_vals ** 2).mean()))
                    else:
                        norm1 = float(np.max(np.abs(term1_vals)))
                        norm2 = float(np.max(np.abs(term2_vals)))
                        norm3 = float(np.max(np.abs(term3_vals)))
                        norm4 = float(np.max(np.abs(term4_vals)))
                        norm5 = float(np.max(np.abs(term5_vals)))
                    term1_vals = term1_vals / max(norm1, norm_eps)
                    term2_vals = term2_vals / max(norm2, norm_eps)
                    term3_vals = term3_vals / max(norm3, norm_eps)
                    term4_vals = term4_vals / max(norm4, norm_eps)
                    term5_vals = term5_vals / max(norm5, norm_eps)

                # Gradient is the weighted sum of term derivatives (dLoss/dX).
                # Use same sign convention as the main optimization loop.
                gradient_vals = (
                    self.FIRST * term1_vals
                    + self.ALPHA * term2_vals
                    + self.BETA * term3_vals
                    + self.GAMMA * term4_vals
                    + self.DELTA * term5_vals
                )
                if grad_max is not None:
                    gmax = float(np.max(np.abs(gradient_vals)))
                    if gmax > 0:
                        gradient_vals = gradient_vals * (float(grad_max) / gmax)

                use_adam = opt_name in ('adam', 'als_then_adam', 'als_each_adam')
                if use_adam:
                    g = gradient_vals
                    if m is None:
                        m = np.zeros_like(g)
                        v = np.zeros_like(g)
                    m = adam_beta1 * m + (1 - adam_beta1) * g
                    v = adam_beta2 * v + (1 - adam_beta2) * (g * g)
                    m_hat = m / (1 - adam_beta1 ** (ite + 1))
                    v_hat = v / (1 - adam_beta2 ** (ite + 1))
                    update = ce_kl_eta * m_hat / (np.sqrt(v_hat) + adam_eps)
                    self.alter_sc_exp.values[:] = self.alter_sc_exp.values - update
                else:
                    self.alter_sc_exp.values[:] = self.alter_sc_exp.values - ce_kl_eta * gradient_vals

                self.alter_sc_exp[self.alter_sc_exp < 0.0] = 0
                if recompute_loss_after_update:
                    self.run_gradient(compute_term3=True, compute_term4=True)

                if normalize_terms:
                    if recompute_loss_after_update:
                        t1 = self.term1_df.values
                        t2 = self.term2_df.values
                        t3 = self.term3_df.values
                        t4 = self.term4_df.values
                        t5 = self.term5_df.values
                        if term_norm == 'l2':
                            norm1 = float(np.sqrt((t1 ** 2).sum()))
                            norm2 = float(np.sqrt((t2 ** 2).sum()))
                            norm3 = float(np.sqrt((t3 ** 2).sum()))
                            norm4 = float(np.sqrt((t4 ** 2).sum()))
                            norm5 = float(np.sqrt((t5 ** 2).sum()))
                        elif term_norm in ('rms', 'l2_mean'):
                            norm1 = float(np.sqrt((t1 ** 2).mean()))
                            norm2 = float(np.sqrt((t2 ** 2).mean()))
                            norm3 = float(np.sqrt((t3 ** 2).mean()))
                            norm4 = float(np.sqrt((t4 ** 2).mean()))
                            norm5 = float(np.sqrt((t5 ** 2).mean()))
                        else:
                            norm1 = float(np.max(np.abs(t1)))
                            norm2 = float(np.max(np.abs(t2)))
                            norm3 = float(np.max(np.abs(t3)))
                            norm4 = float(np.max(np.abs(t4)))
                            norm5 = float(np.max(np.abs(t5)))
                    loss1_eff = loss1 / max(norm1, norm_eps)
                    loss2_eff = loss2 / max(norm2, norm_eps)
                    loss3_eff = self.loss3 / max(norm3, norm_eps)
                    loss4_eff = self.loss4 / max(norm4, norm_eps)
                    loss5_eff = self.loss5 / max(norm5, norm_eps)
                else:
                    loss1_eff, loss2_eff = loss1, loss2
                    loss3_eff, loss4_eff, loss5_eff = self.loss3, self.loss4, self.loss5

                loss = (
                    self.FIRST * loss1_eff + self.ALPHA * loss2_eff + self.BETA * loss3_eff
                    + self.GAMMA * loss4_eff + self.DELTA * loss5_eff
                )
                loss_raw = (
                    self.FIRST * loss1 + self.ALPHA * loss2 + self.BETA * self.loss3
                    + self.GAMMA * self.loss4 + self.DELTA * self.loss5
                )
                if loss < best_loss:
                    best_loss = loss
                    best_state = self.alter_sc_exp.copy()
                result_rows.append(_build_loss_row(
                    loss1_eff, loss2_eff, loss3_eff, loss4_eff, loss5_eff, loss,
                    loss1, loss2, self.loss3, self.loss4, self.loss5, loss_raw,
                    loss6_eff_val = 0.0, loss6_raw_val = 0.0, include_spot_sum = False,
                ))
                result_idx.append(ite)

        self.alter_sc_exp[self.alter_sc_exp < 0.0] = 0
        result = pd.DataFrame(result_rows, columns=res_col, index=result_idx)
        # recompute final loss on the updated state
        self.run_gradient(compute_term3=True, compute_term4=True)
        if normalize_terms:
            t1 = self.term1_df.values
            t2 = self.term2_df.values
            t3 = self.term3_df.values
            t4 = self.term4_df.values
            t5 = self.term5_df.values
            if term_norm == 'l2':
                norm1 = float(np.sqrt((t1 ** 2).sum()))
                norm2 = float(np.sqrt((t2 ** 2).sum()))
                norm3 = float(np.sqrt((t3 ** 2).sum()))
                norm4 = float(np.sqrt((t4 ** 2).sum()))
                norm5 = float(np.sqrt((t5 ** 2).sum()))
            elif term_norm in ('rms', 'l2_mean'):
                norm1 = float(np.sqrt((t1 ** 2).mean()))
                norm2 = float(np.sqrt((t2 ** 2).mean()))
                norm3 = float(np.sqrt((t3 ** 2).mean()))
                norm4 = float(np.sqrt((t4 ** 2).mean()))
                norm5 = float(np.sqrt((t5 ** 2).mean()))
            else:
                norm1 = float(np.max(np.abs(t1)))
                norm2 = float(np.max(np.abs(t2)))
                norm3 = float(np.max(np.abs(t3)))
                norm4 = float(np.max(np.abs(t4)))
                norm5 = float(np.max(np.abs(t5)))
            loss1_eff = self.loss1 / max(norm1, norm_eps)
            loss2_eff = self.loss2 / max(norm2, norm_eps)
            loss3_eff = self.loss3 / max(norm3, norm_eps)
            loss4_eff = self.loss4 / max(norm4, norm_eps)
            loss5_eff = self.loss5 / max(norm5, norm_eps)
        else:
            loss1_eff = self.loss1
            loss2_eff = self.loss2
            loss3_eff = self.loss3
            loss4_eff = self.loss4
            loss5_eff = self.loss5
        loss = (
            self.FIRST * loss1_eff + self.ALPHA * loss2_eff + self.BETA * loss3_eff
            + self.GAMMA * loss4_eff + self.DELTA * loss5_eff
        )
        if loss > best_loss and ce_kl_iters <= 0:
            # restore best state and ensure losses are recomputed on that state
            self.alter_sc_exp = best_state
            loss = best_loss
            try:
                # recompute term dataframes and losses for the restored best_state
                self.run_gradient(compute_term3=True, compute_term4=True)
            except Exception:
                # if recompute fails, continue and rely on previously computed losses
                pass
        # persist refined expression for downstream steps
        self.alter_sc_exp.to_csv(f'{self.save_path}/refined_sc_exp.tsv',sep = '\t',header=True,index=True)
        # Ensure the written loss table reflects the recomputed final/internal loss
        # Compute final normalized/raw losses consistent with the loop logic above.
        try:
            if normalize_terms:
                t1 = self.term1_df.values
                t2 = self.term2_df.values
                t3 = self.term3_df.values
                t4 = self.term4_df.values
                t5 = self.term5_df.values
                if term_norm == 'l2':
                    norm1 = float(np.sqrt((t1 ** 2).sum()))
                    norm2 = float(np.sqrt((t2 ** 2).sum()))
                    norm3 = float(np.sqrt((t3 ** 2).sum()))
                    norm4 = float(np.sqrt((t4 ** 2).sum()))
                    norm5 = float(np.sqrt((t5 ** 2).sum()))
                elif term_norm in ('rms', 'l2_mean'):
                    norm1 = float(np.sqrt((t1 ** 2).mean()))
                    norm2 = float(np.sqrt((t2 ** 2).mean()))
                    norm3 = float(np.sqrt((t3 ** 2).mean()))
                    norm4 = float(np.sqrt((t4 ** 2).mean()))
                    norm5 = float(np.sqrt((t5 ** 2).mean()))
                else:
                    norm1 = float(np.max(np.abs(t1)))
                    norm2 = float(np.max(np.abs(t2)))
                    norm3 = float(np.max(np.abs(t3)))
                    norm4 = float(np.max(np.abs(t4)))
                    norm5 = float(np.max(np.abs(t5)))
                loss1_eff = self.loss1 / max(norm1, norm_eps)
                loss2_eff = self.loss2 / max(norm2, norm_eps)
                loss3_eff = self.loss3 / max(norm3, norm_eps)
                loss4_eff = self.loss4 / max(norm4, norm_eps)
                loss5_eff = self.loss5 / max(norm5, norm_eps)
            else:
                loss1_eff = self.loss1
                loss2_eff = self.loss2
                loss3_eff = self.loss3
                loss4_eff = self.loss4
                loss5_eff = self.loss5

            final_loss = (
                self.FIRST * loss1_eff + self.ALPHA * loss2_eff + self.BETA * loss3_eff
                + self.GAMMA * loss4_eff + self.DELTA * loss5_eff
            )
            final_loss_raw = (
                self.FIRST * self.loss1 + self.ALPHA * self.loss2 + self.BETA * self.loss3
                + self.GAMMA * self.loss4 + self.DELTA * self.loss5
            )
        except Exception as _e_outer:
            import traceback
            print('Warning: failed computing normalized final loss before writing loss.tsv:', _e_outer)
            traceback.print_exc()
            # fall back to last-recorded values (if any)
            final_loss = None
            final_loss_raw = None

        # Replace the last per-iteration row with the recomputed final state so on-disk final row matches internal totals.
        try:
            t6_final = self.term6_df.values if getattr(self, 'term6_df', None) is not None else None
            if normalize_terms and t6_final is not None and getattr(self, 'SPOT_SUM', 0.0) > 0:
                if term_norm == 'l2':
                    norm6 = float(np.sqrt((t6_final ** 2).sum()))
                elif term_norm in ('rms', 'l2_mean'):
                    norm6 = float(np.sqrt((t6_final ** 2).mean()))
                else:
                    norm6 = float(np.max(np.abs(t6_final)))
                loss6_eff_final = self.loss6 / max(norm6, norm_eps)
            elif t6_final is not None and getattr(self, 'SPOT_SUM', 0.0) > 0:
                loss6_eff_final = self.loss6
            else:
                loss6_eff_final = 0.0
            include_final_spot_sum = t6_final is not None and getattr(self, 'SPOT_SUM', 0.0) > 0
            if include_final_spot_sum:
                final_loss = final_loss + self.SPOT_SUM * loss6_eff_final
                final_loss_raw = final_loss_raw + self.SPOT_SUM * self.loss6
            final_row = _build_loss_row(
                loss1_eff, loss2_eff, loss3_eff, loss4_eff, loss5_eff, final_loss,
                self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, final_loss_raw,
                loss6_eff_val = loss6_eff_final, loss6_raw_val = self.loss6,
                include_spot_sum = include_final_spot_sum,
            )
            if len(result_rows) > 0:
                # replace the last recorded row
                result_rows[-1] = final_row
                # keep the existing final index
            else:
                # no per-iteration rows recorded; append as the only row
                result_rows.append(final_row)
                result_idx.append(0)
        except Exception as _e_replace:
            import traceback
            print('Warning: failed to replace final loss row (falling back to append):', _e_replace)
            traceback.print_exc()
            try:
                result_rows.append(_build_loss_row(
                    loss1_eff, loss2_eff, loss3_eff, loss4_eff, loss5_eff, final_loss,
                    self.loss1, self.loss2, self.loss3, self.loss4, self.loss5, final_loss_raw,
                    loss6_eff_val = 0.0, loss6_raw_val = 0.0, include_spot_sum = False,
                ))
                result_idx.append(len(result_idx))
            except Exception:
                pass

        # Build the result DataFrame (preserve user index types where possible)
        try:
            result = pd.DataFrame(result_rows, columns=res_col, index=result_idx)
        except Exception as _e_build:
            import traceback
            print('Error building loss DataFrame; falling back to best-effort write:', _e_build)
            traceback.print_exc()
            try:
                result = pd.DataFrame(result_rows, columns=res_col)
            except Exception:
                result = pd.DataFrame([], columns=res_col)

        # Write atomically: write to temp file then replace the target
        try:
            import tempfile
            import os as _os
            tmp_fd, tmp_path = tempfile.mkstemp(dir=self.save_path, prefix='loss-', suffix='.tsv')
            os.close(tmp_fd)
            result.to_csv(tmp_path, sep='\t', header=True, index=True)
            _os.replace(tmp_path, _os.path.join(self.save_path, 'loss.tsv'))
        except Exception as _e_write:
            # If atomic write fails, attempt direct write and log
            import traceback
            print('Warning: atomic write of loss.tsv failed, attempting direct write:', _e_write)
            traceback.print_exc()
            try:
                result.to_csv(f'{self.save_path}/loss.tsv', sep='\t', header=True, index=True)
            except Exception as _e_final:
                print('Error: failed to write loss.tsv:', _e_final)
                traceback.print_exc()

        # expose result in-memory
        self.result = result
        if self.st_tp != 'slide-seq':
            self.sc_coord, max_shape, _, sparse_A, ans = self._aff_embedding_cached(self.alter_sc_exp)
            if final_refine and self._embedding_compute_shape and ans is not None:
                n_neighbors = int(np.round((self.left_range + 1) * 15))
                self.sc_coord, refined_shape = optimizers.refine_embedding_local(
                    sparse_A,
                    ans,
                    self.sc_coord,
                    n_neighbors=n_neighbors,
                    n_epochs=refine_epochs,
                    n_runs=refine_runs,
                    jitter=refine_jitter,
                )
                if refined_shape is not None:
                    max_shape = max(max_shape, refined_shape)
            _, sc_spot_center = optimizers.sc_prep(self.st_coord, self.sc_agg_meta)
            self.sc_agg_meta[['st_x','st_y']] = sc_spot_center
            self.sc_agg_meta = optimizers.center_shift_embedding(self.sc_coord, self.sc_agg_meta, max_dist = 1)
            self.sc_coord = self.sc_agg_meta[['adj_spex_UMAP1','adj_spex_UMAP2']].to_numpy()
            if final_refine and self._embedding_compute_shape and ans is not None:
                n_neighbors = int(np.round((self.left_range + 1) * 15))
                self.sc_coord, refined_shape = optimizers.refine_embedding_local(
                    sparse_A,
                    ans,
                    self.sc_coord,
                    n_neighbors=n_neighbors,
                    n_epochs=refine_epochs,
                    n_runs=refine_runs,
                    jitter=refine_jitter,
                )
                self.sc_agg_meta[['adj_spex_UMAP1','adj_spex_UMAP2']] = self.sc_coord
                if refined_shape is not None:
                    max_shape = max(max_shape, refined_shape)
            print(f"{'='*40}")
            print(f"Final Loss:{loss:.5f}")
            if max_shape is None and last_shape is not None:
                max_shape = last_shape
            print(f"Final shape correlation: {max_shape:.2f}")
            
        else:
            # v10
            self.sc_agg_meta[['st_x','st_y']] = self.sc_coord
            self.sc_agg_meta[['adj_spex_UMAP1','adj_spex_UMAP2']] = self.sc_coord
        return self.alter_sc_exp,self.sc_agg_meta

    def compute_total(self, normalize_terms=True, term_norm='rms', norm_eps=1e-8):
        """Return the weighted total loss using the same normalization as used
        internally by `gradient_descent` (`lossX_eff = lossX / normX`).

        This helper lets external code (notebooks/tests) compute totals
        deterministically from the object's current term values.
        """
        try:
            t1 = self.term1_df.values
            t2 = self.term2_df.values
            t3 = self.term3_df.values
            t4 = self.term4_df.values
            t5 = self.term5_df.values
        except Exception:
            return None
        if normalize_terms:
            if term_norm == 'l2':
                norm1 = float(np.sqrt((t1 ** 2).sum()))
                norm2 = float(np.sqrt((t2 ** 2).sum()))
                norm3 = float(np.sqrt((t3 ** 2).sum()))
                norm4 = float(np.sqrt((t4 ** 2).sum()))
                norm5 = float(np.sqrt((t5 ** 2).sum()))
            elif term_norm in ('rms', 'l2_mean'):
                norm1 = float(np.sqrt((t1 ** 2).mean()))
                norm2 = float(np.sqrt((t2 ** 2).mean()))
                norm3 = float(np.sqrt((t3 ** 2).mean()))
                norm4 = float(np.sqrt((t4 ** 2).mean()))
                norm5 = float(np.sqrt((t5 ** 2).mean()))
            else:
                norm1 = float(np.max(np.abs(t1)))
                norm2 = float(np.max(np.abs(t2)))
                norm3 = float(np.max(np.abs(t3)))
                norm4 = float(np.max(np.abs(t4)))
                norm5 = float(np.max(np.abs(t5)))
            l1 = self.loss1 / max(norm1, norm_eps)
            l2 = self.loss2 / max(norm2, norm_eps)
            l3 = self.loss3 / max(norm3, norm_eps)
            l4 = self.loss4 / max(norm4, norm_eps)
            l5 = self.loss5 / max(norm5, norm_eps)
        else:
            l1, l2, l3, l4, l5 = self.loss1, self.loss2, self.loss3, self.loss4, self.loss5
        total = self.FIRST * l1 + self.ALPHA * l2 + self.BETA * l3 + self.GAMMA * l4 + self.DELTA * l5
        if getattr(self, 'term6_df', None) is not None and getattr(self, 'SPOT_SUM', 0.0) > 0:
            t6 = self.term6_df.values
            if normalize_terms:
                if term_norm == 'l2':
                    norm6 = float(np.sqrt((t6 ** 2).sum()))
                elif term_norm in ('rms', 'l2_mean'):
                    norm6 = float(np.sqrt((t6 ** 2).mean()))
                else:
                    norm6 = float(np.max(np.abs(t6)))
                l6 = self.loss6 / max(norm6, norm_eps)
            else:
                l6 = self.loss6
            total = total + self.SPOT_SUM * l6
        return total

    def write_loss_table(self, normalize_terms=True, term_norm='rms', norm_eps=1e-8, filename=None, recompute=True):
        """Atomically write the current internal loss components and total to `loss.tsv`.

        By default this overwrites the target file with a single-row table reflecting
        the object's current losses (after an optional recompute). This is useful
        when external code (for example a notebook) mutates state after
        `gradient_descent()` and you want to persist the canonical, in-memory
        totals to disk.

        Returns the path written on success.
        """
        if filename is None:
            filename = os.path.join(self.save_path, 'loss.tsv')

        if recompute:
            try:
                # Ensure term dataframes and loss scalars reflect current state
                self.run_gradient(compute_term3=True, compute_term4=True)
            except Exception:
                # best-effort: continue using existing values
                pass

        # compute normalized (effective) losses using same logic as compute_total
        try:
            t1 = self.term1_df.values
            t2 = self.term2_df.values
            t3 = self.term3_df.values
            t4 = self.term4_df.values
            t5 = self.term5_df.values
            t6 = self.term6_df.values if getattr(self, 'term6_df', None) is not None else None
        except Exception:
            # nothing to write
            raise RuntimeError('Term dataframes are not available for writing loss table')

        if normalize_terms:
            if term_norm == 'l2':
                norm1 = float(np.sqrt((t1 ** 2).sum()))
                norm2 = float(np.sqrt((t2 ** 2).sum()))
                norm3 = float(np.sqrt((t3 ** 2).sum()))
                norm4 = float(np.sqrt((t4 ** 2).sum()))
                norm5 = float(np.sqrt((t5 ** 2).sum()))
                norm6 = float(np.sqrt((t6 ** 2).sum())) if t6 is not None else 1.0
            elif term_norm in ('rms', 'l2_mean'):
                norm1 = float(np.sqrt((t1 ** 2).mean()))
                norm2 = float(np.sqrt((t2 ** 2).mean()))
                norm3 = float(np.sqrt((t3 ** 2).mean()))
                norm4 = float(np.sqrt((t4 ** 2).mean()))
                norm5 = float(np.sqrt((t5 ** 2).mean()))
                norm6 = float(np.sqrt((t6 ** 2).mean())) if t6 is not None else 1.0
            else:
                norm1 = float(np.max(np.abs(t1)))
                norm2 = float(np.max(np.abs(t2)))
                norm3 = float(np.max(np.abs(t3)))
                norm4 = float(np.max(np.abs(t4)))
                norm5 = float(np.max(np.abs(t5)))
                norm6 = float(np.max(np.abs(t6))) if t6 is not None else 1.0
            l1 = self.loss1 / max(norm1, norm_eps)
            l2 = self.loss2 / max(norm2, norm_eps)
            l3 = self.loss3 / max(norm3, norm_eps)
            l4 = self.loss4 / max(norm4, norm_eps)
            l5 = self.loss5 / max(norm5, norm_eps)
            l6 = self.loss6 / max(norm6, norm_eps) if t6 is not None else 0.0
        else:
            l1, l2, l3, l4, l5 = self.loss1, self.loss2, self.loss3, self.loss4, self.loss5
            l6 = self.loss6 if t6 is not None else 0.0
        total = self.FIRST * l1 + self.ALPHA * l2 + self.BETA * l3 + self.GAMMA * l4 + self.DELTA * l5
        total_raw = self.FIRST * self.loss1 + self.ALPHA * self.loss2 + self.BETA * self.loss3 + self.GAMMA * self.loss4 + self.DELTA * self.loss5
        data = {
            'loss1': [self.FIRST * l1],
            'loss2': [self.ALPHA * l2],
            'loss3': [self.BETA * l3],
            'loss4': [self.GAMMA * l4],
            'loss5': [self.DELTA * l5],
            'total': [total],
            'loss1_raw': [self.FIRST * self.loss1],
            'loss2_raw': [self.ALPHA * self.loss2],
            'loss3_raw': [self.BETA * self.loss3],
            'loss4_raw': [self.GAMMA * self.loss4],
            'loss5_raw': [self.DELTA * self.loss5],
            'total_raw': [total_raw],
        }
        if t6 is not None and getattr(self, 'SPOT_SUM', 0.0) > 0:
            data['loss6'] = [self.SPOT_SUM * l6]
            data['loss6_raw'] = [self.SPOT_SUM * self.loss6]
            data['total'] = [total + self.SPOT_SUM * l6]
            data['total_raw'] = [total_raw + self.SPOT_SUM * self.loss6]
        result = pd.DataFrame(data, index=[0])

        # atomic write with fallback
        try:
            import tempfile as _tempfile
            import os as _os
            tmp_fd, tmp_path = _tempfile.mkstemp(dir=os.path.dirname(filename), prefix='loss-', suffix='.tsv')
            _os.close(tmp_fd)
            result.to_csv(tmp_path, sep='\t', header=True, index=True)
            _os.replace(tmp_path, filename)
        except Exception:
            try:
                result.to_csv(filename, sep='\t', header=True, index=True)
            except Exception as _e:
                raise

        # expose in-memory result
        self.result = result
        return filename
