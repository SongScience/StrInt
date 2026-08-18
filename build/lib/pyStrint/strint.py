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

    def _check_input(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.species not in ['Human', 'Mouse']:
            raise ValueError('Species should be chosen among either Human or Mouse.')
        if self.lr_df is None:
            self.lr_df = pp.load_lr_df(species = self.species,lr_dir = None)

        utils.check_st_tp(self.st_tp)
        self.st_adata, self.sc_adata, self.sc_ref, self.weight = utils.check_index_str(self.st_adata, self.sc_adata, self.sc_ref, self.weight)
        self.st_adata, self.weight = utils.check_spots(self.st_adata,self.weight)
        self.st_coord = utils.check_st_coord(self.st_adata)
        self.lr_df = utils.align_lr_gene(self)
        utils.check_st_sc_pair(self.st_adata, self.sc_adata)
        self.sc_adata,self.sc_ref = utils.check_sc(self.sc_adata, self.sc_ref)
        utils.check_decon_type(self.weight, self.sc_adata, self.cell_type_key)
        print('Parameters checked!')

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
        self.st_aff_profile_df = optimizers.cal_aff_profile(self.st_exp, self.lr_df)


    @timeit
    def select_cells(self, user_sc_exp = None, user_sc_agg_meta = None, p = 0, mean_num_per_spot = 10, metric = 'correlation', 
                     max_rep = 1, repeat_penalty = 10, seed = 1111):
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
            The metric for evaluation of cell selection. 'rmse' or 'correlation'.
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
            result = self.sc_agg_meta
        else:
            self.repeat_penalty = repeat_penalty
            self.p = p
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
            # 1. subset and filter
            self.filter_st_exp, self.filter_sc_exp = pp.subset_inter(self.st_exp, self.sc_exp)
            # 2. feature selection
            self.sort_genes = cell_selection.feature_sort(self.filter_sc_exp, degree = 2, span = 0.3)
            self.lr_hvg_genes = cell_selection.lr_shared_top_k_gene(self.sort_genes, self.lr_df, k = 3000, keep_lr_per = 1)
            print(f'\t SpexMod selects {len(self.lr_hvg_genes)} feature genes.')

            # 3. scale and norm
            self.trans_id_idx = pd.DataFrame(list(range(self.filter_sc_exp.shape[0])), index = self.filter_sc_exp.index)
            self.hvg_st_exp = self.filter_st_exp.loc[:,self.lr_hvg_genes]
            self.hvg_sc_exp = self.filter_sc_exp.loc[:,self.lr_hvg_genes]
            norm_hvg_st = cell_selection.norm_center(self.hvg_st_exp)
            norm_hvg_sc = cell_selection.norm_center(self.hvg_sc_exp)
            self.csr_st_exp = csr_matrix(norm_hvg_st)
            self.csr_sc_exp = csr_matrix(norm_hvg_sc)
            # all lr that exp in st
            self.lr_df_align = self.lr_df[self.lr_df[0].isin(self.filter_st_exp.columns) & self.lr_df[1].isin(self.filter_st_exp.columns)].copy()
            # 4. init cell selection
            self.spot_cell_dict, self.init_cor, self.picked_time = cell_selection.init_solution(self.num, self.filter_st_exp.index.tolist(),
                self.csr_st_exp,self.csr_sc_exp,self.sc_meta[self.cell_type_key],self.trans_id_idx,self.repeat_penalty)
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
            if self.p == 0:
                self.sum_sc_agg_exp = cell_selection.get_sum_sc_agg(norm_hvg_sc, result, norm_hvg_st)
                self.sc_agg_aff_profile_df = optimizers.cal_aff_profile(self.sum_sc_agg_exp, self.lr_df_align)
                result, self.after_picked_time = cell_selection.reselect_cell(norm_hvg_st, self.spots_nn_lst, self.st_aff_profile_df, 
                            norm_hvg_sc, self.csr_sc_exp, self.sc_meta, self.trans_id_idx,
                            self.sum_sc_agg_exp, self.sc_agg_aff_profile_df,
                            result, self.picked_time, self.lr_df_align, 
                            p = self.p, repeat_penalty = self.repeat_penalty, metric = metric)
            else:
                for i in range(max_rep):
                    self.sum_sc_agg_exp = cell_selection.get_sum_sc_agg(self.filter_sc_exp,result,self.filter_st_exp)
                    self.sc_agg_aff_profile_df = optimizers.cal_aff_profile(self.sum_sc_agg_exp, self.lr_df_align)
                    result, self.after_picked_time = cell_selection.reselect_cell(self.filter_st_exp, self.spots_nn_lst, self.st_aff_profile_df, 
                                self.filter_sc_exp, self.csr_sc_exp, self.sc_meta, self.trans_id_idx,
                                self.sum_sc_agg_exp, self.sc_agg_aff_profile_df,
                                result, self.picked_time, self.lr_df_align, 
                                p = self.p, repeat_penalty = self.repeat_penalty, metric = metric)
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
    def run_gradient(self):
        # 1. First term
        self.term1_df,self.loss1 = optimizers.cal_term1(self.alter_sc_exp,self.sc_agg_meta,self.st_exp,self.svg,self.W_HVG)
        # print('First-term calculation done!')

        # 2. Second term
        self.term2_df,self.loss2 = optimizers.cal_term2(self.alter_sc_exp,self.sc_ref)
        # print('Second-term calculation done!')

        # 3. Third term, closer cells have larger affinity
        if not (self.st_tp == 'slide-seq' and hasattr(self, 'sc_knn')):
            # if slide-seq and already have found sc_knn
            # dont do it again
            self.sc_dist = pd.DataFrame(distance_matrix(self.sc_coord,self.sc_coord), index = self.alter_sc_exp.index, columns = self.alter_sc_exp.index)
            # 3.2 get c' = N(c)
            self.sc_knn = optimizers.findCellKNN(self.st_coord,self.st_tp,self.sc_agg_meta,self.sc_coord,self.K)
            utils.check_empty_dict(self.sc_knn)
        # 3.3 get the paring genes (g') of gene g for each cells
        self.rl_agg = optimizers.generate_LR_agg(self.alter_sc_exp,self.lr_df)
        # 3.4 get the affinity
        self.aff = optimizers.calculate_affinity_mat(self.lr_df, self.alter_sc_exp.T)
        np.fill_diagonal(self.aff,0)
        self.aff = pd.DataFrame(self.aff, index = self.sc_agg_meta.index, columns=self.sc_agg_meta.index)
        # 3.5 Calculate the derivative
        self.term3_df,self.loss3 = optimizers.cal_term3(self.alter_sc_exp,self.sc_knn,self.aff,self.sc_dist,self.rl_agg)
        # print('Third term calculation done!')

        # 4. Fourth term, towards spot-spot affinity profile
        self.term4_df,self.loss4 = optimizers.cal_term4(self.st_exp,self.sc_knn,self.st_aff_profile_df,self.alter_sc_exp,
                                                        self.sc_agg_meta,self.spot_cell_dict,self.lr_df_align)
        # print('Fourth term calculation done!')
        
        # 5. Fifth term, norm2 regulization
        self.term5_df,self.loss5 = optimizers.cal_term5(self.alter_sc_exp)
        

    @timeit
    def init_grad(self):
        if isinstance(self.init_sc_embed, pd.DataFrame):
            self.sc_coord = utils.check_sc_coord(self.init_sc_embed)
            print('Using user provided init sc_coord.')
        else:
            print('Initialize cell coordinates by affinity embedding...')
            self.sc_coord,max_shape,_,_,_ = optimizers.aff_embedding(self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                            self.save_path,self.left_range,self.right_range,self.steps,self.dim,verbose = False)
            print(f"Initial shape correlation: {max_shape:.2f}")
            # print(f"{'='*50}\n")
        self.run_gradient()
        # v5 calculte the initial loss of each term to balance their force.
        adj2,adj3,adj4,adj5 = optimizers.loss_adj(self.loss1,self.loss2,self.loss3,self.loss4,self.loss5)
        self.ALPHA,self.BETA,self.GAMMA,self.DELTA = self.ALPHA*adj2,self.BETA*adj3,self.GAMMA*adj4,self.DELTA*adj5
        self.sc_agg_meta[['UMAP1','UMAP2']] = self.sc_coord
        # print('Hyperparameters adjusted.')


    @timeit
    def gradient_descent(self, p1 = 0.05, p2 = 0.65, p3 = 0.2, p4 = 0.1, 
                         delta = 0.1, eta = 0.0005, 
                        init_sc_embed = False,
                        iteration = 20, k = 2, W_HVG = 2,
                        left_range = 1, right_range = 2, steps = 1, dim = 2):
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
        if not isinstance(self.sc_ref, np.ndarray):
            self.sc_ref = np.array(self.sc_ref.loc[self.sc_agg_meta['sc_id']])
        res_col = ['loss1','loss2','loss3','loss4','loss5','total']
        result = pd.DataFrame(columns=res_col)
        if self.st_tp == 'slide-seq':
            # cell coord as spot coord
            self.init_sc_embed = self.st_coord.loc[self.sc_agg_meta['spot']]
            self.init_sc_embed.index = self.sc_agg_meta.index
        self.init_grad()
        ######### init done ############
        for ite in range(self.iteration):
            # print(f'-----Start iteration {ite} -----')
            if self.st_tp != 'slide-seq':
                self.sc_coord,max_shape,_,_,_ = optimizers.aff_embedding(self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                                self.save_path,self.left_range,self.right_range,self.steps,self.dim)
            self.run_gradient()
            # TODO revision test added term1 hyperparameter
            gradient = self.FIRST*self.term1_df - self.ALPHA*self.term2_df + self.BETA*self.term3_df + self.GAMMA*self.term4_df + self.DELTA*self.term5_df
            self.alter_sc_exp = self.alter_sc_exp - self.ETA * gradient
            self.alter_sc_exp[self.alter_sc_exp<0.5] = 0
            # TODO revision test added term1 hyperparameter
            # print(f'---{ite} self.loss4 {self.loss4} self.GAMMA {self.GAMMA} self.GAMMA*self.loss4 {self.GAMMA*self.loss4}')
            loss = self.FIRST*self.loss1 + self.ALPHA*self.loss2 + self.BETA*self.loss3 + self.GAMMA*self.loss4 + self.DELTA*self.loss5
            tmp = pd.DataFrame(np.array([[self.GAMMA*self.loss4,self.FIRST*self.loss1,self.ALPHA*self.loss2,self.BETA*self.loss3,self.DELTA*self.loss5,loss]]),columns = res_col, index = [ite])
            result = pd.concat((result,tmp),axis=0)
            if ite == 0:
                print(f"Initial loss: {loss:.5f}")
            # print(f"{'='*40}")
            # print(f"Iteration {ite}...")
            # print(f"Shape Correlation: {max_shape:.4f}")
            # print(f"Total Loss:{loss:.5f}")

        self.alter_sc_exp[self.alter_sc_exp < 1] = 0  
        self.alter_sc_exp.to_csv(f'{self.save_path}/refined_sc_exp.tsv',sep = '\t',header=True,index=True)
        result.to_csv(f'{self.save_path}/loss.tsv',sep = '\t',header=True,index=True)
        self.result = result
        if self.st_tp != 'slide-seq':
            self.sc_coord,max_shape,_,_,_ = optimizers.aff_embedding(self.alter_sc_exp,self.st_coord,self.sc_agg_meta,self.lr_df,
                                self.save_path,self.left_range,self.right_range,self.steps,self.dim)
            _, sc_spot_center = optimizers.sc_prep(self.st_coord, self.sc_agg_meta)
            self.sc_agg_meta[['st_x','st_y']] = sc_spot_center
            self.sc_agg_meta = optimizers.center_shift_embedding(self.sc_coord, self.sc_agg_meta, max_dist = 1)
            print(f"{'='*40}")
            print(f"Final Loss:{loss:.5f}")
            print(f"Final shape correlation: {max_shape:.2f}")
            
        else:
            # v10
            self.sc_agg_meta[['st_x','st_y']] = self.sc_coord
            self.sc_agg_meta[['adj_spex_UMAP1','adj_spex_UMAP2']] = self.sc_coord
        return self.alter_sc_exp,self.sc_agg_meta