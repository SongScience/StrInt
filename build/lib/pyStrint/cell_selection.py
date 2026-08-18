import pandas as pd
import numpy as np
import time
import logging
from scipy.sparse import csr_matrix

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
    np.random.seed(seed)
    [m,n] = mat_orig.shape
    mat = mat_orig.copy()
    # m - spot number
    # n - cell type number
    for i in range(m):
        for j in range(n):
            # loop through each entry
            tmp = mat.iloc[i,j]
            if tmp!=0:
                #print(tmp)
                c = np.floor(tmp)
                # if entry is integer, pass
                if c == tmp:
                    continue
                else:  
                    d = np.ceil(tmp)
                    #print(c,d)
                    new = np.random.choice([c,d], p=[d-tmp,tmp-c])
                    mat.iloc[i,j] = new
        if mat.iloc[i].sum(axis = 0) == 0:
            # at least one cell
            arg_max = mat_orig.iloc[i].argmax()
            # print(f'Spot number {i} has one cell')
            mat.iloc[i,arg_max] = 1
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
def init_solution(cell_type_num, spot_idx, csr_st_exp, csr_sc_exp, meta_df, trans_id_idx, T_HALF):
    spot_i = -1
    picked_index = {}
    correlations = []
    sc_index = np.array(meta_df.index)
    meta_df = np.array(meta_df)
    picked_time = pd.DataFrame(np.zeros(len(sc_index)), index = sc_index)
    for spot_name in spot_idx:
        spot_i += 1
        prob = half_life_prob(t = picked_time[0].values,T = T_HALF)
        Es = csr_st_exp[spot_i]
        cor_st_sc = Es.dot(csr_sc_exp.T).toarray()[0]
        adj_cor = cor_st_sc * prob
        cor_tp = np.vstack((adj_cor, meta_df, sc_index, cor_st_sc)).T
        sort_cor_tp = cor_tp[cor_tp[:, 0].argsort()][::-1]
        w_i = cell_type_num.loc[spot_name]
        est_type = pd.DataFrame(w_i[w_i != 0])
        picked_index[spot_name] = []
        # print(spot_name)
        # print(sort_cor_tp)
        for index, row in est_type.iterrows():
            # print(f'{index},{row}')
            selected_idx = np.where(sort_cor_tp[:,1] == index)[0][0:int(row[spot_name])]
            # modify picked time
            selected_cell_id = list(sort_cor_tp[selected_idx][:,2])
            picked_time.loc[selected_cell_id] += 1
            picked_index[spot_name].extend(selected_cell_id)
        candi_idx = id_to_idx(trans_id_idx, picked_index[spot_name]) 
        # print(f'picked_index{picked_index[spot_name]}')
        # print(f'candi_idx{candi_idx}')
        agg_exp = csr_sc_exp[candi_idx].sum(axis = 0)
        cor = np.corrcoef(Es.toarray(),np.array(agg_exp))[0,1]
        correlations.append(cor)
        # break
    print(f'\t Init solution: max - {np.max(correlations):.4f}, \
    mean - {np.mean(correlations):.4f}, \
    min - {np.min(correlations):.4f}')
    picked_time.columns = ['count']
    return picked_index, correlations, picked_time


def eva_metric(arr1, arr2, metric = 'correlation'):
    # df1, df2: two array with same length
    # return the correlation between two dataframes
    if metric == 'correlation':
        val = np.corrcoef(arr1, arr2)[0,1]
    
    if metric == 'rmse':
        val = -1 * np.sqrt(np.mean((arr1 - arr2)**2))
    return val


@timeit
def reselect_cell(st_exp, spots_nn_lst, st_aff_profile_df, 
                  sc_exp, csr_sc_exp, sc_meta, trans_id_idx,
                  sum_sc_agg_exp, sc_agg_aff_profile_df, 
                  init_sc_df, init_picked_time, lr_df, p = 0.1,repeat_penalty = 10, metric = 'correlation'):
    '''
    Reselect cells from sc exp data for higher exp and interface correlation
    p: weight of interface correlation
    repeat_penalty: penalty for repeated selection of each cell, if set as 10, the prob of being selected will decrease by half after 10 times.

    No repeat
    Runtime: 20s for each spot with 10 cells; 2s each cell.

    '''
    tp_idx_dict = get_tp_idx_dict(sc_meta)
    new_spot_cell_dict = {}
    spot_idx_lst = list(st_exp.index)
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
        norm_s_exp = s_exp/np.std(s_exp)
        ########## SC ########## 
        s_sc_agg_sum = sum_sc_agg_exp.loc[spot]
        norm_s_sc_agg_sum = s_sc_agg_sum/np.std(s_sc_agg_sum)
        # generate baseline corr
        # TODO rmse add
        # max_exp_cor = np.corrcoef(norm_s_exp,norm_s_sc_agg_sum)[0][1]
        max_exp_cor = eva_metric(norm_s_exp,norm_s_sc_agg_sum)
        # print(f'Baseline cor of spot {spot} is {max_exp_cor}')
        ###### Interface ########
        nn_spot = spots_nn_lst[spot]
        a_ss = st_aff_profile_df.loc[(spot,nn_spot),:]
        # replace the no-aff cells
        sum_a_ss = a_ss.sum(axis = 1)
        sum_a_ss = sum_a_ss[sum_a_ss!=0]
        nn_spot = sum_a_ss.index.get_level_values(1).tolist()
        # reselect nn_spot
        a_ss = st_aff_profile_df.loc[(spot,nn_spot),:]
        a_cc = sc_agg_aff_profile_df.loc[(spot,nn_spot),:]
        spot_cell_lst = init_sc_df[init_sc_df['spot'] == spot]['sc_id'].tolist()
        # print(f'orig spot_cell_lst {spot_cell_lst}')
        if nn_spot == [] or a_cc.sum().sum() == 0 or p == 0:
            picked_time, spot_cell_lst, exp_cor = expSwap_SPROUT(spot_cell_lst, csr_sc_exp, sc_meta, trans_id_idx, tp_idx_dict,
                        s_exp, picked_time, gene_num, repeat_penalty, metric = metric)
            max_aff_cor = 0
            max_mix_cor = max_exp_cor 
            mix_corr = exp_cor
            inter_cor = 0
        else:
            max_aff_cor, max_mix_cor = cal_baseline_aff(a_ss, a_cc, max_exp_cor,p = p, metric=metric)
        # for each cell in spot
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
    result.index = range(len(result))
    result.index = result.index.map(str)
    correlations = result['exp_cor_after']
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
def cellReplaceByExp(spot_cell_lst, sc_exp, sc_meta, tp_idx_dict,
                        s_exp, picked_time,
                        repeat_penalty):
    '''
    Default mode, replace cell by highest exp and affinity correlation
    '''
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