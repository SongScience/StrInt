import pickle
import numpy as np
import pandas as pd
# import time
# import logging
import os


def scale_01(x,a,b):
    MAX = np.max(x)
    MIN = np.min(x)
    res = (b-a)*(x-MIN)/(MAX-MIN)+a
    return res


def scale_global_MIN_MAX(df,MIN,MAX):
    df_max = df.max().max()
    df_min = df.min().min()
    df_01 = (df - df_min)/(df_max - df_min)
    res = df_01*(MAX - MIN) - MIN
    return res
    
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    pkl_file = open(filename, 'rb')
    data = pickle.load(pkl_file)
    return data


def check_index_str(st_adata, sc_adata, sc_ref,weight):
    if st_adata.obs.index.dtype != 'object':
        st_adata.obs.index = st_adata.obs.index.map(str)
        print(f'The spot/cell index of st_adata is not str, changed to str for consistency.')
    if sc_adata.obs.index.dtype != 'object':
        sc_adata.obs.index = sc_adata.obs.index.map(str)
        print(f'The cell index of sc_adata is not str, changed to str for consistency.')
    if sc_ref.index.dtype != 'object':
        sc_ref.index = sc_ref.index.map(str)
        print(f'The cell index of sc_ref is not str, changed to str for consistency.')
    if weight.index.dtype != 'object':
        weight.index = weight.index.map(str)
        print(f'The cell index of sc_ref is not str, changed to str for consistency.')    
    return st_adata, sc_adata, sc_ref, weight


def check_spots(st_adata,weight):
    # st, decon spot match
    if (weight.index == st_adata.obs.index).all():
        pass
    elif len(set(weight.index).intersection(set(st_adata.obs.index))) == 0:
        raise ValueError('No spot intersection found between st_adata and weight file.')
    else:
        # shared_idx = list(set(weight.index).intersection(set(st_adata.obs.index)))
        shared_idx = st_adata.obs.index.isin(weight.index)
        st_adata = st_adata[shared_idx,:]
        weight = weight.loc[shared_idx]
        weight = weight.reindex(st_adata.obs.index)
        print(f'Spot index in weight matrix is different from ST expression\'s.\n Adjusted to {len(shared_idx)} shared spots.')
    return st_adata, weight


def check_sc_st_spot(sc_adata,st_adata):
    spots = st_adata.obs.index
    sc_spots = sc_adata.obs['spot']
    inter_spots = set(spots).intersection(set(sc_spots))
    if len(inter_spots) == 0: 
        raise ValueError(f'st_adata has {len(set(spots))} spots/cells, {len(inter_spots)} found in sc_adata.obs.spot.')
    elif len(inter_spots) != len(spots):
        print(f'st_adata has {len(set(spots))} spots/cells, {len(inter_spots)} found in sc_adata.obs.spot, subset to intersect spots/cells.')
        st_adata = st_adata[st_adata.obs.index.isin(inter_spots), :]
        idx = sc_adata.obs[sc_adata.obs['spot'].isin(inter_spots)].index
        sc_adata = sc_adata[idx,:]
    return sc_adata,st_adata


def check_st_coord(st_adata):
    st_meta = st_adata.obs.copy()
    if 'x' and 'y' in st_meta.columns:
        st_coord = st_meta[['x','y']]
    elif 'row' and 'col' in st_meta.columns:
        st_coord = st_meta[['row','col']]
    else:
        raise ValueError(
            f'st_adata expected to have two columns either name x y or row col to represent spatial coordinates, but got None in {st_meta.shape[1]} columns.')
    st_coord.columns = ['x','y']
    return st_coord


def check_empty_dict(mydict):
    if not any(mydict.values()):
        raise ValueError(
            "No cell has neighbor, check parameter st_tp")



def align_lr_gene(self):
    lr_df = self.lr_df
    genes = list(self.sc_adata.var.index)
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    return lr_df

def check_sc_meta_col(sc_adata):
    # TODO for other software input
    '''
    sc_meta should have 'sc_id' and 'spot' columns
    if have x and y return false for running embedding as input coordinates
    '''
    col = sc_adata.obs.columns
    if 'sc_id' not in col:
        raise ValueError(f"Expected [sc_id] as cell index column in sc_adata.obs, not found.")
    if 'spot' not in col:
        raise ValueError(f"Expected [spot] as cell's spot column in sc_adata.obs, not found.")
    if 'celltype' not in col:
        raise ValueError(f"Expected [celltype] as cell-type column in sc_adata.obs, not found.")

    if sc_adata.obs['sc_id'].dtype != 'object':
        sc_adata.obs['sc_id'] = sc_adata.obs['sc_id'].astype('str')
        print(f'The sc_id in sc_adata.obs.sc_id is not str, changed to str for consistency.')
    if sc_adata.obs['spot'].dtype != 'object':
        sc_adata.obs['spot'] = sc_adata.obs['spot'].map(str)
        print(f'The spot/cell index in sc_adata.obs.spot is not str, changed to str for consistency.')
    return sc_adata


def check_celltype(sc_adata):
    '''
    sc_meta should have 'celltype' columns
    '''
    col = sc_adata.obs.columns
    if 'celltype' not in col:
        raise ValueError(f"Expected celltype as cell-type column in obs, not found.")


def check_sc_coord(init_sc_embed):
    if not isinstance(init_sc_embed, pd.DataFrame):
        init_sc_embed = pd.DataFrame(init_sc_embed)
    
    if init_sc_embed.shape[1] == 2:
        sc_coord = init_sc_embed
    else:
        # subset only coordinates and rename
        if 'x' and 'y' in init_sc_embed.columns:
            sc_coord = init_sc_embed[['x','y']]
        elif 'row' and 'col' in init_sc_embed.columns:
            sc_coord = init_sc_embed[['row','col']]
        else:
            raise ValueError(
                f'st_adata expected two columns either name x y or row col to represent spatial coordinates, but got None in {init_sc_embed.shape[1]} columns.')
    sc_coord.columns = ['x','y']
    return sc_coord

def check_st_tp(st_tp):
    if (st_tp != 'visium') and (st_tp != 'st') and (st_tp != 'slide-seq'):
        raise ValueError(
            f'st_tp should be choose among either visium or st or slide-seq, get {st_tp}')


def check_st_sc_pair(st_adata, sc_adata):
    if len(set(st_adata.var.index).intersection(set(sc_adata.var.index)))<10:
        # st_exp.columns = map(lambda x: str(x).upper(), st_exp.columns)
        # sc_exp.columns = map(lambda x: str(x).upper(), sc_exp.columns)
        raise ValueError(
            f'The shared gene of ST and SC expression data is less than 10, check if they are of the same species.')
    

def check_sc(sc_adata, sc_ref):
    '''
    Check if sc_ref and sc_adata has zero row sum, if so remove them from sc_ref and sc_adata
    Check if sc_ref and sc_adata has same genes, if not remove them from sc_ref and sc_adata
    '''
    if (sc_ref.sum(axis=1) == 0).any():
        idx = sc_ref[sc_ref.sum(axis=1) == 0].index
        sc_ref = sc_ref.loc[~(sc_ref.sum(axis=1) == 0), :]
        sc_adata = sc_adata[sc_ref.index, :]
        print(f'Zero row sum found in sc_ref, {idx.tolist()} are therefore removed.')
    if (sc_adata.X.sum(axis = 1) == 0).any():
        boolean = sc_adata.X.sum(axis = 1)
        nonzeros = boolean != 0
        idx = sc_adata[~nonzeros].obs.index
        sc_adata = sc_adata[nonzeros, :]
        sc_ref = sc_ref.loc[sc_adata.obs.index, :]
        print(f'Zero row sum found in sc_adata, {idx.tolist()} are therefore removed.')
    if not sc_ref.columns.equals(sc_adata.var_names.to_list()):
        genes = set(sc_adata.var_names).intersection(set(sc_ref.columns))
        if len(genes) == 0:
            raise ValueError(
                f'sc_ref and sc_adata has different genes, no shared genes found.')
        else:
            sc_ref = sc_ref[list(genes)]
            sc_adata = sc_adata[:,sc_ref.columns]
        print(f'sc_ref and sc_adata has different genes, both data are subset to {sc_adata.shape[1]} genes.')
    return sc_adata, sc_ref

########## preprocessing ###################
def pear(D,D_re):
    tmp = np.corrcoef(D.flatten(order='C'), D_re.flatten(order='C'))
    return tmp[0,1] 


def check_decon_type(weight, sc_adata, cell_type_key):
    if len(set(weight.columns).intersection(set(sc_adata.obs[cell_type_key]))) != len(set(weight.columns)):
        raise ValueError(
            f'Cell type in weight matrix is different from single-cell meta file.')
    if 'celltype' not in sc_adata.obs.columns:
        sc_adata.obs['celltype'] = sc_adata.obs[cell_type_key]
        print(f'celltype column not found in sc_adata.obs, added from {cell_type_key}.')


def check_weight_sum_to_one(matrix):
    # check if the gene sum is
    check = False
    row = matrix.shape[0]
    if np.sum(np.sum(matrix,axis = 1)) == row:
        check = True
    return check


def check_decon_sum(weight):
    if check_weight_sum_to_one(weight):
        return weight
    else:
        print(f'Deconvolution matrix does not sum to 1, normlized.')
        df_normalized = weight.div(weight.sum(axis=1), axis=0)
        return df_normalized
    



######## Utility functions ########
def find_executable_in_path(executable_name):
    """Find the full path of an executable in the system PATH"""
    path_dirs = os.environ.get('PATH', '').split(':')
    for path_dir in path_dirs:
        if path_dir:  # Skip empty paths
            executable_path = os.path.join(path_dir, executable_name)
            if os.path.exists(executable_path) and os.access(executable_path, os.X_OK):
                return executable_path
    return None
