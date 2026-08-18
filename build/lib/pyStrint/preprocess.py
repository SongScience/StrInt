import pickle
import numpy as np
import pandas as pd
import anndata
# import time
# import logging
import os


def read_csv_tsv(filename):
    if ('csv' in filename) or ('.log' in filename):
        tmp = pd.read_csv(filename, sep = ',',header = 0,index_col=0)
    else:
        tmp = pd.read_csv(filename, sep = '\t',header = 0,index_col=0)
    return tmp


def scale_sum(x,SUM):
    res = x.divide(x.sum(axis = 1),axis=0)
    return res*SUM


def load_lr_df(species = 'Human',lr_dir = None):
    if lr_dir:
        lr_df = read_csv_tsv(lr_dir)
    else:
        lr_path = os.path.dirname(os.path.realpath(__file__)) + '/LR/'
        # print(lr_path)
        if species in ['Human','Mouse']:
            # to lowercase
            species = species.lower()
            lr_df = pd.read_csv(f'{lr_path}/{species}_LR_pairs.txt',sep='\t',header=None)
        else:
            raise ValueError(f'Currently only support Human and Mouse, get {species}')
    return lr_df



def make_adata(mat, meta, species, save_path=None, save_adata=False):
    """
    Create AnnData object from expression matrix and metadata
    
    Parameters:
    -----------
    mat : pd.DataFrame
        Expression matrix, cells x genes
    meta : pd.DataFrame
        Cell metadata
    species : str
        Species name, 'Mouse' or 'Human'
    save_path : str, optional
        Path to save outputs
    save_adata : bool, default False
        Whether to save AnnData object
        
    Returns:
    --------
    adata : anndata.AnnData
        Processed AnnData object
    """
    
    # Work on copies to avoid modifying original data
    meta = meta.copy()
    mat = mat.copy()
    
    # Ensure indices are strings
    meta.index = meta.index.astype(str)
    mat.index = mat.index.astype(str)
    
    # Align matrix with metadata
    mat = mat.loc[meta.index]
    
    # Create AnnData object
    # adata = anndata.AnnData(X=mat.values, dtype=np.float32)
    adata = anndata.AnnData(X=mat.values)
    adata.obs = meta
    adata.obs_names = meta.index
    
    # Set gene information
    adata.var = pd.DataFrame(index=mat.columns)
    adata.var['symbol'] = mat.columns.tolist()
    adata.var_names = adata.var['symbol']
    
    # Handle mitochondrial genes
    mt_prefixes = {'Mouse': 'mt-', 'Human': 'MT-'}
    if species not in mt_prefixes:
        raise ValueError(f"Unsupported species: {species}. Supported: {list(mt_prefixes.keys())}")
    
    mt_prefix = mt_prefixes[species]
    adata.var['MT_gene'] = adata.var['symbol'].str.startswith(mt_prefix)
    
    # Store MT gene expression and remove from main matrix
    mt_mask = adata.var['MT_gene'].values
    if mt_mask.any():
        adata.obsm['MT'] = adata[:, mt_mask].X.toarray()
        adata = adata[:, ~mt_mask].copy()
    
    # Set species information
    adata.uns['species'] = species
    
    # Handle save path
    if save_path:
        save_path = os.path.abspath(save_path)
        os.makedirs(save_path, exist_ok=True)
        
        figpath = os.path.join(save_path, 'figures/')
        os.makedirs(figpath, exist_ok=True)
        
        adata.uns['save_path'] = save_path
        adata.uns['figpath'] = figpath
        
        if save_adata:
            adata_path = os.path.join(save_path, 'adata.h5ad')
            adata.write(adata_path)
            print(f"AnnData object saved to: {adata_path}")
    
    print(f"AnnData object created:")
    print(f"  - Cells: {adata.n_obs}")
    print(f"  - Genes: {adata.n_vars}")
    print(f"  - Species: {species}")
    if mt_mask.any():
        print(f"  - MT genes removed: {mt_mask.sum()}")
    
    return adata

# def make_adata(mat,meta,species,save_path = None, save_adata = False):
#     # mat: exp matrix, should be cells x genes
#     # index should be strictly set as strings
#     meta.index = meta.index.map(str)
#     mat.index = mat.index.map(str)
#     mat = mat.loc[meta.index]
#     adata = anndata.AnnData(mat,dtype=np.float32)
#     adata.obs = meta
#     adata.var = pd.DataFrame(mat.columns.tolist(), columns=['symbol'])
#     adata.var_names = adata.var['symbol'].copy()
#     #sc.pp.filter_cells(adata, min_genes=200)
#     #sc.pp.filter_genes(adata, min_cells=3)
#     # remove MT genes for spatial mapping (keeping their counts in the object)
#     if species == 'Mouse':
#         adata.var['MT_gene'] = [gene.startswith('mt-') for gene in adata.var['symbol']]
#     if species == 'Human':
#         adata.var['MT_gene'] = [gene.startswith('MT-') for gene in adata.var['symbol']]
#     adata.obsm['MT'] = adata[:, adata.var['MT_gene'].values].X.toarray()
#     adata = adata[:, ~adata.var['MT_gene'].values]
#     if save_path:
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
#         figpath = save_path + '/figures/'
#         if not os.path.exists(figpath):
#             os.makedirs(figpath)
#         adata.uns['figpath'] = figpath
#         adata.uns['save_path'] = save_path
#     if save_adata:
#         adata.write(f'{save_path}/adata.h5ad')
#     adata.uns['species'] = species
#     return adata



def data_clean(sc_exp, st_exp):
    # cell x genes
    # 1. remove unexpressed genes
    filtered_sc = sc_exp.loc[:,(sc_exp != 0).any(axis=0)]
    filtered_st = st_exp.loc[:,(st_exp != 0).any(axis=0)]
    st_gene = set(filtered_st.columns)
    sc_gene = set(filtered_sc.columns)
    shared_genes = list(st_gene.intersection(sc_gene))
    filtered_sc1 = filtered_sc.loc[:,shared_genes]
    filtered_st1 = filtered_st.loc[:,shared_genes]
    return filtered_sc1, filtered_st1 


def denoise_genes(sc_exp, st_exp, sc_distribution,species):
    sc_genes = sc_exp.columns.tolist()
    st_genes = st_exp.columns.tolist()
    genes = list(set(sc_genes).intersection(set(st_genes)))
    genes = list(set(genes).intersection(set(sc_distribution.columns)))

    if species == 'Mouse':
        mt = [gene for gene in genes if gene.startswith('mt-')]
    if species == 'Human':
        mt = [gene for gene in genes if gene.startswith('MT-')]
    genes = list(set(genes).difference(set(mt)))
    genes.sort()
    return genes


def subset_inter(st_exp, sc_exp):
    '''
    subset df by the intersection genes between st and sc
    '''
    genes = list(set(st_exp.columns).intersection(set(sc_exp.columns)))
    st_exp = st_exp[genes]
    sc_exp = sc_exp[genes]
    return st_exp, sc_exp


def prep_all_adata(sc_exp = None, st_exp = None, sc_distribution = None, 
                   sc_meta = None, st_coord = None, lr_df = None, SP = 'Human'):
    '''
    1. remove unexpressed genes
    2. select shared genes
    3. transform to adata format
    '''
    # scale all genes to [0,10]
    # v5 
    # SUM = st_exp.sum(axis = 1).mean()
    # v6 from st sum to 1e4
    if (SP != 'Human') and (SP != 'Mouse'):
        raise ValueError(
            f'Species should be choose among either Human or Mouse.')
    if lr_df is None:
        lr_df = load_lr_df(species = SP)
    SUM = 1e4
    # Data Clean
    sc_exp, st_exp = data_clean(sc_exp, st_exp)
    genes = denoise_genes(sc_exp, st_exp, sc_distribution, SP)
    sc_exp = sc_exp[genes]
    st_exp = st_exp[genes]
    sc_distribution = sc_distribution[genes]
    # print(genes)
    # print(lr_df)
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # Adata Preparation
    # 1. SC to adata
    scale_sc_exp = scale_sum(sc_exp,SUM)
    sc_adata = make_adata(scale_sc_exp,sc_meta,SP)
    # 2. ST to adata
    scale_st_exp = scale_sum(st_exp,SUM)
    st_adata = make_adata(scale_st_exp,st_coord,SP)
    # 3. distribution to adata
    sc_ref = scale_sum(sc_distribution,SUM)
    # v6 canceled ref adata
    # sc_ref = prep_adata(scale_poisson_spot,sc_ref_meta,SP)
    if sc_adata.shape[1] == st_adata.shape[1] and st_adata.shape[1] == sc_ref.shape[1]:
        print(f'Data clean is done! Using {st_adata.shape[1]} shared genes .')
    return sc_adata, st_adata, sc_ref, lr_df



def prep_all_adata_merfish(sc_exp = None, st_exp = None, sc_distribution = None, 
                   sc_meta = None, st_coord = None, lr_df = None, SP = 'human'):
    '''
    1. remove unexpressed genes
    2. align genes with sc
    3. transform to adata format
    '''
    # scale all genes to [0,10]
    # v5 
    # SUM = st_exp.sum(axis = 1).mean()
    # v6 from st sum to 1e4
    if (SP != 'Human') and (SP != 'Mouse'):
        raise ValueError(
            f'Species should be choose among either human or mouse.')
    SUM = 1e4
    # Data Clean
    filtered_sc = sc_exp.loc[:,(sc_exp != 0).any(axis=0)]
    genes = list(set(filtered_sc.columns).intersection(set(sc_distribution.columns)))
    # Align genes to SC
    sc_exp = sc_exp[genes]
    # st_exp = st_exp.reindex(genes, axis=1)
    lr_df = lr_df[lr_df[0].isin(genes) & lr_df[1].isin(genes)]
    # Adata Preparation
    # 1. SC to adata
    scale_sc_exp = scale_sum(sc_exp,SUM)
    sc_adata = make_adata(scale_sc_exp,sc_meta,SP)
    # 2. ST to adata
    # 1e5 is too large for merfish data, merfish only has 200~500 genes, gene sum is around 1e3.
    st_gene_sum = int(st_exp.sum(axis = 1).mean())
    st_genes = list(set(st_exp.columns).intersection(set(genes)))
    st_exp = st_exp[st_genes]
    scale_st_exp = scale_sum(st_exp,st_gene_sum)
    st_adata = make_adata(scale_st_exp,st_coord,SP)
    # 3. distribution to adata
    # sc_adata filtered mt genes in prep_adata
    sc_distribution = sc_distribution[sc_adata.var_names.tolist()]
    sc_ref = scale_sum(sc_distribution,SUM)
    # v6 canceled ref adata
    # sc_ref = prep_adata(scale_poisson_spot,sc_ref_meta,SP)
    if sc_adata.shape[1] == sc_ref.shape[1]:
        print(f'Data clean and scale are done! Single-cell data has {sc_adata.shape[1]} genes, spatial data has {st_adata.shape[1]} genes.')
    return sc_adata, st_adata, sc_ref, lr_df




def lr2kegg(lri_df, use_lig_gene = True, use_rec_gene = True):
    '''
    Use both ligand and receptor
    '''
    if use_lig_gene:
        a = lri_df[['ligand','lr_co_exp_num','lr_co_ratio_pvalue']]
        a.columns = ['gene','lr_co_exp_num','lr_co_ratio_pvalue']
    else:
        a = pd.DataFrame(columns = ['gene','lr_co_exp_num','lr_co_ratio_pvalue'])

    if use_rec_gene:
        b = lri_df[['receptor','lr_co_exp_num','lr_co_ratio_pvalue']]
        b.columns = ['gene','lr_co_exp_num','lr_co_ratio_pvalue']
    else:
        b = pd.DataFrame(columns = ['gene','lr_co_exp_num','lr_co_ratio_pvalue'])
    c = pd.concat((a,b))
    c = c.groupby('gene').mean().reset_index()
    return c


def filter_kegg(df, pval_thred = 0.05):
    tmp = df.copy()
    tmp = tmp[tmp['pvalue'] < pval_thred].copy()
    tmp['-log10 pvalue'] = np.log10(tmp['pvalue']) * (-1)
    tmp[['tmp1','tmp2']] = tmp['GeneRatio'].str.split('/',expand=True)
    tmp['GeneRatio'] = tmp['tmp1'].astype(int) / tmp['tmp2'].astype(int)
    tmp['Count'] = tmp['Count'].astype(int)
    tmp['-log10 pvalue'] = tmp['-log10 pvalue'].astype(float)
    tmp = tmp.sort_values('GeneRatio',ascending=False)
    # remove suffix
    tmp['Description'] = tmp['Description'].str.split(' - ', expand=True)[0]
    return tmp



def auto_tune_parameters(ST_SC, LR):
    """
    Description:
        Adjusts and calculates four parameters (p1, p2, p3, p4) based on two input values: 
        ST_SC (scaling factor for p2 and p3) and LR (scaling factor for p1 and p4). 
        The calculations are constrained by baseline values, ratios, and maximum/minimum bounds.

    Input:
        ST_SC: Expected cell number divided by the cell number of scRNA-seq reference.
                More cell in the reference, add weight on p3.
        LR: The scaling factor for p1 and p4.

    Output:
        p1: A fraction (1/3) of the calculated p14.
        p2: A value adjusted based on ST_SC and constrained by minimum and maximum boundaries.
        p3: The remaining value after subtracting p14 and p2 from 1.
        p4: A fraction (2/3) of the calculated p14.
    """
    import numpy as np

    # --- Define baseline values and constraints ---
    baseline_p14 = 0.15        # Baseline value for p1 and p4
    baseline_lr = 0.5          # Baseline value for learning rate (LR)
    target_p14_max = 0.15      # Maximum allowable value for p1 and p4

    # --- Calculate p1 and p4 based on LR ---
    # Scale p14 using the ratio of baseline_p14 to baseline_lr
    scale_p14 = baseline_p14 / baseline_lr
    p14 = min(scale_p14 * LR, target_p14_max)  # Ensure p14 does not exceed the target maximum value

    # --- Split p14 into p1 and p4 ---
    # p1 is 1/3 of p14, and p4 is 2/3 of p14
    p1 = p14 * (1 / 3)
    p4 = p14 * (2 / 3)

    # --- Calculate the remaining value after assigning p14 ---
    remaining = 1 - p14

    # --- Define baseline values and constraints for ST_SC and p2 ---
    baseline_ST_SC = 1         # Baseline value for ST_SC
    baseline_p2 = 0.65         # Baseline value for p2
    min_p2 = 0.65              # Minimum allowable value for p2

    # --- Define constraints for p2 ---
    # p3 must be at least 0.15, or p2 and p3 must maintain a 1:4 ratio
    max_p2 = min(0.8 * remaining, remaining - 0.15)

    # --- Compute the unbounded new value for p2 based on the scaling factor ---
    # Scale p2 using the ratio of baseline_p2 to baseline_ST_SC, adjusted by ST_SC
    new_p2_unbounded = (baseline_p2 / baseline_ST_SC) * ST_SC

    # Ensure p2 does not exceed the maximum allowable value and meets the minimum requirements
    new_p2_unbounded = min(max_p2, new_p2_unbounded)
    p2 = max(new_p2_unbounded, min_p2)

    # --- Assign the remaining value to p3 ---
    p3 = remaining - p2

    # --- Return the calculated parameter values ---
    return np.round(p1,2), np.round(p2,2), np.round(p3,2), np.round(p4,2)