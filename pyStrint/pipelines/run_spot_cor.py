import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
# mpl.rcParams['font.family'] = 'arial'
# import sys
# sys.path.insert(1, '/public/wanwang6/5.Simpute/1.SCC/SpexMod/')
# sys.path.insert(1, '/public/wanwang6/5.Simpute/4.compare_softwares/e.evaluation/0.scripts')
import configargparse
parser = configargparse.ArgumentParser()
parser.add_argument('--configs', required=False, is_config_file=True)
parser.add_argument('-s', '--st-file', dest='inDir', required=True, help='dir name, under this dir must have ./spex folder which contains alter_sc_exp.tsv')
parser.add_argument('-c', '--st-meta', dest='agg_meta', required=False, help='Spatial coordinates of the spatial transcriptomics data')

parser.add_argument('-p', '--tp_key', dest='tp_key', required=True, help='Cell type key in sc_agg_meta')
parser.add_argument('-n', '--name', dest='name', required=False, help='Sample name which will be set as the prefix of output.SCC_ or SCC/')

parser.add_argument('-o', '--out-dir', dest='out_dir', required=False, help='Output file path')
parser.add_argument('-b', '--orig-sc-file', dest='orig_sc_file', required=True, help='Single-cell candidate library of the corresponding ST tissue')
parser.add_argument('-d', '--sc_meta', dest='orig_meta_file', required=False, help='Cell selection')
parser.add_argument('-e', '--orig-type-key', dest='orig_tp_key', required=False, help='The colname of celltype in sc_orig_meta')

parser.add_argument('-a', '--species', dest='species', required=True, default='human',help='If the species is human, default human')
parser.add_argument('-t', '--orig-st-file', dest='orig_st_file', required=True, help='Single-cell candidate library of the corresponding ST tissue')


args = parser.parse_args()

def read_csv_tsv(filename):
    if 'csv' in filename:
        tmp = pd.read_csv(filename, sep = ',',header = 0,index_col=0)
    else:
        tmp = pd.read_csv(filename, sep = '\t',header = 0,index_col=0)
    return tmp


tp_key = args.tp_key
outDir = args.out_dir
############### Functions ###############
def scale_sum(x,SUM):
    res = x.divide(x.sum(axis = 1),axis=0)
    return res*SUM


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


def pear(D,D_re):
    D = np.array(D)
    D_re = np.array(D_re)
    tmp = np.corrcoef(D.flatten(order='C'), D_re.flatten(order='C'))
    return tmp[0,1] 


def cal_spot_sum(exp, st_exp, sc_meta, key):
    exp['spot'] = sc_meta[key].values
    sc_spot_sum = exp.groupby(key).sum()
    del exp[key]
    sc_spot_sum = sc_spot_sum.loc[st_exp.index]
    return sc_spot_sum


def cal_each_spot_cor(st_exp,sc_spot_sum,inter_genes):
    st_exp = st_exp[inter_genes]
    sc_spot_sum = sc_spot_sum[inter_genes]
    cor_lst = []
    for i in range(st_exp.shape[0]):
        cor = pear(st_exp.iloc[i],sc_spot_sum.iloc[i])
        cor_lst.append(cor)
    return cor_lst



def exp_cor_spot(alter_sc_exp,st_exp,sc_meta,species):
    SUM = 1e4
    result = pd.DataFrame()
    alter_sc_exp, st_exp_clean = data_clean(alter_sc_exp, st_exp)
    inter_genes = denoise_genes(alter_sc_exp, st_exp_clean, st_exp_clean, species)
    st_exp_clean = st_exp_clean[inter_genes]
    alter_sc_exp = alter_sc_exp[inter_genes]
    meta_spots = set(sc_meta['spot'].unique())
    inter_spots = np.sort(list(set(st_exp_clean.index).intersection(meta_spots)))
    st_exp_clean = st_exp_clean.loc[inter_spots]
    st_exp_clean = scale_sum(st_exp_clean,SUM)
    alter_sc_exp = scale_sum(alter_sc_exp,SUM)
    sc_spot_sum = cal_spot_sum(alter_sc_exp, st_exp_clean, sc_meta, 'spot')
    # sc_spot_sum = sc_spot_sum.loc[inter_spots]
    cor_lst_alter = cal_each_spot_cor(st_exp_clean,sc_spot_sum,inter_genes)
    tmp = pd.DataFrame(cor_lst_alter,columns = ['cor'],index = inter_spots)
    result = pd.concat((result,tmp),axis = 0)
    return result


def exp_cor_spot_raw(alter_sc_exp,st_exp,sc_meta,species):
    result = pd.DataFrame()
    alter_sc_exp, st_exp_clean = data_clean(alter_sc_exp, st_exp)
    inter_genes = denoise_genes(alter_sc_exp, st_exp_clean, st_exp_clean, species)
    st_exp_clean = st_exp_clean[inter_genes]
    alter_sc_exp = alter_sc_exp[inter_genes]
    meta_spots = set(sc_meta['spot'].unique())
    inter_spots = np.sort(list(set(st_exp_clean.index).intersection(meta_spots)))
    st_exp_clean = st_exp_clean.loc[inter_spots]
    sc_spot_sum = cal_spot_sum(alter_sc_exp, st_exp_clean, sc_meta, 'spot')
    cor_lst_alter = cal_each_spot_cor(st_exp_clean,sc_spot_sum,inter_genes)
    tmp = pd.DataFrame(cor_lst_alter,columns = ['cor'],index = inter_spots)
    result = pd.concat((result,tmp),axis = 0)
    return result

############## Load data ##############
st_exp = read_csv_tsv(args.orig_st_file)
sc_exp = read_csv_tsv(args.orig_sc_file)
# inDir = args.st_file.rsplit('/', 1)[0]
alter_exp_dir = args.inDir
cell_id = 'sc_id'
sc_agg_meta = read_csv_tsv(args.agg_meta)
if 'tsv' in alter_exp_dir or 'csv' in alter_exp_dir:
    cyto_alter_after = read_csv_tsv(alter_exp_dir)
else:
    print('No exp file, Pull cells from original sc_exp')
    cyto_alter_after = sc_exp.loc[sc_agg_meta[cell_id]]
    cyto_alter_after.index = sc_agg_meta.index

cyto_alter_after_cor = exp_cor_spot(cyto_alter_after,st_exp,sc_agg_meta,args.species)
cyto_alter_after_cor_raw = exp_cor_spot_raw(cyto_alter_after,st_exp,sc_agg_meta,args.species)
###########################################
# cyto_alter_after_cor_gene = exp_cor_gene(cyto_alter_after, st_exp, sc_agg_meta, args.species)
# cyto_alter_after_cor_raw_gene = exp_cor_gene_raw(cyto_alter_after, st_exp, sc_agg_meta, args.species)

cyto_alter_after_cor_raw.to_csv(f'{outDir}spot_cor_raw.tsv',sep = '\t',header=True,index=True)
cyto_alter_after_cor.to_csv(f'{outDir}spot_cor_scale.tsv',sep = '\t',header=True,index=True)
# cyto_alter_after_cor_raw_gene.to_csv(f'{outDir}gene_cor_raw.tsv',sep = '\t',header=True,index=True)
# cyto_alter_after_cor_gene.to_csv(f'{outDir}gene_cor_scale.tsv',sep = '\t',header=True,index=True)
