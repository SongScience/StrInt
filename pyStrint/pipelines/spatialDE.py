# /home/grads/wanwang6/anaconda3/envs/scvi-env/bin/python
import numpy as np
import pandas as pd
import NaiveDE
import SpatialDE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--st-file', dest='st_file', required=True, help='Spatial transcriptomics data')
parser.add_argument('-c', '--st-coordinate', dest='st_coord', required=True, help='Spatial coordinates of the spatial transcriptomics data')
parser.add_argument('-n', '--name', dest='name', required=False, help='Sample name which will be set as the prefix of output')
parser.add_argument('-k', '--cluster', dest='n_cluster', required=True, help='Cluster number')
parser.add_argument('-q', '--tp-list-to-group', dest='tp_lst', required=False, help='list of types')
parser.add_argument('-o', '--out-dir', dest='out_dir', required=False, help='Output file path')
args = parser.parse_args()

def read_csv_tsv(filename):
    if 'csv' in filename:
        tmp = pd.read_csv(filename, sep = ',',header = 0,index_col=0)
    else:
        tmp = pd.read_csv(filename, sep = '\t',header = 0,index_col=0)
    return tmp
###################################
############ read file ############
###################################
least_spot_num = 10
gene_num = 3
st_exp = read_csv_tsv(args.st_file)
st_coord_raw = read_csv_tsv(args.st_coord)

# 1. outdir
if args.out_dir is not None:
    outDir = args.out_dir + '/' 
    if not os.path.exists(outDir):
        os.makedirs(outDir)
else:
    outDir = os.getcwd() + '/'
# 2.sample name as prefix
if args.name is not None:
    outDir = f'{outDir}/{args.name}'
    if outDir[-1] == '/':
        # name end with / creat new dir
        if not os.path.exists(outDir):
            os.makedirs(outDir)


if st_coord_raw.shape[1] == 2:
    st_coord = st_coord_raw.copy()
elif st_coord_raw.shape[1] == 1:
    st_coord = pd.read_csv(args.st_coord, sep = ',',header = None,index_col= None)
elif st_coord_raw.shape[1] > 2:
    # larger than 2 => meta
    if 'sc_id' in st_coord_raw.columns.tolist():
        if 'adj_spex_UMAP1' in st_coord_raw.columns:
            st_coord = st_coord_raw[['adj_spex_UMAP1','adj_spex_UMAP2']].copy()
            print('spex')
        else:
            st_coord = st_coord_raw[['adj_UMAP1','adj_UMAP2']].copy()
            print('sprout')
    else:
        # spatalk
        st_coord = st_coord_raw[['x','y']].copy()


if args.tp_lst is not None:
    tp_args = args.tp_lst.split(',')
    if len(tp_args)>1:
        # multiple cluster tryout
        tp_lst = [str(x) for x in tp_args]
    else:
        tp_lst = [str(args.tp_lst)]
    print(tp_lst)
st_coord.columns = ['x','y']
###################################
############ DE finder ############
###################################
st_coord['total_counts'] = st_exp.sum(axis = 1)
norm_expr = NaiveDE.stabilize(st_exp.T).T
resid_expr = NaiveDE.regress_out(st_coord, norm_expr.T, 'np.log(total_counts)').T

X = st_coord[['x', 'y']]
results = SpatialDE.run(X, resid_expr)
results = results.sort_values('qval')
results = results.sort_values('FSV',ascending=False)
sign_results = results.query('qval <0.001')
sign_results.to_csv(f'{outDir}spaDE_sigres.tsv',sep = '\t',header=True,index=True)
# check if this gene expressed in at least 10 spots
tmp = pd.DataFrame(np.sum(st_exp >1))
sign_results = sign_results[sign_results['g'].isin(tmp[tmp[0]>least_spot_num].index)]
sign_results = sign_results.sort_values('FSV',ascending=False)
sign_results[['g', 'l', 'qval']].to_csv(f'{outDir}spaDE_filtered_lst.tsv',sep = '\t',header=True,index=False)

###################################
############  pattern  ############
###################################
k_args = args.n_cluster.split(',')
if len(k_args)>1:
    # multiple cluster tryout
    clusters = [int(x) for x in k_args]
else:
    clusters = [int(args.n_cluster)]
print(clusters)
# sign_results = results.query('qval < 0.001')
for n_clusters in clusters:
    histology_results, patterns = SpatialDE.aeh.spatial_patterns(X, resid_expr, sign_results, C=n_clusters, l=1.8, verbosity=1)
    histology_results.columns = ['g',f'pattern_{n_clusters}',f'membership_{n_clusters}']
    patterns.to_csv(f'{outDir}patterns_k_{n_clusters}.tsv',sep = '\t',header=True,index=False)
    sign_results = pd.concat((sign_results,histology_results.iloc[:,1:]),axis = 1)
    sign_results.to_csv(f'{outDir}histology_k_{n_clusters}.tsv',sep = '\t',header=True,index=False)
    
    top_n_marker_df = pd.DataFrame()
    gene_lst = []
    for group in sign_results[f'pattern_{n_clusters}'].unique():
        print(group)
        tmp = sign_results[sign_results[f'pattern_{n_clusters}'] == group]
        if len(tmp) > gene_num:
            top_n_marker_df = pd.concat((top_n_marker_df,tmp.iloc[:gene_num]))
            gene = list(tmp.iloc[:3]['g'])
            gene_lst.extend(gene)
    top_n_marker_df.to_csv(f'{outDir}/top_n_k_{n_clusters}.tsv',sep = '\t',header=True,index=False)
    ###### gene exp plot ######
    i = 0
    ROW = len(sign_results[f'pattern_{n_clusters}'].unique())
    COL = 3
    ROW_L = 4
    COL_L = 6
    plt.figure(figsize=(COL_L*COL, ROW_L* ROW))
    for gene in gene_lst:
        plt.subplot(ROW, COL, i + 1)
        plt.scatter(st_coord['x'], st_coord['y'], c=norm_expr[gene],s = 15);
        plt.title(gene)
        plt.axis('equal')
        plt.colorbar(ticks=[]);
        i+=1
    plt.savefig(f'{outDir}/top_exp_k_{n_clusters}.pdf')
    ##### clustering pattern plot #######
    plt.figure(figsize=(20, 3))
    for i in range(n_clusters):
        plt.subplot(1, n_clusters, i + 1)
        plt.scatter(st_coord['x'], st_coord['y'], c=patterns[i],s = 15);
        plt.axis('equal')
        plt.title('Pattern {} - {} genes'.format(i, histology_results.query(f'pattern_{n_clusters} == @i').shape[0] ))
        plt.colorbar(ticks=[])
    plt.savefig(f'{outDir}/histology_k_{n_clusters}.pdf')
###################################
# # TODO 把spaGCN的改成spaDE的
# if args.tp_lst is not None:
#     for group in tp_lst:
#         subset_meta = st_coord[st_coord['celltype'] == group]
#         subset_st = st_exp.loc[subset_meta.index]
#         subset_coord = st_coord.loc[subset_meta.index]
#         adata = prep_adata(subset_st,subset_coord,sp)
#         spagcn(adata,l_lst,clusters,subset_meta,tp = f'{group}_')
# else:
#     adata = prep_adata(st_exp,st_coord,sp)
#     spagcn(adata,l_lst,clusters,st_coord,tp = '')