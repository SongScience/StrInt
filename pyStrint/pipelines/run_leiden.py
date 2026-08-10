import sys
# sys.path.insert(1, '/public/wanwang6/5.Simpute/1.SCC/SpexMod/')
sys.path.insert(1, '/public/wanwang6/5.Simpute/1.SCC/SpexMod_v10/') # for E15 test, changed cell reselection
from src import sprout_plus
from src import prep_data as prep
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import sys
import os
import warnings
warnings.simplefilter('ignore')
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
#mpl.rcParams['font.family'] = 'arial'
from sklearn.metrics import adjusted_rand_score
import warnings
warnings.filterwarnings("ignore")
lr_dir = '/public/wanwang6/5.Simpute/SPROUT_fast/LR/'
import configargparse
parser = configargparse.ArgumentParser()
parser.add_argument('--configs', required=False, is_config_file=True)
parser.add_argument('-e', '--type-key', dest='tp_key', required=True, help='The colname of celltype in ST_meta')
parser.add_argument('-q', '--query-tp', dest='query_tp', required=True, help='The celltype to be analyzed')
parser.add_argument('-v', '--alter_exp', dest='alter_exp', required=True, help='ST')
parser.add_argument('-p', '--agg_meta', dest='agg_meta', required=True, help='ST meta')

parser.add_argument('-a', '--species', dest='species', required=True, default='human',help='If the species is human, default human')
parser.add_argument('-n', '--name', dest='name', required=False, help='Sample name which will be set as the prefix of output.SCC_ or SCC/')
parser.add_argument('-o', '--out-dir', dest='out_dir', required=False, help='Output file path')
args = parser.parse_args()


def read_csv_tsv(filename):
    if ('csv' in filename) or ('log' in filename):
        tmp = pd.read_csv(filename, sep = ',',header = 0,index_col=0)
    else:
        tmp = pd.read_csv(filename, sep = '\t',header = 0,index_col=0)
    return tmp
from scanpy._settings import settings
# set the random seed to a fixed value
settings.seed = 42


# st_file = '/public/wanwang6/5.Simpute/1.SCC/1.input/scc_new_ST.tsv'
# st_coord = '/public/wanwang6/5.Simpute/1.SCC/1.input/ST_coord.csv'
# species = 'human'
# name = '1.SCC/'
# orig_sc_file = '/public/wanwang6/5.Simpute/1.SCC/1.input/SCC_SC_exp.tsv'
# orig_meta_file = '/public/wanwang6/5.Simpute/1.SCC/1.input/SCC_SC_meta.tsv'
# orig_tp_key = 'level3_celltype'
# alter_exp = '/public/wanwang6/5.Simpute/4.compare_softwares/b.SpexMod/1.results/SCC_c2l/alter_sc_exp.tsv'
# agg_meta = '/public/wanwang6/5.Simpute/4.compare_softwares/b.SpexMod/1.results/SCC_c2l/new_spexmod_sc_meta.tsv'
# outDir = '/public/wanwang6/5.Simpute/4.compare_softwares/e.evaluation/2.files_res/'

tp_key = args.tp_key
# 1. outdir
outDir = args.out_dir + '/'
if not os.path.exists(outDir):
    os.makedirs(outDir)


def basic_ana(exp,meta,species,tp_key,nn = 20,npc=40,res = 1,min_dist=0.5,spread = 1,init_method = 0):
    init_pos = ['paga', 'spectral', 'random'][init_method]
    exp = exp.loc[meta.index]
    adata = prep.prep_adata(exp,meta,species)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    #sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca_variance_ratio(adata, log=True)
    sc.pp.neighbors(adata, n_neighbors=nn, n_pcs=npc)
    sc.tl.leiden(adata,resolution=res)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos=init_pos,min_dist = min_dist,spread = spread)
    sc.tl.leiden(adata,resolution=res)
    #sc.pl.umap(adata, color=['leiden', tp_key])
    return adata

    
def leiden_ana(adata,nn = 20,npc=40,res = 1,min_dist=0.5,spread = 1,init_method = 0):
    init_pos = ['paga', 'spectral', 'random'][init_method]
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=nn, n_pcs=npc)
    sc.tl.leiden(adata,resolution=res)
    sc.tl.paga(adata)
    sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
    sc.tl.umap(adata, init_pos=init_pos,min_dist = min_dist,spread = spread)
    sc.tl.leiden(adata,resolution=res)
    return adata




result = pd.DataFrame()
leiden_df = pd.DataFrame()
# nn越高群越少
# npc越高群越多
# res越高群越多 0.01-1
for nn in [10, 15, 20, 25]:
    for npc in [20, 40]:
        for res in [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 1]:
            for init_pos in [0, 1, 2]:
                for min_dist in [0.7]:
                    for spread in [1]:
                        print(nn, npc, res, init_pos, min_dist, spread)
                        try:
                            sc_alter = read_csv_tsv(args.alter_exp)
                            sc_agg_meta = read_csv_tsv(args.agg_meta)
                            sc_agg_meta = sc_agg_meta.dropna(axis=0, how='any')
                            if 'SCC' in args.agg_meta:
                                sc_agg_meta['adj_spex_UMAP2'] = sc_agg_meta['adj_spex_UMAP2'] * -1
                            sc_alter_adata = basic_ana(sc_alter, sc_agg_meta, args.species, tp_key, nn=nn, npc=npc,
                                                       res=res, min_dist=min_dist, spread=spread,
                                                       init_method=init_pos)

                            sc.pl.umap(sc_alter_adata, color=['leiden', 'celltype'], save=f'{args.name}_{nn}_{npc}_{res}_{init_pos}_{min_dist}_{spread}.pdf')
                            sc_alter_adata.obs['pivot'] = 1
                            tmp = pd.DataFrame(sc_alter_adata.obs['leiden'])
                            tmp.columns = [f'nn_{nn}_{npc}_{res}_{init_pos}_{min_dist}_{spread}']
                            leiden_df = pd.concat((leiden_df, tmp), axis=1)
                        except Exception as e:
                            print(f'Error occurred: {str(e)}')

leiden_df.to_csv(f'{outDir}/leiden_df.tsv', sep='\t', header=True, index=True)

# EVA 
# python /public/wanwang6/5.Simpute/4.compare_softwares/c.leiden/0.scripts/leiden.py args.alter_exp args.agg_meta args.species args.tp_key args.query_tp args.name args.out_dir
agg_meta = pd.read_csv(args.agg_meta, sep='\t', index_col=0, header=0)
leiden_fn = f'{outDir}/leiden_df.tsv'
leiden_fd = leiden_fn.rsplit('/',1)[0]
query_tp = args.query_tp
tp_key = args.tp_key
if query_tp not in agg_meta[tp_key].unique():
    print(f'Error: {query_tp} not in {tp_key}')
    sys.exit(1)

def cal_ari(orig_meta_file, pred_meta_file, tp_key='level3_celltype'):
    ari = adjusted_rand_score(orig_meta_file[tp_key], pred_meta_file)
    return ari


def add_tp_leiden_df(orig_meta, tp_key=tp_key , leiden_key='leiden', query_tp='CAF'):
    meta = orig_meta.copy()
    
    i = 0
    d = {}
    for k in meta[meta[tp_key] == query_tp][leiden_key].unique():
        d[k] = str(i)
        i += 1

    meta[f'{leiden_key}_map'] = None
    meta.loc[meta[tp_key] == query_tp, f'{leiden_key}_map'] = meta.loc[meta[tp_key] == query_tp, leiden_key].map(d)
    # print(meta[meta[tp_key ] == query_tp][leiden_key].map(d))
    for i in meta.index:
        if meta.loc[i,tp_key] == query_tp:
            meta.loc[i,f'{query_tp}_{leiden_key}'] = meta.loc[i,tp_key] + '_' + meta.loc[i,f'{leiden_key}_map']
        else:
            meta.loc[i,f'{query_tp}_{leiden_key}'] = meta.loc[i,tp_key]
    return meta

# 1. select max ARI
meta = agg_meta.copy()
df = pd.read_csv(leiden_fn, sep='\t', index_col=0, header=0)
df = pd.concat([df, meta], axis=1)
max_ari = 0
for i in df.columns:
    if 'nn' in i:
        df[i] = df[i].astype(str)
        # cal ARI
        ari = cal_ari(meta, df[i], tp_key = tp_key)
        l_caf = df[df[tp_key ] == query_tp].groupby(i).count()
        # ARI and cell type number of CAF
        if len(l_caf) >1:
            # print(i,ari, len(l_caf))
            if ari > max_ari:
                max_ari = ari
                max_col = i
print('max',max_col, max_ari)

# 2. save new meta
meta['leiden'] = df[max_col]
meta = add_tp_leiden_df(meta, tp_key=tp_key, leiden_key='leiden', query_tp=query_tp)
meta.to_csv(f'{leiden_fd}/spexmod_sc_meta_{query_tp}_leiden.tsv', sep='\t', index=True, header=True)

# 3. plot subset
if 'adj_spex_UMAP1' in meta.columns:
    size = 20
    cols = ['adj_spex_UMAP1','adj_spex_UMAP2']
    sns.scatterplot(data = df[df[tp_key]!=query_tp],x = cols[0],y=cols[1],hue = df[max_col],
                        s = size, linewidth= False, palette=['#ccc'])
    sns.scatterplot(data = df[(df[tp_key] == query_tp)],x = cols[0],y=cols[1],
                        s = size, linewidth= False,hue = df[max_col])
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.axis('equal')
    title = f'{leiden_fd}/leiden_{query_tp}.pdf'
    plt.savefig(title)

# import subprocess

# command = f"/home/grads/wanwang6/anaconda3/envs/scvi-env/bin/python /public/wanwang6/5.Simpute/4.compare_softwares/c.leiden/0.scripts/leiden_eva.py {args.agg_meta} {outDir}/leiden_df.tsv {args.query_tp} {args.tp_key}"
# result = subprocess.run(command, shell=True, capture_output=True, text=True)

# # Check the result
# if result.returncode == 0:
#     print("Command executed successfully.")
# else:
#     print("An error occurred while executing the command.")
#     print("Error message:", result.stderr)