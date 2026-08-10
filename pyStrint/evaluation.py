import subprocess
import os
import glob
import pandas as pd
import numpy as np
from . import preprocess as pp


def load_spatalk(result, pvalue_thred = 0.005, tp_map = None):
    if tp_map:
        result['celltype_sender'] = result['celltype_sender'].map(tp_map)
        result['celltype_receiver'] = result['celltype_receiver'].map(tp_map)
    result = result[result['lr_co_ratio_pvalue'] < pvalue_thred].copy()
    # print(result.head(5))
    result[['ligand', 'receptor', 'celltype_sender', 'celltype_receiver']] = result[['ligand', 'receptor', 'celltype_sender', 'celltype_receiver']].astype(str)
    result["name"] = result[['ligand','receptor','celltype_sender','celltype_receiver']].apply("-".join, axis=1)
    # some are repeat with different score somehow
    result = result.groupby(['ligand','receptor','celltype_sender','celltype_receiver',"name"]).mean(numeric_only = True).reset_index()
    result["CCI"] = result[['celltype_sender','celltype_receiver']].apply("-".join, axis=1)
    result["LRI"] = result[['ligand','receptor']].apply("-".join, axis=1)
    return result



def runSpaTalk(adata, rscript_executable = None,
               meta_key = 'celltype', tp_key = None, overwrite = False,
               st_dir = None, st_meta_dir = None, n_cores = 1):
    '''
    This function is to run SpaTalk and add its results on the adata object.
    '''
    if rscript_executable is None:
        rscript_executable = adata.uns.get('rscript_path', None)
        if rscript_executable is None:
            raise ValueError('Rscript executable not found, please set rscript_executable or adata.uns["rscript_path"]')
        
    species = adata.uns['species']
    print(f'Species is {species}.')
    if species is None:
        print('Species is not specified, please specify the species in adata.uns[\'species\']')
        return
    
    save_path = adata.uns['save_path']
    out_f = f'{save_path}/spa/'
    if not tp_key:
        tp_key = adata.uns['tp_key']
    if overwrite or not os.path.exists(f'{out_f}/spatalk_meta.csv'):
        script_path = os.path.dirname(os.path.realpath(__file__)) + '/pipelines/'
        r_script_file = f'{script_path}/run_spatalk_lr.R'
        # TODO change name to sc_count.tsv and sc_meta.tsv
        if not st_dir:
            st_dir = f'{save_path}/refined_sc_exp.tsv'

        if not st_meta_dir:
            st_meta_dir = f'{save_path}/cell_mapping_meta.tsv'

        args = [st_dir,st_meta_dir,st_meta_dir,meta_key,species,out_f,str(n_cores)]
        subprocess.run([rscript_executable, "--vanilla", r_script_file]+ args)

    if not os.path.exists(f'{out_f}/lr_pair.csv'):
        raise ValueError(f'Error in running SpaTalk, excuation halted.')
    else:
        # spatalk changes '-' to '_' in celltype.
        tp4spatalk = adata.obs[meta_key].str.replace('-','_')
        tp4spatalk = adata.obs[meta_key].str.replace(' ','_')
        tp_map = dict(zip(tp4spatalk, adata.obs[tp_key]))
        adata.uns['tp_map_spatalk'] = tp_map
        df = pd.read_csv(f'{out_f}/lr_pair.csv',sep = ',',header=0,index_col=0)
        # print(df['celltype_sender'].unique())
        # change from 0.005 to 0.01
        adata.uns['spatalk'] = load_spatalk(df, pvalue_thred = 0.01,tp_map = tp_map)
        adata.uns['spatalk_meta'] = pp.read_csv_tsv(f'{out_f}/spatalk_meta.csv')
        adata.uns['spatalk_meta'].columns = [x.replace('rawmeta.','') for x in adata.uns['spatalk_meta'].columns]
        

    # no need for return adata

def SpaVis(adata, ligand = '',receptor = '',sender = '',receiver = '',
            label_size = 10, linewidth = 2, sender_color = '', receiver_color = '',
            exp_threshold = None, figsize = (3,6), orientation = 'horizontal'):
    save_path = adata.uns['save_path'] + '/spa/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)     
    script_path = os.path.dirname(os.path.realpath(__file__)) + '/pipelines/'
    r_script_file = f'{script_path}/vis_spatalk.R'
    rscript_executable = adata.uns['rscript_path']
    args = [ligand, receptor, sender, receiver, save_path, sender_color, receiver_color, str(label_size), str(linewidth), str(exp_threshold), 
    str(figsize[0]), str(figsize[1]), orientation]
    subprocess.run([rscript_executable, "--vanilla", r_script_file]+ args)
    print(f'Plots saved in {save_path}')


def AlphaVis(
    adata,
    ligand='',
    receptor='',
    sender='FB',
    receiver='T cell',
    alphatalk_df=None,
    alphatalk_path=None,
    sender_color='',
    receiver_color='',
    label_size=10,
    linewidth=1,
    figsize=(12, 8),
    max_hop=4,
    top_n=200,
    max_nodes_per_hop=10,
):
    """
    AlphaTalk-driven LR cascade network with pathway multi-hop propagation.
    Downstream genes/TF are expanded from receptor using SpaTalk pathway graph.
    """
    save_path = adata.uns['save_path'] + '/alphavis/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if alphatalk_df is None:
        if alphatalk_path is None:
            if 'spatalk' in adata.uns:
                alphatalk_df = adata.uns['spatalk'].copy()
            else:
                raise ValueError("Please provide alphatalk_df or alphatalk_path, or set adata.uns['spatalk'].")
        else:
            alphatalk_df = pp.read_csv_tsv(alphatalk_path)

    required_cols = {'ligand', 'receptor', 'sender_major', 'receiver_major'}
    missing = required_cols.difference(set(alphatalk_df.columns))
    if missing:
        raise ValueError(f"AlphaVis missing required columns: {sorted(missing)}")

    df = alphatalk_df.copy()
    use_cols = ['ligand', 'receptor', 'sender_major', 'receiver_major']
    if 'co_exp_number' in df.columns:
        use_cols.append('co_exp_number')
    df = df[use_cols].copy()
    df = df[(df['sender_major'] == sender) & (df['receiver_major'] == receiver)].copy()
    if len(df) == 0:
        raise ValueError(f'No AlphaTalk rows found for {sender}->{receiver}.')

    # keep strongest rows for stable plotting
    if 'co_exp_number' in df.columns:
        df = df.sort_values('co_exp_number', ascending=False).head(top_n)
    tmp_fn = f'{save_path}/alphatalk_lr_pairs.tsv'
    df.to_csv(tmp_fn, sep='\t', index=False)

    script_path = os.path.dirname(os.path.realpath(__file__)) + '/pipelines/'
    r_script_file = f'{script_path}/vis_alphatalk_cascade.R'
    rscript_executable = adata.uns.get('rscript_path', None)
    if rscript_executable is None:
        raise ValueError('Rscript executable not found, please set adata.uns["rscript_path"]')

    args = [
        tmp_fn,
        save_path,
        ligand,
        receptor,
        sender,
        receiver,
        sender_color,
        receiver_color,
        str(label_size),
        str(linewidth),
        str(figsize[0]),
        str(figsize[1]),
        str(max_hop),
        str(max_nodes_per_hop),
    ]
    subprocess.run([rscript_executable, "--vanilla", r_script_file] + args)
    print(f'Plots saved in {save_path}')


def AlphaSpaVis(
    adata,
    ligand='',
    receptor='',
    sender='FB',
    receiver='T cell',
    alphatalk_df=None,
    alphatalk_path=None,
    sender_color='',
    receiver_color='',
    label_size=10,
    linewidth=1,
    exp_threshold=10,
    figsize=(12, 8),
):
    """
    SpaVis-equivalent plot driven by AlphaTalk results, using SpaTalk's original
    plot_lr_path pipeline for multilayer transmission and style.
    """
    save_path = adata.uns['save_path'] + '/alphaspavis/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if alphatalk_df is None:
        if alphatalk_path is None:
            if 'spatalk' in adata.uns:
                alphatalk_df = adata.uns['spatalk'].copy()
            else:
                raise ValueError("Please provide alphatalk_df or alphatalk_path, or set adata.uns['spatalk'].")
        else:
            alphatalk_df = pp.read_csv_tsv(alphatalk_path)

    # write expression and metadata for R-side SpaTalk object construction
    exp_fn = f'{save_path}/exp.tsv'
    meta_fn = f'{save_path}/meta.tsv'
    alpha_fn = f'{save_path}/alphatalk.tsv'

    adata.to_df().T.to_csv(exp_fn, sep='\t')
    meta = adata.obs.copy()
    needed_cols = ['st_x', 'st_y']
    for c in needed_cols:
        if c not in meta.columns:
            raise ValueError(f"adata.obs missing required column: {c}")
    tp_key = adata.uns.get('tp_key', 'celltype')
    if tp_key not in meta.columns:
        raise ValueError(f"adata.obs missing tp_key column: {tp_key}")
    out_meta = meta[[tp_key, 'st_x', 'st_y']].copy()
    out_meta.columns = ['celltype', 'x', 'y']
    out_meta.insert(0, 'cell', out_meta.index.astype(str))
    out_meta.to_csv(meta_fn, sep='\t', index=False)

    alphatalk_df.to_csv(alpha_fn, sep='\t', index=False)

    script_path = os.path.dirname(os.path.realpath(__file__)) + '/pipelines/'
    r_script_file = f'{script_path}/vis_alphaspavis.R'
    rscript_executable = adata.uns.get('rscript_path', None)
    if rscript_executable is None:
        raise ValueError('Rscript executable not found, please set adata.uns["rscript_path"]')

    species = adata.uns.get('species', 'Human')
    args = [
        exp_fn,
        meta_fn,
        alpha_fn,
        save_path,
        species,
        ligand,
        receptor,
        sender,
        receiver,
        sender_color,
        receiver_color,
        str(label_size),
        str(linewidth),
        str(exp_threshold),
        str(figsize[0]),
        str(figsize[1]),
    ]
    subprocess.run([rscript_executable, "--vanilla", r_script_file] + args)
    print(f'Plots saved in {save_path}')

# def SpaVis(adata, ligand = '',receptor = '',sender = '',receiver = ''):
#     save_path = adata.uns['save_path'] + '/spa/'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)     
#     script_path = os.path.dirname(os.path.realpath(__file__)) + '/pipelines/'
#     r_script_file = f'{script_path}/vis_spatalk.R'
#     rscript_executable = adata.uns['rscript_path']
#     args = [ligand,receptor,sender,receiver, save_path]
#     subprocess.run([rscript_executable, "--vanilla", r_script_file]+ args)
#     print(f'Plots saved in {save_path}')


def generate_tp_lri(adata,col4Rec,sender_order,receiver_order):
    '''
    This function is used to generate the LRI for each cell type pair.
    '''
    draw_lr = adata.uns['spatalk']
    if sender_order == None:
        sender_order = col4Rec.index.tolist()
    if receiver_order == None:
        receiver_order = col4Rec.columns.tolist()
    
    sender_order = [sender for sender in sender_order if sender in col4Rec.index.tolist()]
    receiver_order = [receiver for receiver in receiver_order if receiver in col4Rec.columns.tolist()]

    data = col4Rec.loc[sender_order,receiver_order]
    data = data.div(data.sum(axis=1), axis=0)
    
    col4Rec_melt = data.reset_index()
    col4Rec_melt = col4Rec_melt.melt(id_vars = 'sender_tp',value_vars = col4Rec_melt.columns,var_name = 'receiver_tp',value_name = 'CCI')
    draw_target_pattern = col4Rec_melt.copy()
    draw_target_pattern['LRI'] = 0

    for idx,row in draw_target_pattern.iterrows():
        sender = row['sender_tp']
        receiver = row['receiver_tp']
        count = row['LRI']
        tmp = draw_lr[(draw_lr['celltype_sender'] == sender)&(draw_lr['celltype_receiver'] == receiver)]
        if tmp.shape[0] != 0:
            draw_target_pattern.loc[idx,'LRI'] = int(len(tmp['LRI']))
    draw_target_pattern.columns = ['Sender','Receiver','CCI','LRI']
    # print(draw_target_pattern.head())
    draw_target_pattern = draw_target_pattern[draw_target_pattern['Sender'].isin(sender_order) & 
                                            draw_target_pattern['Receiver'].isin(receiver_order)]

    draw_target_pattern['Sender'] = pd.Categorical(draw_target_pattern['Sender'], categories=sender_order, ordered=True)
    draw_target_pattern['Receiver'] = pd.Categorical(draw_target_pattern['Receiver'], categories=receiver_order, ordered=True)
    draw_target_pattern = draw_target_pattern.sort_values(by=['Sender', 'Receiver'])
    return draw_target_pattern


def generate_cci(adata, tp_key = None, return_df = False):
    save_path = adata.uns['save_path']+'/spa/'
    if not tp_key:
        tp_key = adata.uns['tp_key']
        tp_map = adata.uns['tp_map_spatalk']
    # print(save_path)

    cellpair = pp.read_csv_tsv(f'{save_path}/cellpair.csv')
    cellpair[['sender_tp','receiver_tp']] = cellpair['Name'].str.split(' -- ',expand = True)
    cellpair['sender_tp'] = cellpair['sender_tp'].map(tp_map)
    cellpair['receiver_tp'] = cellpair['receiver_tp'].map(tp_map)
    inter_tp = set(cellpair['sender_tp'].unique()).intersection(set(adata.obs[tp_key].unique()))
    if len(inter_tp) < len(adata.obs[tp_key].unique())/2:
        print(f'The cell type in the spatalk cellpair file is not consistent with the {tp_key} cell type in the adata object.')

    nn_df = cellpair.groupby(['sender_tp','receiver_tp']).count().reset_index()
    nn_df = nn_df.pivot(index = 'sender_tp',columns = 'receiver_tp',values = 'Name')
    nn_df.fillna(0,inplace = True)
    nn_df = nn_df.astype(int)
    celltype_nn = nn_df.reset_index()
    celltype_nn = celltype_nn.melt(id_vars = 'sender_tp',value_vars = celltype_nn.columns[1:],var_name = 'receiver_tp',value_name = 'CCI')
    # cell type x cell type, CCI percentage
    col4Rec = nn_df/nn_df.sum(axis = 0)
    row4Send = (nn_df.T/nn_df.sum(axis = 1)).T

    adata.uns['rec_per'] = col4Rec
    adata.uns['send_per'] = row4Send
    adata.uns['cellpair'] = cellpair
    adata.uns['celltype_nn'] = celltype_nn

    if return_df:
        return col4Rec,row4Send


def generate_cci_alphatalk(
    adata,
    alphatalk_df=None,
    alphatalk_path=None,
    tp_key=None,
    sender_col="cell_sender",
    receiver_col="cell_receiver",
    weight_col=None,
    return_df=False,
):
    """
    Generate CCI summary from AlphaTalk outputs.

    This function provides the same core outputs as ``generate_cci``:
    - adata.uns['rec_per']
    - adata.uns['send_per']
    - adata.uns['cellpair']
    - adata.uns['celltype_nn']

    Parameters
    ----------
    adata : AnnData
        AnnData object to store CCI summaries.
    alphatalk_df : pd.DataFrame, optional
        AlphaTalk interaction table. If None, read from ``alphatalk_path``.
    alphatalk_path : str, optional
        Path to AlphaTalk csv/tsv output. Used when ``alphatalk_df`` is None.
    tp_key : str, optional
        Cell type key in ``adata.obs`` for consistency check.
    sender_col : str
        Sender cell type column in AlphaTalk table.
    receiver_col : str
        Receiver cell type column in AlphaTalk table.
    weight_col : str, optional
        If provided, use weighted aggregation by summing ``weight_col`` within
        each sender-receiver pair instead of raw interaction counts.
    return_df : bool
        If True, return ``(col4Rec, row4Send)``.
    """
    if alphatalk_df is None:
        if alphatalk_path is None:
            save_path = adata.uns.get("save_path", None)
            if save_path is None:
                raise ValueError("Please provide alphatalk_df or alphatalk_path, or set adata.uns['save_path'].")
            candidates = [
                f"{save_path}/alphatalk_lr_score.csv",
                f"{save_path}/alphatalk/lr_score.csv",
                f"{save_path}/ana/alphatalk/tables/refined_lr_st_filter.csv",
                f"{save_path}/ana/alphatalk/tables/original_lr_st_filter.csv",
            ]
            alphatalk_path = next((p for p in candidates if os.path.exists(p)), None)
            if alphatalk_path is None:
                raise ValueError("Cannot find AlphaTalk result file automatically. Please set alphatalk_path.")
        alphatalk_df = pp.read_csv_tsv(alphatalk_path)

    if sender_col not in alphatalk_df.columns or receiver_col not in alphatalk_df.columns:
        raise ValueError(
            f"AlphaTalk result must contain '{sender_col}' and '{receiver_col}' columns. "
            f"Current columns: {alphatalk_df.columns.tolist()}"
        )

    if tp_key is None:
        tp_key = adata.uns.get("tp_key", None)

    cellpair = alphatalk_df.copy()
    cellpair["sender_tp"] = cellpair[sender_col].astype(str)
    cellpair["receiver_tp"] = cellpair[receiver_col].astype(str)
    cellpair["Name"] = cellpair["sender_tp"] + " -- " + cellpair["receiver_tp"]

    if tp_key is not None and tp_key in adata.obs.columns:
        inter_tp = set(cellpair["sender_tp"].unique()).intersection(set(adata.obs[tp_key].unique()))
        if len(adata.obs[tp_key].unique()) > 0 and len(inter_tp) < len(adata.obs[tp_key].unique()) / 2:
            print(f"The cell type in AlphaTalk results is not consistent with the {tp_key} cell type in the adata object.")

    if weight_col is not None:
        if weight_col not in cellpair.columns:
            raise ValueError(
                f"weight_col='{weight_col}' not found in AlphaTalk table. "
                f"Current columns: {cellpair.columns.tolist()}"
            )
        tmp = cellpair[["sender_tp", "receiver_tp", weight_col]].copy()
        tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(0.0)
        nn_df = tmp.groupby(["sender_tp", "receiver_tp"])[weight_col].sum().reset_index(name="CCI")
    else:
        nn_df = cellpair.groupby(["sender_tp", "receiver_tp"]).size().reset_index(name="CCI")

    nn_df = nn_df.pivot(index="sender_tp", columns="receiver_tp", values="CCI")
    nn_df.fillna(0, inplace=True)
    if weight_col is None:
        nn_df = nn_df.astype(int)
    else:
        nn_df = nn_df.astype(float)
    celltype_nn = nn_df.reset_index()
    celltype_nn = celltype_nn.melt(
        id_vars="sender_tp",
        value_vars=celltype_nn.columns[1:],
        var_name="receiver_tp",
        value_name="CCI",
    )
    # cell type x cell type, CCI percentage
    col4Rec = nn_df / nn_df.sum(axis=0)
    row4Send = (nn_df.T / nn_df.sum(axis=1)).T
    col4Rec = col4Rec.fillna(0)
    row4Send = row4Send.fillna(0)

    adata.uns["rec_per"] = col4Rec
    adata.uns["send_per"] = row4Send
    adata.uns["cellpair"] = cellpair
    adata.uns["celltype_nn"] = celltype_nn

    if return_df:
        return col4Rec, row4Send


# def runKEGG(adata, rscript_executable = '/apps/software/R/4.2.0-foss-2021b/bin/Rscript', input_fn = None):
#     save_path = adata.uns['save_path']
#     out_f = f'{save_path}/kegg/'
#     if not os.path.exists(out_f):
#         os.makedirs(out_f)
        
#     script_path = os.path.dirname(os.path.realpath(__file__)) + '/pipelines/'
#     r_script_file = f'{script_path}/kegg.R'
#     if input_fn:
#          args = [out_f,'mouse',input_fn]
#     else:
#         # no file specified, run all kegg file under out_F
#         args = [out_f,'mouse']
#     subprocess.run([rscript_executable, "--vanilla", r_script_file]+ args)   



def runKEGG(adata, rscript_executable = None, input_fn = None, input_df = None, df_name = None):
    import subprocess
    save_path = adata.uns['save_path']
    out_f = f'{save_path}/kegg/'
    if not os.path.exists(out_f):
        os.makedirs(out_f)
    species = adata.uns['species']
    if rscript_executable is None:
        rscript_executable = adata.uns.get('rscript_path', None)
        if rscript_executable is None:
            raise ValueError('Rscript executable not found, please set rscript_executable or adata.uns["rscript_path"]')
        
    script_path = os.path.dirname(os.path.realpath(__file__)) + '/pipelines/'
    r_script_file = f'{script_path}/kegg.R'
    if input_df is not None:
        tmp = pp.lr2kegg(input_df, use_lig_gene = True, use_rec_gene = True).reset_index()
        if df_name is None:
            # fn = f'{out_f}/tmp_kegg.tsv'
            fn = f'tmp_kegg.tsv'
            print(f'Variable input_fn have no file name specified, save to {fn}')   
        else:
            # fn = f'{out_f}/{df_name}_kegg.tsv'
            fn = f'{df_name}_kegg.tsv'
        tmp.to_csv(f'{out_f}/{fn}', index = True,sep = '\t',header = True)
        args = [out_f,species,fn]
    else:
        if input_fn:
            args = [out_f,species,input_fn]
        else:
        # no file specified, run all kegg file under out_F
            args = [out_f,species]
    subprocess.run([rscript_executable, "--vanilla", r_script_file]+ args)
    output = fn.split('_kegg.tsv')[0]
    print(output)

    enrichment_fn = f'{out_f}{output}_kegg_enrichment.tsv'
    geneid_fn = f'{out_f}{output}_kegg_geneID.tsv'
    if not os.path.exists(enrichment_fn):
        # Only allow local fallback under current run directory.
        fallback_candidates = [
            f'{out_f}kegg_enrich_{output}.tsv',
            f'{out_f}{output}_kegg_enrichment_backup.tsv',
        ]
        for candidate in fallback_candidates:
            if os.path.exists(candidate):
                enrichment_fn = candidate
                print(f'Use local fallback KEGG result: {candidate}')
                break

    if not os.path.exists(enrichment_fn):
        return pd.DataFrame(columns=['category', 'subcategory', 'ID', 'Description', 'GeneRatio', 'BgRatio',
                                     'RichFactor', 'FoldEnrichment', 'zScore', 'pvalue', 'p.adjust', 'qvalue',
                                     'geneID', 'Count', 'geneSymbol'])

    kegg_res = pd.read_csv(enrichment_fn, sep='\t', header=0)
    if kegg_res.columns[0].startswith('Unnamed'):
        kegg_res = kegg_res.drop(columns=kegg_res.columns[0])

    if os.path.exists(geneid_fn):
        geneid = pd.read_csv(geneid_fn, sep='\t', header=0, index_col=0)
    else:
        geneid = pd.DataFrame(columns=['ENTREZID', 'SYMBOL'])
    gene_dict = dict(zip(geneid['ENTREZID'],geneid['SYMBOL']))
    kegg_res.index = range(len(kegg_res))
    if 'geneSymbol' not in kegg_res.columns:
        for index, row in kegg_res.iterrows():
            if 'geneID' not in row or pd.isna(row['geneID']):
                continue
            gene_ids = str(row['geneID']).split('/')
            symbols = []
            for gene_id in gene_ids:
                try:
                    symbols.append(gene_dict.get(int(gene_id), gene_id))
                except ValueError:
                    symbols.append(gene_id)
            kegg_res.loc[index, 'geneSymbol'] = '/'.join(symbols)
    return kegg_res



def lri_kegg_enrichment(adata, target_sender = [], target_receiver = [], 
                        unique_lri = None,use_lig_gene = True, use_rec_gene = True,
                        overwrite = False):
    '''
    unique_lri: Default is None, plot all LRI
                If set as True plot unique LRI with auto calculated unique_count
                If set as integer, unique_count set as this integer
                unique_count is the maximum number of occurrences of LRI among all CCIs
    '''
    lri_df = adata.uns['spatalk']
    save_path = adata.uns['save_path']
    out_f = f'{save_path}/kegg/'
    if not os.path.exists(out_f):
        os.makedirs(out_f)

    if len(target_sender) == 0:
        target_sender = lri_df['celltype_sender'].unique()
    
    if len(target_receiver) == 0:
        target_receiver = lri_df['celltype_receiver'].unique()

    lri_df = lri_df[lri_df['celltype_sender'].isin(target_sender) & lri_df['celltype_receiver'].isin(target_receiver)].copy()
    if unique_lri:
        df = lri_df.groupby(['celltype_sender','LRI']).count()
        df.reset_index(inplace = True)
        if isinstance(unique_lri,bool):
            uniq_count_thred = int(np.round(df["ligand"].mean()))
            print('boolean',uniq_count_thred)
        elif isinstance(unique_lri,int):
            uniq_count_thred = unique_lri
            print('int',uniq_count_thred)
        else:
            raise ValueError('unique_lri should be bool or int')
        loose_uniq_lri = df[df['ligand'] <= uniq_count_thred]['LRI'].tolist()
        lri_df = lri_df[lri_df['LRI'].isin(loose_uniq_lri)].copy()
    else:
        uniq_count_thred = 'all'
    print(lri_df.groupby(['LRI']).count().sort_values(by = 'ligand'))
    lig_gene = 'L' if use_lig_gene else 'n'
    rec_gene = 'R' if use_rec_gene else 'n'
    
    for sender in target_sender:
        for rec in target_receiver:
            target_lri = lri_df[(lri_df['celltype_sender'] == sender) & (lri_df['celltype_receiver'] == rec)].copy()
            # print(target_lri.head(5))
            # incase of invalid file name
            sender = sender.replace('/','_')
            rec = rec.replace('/','_')
            tmp = pp.lr2kegg(target_lri, use_lig_gene = use_lig_gene, use_rec_gene = use_rec_gene).reset_index()
            input_fn = f'{out_f}/{sender}_{rec}_{lig_gene}_{rec_gene}_{uniq_count_thred}_kegg.tsv'
            out_fn = f'{out_f}/{sender}_{rec}_{lig_gene}_{rec_gene}_{uniq_count_thred}_kegg_enrichment.tsv'
            if not os.path.exists(out_fn) or overwrite:
                tmp.to_csv(input_fn,index = True,sep = '\t',header = True)
                runKEGG(adata, rscript_executable = '/apps/software/R/4.2.0-foss-2021b/bin/Rscript', input_fn = input_fn)

    # load
    if 'kegg_enrichment' in adata.uns.keys():
        kegg_res = adata.uns['kegg_enrichment']
    else:
        kegg_res = pd.DataFrame()

    if 'gene_dict' in adata.uns.keys():
        gene_dict = adata.uns['gene_dict']
    else:
        gene_dict = {}

    geneid = pd.DataFrame()
    # for query_tp in glia:
    for sender in target_sender:
        for rec in target_receiver:
            sender_fn = sender.replace('/','_')
            rec_fn = rec.replace('/','_')
            tmp = pd.read_csv(f"{out_f}/{sender_fn}_{rec_fn}_{lig_gene}_{rec_gene}_{uniq_count_thred}_kegg_enrichment.tsv",sep = '\t',header=0,index_col=0)
            tmp = pp.filter_kegg(tmp,pval_thred = 0.05)
            tmp['celltype_sender'] = sender
            tmp['celltype_receiver'] = rec
            tmp['used_genes'] = f'{lig_gene}_{rec_gene}'
            kegg_res = pd.concat((kegg_res,tmp))

            tmp = pd.read_csv(f"{out_f}/{sender_fn}_{rec_fn}_{lig_gene}_{rec_gene}_{uniq_count_thred}_kegg_geneID.tsv",sep = '\t',header=0,index_col=0)
            geneid = pd.concat((geneid,tmp))
    tmp_gene_dict = dict(zip(geneid['ENTREZID'],geneid['SYMBOL']))
    gene_dict.update(tmp_gene_dict)
    kegg_res.index = range(len(kegg_res))
    for index, row in kegg_res.iterrows():
        gene_ids = row['geneID'].split('/')
        symbols = [gene_dict[int(gene_id)] for gene_id in gene_ids]
        new_gene_ids = '/'.join(symbols)
        kegg_res.loc[index, 'geneSymbol'] = new_gene_ids
    adata.uns['kegg_enrichment'] = kegg_res  
    adata.uns['gene_dict'] = gene_dict
    return kegg_res, gene_dict



def rowmax(df):
    # df = df.iloc[:,:3]
    return df.idxmax(axis=1)


def load_pattern(adata,k = 3):
    fn_dir = adata.uns['save_path'] + f'/spade/patterns_k_{k}.tsv'
    spex_pattern = pp.read_csv_tsv(fn_dir)
    if fn_dir.endswith('tsv'): # mine index is the first col
        spex_pattern.reset_index(inplace=True)
    spex_pattern['pattern'] = rowmax(spex_pattern)
    # print(spex_pattern['pattern'].unique())
    spex_pattern['pattern'] = 'Pattern ' + spex_pattern['pattern'].astype(str)
    if 'spot' in spex_pattern.columns:
        spex_pattern['spot'] = adata.obs['spot']
    adata.obs[f'pattern'] = spex_pattern['pattern'].values
    # scale pattern value to 0~1
    spex_pattern.iloc[:,:k] = spex_pattern.iloc[:,:k].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    adata.uns['pattern'] = spex_pattern
    return spex_pattern


def load_histo(adata,k = 3):
    fn_dir = adata.uns['save_path'] + f'/spade/histology_k_{k}.tsv'
    spex_histo = pp.read_csv_tsv(fn_dir)
    spex_histo.reset_index(inplace = True)
    spex_histo.index = spex_histo['g']
    spex_histo = spex_histo[spex_histo['pval']<0.05]
    if f'pattern_{k}' not in spex_histo.columns:
        spex_histo[f'pattern_{k}'] = spex_histo[f'pattern']
        spex_histo[f'membership_{k}'] = spex_histo[f'membership']
        del spex_histo['membership']
        del spex_histo['pattern']
    adata.uns['pattern_genes'] = spex_histo
    return spex_histo


def load_moran(adata):
    fn_dir = adata.uns['save_path'] + f'/moran/moranI.tsv'
    moran = pp.read_csv_tsv(fn_dir)
    adata.var = pd.concat((adata.var,moran['I']),axis = 1)
    return pd.DataFrame(moran['I'])


def run_GSEA(after_adata,target_col = 'CAF_leiden',sorter = ['CAF_1','CAF_0'],
             gene_set = 'GO_Biological_Process_2021'):
    import gseapy as gp
    import scanpy as sc
    if 'lognorm' not in after_adata.layers.keys():
        sc.pp.normalize_total(after_adata, target_sum=1e4)
        sc.pp.log1p(after_adata)
        after_adata.layers['lognorm'] = after_adata.X

    after_adata.obs[target_col] = pd.Categorical(after_adata.obs[target_col], categories=sorter, ordered=True)
    indices = after_adata.obs.sort_values([target_col]).index
    after_adata = after_adata[indices,:]
    res = gp.gsea(data=after_adata.to_df().T,
        gene_sets=gene_set,
        cls=after_adata.obs[target_col],
        permutation_num=10000,
        # permutation_type='phenotype',
        permutation_type='gene_set',
        outdir=None,
        method='s2n', # signal_to_noise
        threads= 16)
    # print(res.res2d.head(10))
    return res



################## SpotCor ##################

def runSpotCor(adata, python_executable = None, recon_exp_file = None, recon_meta_file = None,
               orig_sc_file = None, orig_st_file = None, species = None):
    import subprocess
    tp_key = adata.uns['tp_key']

    save_path = adata.uns['save_path']
    out = f'{save_path}/spot_cor/'
    if not os.path.exists(out):
        os.makedirs(out)
    
    if species is None:
        species = adata.uns['species']
    script_path = os.path.dirname(os.path.realpath(__file__)) + '/pipelines/'
    script_file = f'{script_path}/run_spot_cor.py'
    if not 'python_path' in adata.uns and not python_executable:
        raise ValueError('python path is not provided, please provide with parameter python_path')
    else:
        python_executable = python_executable or adata.uns.get('python_path')
        if not python_executable:
            raise ValueError('python path is not provided, please provide with parameter python_path')
        subprocess.run([python_executable, \
        script_file, \
        '-s', recon_exp_file, \
        '-c', recon_meta_file, \
        '-p', tp_key, \
        '-o', out, \
        '-b', orig_sc_file, \
        '-t', orig_st_file, \
        '-a', species])
    df = pp.read_csv_tsv(f'{out}/spot_cor_scale.tsv')
    return df
