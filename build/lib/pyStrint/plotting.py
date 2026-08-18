# import anndata
# import os

# from scipy.sparse import csr_matrix
# import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from . import evaluation as eva
from matplotlib_venn import venn2
# from matplotlib_venn import venn3
# import plotly.express as px
# import math
# from . import utils

def sc_celltype(adata, color_map = None, tp_key = None, swap_xy = False,
            subset_idx = None, legend = False, figsize = (4,4), rect = False,
            savefig = False, size = 10, alpha = 0.8,title = None,theme = 'white'):
    '''
    @ Wang Jingwan 0314
    This function is used to draw scatter plot of single cell data.
    '''

    sc_agg_meta = adata.obs.copy()
    sc_agg_meta['pivot'] = 1
    if tp_key is None:
        tp_key = adata.uns['tp_key']
    if color_map is None:
        if 'major_color_map' in adata.uns.keys():
            color_map = adata.uns['major_color_map']
    #sort celltype number from large to small from sc_agg_meta
    sc_agg_meta['celltype_num'] = sc_agg_meta.groupby(tp_key)['pivot'].transform('count')
    #draw scater plot of each celltype in a order of celltype_num, large to small
    sc_agg_meta = sc_agg_meta.sort_values(by=['celltype_num'],ascending=False)
    sc_agg_meta[tp_key] = sc_agg_meta[tp_key].astype(object)

    if 'adj_spex_UMAP1' in sc_agg_meta.columns:
        cols = ['adj_spex_UMAP1','adj_spex_UMAP2']
    elif 'adj_UMAP1' in sc_agg_meta.columns:
        cols = ['adj_UMAP1','adj_UMAP2']
    elif 'st_x' in sc_agg_meta.columns:
        cols = ['st_x','st_y']
    elif 'col' in sc_agg_meta.columns:
        cols = ['row','col']
    elif 'X' in sc_agg_meta.columns:
        cols = ['X','Y']
    else:
        cols = ['x','y']
    if swap_xy == True:
        cols = [cols[1],cols[0]]
    # print(cols)
    plt.figure(figsize=figsize)
    with sns.axes_style("white"):
        if subset_idx is None:
            sns.scatterplot(data=sc_agg_meta, x=cols[0], y=cols[1],hue = tp_key, 
                            s = size,alpha = alpha,palette=color_map,edgecolor = None)
        else:
            subset_data = sc_agg_meta.copy()
            subset_data['subset'] = 'Other cells'
            subset_data.loc[subset_idx,'subset'] = subset_data.loc[subset_idx,tp_key]
            sns.scatterplot(data=subset_data[subset_data['subset'] == 'Other cells'], x=cols[0], y=cols[1],hue = 'subset', 
                            s = 5,alpha = 1,palette=['#cccccc'],edgecolor = None)
            sns.scatterplot(data=subset_data.loc[subset_idx], x=cols[0], y=cols[1],hue = 'subset', 
                            s = size+5,alpha = alpha,palette=color_map,edgecolor = None)
            if rect: 
                if isinstance(rect, bool):
                    x_min,y_min = subset_data.loc[subset_idx][cols].min()
                    x_max,y_max = subset_data.loc[subset_idx][cols].max()
                elif isinstance(rect, list):
                    x_min,y_min = rect[0]
                    x_max,y_max = rect[1]
                rect_width = x_max - x_min  # Width of the rectangle
                rect_height = y_max - y_min  # Height of the rectangle
                rect_color = 'red'  # Rectangle color 
                rectangle = plt.Rectangle((x_min, y_min), rect_width, rect_height, fill=False, color=rect_color, linewidth=2)
                plt.gca().add_patch(rectangle)
        if legend:
            if sc_agg_meta[tp_key].nunique() > 18:
                ncol = 2
            else:
                ncol = 1
            handles, labels = plt.gca().get_legend_handles_labels()
            # Sort the handles and labels based on the labels
            handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: (x[1] == "Other cells", x[1]))
            handles_sorted, labels_sorted = zip(*handles_labels_sorted)
            # Create a new legend with the sorted handles and labels
            leg = plt.legend(handles_sorted, labels_sorted,loc='center left', bbox_to_anchor=(0.99, 0.5),
                             ncol=ncol, handletextpad=0.5,columnspacing=0.4,labelspacing=0.1,
                             fontsize = 16,markerscale = 2,handlelength = 0.5)
            leg.get_frame().set_linewidth(0.0)  # Remove legend frame
        else:
            plt.legend([],[], frameon=False)  
        plt.title(title,fontsize=21)
        plt.axis('equal')
    # plt.tight_layout()
    if theme == 'white':
        # sns.despine(left=True, bottom=True)
        plt.xlabel('',fontsize=16)
        plt.ylabel('',fontsize=16)
        plt.xticks([],fontsize=14)
        plt.yticks([],fontsize=14)
        # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    
    if savefig:
        if isinstance(savefig,str):
            if '/' in savefig:
                savefig = f'{savefig}'
            else:
                save_path = adata.uns['figpath']
                savefig = f'{save_path}/{savefig}'
        else:
            save_path = adata.uns['figpath']
            savefig = f'{save_path}/celltype_sc.pdf'
        plt.savefig(f'{savefig}', bbox_inches='tight')
    plt.show()
    plt.clf()



def decide_figsize(x = None, y = None,unit = 4):
    # small one be the unit
    xrange = x.max() - x.min()
    yrange = y.max() - y.min()
    # col_L is x
    # row_L is y
    if xrange < yrange:
        ROW_L = round(unit * (yrange/xrange))
        COL_L = unit
    else:
        COL_L = round(unit * (xrange/yrange))
        ROW_L = unit
    return ROW_L,COL_L


def sc_subtype(adata,color_map = None,tp_key = 'celltype', target_tp = None, size = 20,
                   theme_white = True, savefig = False, COL = None, hue = None):
    sc_agg_meta = adata.obs.copy()
    if hue is None:
        sc_agg_meta['pivot'] = 1
        hue = 'pivot'

    if target_tp is None:
        target_tp = sc_agg_meta[tp_key].unique().tolist()
        name = 'all'
    else:
        name = target_tp[0]
    
    if COL is None:
        # e.g. sqrt(9) = 3x3
        # e.g. sqrt(12) = 3x4 (lesser row more col, thus np.floor)
        ROW = int(np.floor(np.sqrt(len(target_tp))))
        COL = int(np.ceil(len(target_tp) / ROW))
    else:
        ROW = int(np.ceil(len(target_tp) / COL))

    if 'adj_spex_UMAP1' in sc_agg_meta.columns:
        cols = ['adj_spex_UMAP1','adj_spex_UMAP2']
    elif 'adj_UMAP1' in sc_agg_meta.columns:
        cols = ['adj_UMAP1','adj_UMAP2']
    else:
        cols = ['x','y']

    if 'st_x' in sc_agg_meta.columns:
        st_cols = ['st_x','st_y']
    elif 'x' in sc_agg_meta.columns:
        st_cols = ['x','y']
    else:
        st_cols = cols
    i = 0
    ROW_L,COL_L = decide_figsize(x = sc_agg_meta['st_x'], y = sc_agg_meta['st_y'])
    plt.figure(figsize=(COL_L*COL, ROW_L* ROW))
    for tp in target_tp:
        with sns.axes_style("white"):
            plt.subplot(ROW, COL, i + 1)
            sns.scatterplot(data=sc_agg_meta[[st_cols[0],st_cols[1],'pivot']].drop_duplicates(), x=st_cols[0], y=st_cols[1],hue = hue, 
                        s = size,alpha = 0.8,palette=['#ccc'],edgecolor = None
                        )
            sns.scatterplot(data=sc_agg_meta[sc_agg_meta[tp_key] == tp], x=cols[0], y=cols[1],hue = hue, 
                            s = size,alpha = 0.8,palette=[color_map[tp]],edgecolor = None
                            )
            if theme_white:
                plt.legend([],[], frameon=False)  
                # sns.despine(left=True, bottom=True)
                plt.tick_params(left=False, bottom=False, top = False, right = False)
                plt.tight_layout()
                plt.xlabel('',fontsize=16)
                plt.ylabel('',fontsize=16)
                plt.xticks([],fontsize=14)
                plt.yticks([],fontsize=14)
            else:
                plt.legend([],[], frameon=False)  
                plt.axis('equal')
                plt.xlabel('x',fontsize=16)
                plt.ylabel('y',fontsize=16)
                plt.subplots_adjust(wspace=0.4,hspace=0.4)
        plt.title(f'{tp}',fontsize=22)
        i+=1

    if savefig:
        save_path = adata.uns['figpath']
        savefig = f'{save_path}/{name}_sc_solo.pdf'
        plt.savefig(f'{savefig}', bbox_inches='tight')
    plt.show()
    plt.clf()




def boxplot(adata, metric='', palette_dict=None, sub_idx=None,
            x='method', y='pair_num', figsize=(2.4, 3),
            ylabel='LRI count', dodge=False, 
            test='t-test_ind', rotate_x=False,
            savefig=False, title=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    
    # Get the appropriate dataframe
    if (metric == 'spot_cor') or (metric == 'gene_cor'):
        draw_df = adata.uns[metric]
    elif metric == 'moran':
        draw_df = adata.uns[metric]
    else:
        draw_df = adata
    
    if sub_idx is not None:
        draw_df = draw_df.loc[sub_idx]

    # Create figure
    plt.figure(figsize=figsize)
    
    # Create boxplot with corrected palette usage
    ax = sns.boxplot(data=draw_df,
                    x=x,
                    y=y,
                    hue=x,  # Set hue to x
                    dodge=dodge,
                    palette=palette_dict,
                    showfliers=False,
                    width=0.8)

    # Statistical annotation
    groups = draw_df[x].unique()
    if len(groups) == 2:
        group1_data = draw_df[draw_df[x] == groups[0]][y]
        group2_data = draw_df[draw_df[x] == groups[1]][y]
        
        if test == 't-test_ind':
            stat, pval = stats.ttest_ind(group1_data, group2_data)
        elif test == 'Mann-Whitney':
            stat, pval = stats.mannwhitneyu(group1_data, group2_data)
        elif test == 'Wilcoxon':
            stat, pval = stats.wilcoxon(group1_data, group2_data)
        
        y_max = draw_df[y].max()
        y_pos = y_max + (y_max * 0.05)
        
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'
        else:
            sig = 'ns'
            
        plt.plot([0, 0, 1, 1], 
                [y_pos-(y_max * 0.05), y_pos-(y_max * 0.03), y_pos-(y_max * 0.03), y_pos-(y_max * 0.05)], 
                color='gray', lw=1)
        plt.text(0.5, y_pos-(y_max * 0.03), sig, color = 'gray',
                horizontalalignment='center', fontsize=12)

    # Formatting
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('', size=16)
    plt.ylabel(ylabel, size=16)
    
    if title:
        plt.title(title, size=16)
    
    if rotate_x:
        plt.xticks(rotation=90)

    # Legend handling
    # if not legend:
    plt.legend([], [], frameon=False)

    # Save figure if requested
    if savefig:
        if not isinstance(adata, pd.DataFrame):
            save_path = adata.uns['figpath']
            savefig = f'{save_path}/{metric}_box.pdf'
            plt.savefig(savefig, bbox_inches='tight')
        else:
            plt.savefig(savefig, bbox_inches='tight')
    
    return ax


# def boxplot(adata, metric = '', palette_dict = None, sub_idx = None,
#                  x = 'method', y = 'pair_num', hue='method',figsize = (2.4, 3),
#                  ylabel='LRI count', dodge=False,legend = False,
#                  test = 't-test_ind',rotate_x = False,
#                  savefig = False, title = None):
#     '''
#     @ Wang Jingwan 0314
#     test method can choose from 
#     't-test_welch', 't-test_paired', 'Mann-Whitney', 'Mann-Whitney-gt', 
#     'Mann-Whitney-ls', 'Levene', 'Wilcoxon', 'Kruskal', 'Brunner-Munzel'
#     seaborn==0.11.0
#     '''
#     if (metric == 'spot_cor') or (metric == 'gene_cor'):
#         draw_df = adata.uns[metric]
#     elif metric == 'moran':
#         # draw_df = pd.DataFrame(adata.var['I'])
#         draw_df = adata.uns[metric]
#     else:
#         draw_df = adata
    
#     if sub_idx:
#         draw_df = draw_df.loc[sub_idx]

#     from statannotations.Annotator import Annotator

#     plt.figure(figsize=figsize)
#     ax = sns.boxplot(x=x, y=y, data=draw_df, hue=hue, 
#                      dodge=dodge, palette=palette_dict,
#                      width=.8)
#     pairs = [tuple(draw_df[x].unique())]
#     annot = Annotator(ax, pairs, x=x, y=y, data=draw_df)
#     annot.configure(test=test, comparisons_correction="BH",correction_format="replace")
#     annot.apply_test()
#     annot.annotate()
#     plt.tight_layout()
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.xlabel('', size=16)
#     plt.ylabel(ylabel, size=16)
#     plt.title(title,size = 16)
#     if rotate_x:
#         plt.xticks(rotation=90)

#     if legend:
#         # plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
#         leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2,
#                         handletextpad=0.3,columnspacing=0.3,fontsize = 14)
#         leg.get_frame().set_linewidth(0.0)  # Remove legend frame
#     else:
#         plt.legend([],[], frameon=False)

#     if savefig:
#         if not isinstance(adata, pd.DataFrame):
#             save_path = adata.uns['figpath']
#             savefig = f'{save_path}/{metric}_box.pdf'
#             plt.savefig(f'{savefig}', bbox_inches='tight')
#         else:
#             plt.savefig(f'{savefig}', bbox_inches='tight')


def exp_violin(adata, gene=None, tp_key=None, types=None,
               palette_dict=None, test='t-test_ind',
               highlight=[], log=True, scale=True,
               figsize=(6, 4), x_rotate=False,
               savefig=False, title=''):
    from statannotations.Annotator import Annotator
    from itertools import combinations
    df = adata[:, gene].to_df()

    if log:
        df = np.log(df + 1)

    if scale:
        df = (df - df.mean()) / df.std()

    df = pd.concat((df, adata.obs[tp_key]), axis=1)
    if types:
        df = df[df[tp_key].isin(types)]
        print(df.head(5))
        print(types)
    plt.figure(figsize=figsize)
    ax = sns.violinplot(data=df, x=tp_key, y=gene,
                        order=types, palette=palette_dict,
                        linewidth=.9, cut=0, scale='width')
    if highlight:
        pairs = [(cell_type1, cell_type2) for cell_type1 in highlight for cell_type2 in types if cell_type1 != cell_type2]
    else:
        pairs = list(combinations(types, 2))

    annot = Annotator(ax, pairs, x=tp_key, y=gene, data=df,order = types)
    annot.configure(test=test, comparisons_correction="BH", correction_format="replace")
    annot.apply_test()
    annot.annotate()

    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('')
    plt.ylabel(gene, size=21)
    plt.title(title, size=21)

    if x_rotate:
        plt.xticks(rotation=90)

    if savefig:
        save_path = adata.uns['figpath']
        savefig = f'{save_path}/{title}_exp_violin.pdf'
        plt.savefig(f'{savefig}', bbox_inches='tight')



def highlight_y(draw_df, y, y_highlight):
    arr = draw_df[y].unique()
    highlighted_ytick = np.where(np.isin(arr, y_highlight))[0]
    # print(highlighted_ytick) # The ytick to be highlighted
    # Customize the ytick color
    yticks, _ = plt.yticks()
    ytick_labels = plt.gca().get_yticklabels()
    for ytick, ytick_label in zip(yticks, ytick_labels):
        # print(ytick)
        if ytick in highlighted_ytick:
            ytick_label.set_color('red')


def highlight_x(draw_df, x, x_highlight):
    arr = draw_df[x].unique()
    highlighted_xtick = np.where(np.isin(arr, x_highlight))[0]
    # print(highlighted_xtick) # The ytick to be highlighted
    # Customize the ytick color
    xticks, _ = plt.xticks()
    xtick_labels = plt.gca().get_xticklabels()
    for xtick, xtick_label in zip(xticks, xtick_labels):
        # print(ytick)
        if xtick in highlighted_xtick:
            xtick_label.set_color('red')


# def cal_even_div_legend(draw_df, hue = None, div_num = 3):
#     start = draw_df[hue].min()
#     end = draw_df[hue].max()
#     # Calculate the section size
#     section_size = (end - start) / div_num
#     # Calculate the three numbers in the sections
#     hue_numbers = np.linspace(start + section_size, end - section_size, 3)
#     return hue_numbers


def draw_bubble(draw_df, x=None, y="Description", x_highlight = None, y_highlight=None, 
                savefig=None, figsize=None, legend=False, xrotate=False,
                showlabel=False, xlabel = None, ylabel = None, title=None, 
                  cmap='viridis',hue='Count', size='GeneRatio',vmin = None, vmax = None, 
                  color_bar_pos = [1.1, 0.35, 0.55, 0.35]):
    '''
    Red label item in x, y axis
    highlight = 'T_cells_CD4+'
    or
    highlight = ['T_cells_CD4+','B cell]
    '''
    if figsize is None:
        width = np.max([int(np.ceil(len(draw_df[x].unique()) / 3.5)), 2])+0.2
        height = int(np.ceil(len(draw_df[y].unique())/4)+1)
        if xlabel:
            height += 0.5
        if ylabel:
            width += 0.5
        print(f'Auto figsize {(width,height)}')
    else:
        width = figsize[0]
        height = figsize[1]

    print(width,height)
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(width, height))
        ax = sns.scatterplot(data = draw_df, x = x, y = y, hue = hue, 
                        legend = legend,
                        palette = cmap, 
                        size = size, sizes = (50,300)
                        )
        
        highlight_x(draw_df, x, x_highlight)
        highlight_y(draw_df, y, y_highlight)

        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)

        if legend:
            sizes = np.percentile(draw_df[size], [25, 50, 75, 100])
            sizes = np.round(sizes, 2)
            labels = np.percentile(range(50,300), [25, 50, 75, 100])
            labels = np.round(labels, 0)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='black', markersize=labels[i]/20, label=sizes[i]) 
                            for i in range(len(sizes))]
            plt.legend(handles=legend_elements, title=size, title_fontsize=14, loc='center left', bbox_to_anchor=(1.2, 0.4), 
                       fontsize=14, handlelength = 0.5, handletextpad=0.5,labelspacing=0.2)
        
            import matplotlib.colors
            if isinstance(vmin,int):
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = matplotlib.colors.Normalize(vmin=min(draw_df[hue]), vmax=max(draw_df[hue]))
            print(norm)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # plt.colorbar(sm)
            colorbar = plt.colorbar(sm, shrink=1,ax=ax, orientation='horizontal')
            # colorbar = plt.colorbar(sm, cax=cbar_ax, orientation='vertical', pad=0.05)
            colorbar.ax.set_title(hue, fontsize = 12)
            # colorbar.ax.set_position([1.1, 0.35, 0.55, 0.35]) # kegg
            colorbar.ax.set_position(color_bar_pos)

        if showlabel == False:
            plt.xlabel('')
            plt.ylabel('')
        else:
            if xlabel:
                plt.xlabel(xlabel,fontsize = 22)
            else:
                plt.xlabel('',fontsize = 22)

            if ylabel:
                plt.ylabel(ylabel,fontsize = 22)
            else:
                plt.ylabel('',fontsize = 22)

        if title:
            plt.title(title,fontsize=16)

        if xrotate:
            plt.xticks(rotation=90)

        plt.xlim(-0.5, len(draw_df[x].unique())-0.5) 
        plt.ylim(len(draw_df[y].unique())-0.5,-0.5)   
        # if len(draw_df[x].unique()) == 2:
        #     plt.xlim(-0.5, 1.5)
        #     # plt.ylim(10, -0.8)
        # elif len(draw_df[x].unique()) == 3:
        #     plt.xlim(-0.5, 2.5)
        if savefig:
            plt.savefig(f'{savefig}', bbox_inches='tight')
        plt.show()
        plt.clf()


def old_draw_bubble(draw_df, x=None, y="Description", x_highlight = None, y_highlight=None, 
                savefig=None, figsize=None, legend=False, xrotate=False,
                showlabel=False, xlabel = None, ylabel = None, title=None, 
                  cmap='viridis',hue='Count', size='GeneRatio',vmin = None, vmax = None):
    '''
    Red label item in x, y axis
    highlight = 'T_cells_CD4+'
    or
    highlight = ['T_cells_CD4+','B cell]
    '''
    if figsize is None:
        width = np.max([int(np.ceil(len(draw_df[x].unique()) / 3.5)), 2])+0.2
        height = int(np.ceil(len(draw_df[y].unique())/4)+1)
        if xlabel:
            height += 0.5
        if ylabel:
            width += 0.5
        print(f'Auto figsize {(width,height)}')
    else:
        width = figsize[0]
        height = figsize[1]

    # print(width,height)
    with sns.axes_style("whitegrid"):
        plt.figure(figsize=(width, height))
        ax = sns.scatterplot(data = draw_df, x = x, y = y, hue = hue, 
                        legend = legend,
                        palette = cmap, 
                        size = size, sizes = (50,300)
                        )
        
        highlight_x(draw_df, x, x_highlight)
        highlight_y(draw_df, y, y_highlight)

        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)

        if legend:
            sizes = np.percentile(draw_df[size], [25, 50, 75, 100])
            sizes = np.round(sizes, 2)
            labels = np.percentile(range(50,300), [25, 50, 75, 100])
            labels = np.round(labels, 0)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor='black', markersize=labels[i]/20, label=sizes[i]) 
                            for i in range(len(sizes))]
            plt.legend(handles=legend_elements, title=size, title_fontsize=14, loc='center left', bbox_to_anchor=(1.25, 0.5), 
                       fontsize=14, handlelength = 0.5, handletextpad=0.5,labelspacing=0.2)
        
            import matplotlib.colors
            if isinstance(vmin,int):
                norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = matplotlib.colors.Normalize(vmin=min(draw_df[hue]), vmax=max(draw_df[hue]))
            print(norm)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            # plt.colorbar(sm)
            colorbar = plt.colorbar(sm, shrink=0.6,ax=ax)
            colorbar.ax.set_title(hue, fontsize = 14)
        if showlabel == False:
            plt.xlabel('')
            plt.ylabel('')
        else:
            if xlabel:
                plt.xlabel(xlabel,fontsize = 22)
            else:
                plt.xlabel('',fontsize = 22)

            if ylabel:
                plt.ylabel(ylabel,fontsize = 22)
            else:
                plt.ylabel('',fontsize = 22)

        if title:
            plt.title(title,fontsize=22)

        if xrotate:
            plt.xticks(rotation=90)

        plt.xlim(-0.5, len(draw_df[x].unique())-0.5) 
        plt.ylim(len(draw_df[y].unique())-0.5,-0.5)   
        # if len(draw_df[x].unique()) == 2:
        #     plt.xlim(-0.5, 1.5)
        #     # plt.ylim(10, -0.8)
        # elif len(draw_df[x].unique()) == 3:
        #     plt.xlim(-0.5, 2.5)
        if savefig:
            plt.savefig(f'{savefig}', bbox_inches='tight')
        plt.show()
        plt.clf()




def celltype_cci(adata, is_sender_per = True, figsize = (3.6,3.6), 
                 target_sender = None, target_receiver = None, y_highlight='', x_highlight='',
                cmap = 'Reds', hue = 'LRI',showlabel = True,
                vmin = None, vmax = None,
                size = 'CCI',xrotate = True, legend = True, color_bar_pos = [1.1, 0.35, 0.55, 0.35],
                savefig = False, title = ''):
    '''
    is_sender_per: if True, use the cci percentage file that sender row sum is 1 [send_per].
    '''
    if is_sender_per:
       df = adata.uns['send_per']
       name = 'sender'
    else:
        df = adata.uns['rec_per']
        name = 'receiver'
    # TODO move lri here, to subset 
    draw_df = eva.generate_tp_lri(adata,df,target_sender,target_receiver)
    if savefig:
        save_path = adata.uns['figpath']
        savefig = f'{save_path}/{name}_CCI.pdf'
    
    draw_bubble(draw_df, x='Receiver', y='Sender', x_highlight = x_highlight, y_highlight=y_highlight,
                    savefig = savefig, title = title, figsize = figsize, legend = legend,
                    vmin = vmin, vmax = vmax,
                    showlabel = showlabel, xrotate = xrotate, cmap=cmap, hue=hue, size=size, color_bar_pos = color_bar_pos)
    

def clustermap(df, index = 'ligand', col = 'receptor', value = 'lr_co_exp_num',
                 aggfunc = 'sum', log = True, scale = False, row_cluster = False, col_cluster = False,
                 highlight = None,  cmap = 'coolwarm', title = '', rotate_x = True,
                 xticks = False, yticks = True, xlabel = True, ylabel = True, cbar_x = 1.2,
                 figsize = (5,5),col_colors = None, row_colors = None,
                 savefig = False):
    '''
    df: dataframe
    if df has there columns and required to be pivot, then use the following parameters
    otherwise set aggfunc as None
        index: row index
        col: column index
        value: value to fill the pivot table
        aggfunc: how to aggregate the value
    log: log the value or not
    row_cluster: cluster row or not
    col_cluster: cluster column or not
    highlight: highlight the rows
    title: title of the plot
    cmap: color map of heatmap
    col_color: color of the column [sorted by the df row's order]
    xticks: show xticks or not
    yticks: show yticks or not
    savefig: output directory
    '''
    if aggfunc is None:
        n_tp_lri = df.copy()
    else:
        if aggfunc == 'sum':
            n_tp_lri = pd.crosstab(index=df[index], columns = df[col],values=df[value],aggfunc = sum)
    n_tp_lri = n_tp_lri.fillna(0)
    if log:
        n_tp_lri = np.log(n_tp_lri + 1)
        legend_label = f'{value} (log)'
    else:
        legend_label = f'{value}'

    if scale == True:
        n_tp_lri = n_tp_lri.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        
    clustermap = sns.clustermap(n_tp_lri,row_cluster=row_cluster,col_cluster=col_cluster,
               standard_scale=None,dendrogram_ratio=0.001,cmap = cmap, col_colors = col_colors,row_colors = row_colors,
               figsize=figsize, cbar_pos=(cbar_x, 0.5, 0.02, 0.2),cbar_kws={'orientation': 'vertical','label':legend_label})
    if not xticks:
        clustermap.ax_heatmap.set_xticklabels([])
        clustermap.ax_heatmap.set_xticks([])
    else:
        clustermap.ax_heatmap.xaxis.label.set_size(18)
        clustermap.ax_heatmap.xaxis.set_tick_params(labelsize=16)

    
    if not yticks:
        clustermap.ax_heatmap.set_yticklabels([])
        clustermap.ax_heatmap.set_yticks([])
    else:
        clustermap.ax_heatmap.yaxis.label.set_size(18)
        clustermap.ax_heatmap.yaxis.set_tick_params(labelsize=16)

    if not xlabel:
        clustermap.ax_heatmap.set_xlabel('')
    else:
        if xlabel != True:
            clustermap.ax_heatmap.set_xlabel(xlabel)
    if not ylabel:
        clustermap.ax_heatmap.set_ylabel('')
    else:
        if ylabel != True:
            clustermap.ax_heatmap.set_ylabel(ylabel)

    if highlight is not None:
        arr = n_tp_lri.index.to_list()
        if row_cluster:
            # Reorganize the index labels based on the cluster order
            reordered_index = [arr[i] for i in clustermap.dendrogram_row.reordered_ind]
            # Customize the ytick color and font weight
            yticks, _ = plt.yticks()
            ytick_labels = clustermap.ax_heatmap.get_yticklabels()
            for index, ytick_label in enumerate(ytick_labels):
                if reordered_index[index] in highlight:
                    ytick_label.set_color('red')
                    ytick_label.set_weight('bold')
        else:
            highlighted_ytick = np.where(np.isin(arr, highlight))[0]
            print(highlighted_ytick) # The ytick to be highlighted
            # Customize the ytick color
            _, yticks = plt.yticks()
            print(yticks)
            ytick_labels = clustermap.ax_heatmap.get_yticklabels()
            for index, ytick_label in enumerate(ytick_labels):
                if arr[index] in highlight:
                    ytick_label.set_color('red')
                    ytick_label.set_weight('bold')
    if title:
        clustermap.ax_heatmap.set_title(title,fontsize=18, y = 1.05) 

    if savefig:
        plt.savefig(f'{savefig}', bbox_inches='tight')

    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)

    if rotate_x:
        # print('r')
        plt.xticks(fontsize=16, rotation=90)
    plt.show()


def lri_heatmap(adata, is_sender = False, target_sender = [], target_receiver = [], title = None, unique_lri = False,
                figsize = (8,3), log = True, row_cluster = True, col_cluster = False,
                highlight_lri = None, cmap = 'Reds',cellTypeAsCol = True,
                xticks = True, yticks = True,savefig = False):

    lri_df = adata.uns['spatalk']
    if len(target_sender) == 0:
        target_sender = lri_df['celltype_sender'].unique()
    
    if len(target_receiver) == 0:
        target_receiver = lri_df['celltype_receiver'].unique()

    if is_sender:
        col_type = 'celltype_sender'    
    else:
        col_type = 'celltype_receiver'

    lri_df = lri_df[lri_df['celltype_sender'].isin(target_sender) & lri_df['celltype_receiver'].isin(target_receiver)].copy()
    if unique_lri:
        df = lri_df.groupby('LRI').count()
        uniq_lri = df[df['ligand'] == 1].index.tolist()
        if isinstance(unique_lri,bool):
            uniq_count_thred = np.round(df["ligand"].mean())
        elif isinstance(unique_lri,int):
            uniq_count_thred = unique_lri
        else:
            raise ValueError('unique_lri should be bool or int')
        loose_uniq_lri = df[df['ligand'] <= uniq_count_thred].index.tolist()
        lri_df = lri_df[lri_df['LRI'].isin(loose_uniq_lri)].copy()
        if uniq_count_thred>1:
            highlight = uniq_lri + (highlight_lri or [])
        else:
            highlight = highlight_lri
    else:
        highlight = highlight_lri

    if savefig:
        save_path = adata.uns['figpath']
        savefig = f'{save_path}/{title}_LRI_co_exp_heatmap.pdf'
    if cellTypeAsCol:
        clustermap(lri_df,index = 'LRI', col = col_type, value = 'lr_co_exp_num',
                        aggfunc = 'sum', log = log, row_cluster = row_cluster, col_cluster = col_cluster,
                        figsize = figsize, cmap = cmap, title = title, savefig = savefig,
                        highlight = highlight, xticks = xticks, yticks = yticks)        
    else:
        clustermap(lri_df,index = col_type, col = 'LRI', value = 'lr_co_exp_num',
                        aggfunc = 'sum', log = log, row_cluster = row_cluster, col_cluster = col_cluster,
                        figsize = figsize, cmap = cmap, title = title, savefig = savefig,
                        highlight = highlight, xticks = xticks, yticks = yticks)
    return lri_df



def top_kegg_enrichment(adata, top_n = 10, groupby_sender = False,
                        target_sender = [], target_receiver = [],target_path = [], 
                        use_lig_gene = True, use_rec_gene = True,
                        cmap = 'RdPu', hue = '-log10 pvalue', size = 'GeneRatio',
                        legend = True, figsize = None, savefig = False,
                        color_bar_pos = [1.1, 0.35, 0.55, 0.35]):
    # TODO discard cancer related
    kegg_res = adata.uns['kegg_enrichment']

    lig_gene = 'L' if use_lig_gene else 'n'
    rec_gene = 'R' if use_rec_gene else 'n'
    used_genes = f'{lig_gene}_{rec_gene}'
    kegg_res = kegg_res[kegg_res['celltype_sender'].isin(target_sender) & 
                        kegg_res['celltype_receiver'].isin(target_receiver) & 
                        (kegg_res['used_genes'] == used_genes)]
    if groupby_sender:
        show_row = 'celltype_sender'
        xlabel = 'Sender'
        target_row = 'celltype_receiver'
        targets = target_receiver
    else:
        show_row = 'celltype_receiver'
        xlabel = 'Receiver'
        target_row = 'celltype_sender'
        targets = target_sender        

    # TODO subset by sender, plot two with title why plot together
    top_keggs = pd.DataFrame()
    for target in targets:
        title = f'{target}'
        top_kegg = kegg_res[kegg_res[target_row] == target].copy()
        top_kegg = top_kegg.groupby([show_row]).apply(lambda x: x.nlargest(top_n, 'Count')).reset_index(drop=True)
        uniq_path = list(set(top_kegg['Description']))
        # incase some path shows but not in top
        top_kegg = kegg_res[kegg_res['Description'].isin(uniq_path)].copy()
        # top_kegg = top_kegg[~top_kegg['Description'].isin(discard)]
        top_kegg = top_kegg.sort_values(by = [show_row,'Count'],ascending = [True,False])
        top_keggs = pd.concat([top_keggs,top_kegg]) 
        draw_bubble(top_kegg, x = show_row, y_highlight = target_path,
                    y = "Description",cmap = cmap, hue = hue,
                    size = size,xrotate = True,legend = legend, 
                    title = title, showlabel = True,
                    xlabel = xlabel,figsize = figsize,savefig = savefig,
                    color_bar_pos = color_bar_pos)
        plt.clf()
    return top_keggs



def keggPath_lri(adata, groupby_sender = False, value = 'lr_co_exp_num', thred = 5, top_n_each = 999,
                target_sender = [], target_receiver = [], target_path = '', 
                use_lig_gene = True, use_rec_gene = True,
                savefig = False, show_title = True, unique_lri = None,
                figsize = None, log = True, row_cluster = True, col_cluster = False,
                highlight = None, cmap = 'Reds',cellTypeAsCol = True,
                xticks = True, yticks = True):
    kegg_res = adata.uns['kegg_enrichment']
    lri_df = adata.uns['spatalk']

    lig_gene = 'L' if use_lig_gene else 'n'
    rec_gene = 'R' if use_rec_gene else 'n'
    used_genes = f'{lig_gene}_{rec_gene}'

    kegg_res = kegg_res[kegg_res['celltype_sender'].isin(target_sender) & 
                        kegg_res['celltype_receiver'].isin(target_receiver) & 
                        (kegg_res['used_genes'] == used_genes) &
                        (kegg_res['Description'] == target_path)]
    if len(kegg_res) == 0:
        raise ValueError(f'No {target_path} found in the enrichment result.')
    
    lri_df = lri_df[lri_df['celltype_sender'].isin(target_sender) &
                    lri_df['celltype_receiver'].isin(target_receiver)]
    
    if unique_lri:
        df = lri_df.groupby(['celltype_sender','LRI']).count()
        df.reset_index(inplace = True)
        if isinstance(unique_lri,bool):
            uniq_count_thred = int(np.round(df["ligand"].mean()))
            # print('boolean',uniq_count_thred)
        elif isinstance(unique_lri,int):
            uniq_count_thred = unique_lri
            # print('int',uniq_count_thred)
        else:
            raise ValueError('unique_lri should be bool or int')
        loose_uniq_lri = df[df['ligand'] <= uniq_count_thred]['LRI'].tolist()
        lri_df = lri_df[lri_df['LRI'].isin(loose_uniq_lri)].copy()

    
    path_genes = []
    for _, row in kegg_res.iterrows():
        tmp_genes = row['geneSymbol'].split('/')
        # print(len(tmp_genes))
        path_genes.extend(tmp_genes)
    path_genes = list(set(path_genes))

    if (lig_gene == 'L') and (rec_gene == 'R'):
        target_df = lri_df[lri_df['ligand'].isin(path_genes) & lri_df['receptor'].isin(path_genes)]
    elif (lig_gene == 'L') and (rec_gene == 'n'):
        target_df = lri_df[lri_df['ligand'].isin(path_genes)]
    elif (lig_gene == 'n') and (rec_gene == 'R'):
        target_df = lri_df[lri_df['receptor'].isin(path_genes)]
    else:
        raise ValueError('Both use_lig_gene and use_rec_gene are False. No gene used.')
    
    target_df = target_df[target_df[value]>thred].copy()
    target_df = target_df.groupby('CCI').apply(lambda x: x.nlargest(top_n_each, value)).reset_index(drop=True)

    if savefig:
        save_path = adata.uns['figpath']
        savefig = f'{save_path}/LRI_{target_path}_heatmap.pdf'

    if groupby_sender:
        show_row = 'celltype_sender'
    else:
        show_row = 'celltype_receiver'

    if show_title:
        title = target_path

    if cellTypeAsCol:
        clustermap(target_df,index = 'LRI', col = show_row, value = value,
                        aggfunc = 'sum', log = log, row_cluster = row_cluster, col_cluster = col_cluster,
                        figsize = figsize, cmap = cmap, title = title, savefig = savefig,
                        highlight = highlight, xticks = xticks, yticks = yticks)        
    else:
        clustermap(lri_df,index = 'celltype_receiver', col = 'LRI', value = value,
                        aggfunc = 'sum', log = log, row_cluster = row_cluster, col_cluster = col_cluster,
                        figsize = figsize, cmap = cmap, title = title, savefig = savefig,
                        highlight = highlight, xticks = xticks, yticks = yticks)
    return target_df



def keggPaths_lri_flow(adata,target_sender = [], target_receiver = [], target_path = [], 
                use_lig_gene = True, use_rec_gene = True,value = 'lr_co_exp_num', thred = 5, top_n_each = 5):
    kegg_res = adata.uns['kegg_enrichment']
    lri_df = adata.uns['spatalk']

    lig_gene = 'L' if use_lig_gene else 'n'
    rec_gene = 'R' if use_rec_gene else 'n'
    used_genes = f'{lig_gene}_{rec_gene}'

    kegg_res = kegg_res[kegg_res['celltype_sender'].isin(target_sender) & 
                        kegg_res['celltype_receiver'].isin(target_receiver) & 
                        (kegg_res['used_genes'] == used_genes) &
                        (kegg_res['Description'].isin(target_path))]
    if len(kegg_res) == 0:
        raise ValueError(f'No {target_path} found in the enrichment result.')
    
    lri_df = lri_df[lri_df['celltype_sender'].isin(target_sender) &
                    lri_df['celltype_receiver'].isin(target_receiver)]
    
    
    all_target_df = pd.DataFrame()
    for path in target_path:
        path_genes = []
        tmp_kegg = kegg_res[kegg_res['Description'] == path]
        tmp_genes = tmp_kegg['geneSymbol'].values
        for _, row in tmp_kegg.iterrows():
            tmp_genes = row['geneSymbol'].split('/')
            # print(len(tmp_genes))
            path_genes.extend(tmp_genes)
        path_genes = list(set(path_genes))
        print(len(path_genes))
        if (lig_gene == 'L') and (rec_gene == 'R'):
            target_df = lri_df[lri_df['ligand'].isin(path_genes) | lri_df['receptor'].isin(path_genes)].copy()
        elif (lig_gene == 'L') and (rec_gene == 'n'):
            target_df = lri_df[lri_df['ligand'].isin(path_genes)].copy()
        elif (lig_gene == 'n') and (rec_gene == 'R'):
            target_df = lri_df[lri_df['receptor'].isin(path_genes)].copy()
        target_df['pathway'] = path
        all_target_df = pd.concat([all_target_df,target_df])
    all_target_df = all_target_df[all_target_df[value]>thred].copy()
    all_target_df = all_target_df.groupby(['CCI','pathway']).apply(lambda x: x.nlargest(top_n_each, value)).reset_index(drop=True)
    all_target_df = all_target_df[['LRI','ligand','receptor','pathway']].drop_duplicates()
    return all_target_df




def draw_lr_flow2(df, left_panel = 'ligand', right_panel = 'receptor',
                 figsize = (10,10)):
    import plotly.graph_objects as go
    ligs = list(df[left_panel].unique())
    recs = list(df[right_panel].unique())
    labels = list(df[left_panel].unique())
    label_rec = list(df[right_panel].unique())
    labels.extend(label_rec)
    # define source and target indices for each link
    source = df[left_panel].astype('category').cat.codes.tolist()
    target = df[right_panel].astype('category').cat.codes.tolist()
    target = [x + len(set(source)) for x in target]
    value = [np.random.randint(1, 2) for _ in range((len(df)))]
    trace = go.Sankey(
        node=dict(
            pad=5,
            thickness=20,
            line=dict(color='black', width=0.1),
            label=labels,
            color=["#F56867"]*len(ligs) + ["#3A84E6"]*len(recs)
        ),
        link=dict(
            source=source, # indices correspond to labels, eg A1, A2, A1, B1 
            target=target, 
            value=value,
        )
    )
    # create layout
    layout = go.Layout(
        title='',
        font=dict(size=18)
    )
    # create figure
    fig = go.Figure(data=[trace], layout=layout)
    width = figsize[0]*100
    height = figsize[1]*100
    fig.update_layout(width=width, height=height)
    # fig.write_image(f'./8.Ref/figures/main/lr_network.pdf') 
    fig.show()
    return


def draw_lr_flow3(df, left_panel = 'ligand', mid_panel = 'receptor', right_panel = 'pathway',
                 figsize = (10,10), outDir = None):
    import plotly.graph_objects as go
    ligs = list(df[left_panel].unique())
    recs = list(df[mid_panel].unique())
    paths = list(df[right_panel].unique())

    labels = list(df[left_panel].unique())
    label_rec = list(df[mid_panel].unique())
    label_path = list(df[right_panel].unique())
    labels.extend(label_rec)
    labels.extend(label_path)
    # print('label',labels)
    label_num_map = dict(zip(labels,range(len(labels))))
    source1 = list(df[left_panel].map(label_num_map))
    target1 = list(df[mid_panel].map(label_num_map))
    target2 = list(df[right_panel].map(label_num_map))
    source = source1 + target1
    target = target1 + target2
    # print('source1',source1)
    # print('target1',target1)
    # print('target2',target2)
    # print([0.1]*len(ligs) + [0.2]*len(recs) + [10]*len(paths))
    value = [np.random.randint(1, 2) for _ in range((len(source)))]
    trace = go.Sankey(
        node=dict(
            pad=5,
            thickness=20,
            line=dict(color='gray', width=0.1),
            label=labels,
            color=["#F56867"]*len(ligs) + ["#3A84E6"]*len(recs) + ["#59BE86"] *len(paths),
            x = [0.1]*len(ligs) + [0.3]*len(recs) + [0.9]*len(paths),
            # y = [0.1]*len(ligs) + [0.1]*len(recs) + [0.1]*len(paths),
            # y = [0.5]*len(ligs) + [0.5]*len(recs) + [0.5]*len(paths),
        ),
        link=dict(
            source=source, # indices correspond to labels, eg A1, A2, A1, B1 
            target=target, 
            value=value,
            color = '#e7e7e7'
        )
    )
    # create layout
    layout = go.Layout(
        title='',
        font=dict(size=24,color='black'),
    )
    # create figure
    fig = go.Figure(data=[trace], layout=layout)
    # print(fig.data[0]['node']['x'])
    # print(fig.data[0]['node']['y'])
    width = figsize[0]*100
    height = figsize[1]*100
    fig.update_layout(width=width, height=height)
    fig.write_image(f'{outDir}')
    fig.show()


def spatial_gene_exp(adata, gene = None, method = None, group = None,
                     log = True, scale = True, size = 10, title = None,
                     hl_type = [], tp_key = 'celltype', rect = False,
                    theme_white = True, legend = True, figsize = (3.4, 2.6)):
    
    exp = adata.to_df()
    meta = adata.obs
    # if gene is list
    if isinstance(gene, list):
        tmp = exp[gene].sum(axis = 1)
        exp[group] = tmp
        gene = group

    cmap = plt.cm.plasma
    cmap = plt.cm.gnuplot2

    # last_color = cmap.colors[0]
    last_color = 'black'
    data = pd.DataFrame(exp[gene],columns=[gene],index = exp.index)

    if log == True:
        data = pd.DataFrame(np.log(data+1),columns=[gene],index = exp.index)

    if scale == True:
        data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    if 'adj_spex_UMAP1' in meta.columns:
        cols = ['adj_spex_UMAP1','adj_spex_UMAP2']
    elif 'adj_UMAP1' in meta.columns:
        cols = ['adj_UMAP1','adj_UMAP2']
    else:
        cols = ['x','y']

    data = pd.concat((data,meta[cols]),axis = 1)
    # plt.figure(figsize=figsize)
    if hl_type:
        hl_idx = adata.obs[adata.obs[tp_key].isin(hl_type)].index
        sub_data = data.loc[hl_idx].copy()
        bg_data = data[~data.index.isin(hl_idx)] 
        with sns.axes_style("white"):
            plt.scatter(x=bg_data[cols[0]], y=bg_data[cols[1]],
                        s=size, linewidth=False, color = '#ccc')
            plt.scatter(x=sub_data[cols[0]], y=sub_data[cols[1]], c=sub_data[gene], 
                        s=size+5, linewidth=False, cmap=cmap)
            title_text = f'{method} {gene} of {" ".join(hl_type)}'
        if rect and isinstance(rect, bool):
            x_min,y_min = sub_data.loc[hl_idx][cols].min()
            x_max,y_max = sub_data.loc[hl_idx][cols].max()
            rect_width = x_max - x_min  # Width of the rectangle
            rect_height = y_max - y_min  # Height of the rectangle
            rect_color = 'red'  # Rectangle color 
            rectangle = plt.Rectangle((x_min, y_min), rect_width, rect_height, fill=False, color=rect_color, linewidth=2)
            plt.gca().add_patch(rectangle)
    else:
        bg_data = data[data[gene] <= 0] 
        if len(bg_data) > 1:
            with sns.axes_style("white"):
                plt.scatter(x=bg_data[cols[0]], y=bg_data[cols[1]], 
                            s=size, linewidth=False, color = last_color) 
            # Filter the data where gene is not equal to 0
                filtered_data = data[data[gene] > 0]
        else:
            filtered_data = data
        with sns.axes_style("white"):
            plt.scatter(x=filtered_data[cols[0]], y=filtered_data[cols[1]], c=filtered_data[gene], 
                        s=size, linewidth=False, cmap=cmap)
            title_text = f'{method} {gene}'

    if title:
        plt.title(title,fontsize=22)
    else:
        plt.title(title_text,fontsize=22)

    if rect and isinstance(rect, list):
        x_min,y_min = rect[0]
        x_max,y_max = rect[1]
        rect_width = x_max - x_min  # Width of the rectangle
        rect_height = y_max - y_min  # Height of the rectangle
        rect_color = 'red'  # Rectangle color 
        rectangle = plt.Rectangle((x_min, y_min), rect_width, rect_height, fill=False, color=rect_color, linewidth=2)
        plt.gca().add_patch(rectangle)

    if legend == True:
        plt.colorbar(ticks = [0,1], label = 'Expression', anchor=(0,0.5))
        
    plt.axis('equal')
    # print(filtered_data[cols[0]].min(),filtered_data[cols[0]].max())
    # print(np.ceil(filtered_data[cols[0]].min()))
    # plt.xlim(np.ceil(filtered_data[cols[0]].min()), np.ceil(filtered_data[cols[0]].max()))
    # plt.xlim(-4.5, -2)

    if theme_white:
        plt.legend([],[], frameon=False)  
        plt.tick_params(left=False, bottom=False, top = False, right = False)
        plt.tight_layout()
        # sns.despine(left=True, bottom=True)
        plt.xlabel('',fontsize=16)
        plt.ylabel('',fontsize=16)
        plt.xticks([],fontsize=14)
        plt.yticks([],fontsize=14)
    else:
        plt.tick_params(left=False, bottom=False, top = False, right = False)
        plt.tight_layout()
        # sns.despine(left=True, bottom=True)
        plt.xlabel('',fontsize=16)
        plt.ylabel('',fontsize=16)
        plt.xticks([],fontsize=14)
        plt.yticks([],fontsize=14)



def LRI_of_CCI(adata_orig, sender = '',receiver = '',ligand = '', receptor = '',
                hue = '',color_map = None,figsize = (6,3),arrow_length = 0.015, size = 30,
                cols = ['adj_spex_UMAP1','adj_spex_UMAP2'], legend = True, show_pair = False,
                title = None, subset = True, savefig = None):
    adata = adata_orig[adata_orig.uns['spatalk_meta'].index.astype(str)]
    cellpair = adata.uns['cellpair']
    cols = ['adj_spex_UMAP1','adj_spex_UMAP2']
    st_cols = ['st_x','st_y']
    adata.obs.index = adata.uns['spatalk_meta']['cell']
    meta = adata.obs.copy()
    # drop Categories 
    meta[hue] = meta[hue].astype(object)
    st_spots = meta[st_cols].drop_duplicates()
    target_cellpair = cellpair[(cellpair['sender_tp'] == sender)&(cellpair['receiver_tp'] == receiver)].copy()
    # print(target_cellpair)
    target_cellpair[ligand] = adata[target_cellpair['cell_sender'],ligand].to_df().values
    target_cellpair[receptor] = adata[target_cellpair['cell_receiver'],receptor].to_df().values
    # print(target_cellpair.head(5))
    draw_df = meta.loc[target_cellpair['cell_sender'].tolist()+target_cellpair['cell_receiver'].tolist()].copy()
    if subset:
    # cells in selected area
        draw_bg=meta[(meta[cols[0]]>=draw_df[cols].min()[cols[0]])&(meta[cols[0]]<=draw_df[cols].max()[cols[0]])&
                (meta[cols[1]]>=draw_df[cols].min()[cols[1]])&(meta[cols[1]]<=draw_df[cols].max()[cols[1]])]
    else:
        draw_bg = meta
    # print(draw_bg[hue].unique())
    celltype_df = draw_bg[draw_bg[hue].isin([sender,receiver])].copy()
    # print(celltype_df[hue].unique())
    if show_pair:
        # only show the pair of cells that detected by spatalk
        target_cellpair_arrow = target_cellpair.copy()
        cellpairs = len(target_cellpair)
        title = f'{sender} - {receiver} Cell pairs: {cellpairs}'
    else:
        target_cellpair_arrow = target_cellpair[(target_cellpair[ligand] >0)&(target_cellpair[receptor] >0)].copy()
        # print(target_cellpair_arrow)
    plt.figure(figsize=figsize)
    sns.scatterplot(data = draw_bg, x = cols[0], y=cols[1], 
                    s = 10, alpha = 0.4,c=['#ccc'],edgecolor = None)
    
    sns.scatterplot(data = celltype_df, x=cols[0], y=cols[1],hue = hue, 
                    s = size,alpha = 1, palette=color_map, edgecolor = None)      
    
    for _,row in target_cellpair_arrow.iterrows():
        start = meta.loc[row['cell_sender']][cols]
        end = meta.loc[row['cell_receiver']][cols]
        color = 'black'
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        plt.arrow(start[0], start[1], dx, dy,
                color=color, width=0.001, length_includes_head=True,
                # head_length=arrow_length, head_width=arrow_length, 
          head_width=arrow_length/2, head_length=arrow_length,
          overhang = 0.2, alpha = 0.8,
          head_starts_at_zero=False)
  
    # plt.legend([],[], frameon=Fase)
    if not title:
        if figsize[0] > figsize[1]:
            # landscape
            plt.title(f'{sender}: {ligand} to {receiver}: {receptor}',fontsize=16)
        else:
            # portrait
            plt.title(f'{sender}: {ligand} to \n {receiver}: {receptor}',fontsize=16)
    else:
        plt.title(title,fontsize=16)
    if legend:
        leg = plt.legend(loc='center left', bbox_to_anchor=(0.99, 0.5),
                                ncol=1, handletextpad=0.5,columnspacing=0.4,labelspacing=0.5,
                                fontsize = 16,markerscale = 2,handlelength = 0.5)
        leg.get_frame().set_linewidth(0.0)  # Remove legend frame
    else:
        plt.legend([],[], frameon=False)
    plt.xlabel('',fontsize=16)
    plt.ylabel('',fontsize=16)
    plt.xticks([],fontsize=14)
    plt.yticks([],fontsize=14)
    plt.axis('equal')
    if savefig:
        save_path = adata.uns['figpath']
        savefig = f'{save_path}/{sender}_{receiver}_{ligand}_{receiver}_arrow.pdf'
        plt.savefig(f'{savefig}', dpi=300, bbox_inches='tight')
        print(f'Figure saved {savefig}')
    plt.show()
    #########################
    if subset:
        plt.figure(figsize=(4,3.2))
        rect_x,rect_y = draw_df[cols].min() # X-coordinate of the bottom-left corner
        rect_y = draw_df[cols].min()[1]  # Y-coordinate of the bottom-left corner
        rect_width = draw_df[cols].max()[0] - rect_x  # Width of the rectangle
        rect_height = draw_df[cols].max()[1] - rect_y  # Height of the rectangle
        rect_color = 'black'  # Rectangle color
        
        sns.scatterplot(data=st_spots, x = st_cols[0], y=st_cols[1],c = ['#ccc'],
                s = 10,alpha = 1,  edgecolor = None)
        # sns.scatterplot(data=meta, x = cols[0], y=cols[1],hue = hue, 
                        # s = 10,alpha = 1, palette=color_map, edgecolor = None)
        sns.scatterplot(data=celltype_df, x=cols[0], y=cols[1],hue = hue, 
                        s = 10,alpha = 1,palette=color_map,edgecolor = None)
        # Draw the rectangle using Matplotlib
        rectangle = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, fill=False, color=rect_color)
        plt.gca().add_patch(rectangle)
        plt.legend([],[], frameon=False)
        plt.title(f'{sender} to {receiver}',fontsize=16)
        sns.despine(left=True, bottom=True)
        plt.xlabel('',fontsize=16)
        plt.ylabel('',fontsize=16)
        plt.xticks([],fontsize=14)
        plt.yticks([],fontsize=14)
        plt.axis('equal')
        if savefig:
            save_path = adata.uns['figpath']
            savefig = f'{save_path}/{sender}_{receiver}_{ligand}_{receiver}_region.pdf'
            plt.savefig(f'{savefig}', dpi=300, bbox_inches='tight')
            print(f'Figure saved {savefig}')
        plt.show()






def patterns(adata, savefig = False, cmap = None, COL = None):
    import re
    pattern = adata.uns['pattern']
    p_names = [col for col in pattern.columns if re.match(r'^\d+$', col)]
    if COL is None:
        ROW = int(np.floor(np.sqrt(len(p_names))))
        COL = int(np.ceil(len(p_names) / ROW))
        ROW = ROW * 3
    else:
        ROW = int(np.ceil(len(p_names) / COL)) *3
    ROW_L = 3
    COL_L = 4
    plt.figure(figsize=(COL_L * COL, ROW_L * ROW))
    i = 0
    colname = ['adj_spex_UMAP1', 'adj_spex_UMAP2']
    size = 5
    df = adata.obs
    for p in p_names:
        plt.subplot(ROW, COL, i + 1)
        plt.scatter(df[colname[0]], df[colname[1]], c=pattern[str(p)], s=size,cmap = cmap)
        plt.axis('equal')
        plt.title(f'Pattern {p}',fontsize=22)
        plt.colorbar(ticks = [0,1],label = 'Standardized expression')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        i += 1
        plt.subplots_adjust(wspace=0.25,hspace=0.5)
    if savefig:
        if isinstance(savefig,str):
            if '/' in savefig:
                savefig = f'{savefig}'
            else:
                save_path = adata.uns['figpath']
                savefig = f'{save_path}/{savefig}'
        else:
            save_path = adata.uns['figpath']
            savefig = f'{save_path}/patterns.pdf'
        plt.savefig(f'{savefig}', dpi=300, bbox_inches='tight')


def celltype_pie(adata, meta, target = None, thred = 5, cmap = None, title = None, savefig = False):
    draw_df = meta.groupby(target).count()
    draw_df.sort_values(draw_df.columns[0], ascending=False, inplace=True)
    
    # Calculate percentages
    percentages = draw_df.iloc[:, 0] / draw_df.iloc[:, 0].sum() * 100
    
    draw_df = pd.DataFrame(percentages)
    draw_df['celltype'] = draw_df.index
    draw_df['celltype'] = draw_df['celltype'].astype(str)
    draw_df.columns = ['per', 'celltype']
    draw_df.loc[draw_df['per'] < thred, 'celltype'] = 'Other cell types'
    # print(draw_df)
    labels = [label if pct >= thred else '' for label, pct in zip(draw_df['celltype'], percentages)]
    # Conditionally set the label format for percentages
    def label_format(pct):
        if pct >= thred:
            return f'{pct:.1f}%'
        else:
            return ''
    
    # Plot the pie chart
    plt.pie(draw_df['per'].values, labels=labels, colors=[cmap[x] for x in draw_df['celltype']],
            autopct=label_format, textprops={'fontsize': 18})
    
    if title:
        plt.title(title)
    
    if savefig:
        save_path = adata.uns['figpath']
        savefig = f'{save_path}/{title}_pie.pdf'
        plt.savefig(f'{savefig}', bbox_inches='tight')
    plt.show()
    plt.clf()



def spatial_gene_score(adata, gene_score = None,
                     log=True,scale = True, spot_agg = False,
                     hl_type = [], tp_key = 'celltype', rect = False,
                     theme_white = True, legend = True, figsize = (3.4, 2.6),):
    meta = adata.obs
    cmap = plt.cm.plasma
    last_color = cmap.colors[0]
    # last_color = 'black'
    if 'adj_spex_UMAP1' in meta.columns:
        cols = ['adj_spex_UMAP1','adj_spex_UMAP2']
        size = 10
    elif 'adj_UMAP1' in meta.columns:
        cols = ['adj_UMAP1','adj_UMAP2']
        size = 10
    else:
        cols = ['x','y']
        size = 20
    
    if spot_agg:
        data = pd.DataFrame(meta.groupby('spot').mean(numeric_only=True)[gene_score])
        cols = ['st_x','st_y']
        size = 40
        coords = meta.groupby('spot').mean(numeric_only=True)[cols]
    else:
        data = pd.DataFrame(meta[gene_score].copy())
        coords = meta[cols]
    
    if log == True:
        data = np.log(data+1)

    if scale == True:
        data = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        
    data = pd.concat((data,coords),axis = 1)
    plt.figure(figsize=figsize)
    if hl_type:
        hl_idx = adata.obs[adata.obs[tp_key].isin(hl_type)].index
        subset_data = data.loc[hl_idx].copy()
        bg_data = data[~data.index.isin(hl_idx)] 
        with sns.axes_style("white"):
            plt.scatter(x=bg_data[cols[0]], y=bg_data[cols[1]],
                        s=size, linewidth=False, color = '#ccc')
            plt.scatter(x=subset_data[cols[0]], y=subset_data[cols[1]], c=subset_data[gene_score], 
                        s=size+5, linewidth=False, cmap=cmap)
            plt.title(f'{gene_score}', fontsize=22)
    else:
        bg_data = data[data[gene_score] <= 0] 
        if len(bg_data) > 1:
            with sns.axes_style("white"):
                plt.scatter(x=bg_data[cols[0]], y=bg_data[cols[1]], 
                            s=size, linewidth=False, color = last_color) 
            # Filter the data where gene is not equal to 0
                filtered_data = data[data[gene_score] > 0]
        else:
            filtered_data = data
        with sns.axes_style("white"):
            plt.scatter(x=filtered_data[cols[0]], y=filtered_data[cols[1]], c=filtered_data[gene_score], 
                        s=size, linewidth=False, cmap=cmap)
            plt.title(f'{gene_score}',fontsize=22)

    if rect: 
        if isinstance(rect, bool):
            x_min,y_min = subset_data.loc[hl_idx][cols].min()
            x_max,y_max = subset_data.loc[hl_idx][cols].max()
        elif isinstance(rect, list):
            x_min,y_min = rect[0]
            x_max,y_max = rect[1]
        rect_width = x_max - x_min  # Width of the rectangle
        rect_height = y_max - y_min  # Height of the rectangle
        rect_color = 'red'  # Rectangle color 
        rectangle = plt.Rectangle((x_min, y_min), rect_width, rect_height, fill=False, color=rect_color, linewidth=2)
        plt.gca().add_patch(rectangle)
        
    if legend:
        if scale:
            cbar = plt.colorbar(ticks = [0,1])
            cbar.set_label(label = 'Scaled score', fontsize=16)  
        else:
            cbar = plt.colorbar()
            cbar.set_label(label = 'Score', fontsize=16)  
        cbar.ax.tick_params(labelsize=16)
        
    plt.axis('equal')

    if theme_white:
        plt.legend([],[], frameon=False)  
        sns.despine(left=True, bottom=True)
        plt.tick_params(left=False, bottom=False, top = False, right = False)
        plt.tight_layout()
        sns.despine(left=True, bottom=True)
        plt.xlabel('',fontsize=16)
        plt.ylabel('',fontsize=16)
        plt.xticks([],fontsize=14)
        plt.yticks([],fontsize=14)
    else:
        plt.tick_params(left=False, bottom=False, top = False, right = False)
        plt.tight_layout()
        # sns.despine(left=True, bottom=True)
        plt.xlabel('',fontsize=16)
        plt.ylabel('',fontsize=16)
        plt.xticks([],fontsize=14)
        plt.yticks([],fontsize=14)
    plt.show()
#######################
###### Comparing ######
#######################
    
def venn2(before_lri,tp1,after_lri,tp2,venn_color,title = None):
    from matplotlib_venn import venn2
    venn = venn2([set(before_lri), set(after_lri)], set_labels=(tp1, tp2))
    for text in venn.set_labels:
        text.set_fontsize(22)
    for text in venn.subset_labels:
        text.set_fontsize(20)
    venn.get_patch_by_id('10').set_color(venn_color[0])  # Set A color
    venn.get_patch_by_id('01').set_color(venn_color[2])  # Set B color
    venn.get_patch_by_id('11').set_color(venn_color[1])  # Overlapping region color
    if title is not None:
        plt.title(title,fontsize=26)
    for patch in venn.patches:
        patch.set_alpha(.8)



def gsea_enrichment(gsea_res, k=1, figsize_heat=(6, 5), figsize_gsea=(4, 4)):
    import gseapy as gp
    res_df = gsea_res.res2d
    res_df = res_df[(res_df['NOM p-val'] < 0.01) & (res_df['ES'] > 0)].copy()
    print(res_df)
    genes = res_df.Lead_genes.loc[k].split(";")
    print(genes)
    ax = gp.heatmap(df=gsea_res.heatmat.loc[genes],
                    z_score=None,
                    title=res_df.Term.loc[k].split(' Hom')[0],
                    figsize=figsize_heat,
                    cmap=plt.cm.viridis,
                    xticklabels=False)
    plt.show()

    term = res_df.Term
    # print(term.loc[k])
    # annot1 = res_df.loc[k,'NES']
    # annot2 = res_df.loc[k,'NOM p-val']
    # print(annot1)
    plt.figure(figsize=figsize_gsea)
    axs = gsea_res.plot(terms=term.loc[k], figsize=figsize_gsea)

    # plt.annotate(f'NES = {annot1:.2f}', xy=(0, -0.35), fontsize=16, color='black')
    # plt.annotate(f'Pval = {annot2:.3f}', xy=(0, -0.65), fontsize=16, color='black')

    plt.show()