# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:27:02 2022

@author: Administrator
"""
import numpy as np
import pandas as pd
import smurf
from sklearn import metrics
from sklearn.cluster import KMeans
# smurf can be downloaded by pip install smurf.imputation
# sc_exp is the library
# sc_exp: row as cell columns as genes
def poisson_prep(filter_sc,n_features,steps,alpha,eps):
    operator = smurf.SMURF(n_features=n_features, steps=steps,alpha=alpha,eps=eps,estimate_only=False)
    d = operator.smurf_impute(filter_sc.T)
    data_imputed = d["estimate"]
    # gene_matrix = d["gene latent factor matrix"]
    # cell_matrix = d["cell latent factor matrix"]
    poisson = data_imputed.T
    return poisson


def cluster_eva(imputed_sc,true_label,steps):
    max_ARI = -1
    n_clusters = len(set(true_label))
    for i in range(steps):
        # label = pd.DataFrame(true_label)
        # label.drop_duplicates(inplace=True)
        model = KMeans(n_clusters=n_clusters,random_state=i*100)
        y_pred = model.fit_predict(imputed_sc)
        ARI = metrics.adjusted_rand_score(y_pred, true_label)
        if ARI >= max_ARI:
            max_ARI = ARI
    return max_ARI


#根据ARI选择最优参数，其中seed_count是用于K-means中选择种子的多少，steps是SMURF进行多少步，n_features_list（列表形式）用于n_features超参数选择。
def runSmurf(sc_exp, true_label, paras, k):
    '''
    paras: Dict of parameters
    true_label: Ref cell-type annotation of sc_exp (numerical)
    k: k-times repeat of KMeans in evaluation
    '''
    col = ['steps','n_features','eps','alpha','k','ari']
    result = pd.DataFrame(columns=col)
    max_ari = -1
    sc_exp = sc_exp.loc[:,(sc_exp != 0).any(axis=0)]
    for i in range(len(paras)):
        steps,n_features,eps,alpha = paras[i]
        poisson = poisson_prep(sc_exp, n_features,steps,alpha,eps)
        ari = cluster_eva(poisson,true_label,k)
        tmp = pd.DataFrame(np.array([[steps,n_features,eps,alpha,k,ari]]),columns = col)
        result = pd.concat((result,tmp),axis=0)
        print(f'When set {steps} steps, {eps} as eps,{alpha} as alpha, and {n_features} features for smurf, the ARI is {ari}')
        if ari > max_ari:
            max_poisson = poisson
            max_ari = ari
            max_feature = n_features
            max_steps = steps
            max_eps = eps
            max_alpha = alpha
    print(f'The max ARI from smurf is {max_ari}')
    return max_poisson,result




