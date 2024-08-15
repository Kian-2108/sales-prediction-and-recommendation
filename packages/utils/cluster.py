from ..utils import utils
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.cm as cm

from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist,squareform,cdist
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import OrdinalEncoder, MultiLabelBinarizer

def pairwise(seq,typ):
    if typ == "euclidean":
        mod_seq = list(seq.fillna(np.inf).apply(lambda x: [x]))
        pwd = pdist(mod_seq)
        pwd = pwd/(1+pwd)
        np.nan_to_num(pwd,False,1)
    elif typ == "matching":
        model = OrdinalEncoder()
        mod_seq = model.fit_transform(pd.DataFrame(seq.fillna(pd.Series(list(seq.index),index=seq.index))))
        pwd = pdist(mod_seq,"jaccard")
    elif typ == "jaccard":
        mod_seq = seq.fillna(pd.Series([["NULL_VALUES"] for _ in seq],index=seq.index)).apply(lambda x: list(set(x).difference({None})))
        model = MultiLabelBinarizer() 
        tdf = pd.DataFrame(model.fit_transform(mod_seq),index=mod_seq.index,columns= model.classes_)
        tdf["NULL_VALUES"] = tdf["NULL_VALUES"].replace(1,np.NaN).fillna(pd.Series([-i-1 for i in range(len(tdf))],index=tdf.index))
        pwd = pdist(tdf.values.tolist(),"jaccard")
    return pwd


def weighted_pairwise(weights,temp_data_path):
    wpwd = None
    for col,wt in weights.items():
        col_pwd = np.load(f"{temp_data_path}/{col}")
        # col_pwd = utils.read_parquet(f"{data_path}/PWD/{col}.parquet")[col].to_numpy()
        if type(wpwd) == type(None):
            wpwd = col_pwd*wt
        else:
            wpwd += col_pwd*wt
        del col_pwd
    return wpwd/sum(weights.values())
    
# def WCSS(pwd,clustering):
#     matrix = squareform(pwd)
#     df = pd.DataFrame(matrix)
#     wcss = 0
#     for cluster in set(clustering):
#         temp = df[clustering==cluster]
#         temp = temp[list(temp.index)]
#         wcss += np.sum(np.square(squareform(temp)))/(2*len(temp))
#     return wcss

def WCSS(wpwd,clusters):
    matrix = pd.DataFrame(squareform(np.square(wpwd)),index=clusters,columns=clusters)
    return np.diag((np.diag(1/np.unique(clusters,return_counts=True)[1]))@matrix.groupby(matrix.index).sum().groupby(matrix.index,axis=1).sum()).sum()/2

def silhouette_scores(wpwd,clustering):
    try:
        return silhouette_samples(squareform(wpwd),clustering)
    except:
        return clustering*0

def silhouette_plot(ss, clusters,weights,metrics):
    n_clusters = len(set(clusters))
    scores = pd.DataFrame(columns=["index", "score", "score_1", "cluster", "color"])
    scores["score"] = ss.copy()
    scores["cluster"] = clusters
    scores["score_1"] = scores["score"] + 1
    scores["cluster_max"] = scores.groupby(["cluster"])[["score"]].transform('max')
    scores = scores.sort_values(["cluster_max","cluster", "score"], ascending=[False, True, False])
    scores["index"] = [_ for _ in range(len(ss))]
    scores["color"] = scores["cluster"].apply(lambda x: str(list(mc.TABLEAU_COLORS.keys())[(x - 1) % 10]))
    fig,ax = plt.subplots(ncols=2, nrows=1, figsize=(20, 15),gridspec_kw={'width_ratios': [3, 1]})
    ax[0].bar(
        data=scores,
        x="index",
        height="score_1",
        width=1.0,
        linewidth=0.0,
        color="color",
        )
    ax[0].axhline(y=1, color="black", linestyle=":")
    ax[0].set_title(f"Silhouette Scores wih {n_clusters} clusters")
    ax[0].set_xlabel("Customers")
    ax[0].set_ylabel("Silhouette Score")
    ax[0].set_ylim([0,2])
    wt_ls = [[feature,weight] for feature,weight in weights.items()]
    wt_ls.append(["",""])
    wt_ls = wt_ls+metrics
    wt_df = pd.DataFrame(wt_ls,columns=["Features","Weights"])
    ax[1].axis("off")
    ax[1].set_title(f"Weights & Metrics of clustering")
    wt_tabel = ax[1].table(cellText=wt_df.values, colLabels=wt_df.columns, loc='center')
    return fig