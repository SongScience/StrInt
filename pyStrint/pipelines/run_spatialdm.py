#!/usr/bin/env python3
import argparse
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import spatialdm as sdm
from itertools import zip_longest
from sklearn.neighbors import NearestNeighbors

from common import (
    build_adata_for_reconstructed,
    build_adata_for_st,
    ensure_dirs,
    runtime_stamp,
    save_json,
    snapshot_scripts,
    write_runtime,
)


def safe_weight_matrix(adata, l=1.2, cutoff=0.2, n_neighbors=None, n_nearest_neighbors=6, single_cell=False):
    """
    Drop-in replacement for SpatialDM weight_matrix, avoiding sparse fancy indexing
    that can segfault on some SciPy builds.
    """
    X_loc = adata.obsm["spatial"]
    if isinstance(X_loc, pd.DataFrame):
        X_loc = X_loc.values
    if n_neighbors is None:
        n_neighbors = n_nearest_neighbors * 31

    def _rbf_graph(dist_graph):
        g = dist_graph.tocsr(copy=True)
        g.data = np.exp(-(g.data ** 2) / (2 * l ** 2))
        if single_cell:
            g.setdiag(np.zeros(g.shape[0]))
        else:
            g.setdiag(np.exp(-(dist_graph.diagonal() ** 2) / (2 * l ** 2)))
        g.data[g.data < cutoff] = 0
        g.eliminate_zeros()
        return g

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree", metric="euclidean").fit(X_loc)
    d = nn.kneighbors_graph(X_loc, mode="distance")
    W = _rbf_graph(d)

    nn0 = NearestNeighbors(n_neighbors=n_nearest_neighbors, algorithm="ball_tree", metric="euclidean").fit(X_loc)
    d0 = nn0.kneighbors_graph(X_loc, mode="distance")
    K = d0.tocsr(copy=True)
    K.data = np.exp(-(K.data ** 2) / (2 * l ** 2))
    if single_cell:
        K.setdiag(np.zeros(K.shape[0]))
    else:
        K.setdiag(np.exp(-(d0.diagonal() ** 2) / (2 * l ** 2)))

    adata.uns["single_cell"] = single_cell
    adata.obsp["weight"] = W * X_loc.shape[0] / W.sum()
    adata.obsp["nearest_neighbors"] = K * X_loc.shape[0] / K.sum()


def extract_lr_local(adata, species: str, min_cell: int, lr_data_dir: Optional[Path] = None):
    """
    SpatialDM extract_lr-compatible implementation with local LR_data support.
    """
    adata.uns["mean"] = "algebra"
    geneinter_name = f"{species}-interaction_input_CellChatDB.csv.gz"
    complex_name = f"{species}-complex_input_CellChatDB.csv"

    geneInter = None
    comp = None

    if lr_data_dir is not None:
        gi_p = lr_data_dir / geneinter_name
        cp_p = lr_data_dir / complex_name
        if gi_p.exists() and cp_p.exists():
            geneInter = pd.read_csv(gi_p, header=0, index_col=0, compression="gzip")
            comp = pd.read_csv(cp_p, header=0, index_col=0)

    if geneInter is None or comp is None:
        # fallback to package builtin data (offline)
        sdm.extract_lr(adata, species, min_cell=min_cell, datahost="package")
        return

    geneInter = geneInter.sort_values("annotation")
    ligand = geneInter.ligand.values
    receptor = geneInter.receptor.values
    geneInter = geneInter.copy()
    geneInter.pop("ligand")
    geneInter.pop("receptor")

    keep = []
    for i in range(len(ligand)):
        for n in [ligand, receptor]:
            lr_name = n[i]
            if lr_name in comp.index:
                vals = comp.loc[lr_name].dropna().values
                n[i] = vals[pd.Series(vals).isin(adata.var_names)]
            else:
                n[i] = pd.Series(lr_name).values[pd.Series(lr_name).isin(adata.var_names)]

        if (len(ligand[i]) > 0) and (len(receptor[i]) > 0):
            meanL = adata[:, ligand[i]].X.mean(axis=1)
            meanR = adata[:, receptor[i]].X.mean(axis=1)
            keep.append((sum(meanL > 0) >= min_cell) and (sum(meanR > 0) >= min_cell))
        else:
            keep.append(False)

    ind = geneInter[keep].index
    adata.uns["ligand"] = pd.DataFrame.from_records(zip_longest(*pd.Series(ligand[keep]).values)).transpose()
    adata.uns["ligand"].columns = [f"Ligand{i}" for i in range(adata.uns["ligand"].shape[1])]
    adata.uns["ligand"].index = ind
    adata.uns["receptor"] = pd.DataFrame.from_records(zip_longest(*pd.Series(receptor[keep]).values)).transpose()
    adata.uns["receptor"].columns = [f"Receptor{i}" for i in range(adata.uns["receptor"].shape[1])]
    adata.uns["receptor"].index = ind
    adata.uns["num_pairs"] = len(ind)
    adata.uns["geneInter"] = geneInter.loc[ind]
    if adata.uns["num_pairs"] == 0:
        raise ValueError("No effective RL. Please check input count matrix/species.")


def run_one(
    condition: str,
    adata,
    out_dir: Path,
    species: str,
    l: float,
    cutoff: float,
    min_cell: int,
    n_perm: int,
    nproc: int,
    fdr_threshold: float,
    lr_data_dir: Optional[Path],
):
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime = []

    np.random.seed(0)

    print(f"[{runtime_stamp()}] {condition}: step=weight_matrix start", flush=True)
    t0 = runtime_stamp()
    safe_weight_matrix(adata, l=l, cutoff=cutoff, single_cell=False)
    runtime.append({"step": "weight_matrix", "start": t0, "end": runtime_stamp()})
    print(f"[{runtime_stamp()}] {condition}: step=weight_matrix done", flush=True)

    print(f"[{runtime_stamp()}] {condition}: step=extract_lr start", flush=True)
    t0 = runtime_stamp()
    extract_lr_local(adata, species, min_cell=min_cell, lr_data_dir=lr_data_dir)
    runtime.append({"step": "extract_lr", "start": t0, "end": runtime_stamp()})
    print(f"[{runtime_stamp()}] {condition}: step=extract_lr done", flush=True)

    print(f"[{runtime_stamp()}] {condition}: step=spatialdm_global start", flush=True)
    t0 = runtime_stamp()
    sdm.spatialdm_global(adata, n_perm, specified_ind=None, method="z-score", nproc=nproc)
    runtime.append({"step": "spatialdm_global", "start": t0, "end": runtime_stamp()})
    print(f"[{runtime_stamp()}] {condition}: step=spatialdm_global done", flush=True)

    print(f"[{runtime_stamp()}] {condition}: step=sig_pairs start", flush=True)
    t0 = runtime_stamp()
    sdm.sig_pairs(adata, method="z-score", fdr=True, threshold=fdr_threshold)
    runtime.append({"step": "sig_pairs", "start": t0, "end": runtime_stamp()})
    print(f"[{runtime_stamp()}] {condition}: step=sig_pairs done", flush=True)

    print(f"[{runtime_stamp()}] {condition}: step=spatialdm_local start", flush=True)
    t0 = runtime_stamp()
    sdm.spatialdm_local(adata, n_perm=n_perm, method="z-score", specified_ind=None, nproc=nproc)
    runtime.append({"step": "spatialdm_local", "start": t0, "end": runtime_stamp()})
    print(f"[{runtime_stamp()}] {condition}: step=spatialdm_local done", flush=True)

    print(f"[{runtime_stamp()}] {condition}: step=sig_spots start", flush=True)
    t0 = runtime_stamp()
    try:
        sdm.sig_spots(adata, method="z-score", fdr=True, threshold=fdr_threshold)
        sig_spot_mode = "fdr"
    except Exception:
        sdm.sig_spots(adata, method="z-score", fdr=False, threshold=fdr_threshold)
        sig_spot_mode = "pvalue"
    runtime.append({"step": "sig_spots", "start": t0, "end": runtime_stamp(), "mode": sig_spot_mode})

    global_res = adata.uns["global_res"].copy()
    global_i = adata.uns.get("global_I", pd.DataFrame())
    local_stat = adata.uns.get("local_stat", pd.DataFrame())
    local_z = adata.uns.get("local_z_p", pd.DataFrame())

    global_res.to_csv(out_dir / "spatialdm.csv")
    if isinstance(global_i, pd.DataFrame) and not global_i.empty:
        global_i.to_csv(out_dir / "global_I.csv")
    if isinstance(local_stat, pd.DataFrame) and not local_stat.empty:
        local_stat.to_csv(out_dir / "local_stat.csv")
    if isinstance(local_z, pd.DataFrame) and not local_z.empty:
        local_z.to_csv(out_dir / "local_z_p.csv")

    # Save binarized hotspots (from notebook workflow)
    if "local_method" in adata.uns and "local_perm_p" in adata.uns:
        adata.uns["local_perm_p"][adata.uns["local_perm_p"] > 0.1] = 1
        if adata.uns["local_method"] == "z-score":
            adata.uns["local_z_p"][adata.uns["local_z_p"] > 0.1] = 1
            bin_spots = 1 - adata.uns["local_z_p"].astype(float)
        else:
            bin_spots = 1 - adata.uns["local_perm_p"].astype(float)
        bin_spots.to_csv(out_dir / "bin_spots.csv")

    # Match notebook behavior: coerce problematic uns tables before h5ad write.
    adata_h5 = adata.copy()
    for k in ["ligand", "receptor", "geneInter", "global_res"]:
        if k in adata_h5.uns and isinstance(adata_h5.uns[k], pd.DataFrame):
            adata_h5.uns[k] = adata_h5.uns[k].astype(str)
    if "local_fdr" in adata_h5.uns and isinstance(adata_h5.uns["local_fdr"], pd.DataFrame):
        adata_h5.uns["local_fdr"] = adata_h5.uns["local_fdr"].astype(str)
    if "local_stat" in adata_h5.uns and isinstance(adata_h5.uns["local_stat"], dict):
        if "n_spots" in adata_h5.uns["local_stat"]:
            adata_h5.uns["local_stat"]["n_spots"] = ""
        if "local_fdr" in adata_h5.uns["local_stat"]:
            adata_h5.uns["local_stat"]["local_fdr"] = ""
    adata_h5.write_h5ad(out_dir / "spatialdm.h5ad")
    write_runtime(out_dir / "runtime_steps.csv", runtime)

    summary = {
        "condition": condition,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_lri_total": int(len(global_res)),
        "n_lri_selected": int(global_res["selected"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        if "selected" in global_res.columns
        else None,
        "sig_spot_mode": sig_spot_mode,
        "out_dir": str(out_dir),
    }
    save_json(summary, out_dir / "run_summary.json")


def main():
    ap = argparse.ArgumentParser(description="Run SpatialDM for SCC condition")
    ap.add_argument("--condition", required=True, choices=["st", "original", "refined"])
    ap.add_argument("--out", default="/bigdata/disk2/xtsong-data/strint/SCC_results/ana/spatialdm")
    ap.add_argument("--st-exp", default="/bigdata/disk2/xtsong-data/strint/SCC_results/inputs/ST_exp.tsv")
    ap.add_argument("--st-coord", default="/bigdata/disk2/xtsong-data/strint/SCC_results/inputs/ST_coord.csv")
    ap.add_argument("--original-exp", default="/bigdata/disk2/xtsong-data/strint/SCC_results/original/before_sc_exp.tsv")
    ap.add_argument("--original-meta", default="/bigdata/disk2/xtsong-data/strint/SCC_results/original/sc_agg_meta.tsv")
    ap.add_argument("--refined-exp", default="/bigdata/disk2/xtsong-data/strint/SCC_results/refined/refined_sc_exp.tsv")
    ap.add_argument("--refined-meta", default="/bigdata/disk2/xtsong-data/strint/SCC_results/refined/sc_agg_meta.tsv")
    ap.add_argument("--species", default="human")
    ap.add_argument("--lr-data-dir", default="/home/dqw_sxt/ALgo/SpatialDM/spatialdm/datasets/LR_data")
    ap.add_argument("--l", type=float, default=1.2)
    ap.add_argument("--cutoff", type=float, default=0.2)
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--nproc", type=int, default=10)
    ap.add_argument("--fdr-threshold", type=float, default=0.05)
    args = ap.parse_args()

    out = ensure_dirs(Path(args.out))
    snapshot_scripts(Path(__file__).resolve().parent, out["scripts"])
    cond_name = "ST" if args.condition == "st" else args.condition
    cond_dir = out["intermediate"] / cond_name
    cond_dir.mkdir(parents=True, exist_ok=True)

    st_exp = Path(args.st_exp)
    st_coord = Path(args.st_coord)
    original_exp = Path(args.original_exp)
    original_meta = Path(args.original_meta)
    refined_exp = Path(args.refined_exp)
    refined_meta = Path(args.refined_meta)
    lr_data_dir = Path(args.lr_data_dir) if args.lr_data_dir else None

    if not st_exp.exists():
        alt = Path("/bigdata/disk2/xtsong-data/strint/SCC_results/inputs/ST_exp.tsv")
        if alt.exists():
            st_exp = alt
    if not st_coord.exists():
        alt = Path("/bigdata/disk2/xtsong-data/strint/SCC_results/inputs/ST_coord.csv")
        if alt.exists():
            st_coord = alt
    if not original_exp.exists():
        alt = Path("/bigdata/disk2/xtsong-data/strint/SCC_results/original/before_sc_exp.tsv")
        if alt.exists():
            original_exp = alt
    if not original_meta.exists():
        alt = Path("/bigdata/disk2/xtsong-data/strint/SCC_results/original/sc_agg_meta.tsv")
        if alt.exists():
            original_meta = alt
    if not refined_exp.exists():
        alt = Path("/bigdata/disk2/xtsong-data/strint/SCC_results/refined/refined_sc_exp.tsv")
        if alt.exists():
            refined_exp = alt
    if not refined_meta.exists():
        alt = Path("/bigdata/disk2/xtsong-data/strint/SCC_results/refined/sc_agg_meta.tsv")
        if alt.exists():
            refined_meta = alt

    try:
        if args.condition == "st":
            adata = build_adata_for_st(st_exp, st_coord)
        elif args.condition == "original":
            adata = build_adata_for_reconstructed(original_exp, original_meta, st_coord)
        else:
            adata = build_adata_for_reconstructed(refined_exp, refined_meta, st_coord)

        run_one(
            condition=cond_name,
            adata=adata,
            out_dir=cond_dir,
            species=args.species,
            l=args.l,
            cutoff=args.cutoff,
            min_cell=args.min_cell,
            n_perm=args.n_perm,
            nproc=args.nproc,
            fdr_threshold=args.fdr_threshold,
            lr_data_dir=lr_data_dir,
        )
        print(f"[{runtime_stamp()}] SpatialDM completed for {cond_name}: {cond_dir}")
    except Exception as e:
        err_log = out["logs"] / f"run_spatialdm_{cond_name.lower()}_error.log"
        with open(err_log, "w", encoding="utf-8") as f:
            f.write(f"[{runtime_stamp()}] {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
