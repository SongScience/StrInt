#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


DEFAULT_BASE = Path("/bigdata/disk2/xtsong-data/strint/SCC_results")
DEFAULT_OUT = DEFAULT_BASE / "ana" / "SPF" / "ST"
RNG = np.random.default_rng(7)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate original vs refined against ST ground truth.")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--marker-topk", type=int, default=6)
    p.add_argument("--hvg-topk", type=int, default=400)
    p.add_argument("--random-topk", type=int, default=400)
    p.add_argument("--lr-p-cut", type=float, default=0.01)
    return p.parse_args()


def scale_sum(df, total=1e4):
    sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(sums, axis=0).fillna(0.0) * total


def read_matrix(path, usecols=None):
    return pd.read_csv(path, sep="\t", index_col=0, usecols=usecols)


def read_subset(path, genes):
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    keep = [header[0]] + [g for g in genes if g in header]
    return read_matrix(path, usecols=keep)


def clean_gene_mask(genes):
    genes = pd.Index(genes)
    bad = (
        genes.str.startswith("MT-")
        | genes.str.startswith("RPS")
        | genes.str.startswith("RPL")
        | genes.str.startswith("RP")
        | genes.str.startswith("MIR")
        | genes.str.contains(r"\.", regex=True)
    )
    return ~bad


def paired_wilcoxon(after, before):
    mask = np.isfinite(after) & np.isfinite(before)
    if mask.sum() < 2:
        return np.nan
    diff = after[mask] - before[mask]
    if np.allclose(diff, 0):
        return 1.0
    return wilcoxon(after[mask], before[mask], alternative="greater").pvalue


def format_p(p):
    if not np.isfinite(p):
        return "NA"
    if p < 1e-16:
        return "<1e-16"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def pearson_cols(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    denom = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
    out = np.full(a.shape[1], np.nan, dtype=np.float64)
    valid = denom > 0
    out[valid] = (a[:, valid] * b[:, valid]).sum(axis=0) / denom[valid]
    return out


def build_marker_genes(sc_scale, celltypes, topk):
    expr = sc_scale.to_numpy(dtype=np.float32)
    genes = sc_scale.columns.to_numpy()
    good_mask = np.asarray(clean_gene_mask(genes), dtype=bool)
    markers = []
    rows = []
    eps = 1e-3

    for ct in sorted(celltypes.unique()):
        mask = (celltypes == ct).to_numpy()
        if mask.sum() < 8 or (~mask).sum() < 8:
            continue
        in_expr = expr[mask]
        out_expr = expr[~mask]
        mean_in = in_expr.mean(axis=0)
        mean_out = out_expr.mean(axis=0)
        pct_in = (in_expr > 0).mean(axis=0)
        pct_out = (out_expr > 0).mean(axis=0)
        logfc = np.log2((mean_in + eps) / (mean_out + eps))
        score = logfc + 0.25 * np.clip(pct_in - pct_out, 0, None)
        keep = good_mask & (pct_in >= 0.2) & (logfc >= 1.0) & (mean_in >= 1.0)
        idx = np.where(keep)[0]
        if idx.size == 0:
            continue
        ranked = idx[np.argsort(score[idx])[::-1][:topk]]
        for j in ranked:
            markers.append(genes[j])
            rows.append(
                {
                    "gene": genes[j],
                    "celltype": ct,
                    "mean_in": float(mean_in[j]),
                    "mean_out": float(mean_out[j]),
                    "pct_in": float(pct_in[j]),
                    "pct_out": float(pct_out[j]),
                    "log2_fc": float(logfc[j]),
                    "score": float(score[j]),
                }
            )

    marker_df = pd.DataFrame(rows).sort_values(["celltype", "score"], ascending=[True, False])
    return sorted(set(markers)), marker_df


def pick_random_genes(candidate_genes, st_scale, n_pick):
    if len(candidate_genes) <= n_pick:
        return sorted(candidate_genes)
    gene_means = st_scale[candidate_genes].mean(axis=0).sort_values()
    trimmed = gene_means.iloc[int(0.1 * len(gene_means)): int(0.9 * len(gene_means))]
    pool = trimmed.index.to_numpy() if trimmed.size >= n_pick else gene_means.index.to_numpy()
    return sorted(RNG.choice(pool, size=n_pick, replace=False).tolist())


def load_and_aggregate(path, mapping, genes):
    df = read_subset(path, genes)
    df.index = df.index.astype(str)
    common_cells = mapping.index.intersection(df.index)
    if len(common_cells) == 0:
        raise ValueError(f"No overlapping cells between mapping and {path}")
    tmp_mapping = mapping.loc[common_cells].copy()
    tmp_df = df.loc[common_cells].copy()
    grouped = pd.DataFrame(tmp_df.to_numpy(), index=tmp_mapping["spot"].astype(str).to_numpy(), columns=tmp_df.columns)
    grouped = grouped.groupby(level=0).sum()
    return np.log1p(scale_sum(grouped)).astype(np.float32)


def choose_examples(stats_df):
    examples = []
    for group in ["Marker", "LR", "HVG"]:
        sub = stats_df[(stats_df["group"] == group) & np.isfinite(stats_df["delta_spf"])]
        if len(sub) == 0:
            continue
        examples.append(sub.sort_values("delta_spf", ascending=False).iloc[0]["gene"])
    return examples[:3]


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    st_path = args.base / "inputs" / "ST_exp.tsv"
    sc_path = args.base / "inputs" / "SC_exp.tsv"
    sc_meta_path = args.base / "inputs" / "SC_meta.tsv"
    original_path = args.base / "original" / "before_sc_exp.tsv"
    refined_path = args.base / "refined" / "refined_sc_exp.tsv"
    mapping_path = args.base / "refined" / "cell_mapping_meta.tsv"
    lr_paths = [
        args.base / "ana" / "alphatalk" / "tables" / "refined_lr_st_filter.csv",
        args.base / "ana" / "alphatalk" / "tables" / "original_lr_st_filter.csv",
    ]

    st_raw = read_matrix(st_path)
    st_raw.index = st_raw.index.astype(str)
    st_scale = np.log1p(scale_sum(st_raw)).astype(np.float32)

    sc_raw = read_matrix(sc_path)
    sc_meta = pd.read_csv(sc_meta_path, sep="\t", index_col=0)
    sc_raw.index = sc_raw.index.astype(str)
    sc_meta.index = sc_meta.index.astype(str)
    sc_meta = sc_meta.loc[sc_raw.index.intersection(sc_meta.index)]
    sc_raw = sc_raw.loc[sc_meta.index]
    if "level3_celltype" in sc_meta.columns:
        sc_meta["celltype"] = sc_meta["level3_celltype"].astype(str)
    elif "celltype" in sc_meta.columns:
        sc_meta["celltype"] = sc_meta["celltype"].astype(str)
    else:
        raise ValueError("SC_meta.tsv missing celltype column")
    sc_scale = np.log1p(scale_sum(sc_raw)).astype(np.float32)

    marker_genes, marker_detail = build_marker_genes(sc_scale, sc_meta["celltype"], args.marker_topk)
    marker_detail.to_csv(out_dir / "marker_gene_definition.csv", index=False)

    hvg_series = st_scale.var(axis=0).sort_values(ascending=False)
    lr_gene_set = set()
    for lr_path in lr_paths:
        if not lr_path.exists():
            continue
        lr = pd.read_csv(lr_path)
        if "lr_co_ratio_pvalue" in lr.columns:
            lr = lr.loc[lr["lr_co_ratio_pvalue"] <= args.lr_p_cut]
        elif "co_exp_p" in lr.columns:
            lr = lr.loc[lr["co_exp_p"] <= args.lr_p_cut]
        lr_gene_set.update(lr["ligand"].astype(str))
        lr_gene_set.update(lr["receptor"].astype(str))

    universe = set(st_scale.columns)
    marker_set = set(g for g in marker_genes if g in universe)
    lr_set = set(g for g in lr_gene_set if g in universe)
    hvg_candidates = [g for g in hvg_series.index if g not in marker_set and g not in lr_set and clean_gene_mask([g])[0]]
    hvg_genes = hvg_candidates[: args.hvg_topk]
    random_candidates = [
        g for g in st_scale.columns
        if g not in marker_set and g not in lr_set and g not in set(hvg_genes) and clean_gene_mask([g])[0]
    ]
    random_genes = pick_random_genes(random_candidates, st_scale, args.random_topk)

    group_rows = []
    for gene in sorted(marker_set):
        group_rows.append({"gene": gene, "group": "Marker"})
    for gene in sorted(lr_set - marker_set):
        group_rows.append({"gene": gene, "group": "LR"})
    for gene in hvg_genes:
        group_rows.append({"gene": gene, "group": "HVG"})
    for gene in random_genes:
        group_rows.append({"gene": gene, "group": "Random"})
    group_df = pd.DataFrame(group_rows).drop_duplicates(subset=["gene"], keep="first")

    common_genes = [g for g in group_df["gene"].tolist() if g in st_scale.columns]
    group_df = group_df[group_df["gene"].isin(common_genes)].copy()
    selected_genes = group_df["gene"].tolist()

    mapping = pd.read_csv(mapping_path, sep="\t", index_col=0)
    mapping.index = mapping.index.astype(str)

    st_sel = st_scale[selected_genes].copy()
    original = load_and_aggregate(original_path, mapping, selected_genes)
    refined = load_and_aggregate(refined_path, mapping, selected_genes)

    common_genes2 = st_sel.columns.intersection(original.columns).intersection(refined.columns)
    group_df = group_df[group_df["gene"].isin(common_genes2)].copy()
    selected_genes = group_df["gene"].tolist()
    st_sel = st_sel[selected_genes]
    original = original[selected_genes]
    refined = refined[selected_genes]

    common_spots = st_sel.index.intersection(original.index).intersection(refined.index)
    st_sel = st_sel.loc[common_spots]
    original = original.loc[common_spots]
    refined = refined.loc[common_spots]

    stats_df = group_df.copy()
    stats_df["original_spf"] = pearson_cols(original.to_numpy(), st_sel.to_numpy())
    stats_df["refined_spf"] = pearson_cols(refined.to_numpy(), st_sel.to_numpy())
    stats_df["delta_spf"] = stats_df["refined_spf"] - stats_df["original_spf"]
    stats_df.to_csv(out_dir / "per_gene_pattern_similarity.csv", index=False)

    summary_rows = []
    for group in ["Marker", "LR", "HVG", "Random"]:
        sub = stats_df[stats_df["group"] == group]
        before_vals = sub["original_spf"].to_numpy()
        after_vals = sub["refined_spf"].to_numpy()
        delta_vals = sub["delta_spf"].to_numpy()
        summary_rows.append(
            {
                "group": group,
                "n_genes": int(len(sub)),
                "mean_original": float(np.nanmean(before_vals)),
                "mean_refined": float(np.nanmean(after_vals)),
                "mean_delta_spf": float(np.nanmean(delta_vals)),
                "fraction_improved": float(np.nanmean(delta_vals > 0)),
                "wilcoxon_p_greater": float(paired_wilcoxon(after_vals, before_vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    (out_dir / "summary_metrics.json").write_text(summary_df.to_json(orient="records", indent=2))

    examples = choose_examples(stats_df)
    pd.DataFrame({"example_gene": examples}).to_csv(out_dir / "example_genes.csv", index=False)

    sns.set_theme(style="whitegrid")
    palette = {"Marker": "#BF8A6A", "LR": "#7AA6B8", "HVG": "#86A985", "Random": "#A8A8A8"}
    summary_order = [g for g in ["Marker", "LR", "HVG", "Random"] if (stats_df["group"] == g).any()]

    fig_a, ax_a = plt.subplots(figsize=(7.0, 5.8))
    mn = np.nanmin(stats_df[["original_spf", "refined_spf"]].to_numpy())
    mx = np.nanmax(stats_df[["original_spf", "refined_spf"]].to_numpy())
    for group in summary_order:
        sub = stats_df[stats_df["group"] == group]
        ax_a.scatter(sub["original_spf"], sub["refined_spf"], s=18, alpha=0.55, label=f"{group} (n={len(sub)})", color=palette[group], edgecolor="none")
    ax_a.plot([mn, mx], [mn, mx], color="black", linewidth=1, linestyle="--")
    ax_a.set_xlabel("Original SPF")
    ax_a.set_ylabel("Refined SPF")
    ax_a.set_title("Paired SPF scatter by gene group")
    ax_a.legend(frameon=False, loc="lower right", fontsize=9)
    ann_y = 0.98
    for group in summary_order:
        row = summary_df[summary_df["group"] == group].iloc[0]
        ax_a.text(0.02, ann_y, f"{group}: mean Δ={row['mean_delta_spf']:.3f}, improved={row['fraction_improved']:.1%}, p={format_p(row['wilcoxon_p_greater'])}", transform=ax_a.transAxes, ha="left", va="top", fontsize=9, color=palette[group])
        ann_y -= 0.07
    fig_a.tight_layout()
    fig_a.savefig(out_dir / "st_pattern_similarity_A_scatter.pdf", bbox_inches="tight")
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(115 / 25.4, 90 / 25.4))
    plot_df = stats_df.melt(id_vars=["gene", "group"], value_vars=["original_spf", "refined_spf"], var_name="method", value_name="spf")
    plot_df["method"] = plot_df["method"].map({"original_spf": "Original", "refined_spf": "Refined"})
    sns.boxplot(data=plot_df, x="group", y="spf", hue="method", order=summary_order, palette={"Original": "#DAB6A4", "Refined": "#89AFC8"}, showfliers=False, ax=ax_b)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Pearson SPF to ST", fontsize=8)
    ax_b.set_title("SPF distribution by gene group", fontsize=10)
    ax_b.tick_params(axis="both", labelsize=8)
    ax_b.legend(frameon=False, title="", fontsize=8)
    fig_b.tight_layout()
    fig_b.savefig(out_dir / "st_pattern_similarity_B_boxplot.pdf", bbox_inches="tight")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(7.0, 5.2))
    for group in summary_order:
        vals = stats_df.loc[stats_df["group"] == group, "delta_spf"].dropna().sort_values().to_numpy()
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax_c.plot(vals, y, label=group, color=palette[group], linewidth=2)
    ax_c.axvline(0, color="black", linestyle="--", linewidth=1)
    ax_c.set_xlabel("ΔSPF (Refined - Original)")
    ax_c.set_ylabel("Cumulative fraction of genes")
    ax_c.set_title("CDF of ΔSPF")
    ax_c.legend(frameon=False, loc="lower right")
    fig_c.tight_layout()
    fig_c.savefig(out_dir / "st_pattern_similarity_C_cdf.pdf", bbox_inches="tight")
    plt.close(fig_c)

    fig_d, ax_d = plt.subplots(figsize=(6.6, 5.2))
    bars = summary_df.set_index("group").loc[summary_order]
    ax_d.bar(summary_order, bars["fraction_improved"], color=[palette[g] for g in summary_order], alpha=0.85)
    ax_d.set_ylim(0, 1)
    ax_d.set_ylabel("Fraction with Δ > 0")
    ax_d.set_title("One-number summary by gene group")
    for i, group in enumerate(summary_order):
        ax_d.text(i, bars.loc[group, "fraction_improved"] + 0.02, f"{bars.loc[group, 'fraction_improved']:.1%}", ha="center", va="bottom", fontsize=9)
    fig_d.tight_layout()
    fig_d.savefig(out_dir / "st_pattern_similarity_D_fraction_improved.pdf", bbox_inches="tight")
    plt.close(fig_d)

    coords = mapping.copy()
    coords["spot"] = coords["spot"].astype(str)
    coords = coords.groupby("spot")[["adj_spex_UMAP1", "adj_spex_UMAP2"]].mean()
    coords.columns = ["x", "y"]
    coords = coords.loc[coords.index.intersection(common_spots)]

    panels = ["Original", "Refined", "ST", "Delta"]
    if len(examples) > 0:
        fig_e, axes = plt.subplots(len(examples), 4, figsize=(12.5, 3.3 * len(examples)), squeeze=False)
    else:
        fig_e, ax_tmp = plt.subplots(figsize=(8, 2.8))
        ax_tmp.axis("off")
        ax_tmp.text(0.5, 0.5, "No example genes selected", ha="center", va="center")
        axes = np.array([[ax_tmp, ax_tmp, ax_tmp, ax_tmp]])

    for r, gene in enumerate(examples):
        original_vals = original[gene].loc[coords.index].to_numpy()
        refined_vals = refined[gene].loc[coords.index].to_numpy()
        st_vals = st_sel[gene].loc[coords.index].to_numpy()
        delta_vals = refined_vals - original_vals
        vmax = np.nanpercentile(np.concatenate([original_vals, refined_vals, st_vals]), 99)
        norm_expr = Normalize(vmin=0, vmax=max(vmax, 1e-6))
        vmax_delta = np.nanpercentile(np.abs(delta_vals), 99)
        norm_delta = TwoSlopeNorm(vmin=-max(vmax_delta, 1e-6), vcenter=0, vmax=max(vmax_delta, 1e-6))
        arrays = [original_vals, refined_vals, st_vals, delta_vals]
        cmaps = ["viridis", "viridis", "viridis", "coolwarm"]
        norms = [norm_expr, norm_expr, norm_expr, norm_delta]
        group = stats_df.loc[stats_df["gene"] == gene, "group"].iloc[0]
        delta = stats_df.loc[stats_df["gene"] == gene, "delta_spf"].iloc[0]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            sc = ax.scatter(coords["x"], coords["y"], c=arrays[c], cmap=cmaps[c], norm=norms[c], s=8, linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(panel, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{gene}\n{group}\nΔ={delta:.3f}", rotation=0, labelpad=36, va="center", fontsize=10)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)

    fig_e.suptitle("Qualitative examples", fontsize=14, y=0.995)
    fig_e.tight_layout(rect=[0, 0, 1, 0.97])
    fig_e.savefig(out_dir / "st_pattern_similarity_E_examples.pdf", bbox_inches="tight")
    plt.close(fig_e)

    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "n_marker_genes": int((group_df["group"] == "Marker").sum()),
            "n_lr_genes": int((group_df["group"] == "LR").sum()),
            "n_hvg_genes": int((group_df["group"] == "HVG").sum()),
            "n_random_genes": int((group_df["group"] == "Random").sum()),
            "examples": examples,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
