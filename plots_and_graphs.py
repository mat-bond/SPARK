#!/usr/bin/env python3
import logging
import os
from collections import defaultdict
from typing import List, Optional
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np # type: ignore
import matplotlib # type: ignore
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt # type: ignore
from utils import _get_designed_residue_indices_n_offsets

#--------------------------------- Helpers---------------------------------#

def _make_hist(x, y, y_label, file_name, out_dir,x_label: Optional[str] = None):
    # Helper: draw one bar chart and save it
    plt.figure()
    cmap = plt.cm.get_cmap("tab20", len(x))
    colors = cmap(np.arange(len(x)))
    plt.bar(x, y, color=colors, edgecolor="black")

    if x_label is not None: 
        plt.xlabel(x_label)
        plt.title(f"{y_label} vs {x_label}")
    else: 
        plt.xlabel("Linker length (residues)")
        plt.title(f"{y_label} vs linker length")

    plt.ylabel(y_label)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    plt.savefig(path, dpi=300)
    plt.close()
    logging.info("Saved histogram → %s", path)

def _make_scatter(x, y, y_label, file_name, out_dir,x_label: Optional[str] = None,use_jitter: Optional[bool] = True):
    # Create scatter plot with jitter
    plt.figure()

    x_vals = x

    if use_jitter:
        # Add small jitter to x-values to prevent overlapping
        rng = np.random.default_rng(42)
        jitter = 0.1 * (rng.random(len(x)) - 0.5)
        x_vals = [val + j for val, j in zip(x, jitter)]

    plt.scatter(x_vals, y, alpha=0.5)
    if x_label is not None: 
        plt.xlabel(x_label)
        plt.title(f"{y_label} vs {x_label}")
    else: 
        plt.xlabel("Linker length (residues)")
        plt.title(f"{y_label} vs linker length")

    plt.ylabel(y_label)
    plt.tight_layout()
    
    # Add mean and standard deviation bars
    unique_x = sorted(set(x))
    for x_val in unique_x:
        y_vals = [y[i] for i in range(len(x)) if x[i] == x_val]
        if y_vals:
            mean = np.mean(y_vals)
            std = np.std(y_vals)
            plt.errorbar(x_val, mean, yerr=std, fmt='o', color='red', 
                         capsize=5, markersize=8, label=None)
    
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    plt.savefig(path, dpi=300)
    plt.close()
    logging.info("Saved scatter plot → %s", path)

def _make_line_plot(
    x: List[float],
    y: List[float],
    file_name: str,
    out_dir: str,
    x_label: str = "X-axis",
    y_label: str = "Y-axis",
    title: Optional[str] = None,
    line_label: Optional[str] = None,
    color: Optional[str] = None,
    linewidth: float = 2.0
) -> None:
    """
    Create and save a line plot.

    Parameters:
    - x, y: Data for the line.
    - file_name: Filename for saving the plot.
    - out_dir: Directory to save the plot.
    - x_label, y_label: Axis labels.
    - title: Title of the plot.
    - line_label: Label for the legend.
    - color: Line color.
    - linewidth: Thickness of the line.
    """

    plt.figure()

    # Plot line
    plt.plot(x, y, label=line_label, color=color, linewidth=linewidth)

    # Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    if line_label:
        plt.legend()

    plt.grid(True)
    plt.tight_layout()

    # Save
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, file_name)
    plt.savefig(path, dpi=300)
    plt.close()

    logging.info("Saved line plot → %s", path)

def _make_pae_heat_map(out_path,residue_contig,contig,linker_length,matrix):
    """
    Generates a heatmap marking designed residues from a PAE numpy file.
    `path` is the designed PDB file path.
    `out_path` is where the heatmap will be saved.
    """ 
    if matrix is None: return

    # Find designed residues
    designed_indices, offsets = _get_designed_residue_indices_n_offsets(residue_contig,contig,linker_length)

    # Binary mask: 1 = designed residue, 0 = not
    N = matrix.shape[0]

    # Colormaps
    green_white_cmap = LinearSegmentedColormap.from_list("green_white", ["#00441b", "#ffffff"])

    extent_main = [-0.5, N - 0.5, -0.5, N - 0.5]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[0.05, 1],
        height_ratios=[1, 0.05],
        wspace=0.01, hspace=0.01
    )

    # Main heatmap
    ax_main = fig.add_subplot(gs[0, 1])
    im = ax_main.imshow(matrix, cmap=green_white_cmap,
                        interpolation='nearest', origin="lower",
                        vmin=0, vmax=30, extent=extent_main,
                        aspect='auto')
    ax_main.set_title("Predicted Aligned Error (PAE)", pad=15)
    ax_main.set_xlabel("Residue index")
    ax_main.set_ylabel("Residue index")

    # Ensure correct alignment
    ax_main.set_xlim(-0.5, N - 0.5)
    ax_main.set_ylim(-0.5, N - 0.5)

    ax_main.set_aspect('equal', adjustable='box')

    # Setup ticks for designed residues 
    minor_ticks = sorted(designed_indices)

    # Put them as minor ticks on both axes
    ax_main.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax_main.yaxis.set_minor_locator(FixedLocator(minor_ticks))

    # Minor ticks have no labels; just red tick marks
    ax_main.xaxis.set_minor_formatter(NullFormatter())
    ax_main.yaxis.set_minor_formatter(NullFormatter())

    # Style minor ticks (red, short)
    ax_main.tick_params(axis="x", which="minor", length=4, width=1.0, colors="red")
    ax_main.tick_params(axis="y", which="minor", length=4, width=1.0, colors="red")

    # --- Designed residues as real minor ticks (RED) ---
    minor_ticks = sorted(designed_indices)
    ax_main.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax_main.yaxis.set_minor_locator(FixedLocator(minor_ticks))
    ax_main.xaxis.set_minor_formatter(NullFormatter())
    ax_main.yaxis.set_minor_formatter(NullFormatter())
    ax_main.tick_params(axis="x", which="minor", length=4, width=1.0, colors="red")
    ax_main.tick_params(axis="y", which="minor", length=4, width=1.0, colors="red")

    # --- Chain-break markers drawn manually (BLUE), not ticks ---
    from matplotlib.transforms import blended_transform_factory

    # chain break positions in data coords (between residues)
    chain_starts = sorted(int(v) for v in offsets.values())
    chain_breaks = [s - 0.5 for s in chain_starts if 0 < s < N]

    # How long the little tick should be (as a fraction of the axis)
    mark = 0.03

    # x in data, y in axes fraction
    tx = blended_transform_factory(ax_main.transData, ax_main.transAxes)
    # bottom (outside)
    ax_main.vlines(chain_breaks, -mark, 0.0, transform=tx,
                colors="tab:blue", linewidth=1.2, clip_on=False, zorder=5)
    # top (outside)
    ax_main.vlines(chain_breaks, 1.0, 1.0 + mark, transform=tx,
                colors="tab:blue", linewidth=1.2, clip_on=False, zorder=5)

    # y in data, x in axes fraction
    ty = blended_transform_factory(ax_main.transAxes, ax_main.transData)
    # left (outside)
    ax_main.hlines(chain_breaks, -mark, 0.0, transform=ty,
                colors="tab:blue", linewidth=1.2, clip_on=False, zorder=5)
    # right (outside)
    ax_main.hlines(chain_breaks, 1.0, 1.0 + mark, transform=ty,
                colors="tab:blue", linewidth=1.2, clip_on=False, zorder=5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_main, shrink=0.8)
    cbar.set_label("PAE (Å)")

    plt.tight_layout()

    # Save image
    os.makedirs(out_path, exist_ok=True)
    save_path = os.path.join(out_path, "pae_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    logging.info(f"Saved PAE heatmap → {save_path}")

# ──────────────────────────────────────────────────────────────────────
# Build and save four histograms (RMSD, PAE, PDE, pLDDT) and scatter plots
# ──────────────────────────────────────────────────────────────────────
def create_stats_graphs(stats, out_dir):
    """
    Create PNG plots in *out_dir*:
        - Histograms for average values
        - Scatter plots showing distribution and variability
        - Variance plots showing how variability changes with linker length
    """

    if not stats:
        logging.warning("create_stats_graphs: received empty stats list")
        return

    # Prepare data structures
    all_lengths = []
    all_rmsd = []
    all_des_avg_pae = []
    all_des_avg_pde = []
    all_avg_designed_plddt = []
    all_fixed_volume = []
    all_fixed_cavity_averageVolume = []
    all_fixed_cavity_amount = []
    all_designed_total_volume = []
    all_designed_cavity_averageVolume = []
    all_designed_cavity_amount = []
    all_full_pae = []
    all_full_pde = []
    all_full_plddt = []

    # Extract all data points

    for _pdb_path, rmsd, _, des_avg_pae, des_avg_pde, avg_designed_plddt, full_pae, full_pde, full_plddt, af_boltz_rmsd, af_full_pae, af_full_plddt, af_des_pae, af_des_plddt,fixed_total_volume, fixed_cavity_averageVolume, fixed_cavity_amount, designed_total_volume, designed_cavity_averageVolume, designed_cavity_amount, length in stats:
        all_lengths.append(length)
        all_rmsd.append(rmsd)
        all_des_avg_pae.append(des_avg_pae)
        all_des_avg_pde.append(des_avg_pde)
        all_avg_designed_plddt.append(avg_designed_plddt)
        all_fixed_volume.append(fixed_total_volume)
        all_fixed_cavity_averageVolume.append(fixed_cavity_averageVolume)
        all_fixed_cavity_amount.append(fixed_cavity_amount)
        all_designed_total_volume.append(designed_total_volume)
        all_designed_cavity_averageVolume.append(designed_cavity_averageVolume)
        all_designed_cavity_amount.append(designed_cavity_amount)
        all_full_pae.append(full_pae)
        all_full_pde.append(full_pde)
        all_full_plddt.append(full_plddt)

    # Create plots for each metric
    metrics = [
        ("RMSD (Å)", all_rmsd, "rmsd"),
        ("PAE (Å)", all_des_avg_pae, "pae"),
        ("PDE (Å)", all_des_avg_pde, "pde"),
        ("pLDDT (0-1)", all_avg_designed_plddt, "plddt"),
        ("Fixed Residue Total Cavity Volume (Å^3)",all_fixed_volume,"fixed_total_volume"),
        ("Fixed Residue Average Cavity Volume (Å^3)",all_fixed_cavity_averageVolume,"all_fixed_cavity_averageVolume"),
        ("Amount of Fixed Residue Cavities",all_fixed_cavity_amount,"all_fixed_cavity_amount"),
        ("Designed Residue Total Cavity Volume (Å^3)",all_designed_total_volume,"designed_total_volume"),
        ("Designed Residue Average Cavity Volume (Å^3)",all_designed_cavity_averageVolume,"designed_cavity_averageVolume"),
        ("Amount of Designed Residue Cavities",all_designed_cavity_amount,"designed_cavity_amount"),
        ("Full PAE (Å)", all_full_pae, "full_pae"),
        ("Full PDE (Å)", all_full_pde, "full_pde"),
        ("Full pLDDT (0-1)", all_full_plddt, "full_plddt")
    ]

    for y_label, values, base_name in metrics:
        # Skip if no data
        if not values:
            continue
            
        # Histogram per length
        bucket = defaultdict(list)
        for L, val in zip(all_lengths, values):
            bucket[L].append(val)
        
        lengths = sorted(bucket)
        avg_values = [np.mean(bucket[L]) if bucket[L] else 0.0 for L in lengths] # Guard against empty buckets
        
        # Scatter plot showing distribution
        _make_scatter(
            all_lengths, values, y_label,
            f"{base_name}_vs_linker_scatter.png", out_dir
        )

    # Scatter plot of PAE in terms of PLDDT and PDE
    _make_scatter(all_avg_designed_plddt, all_des_avg_pae, "PAE (Å)",f"des_pae_vs_plddt_scatter.png", out_dir,"(design) pLDDT (0-1)",use_jitter=False) 
    _make_scatter(all_des_avg_pde, all_des_avg_pae, "PAE (Å)",f"des_pae_vs_des_pde_scatter.png", out_dir,"(design) PDE (Å)",use_jitter=False) 

    # Plot confidence metrics of the designed regions in terms of those of the corresponding entire structure metrics
    _make_scatter(all_full_pde, all_des_avg_pde, "(design) PDE (Å)",f"des_pde_vs_full_pde_scatter.png", out_dir,"Full PDE (Å)",use_jitter=False)
    _make_scatter(all_full_pae, all_des_avg_pae, "(design) PAE (Å)",f"des_pae_vs_full_pae_scatter.png", out_dir,"Full PAE (Å)",use_jitter=False)
    _make_scatter(all_full_plddt, all_avg_designed_plddt, "(design) pLDDT (0-1)",f"des_plddt_vs_full_plddt_scatter.png", out_dir,"Full pLDDT (0-1)",use_jitter=False)
