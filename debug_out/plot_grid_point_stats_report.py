from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

# Use a non-interactive backend for server environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_RE_BACKTICK_NUM = re.compile(r"`([^`]+)`")


@dataclass
class ReportData:
    global_counts: Dict[str, int]
    overflow_grid_counts: Dict[str, int]
    overflow_image_rate: Tuple[int, int]  # (images_with_overflow, total_images)
    overflow_cell_composition: Dict[str, int]  # {single_class, multi_class}
    top20_class_counts: List[Tuple[str, int]]
    top20_overflow_ratio: List[Tuple[str, float]]  # (class_name, ratio_percent)


def _parse_int(s: str) -> int:
    # Handles "47223", etc.
    return int(s.replace(",", "").strip())


def _parse_report_text(text: str) -> ReportData:
    # Global counts
    global_counts: Dict[str, int] = {}
    # e.g. "- Total images: `16551`"
    m = re.search(r"Total images:\s*`(\d+)`", text)
    if m:
        global_counts["images"] = _parse_int(m.group(1))
    m = re.search(r"Total boxes:\s*`(\d+)`", text)
    if m:
        global_counts["boxes"] = _parse_int(m.group(1))
    m = re.search(r"Total points:\s*`(\d+)`", text)
    if m:
        global_counts["points_total"] = _parse_int(m.group(1))
    m = re.search(r"Center points:\s*`(\d+)`", text)
    if m:
        global_counts["points_center"] = _parse_int(m.group(1))
    m = re.search(r"Corner points:\s*`(\d+)`", text)
    if m:
        global_counts["points_corner"] = _parse_int(m.group(1))

    # Overflow grid counts
    overflow_grid_counts: Dict[str, int] = {}
    for key, pattern in [
        ("overflow_cells_total_gt_2B", r"Cells with total points\s*`>\s*2B`:\s*`(\d+)`"),
        ("overflow_cells_center_gt_B", r"Cells with center points\s*`>\s*B`:\s*`(\d+)`"),
        ("overflow_cells_corner_gt_B", r"Cells with corner points\s*`>\s*B`:\s*`(\d+)`"),
    ]:
        m = re.search(pattern, text)
        if m:
            overflow_grid_counts[key] = _parse_int(m.group(1))

    # Overflow image rate: "Images containing overflow cells: `1710`" and "Total images: `16551`"
    m_over = re.search(r"Images containing overflow cells:\s*`(\d+)`", text)
    m_total = re.search(r"Total images:\s*`(\d+)`", text)
    if m_over and m_total:
        overflow_image_rate = (_parse_int(m_over.group(1)), _parse_int(m_total.group(1)))
    else:
        overflow_image_rate = (0, int(global_counts.get("images", 0)))

    # Overflow cell composition
    overflow_cell_composition: Dict[str, int] = {}
    m_single = re.search(r"Single-class overflow cells:\s*`(\d+)`", text)
    m_multi = re.search(r"Multi-class overflow cells:\s*`(\d+)`", text)
    if m_single:
        overflow_cell_composition["single_class_overflow_cells"] = _parse_int(m_single.group(1))
    if m_multi:
        overflow_cell_composition["multi_class_overflow_cells"] = _parse_int(m_multi.group(1))

    # Top20 frequent classes
    top20_class_counts: List[Tuple[str, int]] = []
    in_section = False
    for line in text.splitlines():
        if line.strip() == "### Most Frequent Classes (All Points, Top 20)":
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("### "):
                break
            # e.g. "1. person (`77880`)"
            m = re.match(r"^\s*(\d+)\.\s*([a-zA-Z0-9_]+)\s*\(`(\d+)`\)\s*$", line)
            if m:
                top20_class_counts.append((m.group(2), _parse_int(m.group(3))))

    # Top20 overflow ratio (multi-class overflow cells composition)
    top20_overflow_ratio: List[Tuple[str, float]] = []
    in_section = False
    for line in text.splitlines():
        if line.strip() == "### Class Ratio in Multi-class Overflow Cells (Top 20)":
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("---"):
                break
            # e.g. "1. diningtable: `4.806%` (`254/5285`)"
            m = re.match(r"^\s*(\d+)\.\s*([a-zA-Z0-9_]+):\s*`([0-9.]+)%`\s*\(`.*?`\)\s*$", line)
            if m:
                top20_overflow_ratio.append((m.group(2), float(m.group(3))))

    return ReportData(
        global_counts=global_counts,
        overflow_grid_counts=overflow_grid_counts,
        overflow_image_rate=overflow_image_rate,
        overflow_cell_composition=overflow_cell_composition,
        top20_class_counts=top20_class_counts,
        top20_overflow_ratio=top20_overflow_ratio,
    )


def _save_bar(
    *,
    out_path: Path,
    title: str,
    x: List[str],
    y: List[float],
    y_label: str,
    rotate_xticks: bool = True,
):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    if rotate_xticks:
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x, rotation=60, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_pie(
    *,
    out_path: Path,
    title: str,
    labels: List[str],
    values: List[int],
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser("Plot PLN grid point stats report")
    ap.add_argument("--input", type=str, default="grid_point_stats_report.md")
    ap.add_argument("--out_dir", type=str, default="grid_point_stats_plots")
    args = ap.parse_args()

    input_path = Path(__file__).resolve().parent / args.input
    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    data = _parse_report_text(text)

    # 1) Global counts
    global_keys_order = [
        ("images", "Images"),
        ("boxes", "Boxes"),
        ("points_total", "Total Points"),
        ("points_center", "Center Points"),
        ("points_corner", "Corner Points"),
    ]
    x1 = []
    y1 = []
    for k, label in global_keys_order:
        if k in data.global_counts:
            x1.append(label)
            y1.append(float(data.global_counts[k]))
    if x1:
        _save_bar(
            out_path=out_dir / "global_counts_bar.png",
            title="Global Counts",
            x=x1,
            y=y1,
            y_label="count",
            rotate_xticks=False,
        )

    # 2) Overflow grid counts
    overflow_map = {
        "overflow_cells_total_gt_2B": "Cells total points > 2B",
        "overflow_cells_center_gt_B": "Cells center points > B",
        "overflow_cells_corner_gt_B": "Cells corner points > B",
    }
    x2, y2 = [], []
    for k, label in overflow_map.items():
        if k in data.overflow_grid_counts:
            x2.append(label)
            y2.append(float(data.overflow_grid_counts[k]))
    if x2:
        _save_bar(
            out_path=out_dir / "overflow_grid_counts_bar.png",
            title="Overflow Grid Counts",
            x=x2,
            y=y2,
            y_label="count",
            rotate_xticks=False,
        )

    # 3) Overflow cell composition (single vs multi)
    if data.overflow_cell_composition:
        labels = []
        values = []
        for k in ["single_class_overflow_cells", "multi_class_overflow_cells"]:
            if k in data.overflow_cell_composition:
                labels.append(k.replace("_", " ").title())
                values.append(int(data.overflow_cell_composition[k]))
        if labels:
            _save_pie(
                out_path=out_dir / "overflow_cell_composition_pie.png",
                title="Overflow Cell Composition",
                labels=labels,
                values=values,
            )

    # 4) Top-20 class frequencies
    if data.top20_class_counts:
        classes = [c for c, _ in data.top20_class_counts]
        counts = [float(v) for _, v in data.top20_class_counts]
        _save_bar(
            out_path=out_dir / "top20_class_counts_bar.png",
            title="Top-20 Class Frequencies (All Points)",
            x=classes,
            y=counts,
            y_label="count",
        )

    # 5) Top-20 overflow ratio percentages
    if data.top20_overflow_ratio:
        classes = [c for c, _ in data.top20_overflow_ratio]
        ratios = [float(v) for _, v in data.top20_overflow_ratio]
        _save_bar(
            out_path=out_dir / "top20_overflow_ratio_bar.png",
            title="Top-20 Class Ratio in Multi-class Overflow Cells",
            x=classes,
            y=ratios,
            y_label="ratio (%)",
        )

    # 6) Overflow image rate text-like bar (we can also draw a small bar)
    if data.overflow_image_rate and data.overflow_image_rate[1] > 0:
        with_over, total = data.overflow_image_rate
        rate = 100.0 * float(with_over) / float(total)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.bar(["Overflow Rate"], [rate], color="tab:orange")
        ax.set_title("Overflow Image Rate")
        ax.set_ylabel("rate (%)")
        ax.set_ylim(0, max(5.0, rate * 1.2))
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for i, v in enumerate([rate]):
            ax.text(i, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "overflow_image_rate_bar.png", dpi=150)
        plt.close(fig)

    print(f"Saved plots to: {out_dir}")
    print("Generated files:")
    for p in sorted(out_dir.glob("*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()

