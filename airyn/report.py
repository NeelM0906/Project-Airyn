"""
Generate the Airyn Step 1 Baseline Report as a PDF with charts.
"""

import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
LOG_FILE = ROOT / "logs" / "5646a646-2fd8-49f8-9f12-50a1e5c69e0d.txt"
CHART_DIR = ROOT / "reports"
CHART_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PDF = ROOT / "reports" / "airyn_baseline_report.pdf"

# ── Colors ─────────────────────────────────────────────────────────────
C_PRIMARY = "#2563EB"
C_SECONDARY = "#7C3AED"
C_ACCENT = "#059669"
C_DARK = "#1E293B"
C_LIGHT = "#F8FAFC"
C_GRAY = "#64748B"

# ── Parse training log ─────────────────────────────────────────────────
def parse_log(path):
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    throughputs = []

    with open(path) as f:
        for line in f:
            if not line.startswith("step:"):
                continue
            m_step = re.match(r"step:(\d+)/", line)
            if not m_step:
                continue
            step = int(m_step.group(1))

            m_train = re.search(r"train_loss:([\d.]+)", line)
            m_val = re.search(r"val_loss:([\d.]+)", line)
            m_tok = re.search(r"tok/s:(\d+)", line)

            if m_val:
                val_steps.append(step)
                val_losses.append(float(m_val.group(1)))
            if m_train:
                train_steps.append(step)
                train_losses.append(float(m_train.group(1)))
            if m_tok:
                throughputs.append((step, int(m_tok.group(1))))

    return train_steps, train_losses, val_steps, val_losses, throughputs


def make_charts(train_steps, train_losses, val_steps, val_losses, throughputs):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ── 1. Loss curves ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_steps, train_losses, color=C_PRIMARY, alpha=0.6, linewidth=0.8, label="Train Loss")
    ax.plot(val_steps, val_losses, color=C_SECONDARY, linewidth=2.0, marker="o", markersize=3, label="Val Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend(frameon=False)
    ax.set_ylim(bottom=2.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = str(CHART_DIR / "loss_curves.png")
    fig.savefig(p1, dpi=200)
    plt.close(fig)

    # ── 2. Val loss only (cleaner) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(val_steps, val_losses, color=C_SECONDARY, linewidth=2.0, marker="o", markersize=4)
    for i, (s, l) in enumerate(zip(val_steps, val_losses)):
        if i % 4 == 0 or i == len(val_steps) - 1:
            ax.annotate(f"{l:.3f}", (s, l), textcoords="offset points",
                       xytext=(0, 10), fontsize=7, ha="center", color=C_GRAY)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Curve", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p2 = str(CHART_DIR / "val_loss.png")
    fig.savefig(p2, dpi=200)
    plt.close(fig)

    # ── 3. Throughput ─────────────────────────────────────────────────
    if throughputs:
        t_steps, t_vals = zip(*throughputs)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(t_steps, [v / 1000 for v in t_vals], color=C_ACCENT, linewidth=1.5)
        ax.set_xlabel("Step")
        ax.set_ylabel("Throughput (k tok/s)")
        ax.set_title("Training Throughput", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p3 = str(CHART_DIR / "throughput.png")
        fig.savefig(p3, dpi=200)
        plt.close(fig)
    else:
        p3 = None

    # ── 4. Benchmark comparison bar chart ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # HellaSwag
    ax = axes[0]
    models = ["Random", "GPT-2 124M\n(40B tok)", "Airyn Base\n(2.6B tok)", "Airyn SFT"]
    hs_vals = [0.25, 0.294, 0.301, 0.303]
    colors = ["#CBD5E1", "#94A3B8", C_PRIMARY, C_SECONDARY]
    bars = ax.bar(models, hs_vals, color=colors, width=0.6)
    for bar, val in zip(bars, hs_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
               f"{val:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("Accuracy (acc_norm)")
    ax.set_title("HellaSwag", fontweight="bold")
    ax.set_ylim(0, 0.4)
    ax.grid(axis="y", alpha=0.3)

    # LAMBADA
    ax = axes[1]
    models = ["GPT-2 124M\n(40B tok)", "Airyn SFT", "Airyn Base\n(2.6B tok)"]
    la_vals = [0.326, 0.397, 0.404]
    colors = ["#94A3B8", C_SECONDARY, C_PRIMARY]
    bars = ax.bar(models, la_vals, color=colors, width=0.6)
    for bar, val in zip(bars, la_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
               f"{val:.1%}", ha="center", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("LAMBADA", fontweight="bold")
    ax.set_ylim(0, 0.55)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p4 = str(CHART_DIR / "benchmarks.png")
    fig.savefig(p4, dpi=200)
    plt.close(fig)

    return p1, p2, p3, p4


# ── PDF Generation ─────────────────────────────────────────────────────
def build_pdf(chart_paths):
    p_loss, p_val, p_throughput, p_bench = chart_paths

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF), pagesize=A4,
        leftMargin=25*mm, rightMargin=25*mm, topMargin=20*mm, bottomMargin=20*mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("Title2", parent=styles["Title"], fontSize=22, spaceAfter=6,
                              textColor=HexColor(C_DARK)))
    styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontSize=12,
                              textColor=HexColor(C_GRAY), spaceAfter=20))
    styles.add(ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16,
                              textColor=HexColor(C_PRIMARY), spaceBefore=16, spaceAfter=8))
    styles.add(ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                              textColor=HexColor(C_DARK), spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                              leading=14, spaceAfter=6))
    styles.add(ParagraphStyle("CodeBlock", parent=styles["Code"], fontSize=8.5,
                              leading=11, backColor=HexColor("#F1F5F9"),
                              borderPadding=6, spaceAfter=8))
    styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5,
                              textColor=HexColor(C_GRAY), alignment=TA_CENTER, spaceAfter=12))

    story = []
    W = doc.width

    # ── Title page ─────────────────────────────────────────────────────
    story.append(Spacer(1, 40*mm))
    story.append(Paragraph("Project Airyn", styles["Title2"]))
    story.append(Paragraph("Step 1 Baseline Report — 124M Parameter GPT on FineWeb-Edu", styles["Subtitle"]))
    story.append(Spacer(1, 10*mm))

    meta = [
        ["Date", "April 2026"],
        ["Authors", "Neelanjan Mitra + Claude (Anthropic)"],
        ["Repository", "github.com/NeelM0906/Project-Airyn"],
        ["Hardware", "1x NVIDIA RTX PRO 6000 Blackwell (102.6 GB)"],
        ["Training Time", "4h 32min"],
        ["Tokens Seen", "2.62B (FineWeb-Edu 10B subset)"],
    ]
    t = Table(meta, colWidths=[35*mm, W - 35*mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor(C_GRAY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ── 1. Executive Summary ───────────────────────────────────────────
    story.append(Paragraph("1. Executive Summary", styles["H1"]))
    story.append(Paragraph(
        "This report documents the Step 1 baseline for Project Airyn — a modular LLM training harness "
        "forked from OpenAI's parameter-golf codebase. We train a ~124M parameter GPT-2 class transformer "
        "on 2.62 billion tokens from FineWeb-Edu using the Muon optimizer, achieving a final validation "
        "loss of <b>3.074</b>. The model surpasses the original GPT-2 124M (trained on 40B tokens) on both "
        "HellaSwag (<b>30.1%</b> vs 29.4%) and LAMBADA (<b>40.4%</b> vs 32.6%), demonstrating the "
        "effectiveness of modern training recipes even at reduced data scale.", styles["Body"]))
    story.append(Paragraph(
        "We additionally perform supervised fine-tuning (SFT) on the Alpaca-cleaned instruction dataset "
        "(51.7K examples), producing a model capable of basic conversational interaction while preserving "
        "benchmark performance (HellaSwag 30.3%, LAMBADA 39.7%).", styles["Body"]))

    # ── 2. Architecture ────────────────────────────────────────────────
    story.append(Paragraph("2. Model Architecture", styles["H1"]))
    story.append(Paragraph(
        "The architecture is a decoder-only Transformer with U-Net style skip connections, "
        "following the parameter-golf baseline design. The first half of the layers (encoder) store "
        "skip activations; the second half (decoder) reuse them in reverse order via learned skip weights.",
        styles["Body"]))

    arch_data = [
        ["Parameter", "Value"],
        ["Total Parameters", "123.6M"],
        ["Vocabulary Size", "50,304 (GPT-2 BPE, padded to 128)"],
        ["Layers", "12 (6 encoder + 6 decoder)"],
        ["Model Dimension", "768"],
        ["Attention Heads", "12 (full MHA, no GQA)"],
        ["Head Dimension", "64"],
        ["MLP Hidden", "3,072 (4x expansion, relu² activation)"],
        ["Sequence Length", "1,024 tokens"],
        ["Positional Encoding", "Rotary (RoPE), base=10000"],
        ["Normalization", "RMSNorm (pre-norm)"],
        ["Embedding", "Tied input/output embeddings"],
        ["Logit Softcap", "30.0 (tanh-based)"],
        ["Weight Precision", "fp32 (CastedLinear: stored fp32, computed bf16)"],
    ]
    t = Table(arch_data, colWidths=[45*mm, W - 45*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(C_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor(C_LIGHT)]),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("2.1 Key Design Decisions", styles["H2"]))
    decisions = [
        "<b>Factory-based Block constructor</b> — Block accepts attn_factory and ffn_factory callables, "
        "enabling per-layer module swaps (e.g., SwiGLU, different attention) without modifying the GPT class.",
        "<b>CastedLinear</b> — Weights stored in fp32 for optimizer state quality, cast to bf16 at matmul "
        "time. This gives the precision benefits of fp32 training with bf16 compute speed.",
        "<b>U-Net skip connections</b> — First-half layers store residual activations; second-half layers "
        "consume them in reverse with learned scalar weights. Improves gradient flow in deeper models.",
        "<b>Logit softcap</b> — tanh(logits/30) * 30 prevents extreme logit values from destabilizing training.",
    ]
    for d in decisions:
        story.append(Paragraph(f"• {d}", styles["Body"]))

    # ── 3. Training Methodology ────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("3. Training Methodology", styles["H1"]))

    story.append(Paragraph("3.1 Optimizer", styles["H2"]))
    story.append(Paragraph(
        "We use a split optimizer strategy from modded-nanogpt:", styles["Body"]))
    opt_data = [
        ["Param Group", "Optimizer", "Learning Rate"],
        ["Embedding (tied)", "Adam (fused)", "0.05"],
        ["Block matrices (2D)", "Muon", "0.04"],
        ["Block scalars/vectors", "Adam (fused)", "0.04"],
    ]
    t = Table(opt_data, colWidths=[40*mm, 40*mm, W - 80*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(C_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor(C_LIGHT)]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "<b>Muon optimizer</b> orthogonalizes gradient updates via a fast 5-step Newton-Schulz iteration, "
        "then applies momentum with Nesterov correction. This is particularly effective for matrix-shaped "
        "parameters (attention projections, MLP layers). Muon momentum warms up from 0.85 to 0.95 over "
        "500 steps.", styles["Body"]))

    story.append(Paragraph("3.2 Learning Rate Schedule", styles["H2"]))
    story.append(Paragraph(
        "Constant LR with warmdown: full learning rate for the first 3,800 steps, then linearly decayed "
        "to zero over the final 1,200 steps (warmdown). No warmup on LR (compile warmup handles the first "
        "20 steps separately).", styles["Body"]))

    story.append(Paragraph("3.3 Data", styles["H2"]))
    data_info = [
        ["Dataset", "FineWeb-Edu (sample-10BT)"],
        ["Tokenizer", "GPT-2 BPE (tiktoken, 50,257 tokens)"],
        ["Train Shards", "99 shards × 100M tokens = 9.9B tokens available"],
        ["Val Shards", "1 shard × 100M tokens"],
        ["Tokens Seen", "2.62B (5,000 steps × 524,288 tokens/step)"],
        ["Sequence Length", "1,024"],
        ["Batch Size", "524,288 tokens (8 gradient accumulation steps)"],
    ]
    t = Table(data_info, colWidths=[35*mm, W - 35*mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor(C_GRAY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)

    story.append(Paragraph("3.4 Infrastructure", styles["H2"]))
    infra = [
        ["GPU", "1x NVIDIA RTX PRO 6000 Blackwell Max-Q (102.6 GB VRAM)"],
        ["PyTorch", "2.12.0.dev20260408+cu128 (nightly)"],
        ["Triton", "3.6.0 (triton-windows)"],
        ["CUDA", "12.8"],
        ["torch.compile", "Enabled (inductor backend)"],
        ["Peak VRAM", "38.5 GB (37.5% of available)"],
        ["Throughput", "160,663 tokens/sec sustained"],
        ["Step Time", "3.263 sec/step average"],
        ["Total Time", "4h 32min (16,316 sec)"],
    ]
    t = Table(infra, colWidths=[35*mm, W - 35*mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor(C_GRAY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)

    # ── 4. Training Results ────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Training Results", styles["H1"]))
    story.append(Image(p_loss, width=W, height=W * 0.56))
    story.append(Paragraph("Figure 1: Training and validation loss over 5,000 steps.", styles["Caption"]))

    story.append(Image(p_val, width=W, height=W * 0.5))
    story.append(Paragraph("Figure 2: Validation loss at each checkpoint (every 250 steps).", styles["Caption"]))

    if p_throughput:
        story.append(Image(p_throughput, width=W, height=W * 0.44))
        story.append(Paragraph("Figure 3: Training throughput (tokens/sec).", styles["Caption"]))

    story.append(Paragraph(
        "The model converges smoothly with no instabilities. Validation loss decreases monotonically from "
        "10.84 to 3.07. The warmdown phase (steps 3800-5000) produces a notable acceleration in loss "
        "improvement, with train loss dropping from 3.2 to 2.9 as the learning rate decays.", styles["Body"]))

    # ── 5. Benchmarks ──────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Benchmark Evaluation", styles["H1"]))
    story.append(Image(p_bench, width=W, height=W * 0.44))
    story.append(Paragraph("Figure 4: Benchmark comparison against GPT-2 124M baseline.", styles["Caption"]))

    bench_data = [
        ["Benchmark", "Airyn Base", "Airyn SFT", "GPT-2 124M", "Random"],
        ["HellaSwag acc_norm", "30.1%", "30.3%", "29.4%", "25.0%"],
        ["LAMBADA accuracy", "40.4%", "39.7%", "32.6%", "~0%"],
        ["LAMBADA perplexity", "35.4", "41.4", "35-40", "~50k"],
        ["Val Loss (CE)", "3.074", "—", "—", "—"],
    ]
    t = Table(bench_data, colWidths=[35*mm] + [(W - 35*mm) / 4] * 4, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor(C_PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#FFFFFF"), HexColor(C_LIGHT)]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("5.1 Analysis", styles["H2"]))
    analysis = [
        "<b>Outperforms GPT-2 124M despite 15x less data.</b> Our model sees 2.6B tokens vs GPT-2's 40B, "
        "yet achieves higher accuracy on both benchmarks. This validates the Muon optimizer and modern "
        "training recipe (CastedLinear, RMSNorm, RoPE, logit softcap).",
        "<b>SFT preserves reasoning.</b> HellaSwag accuracy is maintained after fine-tuning (30.1% → 30.3%), "
        "indicating that instruction-tuning on 52K examples does not catastrophically forget general knowledge.",
        "<b>Mild alignment tax on LAMBADA.</b> SFT reduces LAMBADA accuracy from 40.4% to 39.7% and "
        "increases perplexity from 35.4 to 41.4. This is the expected tradeoff: the model shifts its "
        "distribution toward instruction-following, slightly reducing raw text prediction ability.",
    ]
    for a in analysis:
        story.append(Paragraph(f"• {a}", styles["Body"]))

    # ── 6. SFT Details ─────────────────────────────────────────────────
    story.append(Paragraph("6. Supervised Fine-Tuning", styles["H1"]))
    sft_info = [
        ["Base Checkpoint", "Pretrained Airyn (step 5000, val loss 3.074)"],
        ["Dataset", "yahma/alpaca-cleaned (51,738 examples)"],
        ["Template", "### System: / ### User: / ### Assistant:"],
        ["Epochs", "2"],
        ["Learning Rate", "5e-5 (AdamW, weight decay 0.01)"],
        ["Batch Size", "8"],
        ["Loss Masking", "Only on assistant response tokens"],
        ["Gradient Clipping", "1.0"],
        ["Training Time", "~25 min"],
    ]
    t = Table(sft_info, colWidths=[35*mm, W - 35*mm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor(C_GRAY)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "The SFT model demonstrates basic instruction-following: structured responses, numbered lists, "
        "topically relevant answers. Key limitations include repetition loops, shallow factual knowledge, "
        "and inconsistent identity (does not reliably identify as 'Airyn'). These are expected capacity "
        "constraints of a 124M parameter model.", styles["Body"]))

    # ── 7. Roadmap ─────────────────────────────────────────────────────
    story.append(Paragraph("7. Next Steps", styles["H1"]))
    roadmap = [
        "<b>SwiGLU activation</b> — Replace relu² MLP with SwiGLU via the ffn_factory pattern. "
        "Expected improvement: 0.5-1.0% on benchmarks.",
        "<b>Scale up</b> — Train 350M and 1B+ parameter variants with longer context (2048/4096).",
        "<b>More data</b> — Train on full 10B tokens (currently using 2.6B), then expand to 100B+.",
        "<b>Multi-GPU training</b> — Validate DDP on Linux with NCCL for larger runs.",
        "<b>Checkpoint resume</b> — Add save/load of optimizer state for crash recovery.",
        "<b>Better SFT</b> — Generate Airyn-specific synthetic conversations; explore DPO/RLHF.",
        "<b>Additional evals</b> — ARC-Easy, WinoGrande, MMLU (at larger scale).",
    ]
    for r in roadmap:
        story.append(Paragraph(f"• {r}", styles["Body"]))

    # ── Build ──────────────────────────────────────────────────────────
    doc.build(story)
    print(f"Report saved to: {OUTPUT_PDF}")


def main():
    print("Parsing training log...")
    train_steps, train_losses, val_steps, val_losses, throughputs = parse_log(LOG_FILE)
    print(f"  {len(train_steps)} train entries, {len(val_steps)} val entries")

    print("Generating charts...")
    charts = make_charts(train_steps, train_losses, val_steps, val_losses, throughputs)

    print("Building PDF...")
    build_pdf(charts)


if __name__ == "__main__":
    main()
