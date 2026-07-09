#!/usr/bin/env python3
"""Generate a narrated explainer video for the pims-admm-llm repository.

Pipeline: Pillow slides → edge-tts narration → imageio-ffmpeg MP4.
No system ffmpeg required (uses imageio-ffmpeg binary).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SLIDES = OUT / "slides"
AUDIO = OUT / "audio"
FINAL = OUT / "pims_admm_llm_explainer.mp4"
W, H = 1280, 720
BG = (10, 18, 32)
PANEL = (18, 32, 54)
ACCENT = (255, 170, 48)
CYAN = (64, 196, 255)
GREEN = (72, 210, 140)
WHITE = (236, 242, 250)
MUTED = (160, 176, 200)
RED = (255, 110, 110)

# (filename_stem, title, bullet lines, narration)
SCENES: list[tuple[str, str, list[str], str]] = [
    (
        "01_title",
        "pims-admm-llm",
        [
            "Aspen PIMS-style replacement demo",
            "Block-angular ADMM + multi-LLM agents",
            "Crude → CDU → intermediates → blender",
            "GitHub: joelwwiggins/pims-admm-llm",
        ],
        (
            "This is pims-admm-llm — an open-source Aspen PIMS-style replacement demo. "
            "It combines crude data, linear programming submodels, block-angular ADMM "
            "for shadow pricing, and multi-agent LLMs. Hard constraints stay with the "
            "math solvers. The goal is faster, clearer refinery planning without losing "
            "feasibility or make-buy-sell economics."
        ),
    ),
    (
        "02_problem",
        "The planning problem you already know",
        [
            "Which crudes to buy and how hard to run the CDU?",
            "How to balance tanks, blenders, utilities?",
            "Classic PIMS: one giant LP — slow re-runs, hard to parallelize",
            "Shadow prices are gold for make-buy-sell — if you can trust them fast",
        ],
        (
            "Every planning cycle the refinery must choose crude purchases, distillation "
            "cuts, intermediate routing, product recipes, and utilities. Classic tools "
            "like PIMS solve one giant linear program. It works, but re-runs are slow, "
            "parallelism is limited, and the why behind a plan change is often opaque. "
            "Shadow prices from that model drive make-buy-sell — but only if they arrive "
            "in time for the meeting."
        ),
    ),
    (
        "03_team",
        "Meet the Smart Refinery Planning Team",
        [
            "Boss / Master Coordinator — prices, balance, stop/go",
            "CDU Agent — crude slate, yields, capacity",
            "Blender Agent — recipes, specs, product demand",
            "Tanks / Utilities — inventory and energy (extensions)",
            "Each agent = LP solver (hard rules) + LLM (soft judgment)",
        ],
        (
            "Instead of one black box, we model a planning team. A master coordinator "
            "sets price signals for linking streams. Specialist agents own the CDU, "
            "blender, and later tanks and utilities. Each agent is two things: a reliable "
            "LP calculator that never breaks hard rules, and an LLM brain that can note "
            "nonlinear yield or business context the pure linear model misses."
        ),
    ),
    (
        "04_loop",
        "How one planning cycle works",
        [
            "1. Master broadcasts dual prices λ for naphtha, distillate, gasoil, residue",
            "2. Blocks solve local LPs in parallel under those prices",
            "3. Master updates consensus and duals (ADMM)",
            "4. Stop when balanced — plan + shadow prices ready",
            "LLM never overrides mass balance or capacity",
        ],
        (
            "In one cycle the boss broadcasts dual prices for naphtha, distillate, gasoil, "
            "and residue. Each department solves its local LP in parallel and returns a "
            "proposal plus optional LLM notes. The master updates consensus and duals using "
            "ADMM. When linking streams balance, you get a global plan and economic "
            "shadow prices. The LLM may suggest, but it never overrides mass balance or capacity."
        ),
    ),
    (
        "05_admm",
        "Why ADMM for this architecture",
        [
            "Modern alternative / complement to Dantzig–Wolfe",
            "Explicit dual variables λ = shadow prices",
            "No growing column pool — agent-friendly",
            "Natural parallel / distributed block solves",
            "At LP convergence, λ match PIMS-style marginal values",
        ],
        (
            "We use ADMM — Alternating Direction Method of Multipliers — as the "
            "coordination layer. Compared with classic Dantzig–Wolfe, ADMM keeps duals "
            "explicit, avoids a growing column pool, and maps cleanly onto one agent per "
            "block. At linear-program convergence, those duals have the same economic "
            "meaning as PIMS shadow prices: marginal value of relaxing a linking balance."
        ),
    ),
    (
        "06_demo",
        "Live demo VERDICT (toy slate)",
        [
            "Mono PuLP and multi-block ADMM both feasible",
            "Objective match: 1405.966410 (gap 0%)",
            "Crude: WTI 80 + Mars 40 (Maya 0) on 120 kbd CDU",
            "Products: gasoline 35.76 · diesel 44.91 · fuel oil 36.32",
            "Mono ~6 ms · ADMM ~0.3 s over 21 iterations (toy scale)",
        ],
        (
            "On the synthetic crude slate the demo is green. Monolithic PuLP and multi-block "
            "ADMM are both feasible with identical objective one thousand four hundred five "
            "point nine six. Crude slate is eighty WTI and forty Mars. Product rates match. "
            "At toy scale mono is still faster wall-clock; ADMM is built for scale, agents, "
            "and parallel blocks — not always for winning a two-block CBC race."
        ),
    ),
    (
        "07_shadows",
        "Shadow prices & make-buy-sell",
        [
            "λ = marginal $/bbl of intermediate flexibility",
            "Map duals → value of stream, tank, crude, product demand",
            "Naphtha duals already close mono vs ADMM",
            "Gasoil / residue dual recovery still tightening",
            "Linearity checks: Δobj ≈ dual × ΔRHS (PIMS-like ranges)",
        ],
        (
            "Shadow prices translate directly into make-buy-sell language: what is an "
            "extra barrel of naphtha or gasoil worth at the boundary? What is crude "
            "flexibility or product demand worth? The reporting package maps duals to "
            "those insights and checks local linearity — same idea as PIMS ranges. "
            "Primal plans already match; dual recovery on free-disposal streams is the "
            "next hardening step."
        ),
    ),
    (
        "08_layout",
        "Repository layout",
        [
            "src/pims_admm_llm/models/ — crude loaders + block LPs",
            "src/pims_admm_llm/admm/ — coordinator, residuals, dual recovery",
            "src/pims_admm_llm/agents/ — prompts, JSON schemas, LLM client",
            "src/pims_admm_llm/solvers/ — parallel runners + benchmarks",
            "demos/ + docs/story.md — runnable proof + non-math narrative",
        ],
        (
            "The repository is layered. Models hold crude data and block LPs. The ADMM "
            "package coordinates duals and residuals. Agents wrap prompts and structured "
            "JSON with a stub or OpenAI-compatible client. Solvers handle parallel "
            "execution and benchmarks. Demos prove mono versus ADMM, and docs tell the "
            "story for non-math stakeholders."
        ),
    ),
    (
        "09_run",
        "How to run",
        [
            "cd ~/projects/pims-admm-llm && source .venv/bin/activate",
            "PYTHONPATH=src python -m demos.run_demo",
            "PYTHONPATH=src python -m pytest tests/ -q   # 33 passed",
            "python -m demos.run_agent_layer",
            "python demos/shadow_price_report.py",
        ],
        (
            "To run it: activate the project virtualenv, set PYTHONPATH to src, and run "
            "the demo module. Pytest should show thirty-three passed. The agent layer and "
            "shadow price report demos are separate entry points. Default LLM mode is a "
            "stub so demos work offline; point OpenAI-compatible env vars at Grok or "
            "another provider when you want live notes."
        ),
    ),
    (
        "10_close",
        "What this proves — and what's next",
        [
            "✓ Feasible mono and ADMM with matching objective",
            "✓ Shadow-price reporting + stakeholder story",
            "✓ Multi-agent layer that cannot break hard constraints",
            "Next: dual fidelity, real PIMS data, tanks/multi-period, live Grok",
            "MIT · portfolio-ready MVP",
        ],
        (
            "Bottom line: the MVP proves a PIMS-style plan can be decomposed into agents "
            "and ADMM without losing feasibility or objective. Next waves are tighter dual "
            "match on the full demo, real assay loaders, tank and multi-period blocks, and "
            "live Grok on the agent layer. The math stays in charge. The agents make the "
            "plan explainable. That is pims-admm-llm."
        ),
    ),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_slide(stem: str, title: str, bullets: list[str], index: int, total: int) -> Path:
    SLIDES.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # top accent bar
    d.rectangle([0, 0, W, 8], fill=ACCENT)
    # left rail
    d.rectangle([0, 0, 10, H], fill=CYAN)
    # footer panel
    d.rectangle([0, H - 56, W, H], fill=PANEL)

    f_title = font(40, bold=True)
    f_body = font(26)
    f_small = font(18)
    f_tiny = font(16)

    d.text((48, 36), title, fill=WHITE, font=f_title)
    d.line([(48, 96), (W - 48, 96)], fill=(40, 60, 90), width=2)

    y = 130
    for b in bullets:
        # accent bullet
        d.ellipse([52, y + 10, 66, y + 24], fill=ACCENT)
        # wrap long lines
        lines = textwrap.wrap(b, width=68) or [""]
        for i, line in enumerate(lines):
            d.text((84, y + i * 34), line, fill=WHITE if i == 0 else MUTED, font=f_body)
        y += 34 * max(1, len(lines)) + 18

    d.text((48, H - 38), "pims-admm-llm  ·  block-angular ADMM + multi-LLM refinery planning", fill=MUTED, font=f_tiny)
    d.text((W - 140, H - 38), f"{index}/{total}", fill=CYAN, font=f_small)

    # decorative architecture nodes on title slide
    if stem.startswith("01"):
        nodes = [(980, 200, "Master"), (880, 360, "CDU"), (1080, 360, "Blend"), (980, 500, "λ duals")]
        for x, ny, label in nodes:
            d.ellipse([x - 48, ny - 28, x + 48, ny + 28], outline=CYAN, width=2)
            d.text((x - 36, ny - 10), label, fill=CYAN, font=f_tiny)
        d.line([(980, 228), (880, 332)], fill=ACCENT, width=2)
        d.line([(980, 228), (1080, 332)], fill=ACCENT, width=2)
        d.line([(880, 388), (980, 472)], fill=GREEN, width=2)
        d.line([(1080, 388), (980, 472)], fill=GREEN, width=2)

    path = SLIDES / f"{stem}.png"
    img.save(path, "PNG")
    return path


async def synthesize_audio(stem: str, text: str) -> Path:
    AUDIO.mkdir(parents=True, exist_ok=True)
    path = AUDIO / f"{stem}.mp3"
    import edge_tts

    # Clear professional voice
    communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural", rate="-5%")
    await communicate.save(str(path))
    return path


def audio_duration(path: Path) -> float:
    # Use ffprobe from imageio-ffmpeg if available; else estimate from file size
    import imageio_ffmpeg

    ffprobe = imageio_ffmpeg.get_ffmpeg_exe().replace("ffmpeg", "ffprobe")
    if not os.path.exists(ffprobe):
        # bundled package only ships ffmpeg; parse via ffmpeg -i
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        proc = subprocess.run(
            [ffmpeg, "-i", str(path)],
            capture_output=True,
            text=True,
        )
        # Duration: 00:00:12.34
        import re

        m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", proc.stderr)
        if not m:
            return 8.0
        h, mnt, s = m.groups()
        return int(h) * 3600 + int(mnt) * 60 + float(s)

    proc = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 8.0


def build_video(segments: list[tuple[Path, Path, float]]) -> Path:
    import imageio
    import imageio_ffmpeg
    import numpy as np

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    OUT.mkdir(parents=True, exist_ok=True)

    # Concatenate audio with ffmpeg
    list_file = OUT / "audio_concat.txt"
    with list_file.open("w") as f:
        for _, ap, _ in segments:
            f.write(f"file '{ap.resolve()}'\n")
    full_audio = OUT / "narration_full.mp3"
    subprocess.run(
        [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(full_audio)],
        check=True,
        capture_output=True,
    )

    # Build video track: hold each slide for audio duration (+0.4s pad)
    fps = 24
    frames_dir = OUT / "frames"
    frames_dir.mkdir(exist_ok=True)
    frame_paths: list[Path] = []
    n = 0
    for slide, _ap, dur in segments:
        hold = max(dur + 0.35, 3.0)
        n_frames = max(1, int(math.ceil(hold * fps)))
        # symlink/copy once per frame is heavy — use imageio writer with repeated frames
        img = imageio.imread(slide)
        for _ in range(n_frames):
            # store only index; we'll write via writer
            frame_paths.append(slide)  # reuse path; reader caches
            n += 1

    # Write silent video then mux audio
    silent = OUT / "silent.mp4"
    writer = imageio.get_writer(
        str(silent),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        last = None
        last_arr = None
        for p in frame_paths:
            if p != last:
                last_arr = imageio.imread(p)
                last = p
            writer.append_data(last_arr)
    finally:
        writer.close()

    # Mux
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(silent),
        "-i",
        str(full_audio),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        "-movflags",
        "+faststart",
        str(FINAL),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # re-encode if copy fails
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(silent),
            "-i",
            str(full_audio),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(FINAL),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr[-2000:])

    meta = {
        "output": str(FINAL),
        "scenes": len(segments),
        "fps": fps,
        "bytes": FINAL.stat().st_size if FINAL.exists() else 0,
        "durations": [round(d, 2) for *_, d in segments],
        "total_s": round(sum(d for *_, d in segments), 2),
    }
    (OUT / "build_report.json").write_text(json.dumps(meta, indent=2))
    return FINAL


async def main() -> None:
    total = len(SCENES)
    segments: list[tuple[Path, Path, float]] = []
    for i, (stem, title, bullets, narration) in enumerate(SCENES, 1):
        print(f"[{i}/{total}] slide+tts {stem}")
        slide = draw_slide(stem, title, bullets, i, total)
        audio = await synthesize_audio(stem, narration)
        dur = audio_duration(audio)
        print(f"    slide={slide.name} audio={dur:.2f}s")
        segments.append((slide, audio, dur))

    print("muxing video…")
    out = build_video(segments)
    print(f"VERDICT: video_ok path={out} size={out.stat().st_size}")


if __name__ == "__main__":
    asyncio.run(main())
