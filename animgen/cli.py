"""Command line: ``python -m animgen.cli`` (or ``scripts/generate-animations.sh``)."""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from typing import Optional, Sequence

from animgen import fonts, registry
from animgen.context import Context
from animgen.gallery import write_gallery
from animgen.options import Options, from_env, resolve_out_dir
from animgen.scene import render_scene, save_gif


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="animgen", description="Generate the explanatory GIFs of the LLM-from-scratch lab.")
    p.add_argument("--only", nargs="+", metavar="N|NAME", help="sub-task numbers or name parts (default: all), e.g. --only 2 attention")
    p.add_argument("--list", action="store_true", help="list the animations and exit")
    p.add_argument("--quick", action="store_true", help="fast preview: 640x360, 30 training steps")
    p.add_argument("--out", help="output folder (default: ANIM_OUTPUT_DIR or Docs/subtasks/animations)")
    p.add_argument("--width", type=int, help="pixel width (default: ANIM_WIDTH)")
    p.add_argument("--height", type=int, help="pixel height (default: ANIM_HEIGHT)")
    p.add_argument("--frame-ms", type=int, help="milliseconds per tween frame (default: ANIM_FRAME_MS)")
    p.add_argument("--hold-ms", type=int, help="milliseconds each finished step stays on screen")
    p.add_argument("--steps", type=int, help="training steps behind the training GIFs (default: ANIM_TRAIN_STEPS)")
    p.add_argument("--phrase", help="the sentence that travels through the animations (default: ANIM_PHRASE)")
    p.add_argument("--no-gallery", action="store_true", help="do not write index.html")
    return p


def options_from_args(args: argparse.Namespace, base: Optional[Options] = None) -> Options:
    """Environment defaults, then ``--quick``, then explicit flags (flags win)."""
    opt = base or from_env()
    if args.quick:
        opt = opt.quick()
    changes = {
        "out_dir": resolve_out_dir(args.out) if args.out else None,
        "width": args.width, "height": args.height, "frame_ms": args.frame_ms, "hold_ms": args.hold_ms,
        "train_steps": args.steps, "phrase": args.phrase,
    }
    return replace(opt, **{k: v for k, v in changes.items() if v is not None}).validated()


def run(args: argparse.Namespace, base: Optional[Options] = None, out=sys.stdout) -> int:
    """Render the chosen animations; returns the process exit code (0 = all good)."""
    try:
        scenes = registry.select(args.only)
        if args.list:
            for s in registry.ALL_SCENES:
                print(f"{s.number:>2}  {s.filename:<36} {s.steps} steps  {s.title}", file=out)
            return 0
        opt = options_from_args(args, base)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    fonts.configure(opt.font, opt.font_bold, opt.font_mono)
    ctx = Context(opt)
    failures = 0
    started = time.perf_counter()
    for scene in scenes:
        t0 = time.perf_counter()
        try:
            frames = render_scene(scene, ctx, opt)
            size = save_gif(frames, opt.out_dir / scene.filename)
        except Exception as exc:  # keep going: one broken scene must not hide the others
            failures += 1
            print(f"FAILED {scene.filename}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        print(f"ok  {scene.filename:<36} {len(frames):>3} frames {size / 1024:>7.0f} KiB  {time.perf_counter() - t0:4.1f}s", file=out)
    if not args.no_gallery:
        print(f"gallery: {write_gallery(opt.out_dir)}", file=out)
    print(f"{len(scenes) - failures}/{len(scenes)} animations in {time.perf_counter() - started:.0f}s -> {opt.out_dir}", file=out)
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
