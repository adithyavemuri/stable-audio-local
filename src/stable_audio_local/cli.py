"""CLI for one independent local music engine."""
import argparse
import math
import os
import subprocess
from pathlib import Path
from .runtime import SPEC, run, setup, worker, python_path


def positive(value):
    """Reject invalid durations and counts before loading a model."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parser():
    """Construct this engine's supported command line."""
    result = argparse.ArgumentParser(prog=SPEC["name"], description=SPEC["description"])
    default = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share")) / SPEC["name"]
    result.add_argument("--workspace", type=Path, default=default)
    result.add_argument("--uv", help="Path to uv if it is not on PATH")
    result.add_argument("--dry-run", action="store_true", help="Print commands without downloading or changing files")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("setup", help="Install the isolated tested runtime; does not download model weights")
    sub.add_parser("doctor", help="Test CUDA and model-specific compatibility")
    download = sub.add_parser("download", help="Download pinned official model weights")
    if SPEC["kind"] == "ace":
        download.add_argument("--model", choices=["all", "acestep-v15-xl-turbo", "acestep-v15-xl-base"], default="all")
    else:
        sub.add_parser("login", help="Sign into Hugging Face locally; never pass a token in the command line")
    launch = sub.add_parser("launch", help="Launch the official Gradio interface locally")
    launch.add_argument("--port", type=int, default=SPEC["port"])
    if SPEC["kind"] == "ace":
        launch.add_argument("--model", choices=["acestep-v15-xl-turbo", "acestep-v15-xl-base"], default="acestep-v15-xl-turbo")
    validate = sub.add_parser("validate", help="Read-only WAV signal checks; does not establish musical quality")
    validate.add_argument("paths", type=Path, nargs="+")
    for name in ["benchmark", "lego" if SPEC["kind"] == "ace" else "audio-to-audio"]:
        generation = sub.add_parser(name, help="Export raw float32 decoder WAVs with reproducibility JSON")
        if name != "benchmark":
            generation.add_argument("--source", type=Path, required=True)
        else:
            generation.add_argument("--duration", type=positive, default=30)
        generation.add_argument("--count", type=int, default=3 if name == "benchmark" else 1)
        generation.add_argument("--seed", type=int, default=20261022)
        group = generation.add_mutually_exclusive_group()
        group.add_argument("--prompt")
        group.add_argument("--prompt-file", type=Path)
        generation.add_argument("--output-dir", type=Path)
        generation.add_argument("--overwrite", action="store_true")
        if name == "lego":
            generation.add_argument("--instrument", choices=["drums", "bass"], default="drums")
        if name == "audio-to-audio":
            generation.add_argument("--noise-level", type=float, default=0.5)
    return result


def main(argv=None):
    """Dispatch commands without importing PyTorch into the lightweight package."""
    command_line = parser()
    args = command_line.parse_args(argv)
    args.workspace = args.workspace.expanduser().resolve()
    try:
        if getattr(args, "count", 1) < 1:
            raise ValueError("count must be positive")
        if hasattr(args, "seed") and not 0 <= args.seed <= 2**32 - args.count:
            raise ValueError("seed sequence must fit unsigned 32-bit integers")
        if hasattr(args, "duration") and args.duration > SPEC["max_duration"]:
            raise ValueError(f"duration must not exceed {SPEC['max_duration']} seconds")
        if hasattr(args, "noise_level") and not 0 <= args.noise_level <= 1:
            raise ValueError("noise level must be between 0 and 1")
        if getattr(args, "source", None) and not args.source.is_file():
            raise FileNotFoundError(args.source)
        if args.command == "setup":
            setup(args)
        elif args.command == "login":
            run([python_path(args).with_name("hf.exe" if os.name == "nt" else "hf"), "auth", "login"], args)
        else:
            extra = []
            for key in ("model", "port", "duration", "count", "seed", "prompt", "prompt_file", "output_dir", "source", "instrument", "noise_level"):
                value = getattr(args, key, None)
                if value is not None:
                    extra += ["--" + key.replace("_", "-"), str(value)]
            if getattr(args, "overwrite", False):
                extra += ["--overwrite"]
            if args.command == "validate":
                extra += [str(path.resolve()) for path in args.paths]
            worker(args, args.command, extra)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        command_line.exit(1, f"{SPEC['name']}: {exc}\n")
