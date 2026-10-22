"""Lightweight orchestration for an isolated official model checkout."""
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SPEC = json.loads(Path(__file__).with_name("spec.json").read_text(encoding="utf-8"))


def environment(workspace):
    """Keep caches local while leaving credentials in the user's HF token store."""
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(workspace / "cache" / "huggingface"))
    env.setdefault("HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token"))
    env.setdefault("UV_CACHE_DIR", str(workspace / "cache" / "uv"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(workspace / "tools" / "python"))
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env["PYTHONIOENCODING"] = "utf-8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def run(command, args, cwd=None):
    """Execute an argument list without shell interpolation or global activation."""
    command = [str(part) for part in command]
    if args.dry_run:
        print(json.dumps({"command": command, "cwd": str(cwd) if cwd else None}))
        return
    subprocess.run(command, cwd=cwd, env=environment(args.workspace), check=True)


def python_path(args):
    """Return this engine's dedicated interpreter."""
    repo = args.workspace / SPEC["folder"]
    return repo / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def worker(args, action, extra=()):
    """Run packaged worker files in the engine's own dependency environment."""
    python = python_path(args)
    if not args.dry_run and not python.is_file():
        raise RuntimeError(f"Runtime not installed: {python}; run setup first")
    run([python, Path(__file__).with_name("_worker.py"), action,
         "--workspace", args.workspace, *extra], args)


def uv_path(args):
    """Find uv without installing or replacing a global tool."""
    uv = args.uv or shutil.which("uv")
    if not uv:
        raise RuntimeError("Install uv from https://docs.astral.sh/uv/ or pass --uv PATH")
    return uv


def setup_repo(args):
    """Fetch official code and refuse to change dirty or unofficial checkouts."""
    repo = args.workspace / SPEC["folder"]
    if not args.dry_run:
        if sys.platform != "win32" or platform.machine().lower() not in ("amd64", "x86_64"):
            raise RuntimeError("Automatic setup is tested on native Windows x86-64 only")
        args.workspace.mkdir(parents=True, exist_ok=True)
        if repo.exists():
            origin = subprocess.check_output(["git", "-C", str(repo), "remote", "get-url", "origin"], text=True).strip()
            if origin.removesuffix(".git").rstrip("/").lower() != SPEC["upstream"].removesuffix(".git").lower():
                raise RuntimeError("Existing checkout is not the official upstream repository")
            dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
            if dirty.strip():
                raise RuntimeError("Existing checkout has tracked changes; refusing to change it")
    if not repo.exists():
        run(["git", "clone", "--filter=blob:none", SPEC["upstream"], repo], args)
    run(["git", "-C", repo, "fetch", "--tags", "origin"], args)
    run(["git", "-C", repo, "checkout", "--detach", SPEC["commit"]], args)
    return repo


def setup(args):
    """Install one engine with its tested Python/CUDA pins, without weights."""
    repo = setup_repo(args)
    uv = uv_path(args)
    if SPEC["kind"] == "ace":
        excluded = ["nano-vllm", "lightning", "tensorboard", "lycoris-lora", "peft", "torchcodec", "torchao"]
        command = [uv, "sync", "--frozen", "--python", "3.12", "--no-default-groups"]
        for package in excluded:
            command += ["--no-install-package", package]
        run(command, args, repo)
    else:
        run([uv, "venv", "--allow-existing", "--python", "3.11", repo / ".venv"], args)
        pip = [uv, "pip", "install", "--python", python_path(args)]
        run([*pip, "torch==2.7.1", "torchaudio==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu128"], args)
        run([*pip, "-e", ".[ui]", "scipy", "--constraint", Path(__file__).with_name("constraints-windows.txt")], args, repo)
        worker(args, "doctor", ["--pre-flash"])
        run([*pip, "--no-deps", SPEC["flash_wheel"]], args)
    worker(args, "doctor")
