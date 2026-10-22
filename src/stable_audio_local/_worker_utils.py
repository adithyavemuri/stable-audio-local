"""Runtime-only helpers; no model code or audio is shipped."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROMPT = "Instrumental atmospheric progressive post-rock, haunting clean electric guitar with warm delay and reverb, melodic bass and realistic acoustic drums. Starts spacious, lonely and melancholic, gradually evolving into a powerful cinematic rock arrangement with distorted rhythm guitars and expressive soaring guitar melodies. Natural dynamics, strong harmonic coherence, memorable motifs, dark mysterious atmosphere, approximately 90 BPM, no vocals, no EDM, no shredding."


def digest(path):
    """Hash a source or output without changing it."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_source(source, output):
    """Refuse to overwrite the conditioning recording, including symlinks."""
    if source and Path(source).resolve() == Path(output).resolve():
        raise ValueError("Output must not replace the source recording")


def output_path(args, name):
    """Resolve an output path and require explicit permission to replace it."""
    folder = Path(args.output_dir or Path(args.workspace) / "outputs")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    check_source(args.source, path)
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {path}; choose another folder or --overwrite")
    return path


def source_unchanged(source, original_hash):
    """Verify that inference left the original recording untouched."""
    if source and digest(source) != original_hash:
        raise RuntimeError("Source recording changed during the operation")


def repo_commit(repo, expected):
    """Reject untested upstream versions before using version-specific hooks."""
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise RuntimeError(f"Untested upstream commit {actual}; run setup to use {expected}")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise RuntimeError("Upstream has tracked changes; refusing version-specific inference")
    return actual


def save_record(path, record):
    """Record actual raw WAV checks and measured CUDA usage."""
    import torch
    from _audio_check import inspect_wav
    record.update(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),
                  peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_vram_bytes=torch.cuda.max_memory_reserved(),
                  raw_decoder_capture=True, processing="Raw float32 decoder samples; no added normalization, clipping, EQ, denoising or mastering",
                  validation=inspect_wav(path))
    Path(path).with_suffix(".json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(path), "runtime_seconds": record["runtime_seconds"],
                      "validation": record["validation"]}), flush=True)
    if record["validation"]["status"] == "FAIL":
        raise RuntimeError("Generated WAV failed signal checks")


def validate(paths):
    """Inspect WAVs read-only; return checks without placing files beside inputs."""
    from _audio_check import inspect_wav
    results = [inspect_wav(path) for path in paths]
    print(json.dumps(results, indent=2))
    return 1 if any(item["status"] == "FAIL" for item in results) else 0
