"""Internal dispatcher executed by the isolated model interpreter."""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from _worker_utils import repo_commit, validate

SPEC = json.loads(Path(__file__).with_name("spec.json").read_text())


def doctor(pre_flash=False):
    """Verify actual CUDA execution; additionally test Flash Attention numerics."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; do not replace drivers automatically")
    result = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    if SPEC["kind"] == "stable":
        if sys.platform != "win32" or sys.version_info[:2] != (3, 11) or platform.machine().lower() not in ("amd64", "x86_64"):
            raise RuntimeError("The automatic Flash Attention wheel profile requires Windows AMD64 and Python 3.11")
        if torch.__version__.split("+")[0] != "2.7.1" or torch.version.cuda != "12.8":
            raise RuntimeError("Flash Attention wheel requires torch 2.7.1 and CUDA runtime 12.8")
        if torch.cuda.get_device_capability() not in [(8, 0), (8, 6), (8, 9), (9, 0)]:
            raise RuntimeError("GPU architecture is outside the automatic Flash Attention profile")
        if not pre_flash:
            import flash_attn
            from flash_attn import flash_attn_func
            from torch.nn.functional import scaled_dot_product_attention
            torch.manual_seed(123)
            q, k, v = [torch.randn(1, 256, 4, 64, device="cuda", dtype=torch.float16) for _ in range(3)]
            actual = flash_attn_func(q, k, v)
            reference = scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)).transpose(1,2)
            error = (actual-reference).abs().max().item()
            if not torch.isfinite(actual).all() or error >= 0.01:
                raise RuntimeError("Flash Attention failed the GPU numerical check")
            result.update(flash_attention=flash_attn.__version__, flash_max_absolute_error=error)
    print(json.dumps(result, indent=2))


def launch(args, repo):
    """Launch one engine's official interface on localhost only."""
    if SPEC["kind"] == "ace":
        from _download import verify_weights
        verify_weights(repo, SPEC, args.model)
        subprocess.run([sys.executable, "-m", "acestep.acestep_v15_pipeline", "--config_path", args.model,
            "--init_service", "true", "--init_llm", "true", "--lm_model_path", "acestep-5Hz-lm-0.6B",
            "--backend", "pt", "--device", "cuda", "--offload_to_cpu", "true", "--offload_dit_to_cpu", "false",
            "--quantization", "none", "--use_flash_attention", "false", "--batch_size", "1",
            "--server-name", "127.0.0.1", "--port", str(args.port), "--download-source", "huggingface"], cwd=repo, check=True)
    else:
        from _download import load_model
        from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui
        model = load_model(SPEC)
        interface = create_diffusion_cond_ui(model, gradio_title="Stable Audio 3 Medium")
        interface.queue()
        interface.launch(server_name="127.0.0.1", server_port=args.port, share=False,
            js=getattr(interface, "_sao_js", None), theme=getattr(interface, "_sao_theme", None))


def arguments(argv=None):
    """Parse paths after options consistently on Python 3.11 and 3.12."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["doctor", "download", "launch", "benchmark", "lego", "audio-to-audio", "validate"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--pre-flash", action="store_true")
    parser.add_argument("--model", default="all")
    parser.add_argument("--port", type=int, default=SPEC["port"])
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--instrument", choices=["drums", "bass"], default="drums")
    parser.add_argument("--noise-level", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("paths", nargs="*")
    return parser.parse_intermixed_args(argv)


def main():
    """Load heavy modules only for the selected operation."""
    args = arguments()
    if args.action == "validate":
        return validate(args.paths)
    repo = args.workspace / SPEC["folder"]
    repo_commit(repo, SPEC["commit"])
    sys.path.insert(0, str(repo))
    if args.prompt_file:
        args.prompt = args.prompt_file.read_text(encoding="utf-8")
    if args.action == "download":
        from _download import download
        download(args, repo, SPEC)
    elif args.action == "doctor":
        doctor(args.pre_flash)
    elif args.action == "launch":
        launch(args, repo)
    else:
        doctor()
        from _inference import generate
        generate(args, repo, SPEC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
