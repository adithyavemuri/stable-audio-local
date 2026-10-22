"""Pinned post-trained Medium assets and local-only model construction."""
import json
from pathlib import Path


def snapshot(spec, local_only=False):
    """Resolve only the official Medium and its bundled encoder at a fixed revision."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=spec["model_repo"], revision=spec["model_revision"],
        allow_patterns=["model_config.json", "model.safetensors", "t5gemma-b-b-ul2/*"],
        local_files_only=local_only, max_workers=1))


def download(args, repo, spec):
    """Download gated weights using HF's local login, never a token argument."""
    path = snapshot(spec)
    verify(path, spec)
    print(json.dumps({"model": "medium", "repo": spec["model_repo"], "revision": spec["model_revision"]}))


def verify(path, spec):
    """Reject incomplete main or encoder weights."""
    for filename, size in spec["weights"].items():
        file = path / filename
        if not file.is_file() or file.stat().st_size != size:
            raise RuntimeError(f"Missing or truncated official weight: {file}; run download")


def load_model(spec):
    """Use official loading APIs with a pinned local encoder, without editing weights."""
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.loading_utils import load_diffusion_cond
    path = snapshot(spec, local_only=True)
    verify(path, spec)
    config = json.loads((path / "model_config.json").read_text())
    for condition in config["model"]["conditioning"]["configs"]:
        if condition["type"] == "t5gemma":
            # Both are official constructor arguments. Avoid following a newer remote main.
            condition["config"]["model_path"] = str(path / "t5gemma-b-b-ul2")
            condition["config"]["subfolder"] = None
    model = load_diffusion_cond(config, str(path / "model.safetensors"), device="cuda", model_half=True)
    model.use_lora, model.lora_names = False, []
    return StableAudioModel(model, config, "cuda", True)
