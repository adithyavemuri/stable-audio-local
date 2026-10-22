"""Medium raw-WAV benchmark and full-arrangement audio-to-audio."""
import time
from _raw_capture import RawDecoderCapture
from _worker_utils import PROMPT, digest, output_path, save_record, source_unchanged
from _download import load_model


def generate(args, repo, spec):
    """Generate raw decoder samples and leave input recordings unchanged."""
    import random
    import numpy as np
    import torch
    import soundfile as sf
    source_hash = digest(args.source) if args.source else None
    extra = {}
    duration = args.duration
    if args.source:
        audio, rate = sf.read(args.source, dtype="float32", always_2d=True)
        extra = dict(init_audio=(rate, torch.from_numpy(audio.T.copy())), init_noise_level=args.noise_level)
        duration = len(audio) / rate
    if not 0 < duration <= spec["max_duration"]:
        raise ValueError("Source duration exceeds the supported profile")
    prompt = args.prompt or PROMPT
    paths = [output_path(args, f"{'audio2audio' if args.source else 'text2audio'}_{duration:g}s_seed{args.seed+i}.wav") for i in range(args.count)]
    start = time.perf_counter()
    model = load_model(spec)
    load_seconds = time.perf_counter() - start
    load_peak = torch.cuda.max_memory_allocated()
    capture = RawDecoderCapture(model.same, "decode")
    for index, path in enumerate(paths):
        seed = args.seed + index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        generated = model.generate(prompt=prompt, duration=duration, seed=seed, steps=8,
            cfg_scale=1, batch_size=1, chunked_decode=True, **extra)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        raw = capture.take(generated.shape[-1])
        sf.write(path, raw.numpy().T, model.model_config["sample_rate"], subtype="FLOAT")
        source_unchanged(args.source, source_hash)
        save_record(path, dict(model="medium", source_commit=spec["commit"], seed=seed, prompt=prompt,
            checkpoint_repo=spec["model_repo"], checkpoint_revision=spec["model_revision"],
            parameters=dict(duration=duration, steps=8, cfg_scale=1, batch_size=1,
                chunked_decode=True, model_half=True, init_noise_level=args.noise_level if args.source else None),
            runtime_seconds=elapsed, model_load_seconds=load_seconds,
            model_load_peak_allocated_vram_bytes=load_peak, source_sha256=source_hash, isolated_stem=False))
