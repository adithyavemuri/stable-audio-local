# Stable Audio 3 Medium Local

A small independent Python package for installing, launching and benchmarking the
[official upstream engine](https://github.com/Stability-AI/stable-audio-3) in its own uv-managed environment.
The wrapper itself has no Python dependencies and does not import PyTorch for CLI help.
Model code, checkpoints, credentials, recordings and generated audio are not distributed.

## Install the wrapper

Requires Git, [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer.
The automatic model setup targets **native Windows x86-64 with an NVIDIA CUDA GPU**.
The tested machine was an RTX 3080 Laptop GPU with 16 GB VRAM.

```powershell
uv tool install git+https://github.com/adithyavemuri/stable-audio-local.git@v0.1.1
stable-audio-local --help
stable-audio-local setup
```

`setup` installs dependencies only. Downloading large model weights is a separate step.
It does not replace NVIDIA drivers, global CUDA, global Python or global PyTorch.
Use `--dry-run` before the command to preview an operation without executing it:

```powershell
stable-audio-local --dry-run --uv uv setup
```

## Workspace and existing installations

The default workspace is `%LOCALAPPDATA%\stable-audio-local`. All engine dependencies are isolated
inside the upstream checkout's `.venv`; caches and downloaded Python interpreters remain
inside the chosen workspace. Pass `--workspace` **before** the command to select another
location, including an existing installation with the expected upstream subfolder:

```powershell
stable-audio-local --workspace 'D:\Audio Models\stable-audio-local' setup
stable-audio-local --workspace 'D:\Audio Models\stable-audio-local' launch
```

The expected subfolder is `stable-audio-3`. Existing checkouts must have the
official origin; setup refuses tracked changes. It checks out the documented tested
commit, rather than silently tracking upstream updates. No automatic model updates run
at UI startup. Use a new workspace when experimenting with a different upstream version.

## Download and launch

Request access to [the official post-trained Medium checkpoint](https://huggingface.co/stabilityai/stable-audio-3-medium)
and accept its access terms in your own Hugging Face account, then sign in locally:

```powershell
stable-audio-local login
stable-audio-local download
stable-audio-local doctor
stable-audio-local launch
```

Enter your token only at the interactive HF login prompt; do not put it into scripts.
HF uses its standard user token store or your existing `HF_TOKEN_PATH`/`HF_TOKEN` setting.
The default workspace cache does not contain the token. Downloads are approximately
**10.45 GB**, including Medium and its bundled T5Gemma encoder. This package uses
**`medium`**, not `medium-base`, and pins both the main checkpoint and encoder revision.

Open **http://127.0.0.1:7861** after loading. Stop the foreground server with Ctrl+C.
The official Gradio UI is launched on localhost without a public sharing tunnel.
No LoRA training extra or development dependency group is installed.

### Flash Attention compatibility

Medium requires Flash Attention 2. The Windows setup first installs Python 3.11 and
torch/torchaudio 2.7.1+cu128, checks real CUDA availability and GPU architecture, then
installs the exact Windows CPython 3.11 wheel referenced in ACE-Step's official
`acestep/third_parts/nano-vllm/pyproject.toml`. The binary is community-built by
`sdbds/flash-attention-for-windows`, not built or distributed by this package.
Its release URL is explicit in `spec.json`; no Linux wheel is installed on Windows.
`doctor` executes a real Flash Attention kernel and compares it numerically with
PyTorch attention. A package import or successful process exit alone is not treated
as sufficient compatibility evidence. The automatic profile accepts Ampere, Ada
and Hopper architectures; only the RTX 3080 Laptop setup has been tested locally.

Use the dedicated interpreter or this CLI after setup. An exact upstream `uv sync`
would remove Flash Attention because upstream does not declare it in its project;
use `uv sync --inexact` if deliberately updating and preserve the tested CUDA pins.
This package does not install or require WSL.

## Generate and condition on a recording

```powershell
stable-audio-local benchmark --duration 30 --count 3
stable-audio-local benchmark --prompt 'Instrumental post-rock with clean guitar and acoustic drums' --seed 42 --count 1
stable-audio-local audio-to-audio --source 'D:\Music\guitar.wav' --noise-level 0.5
```

Generation uses 8 steps, CFG 1, FP16, Flash Attention 2 and chunked decoding.
Audio-to-audio uses the official `(sample_rate, tensor)` conditioning workflow and
the source duration. It can reinterpret the whole arrangement; it does not preserve
the original guitar sample-for-sample and does not promise isolated instrument stems.
The input file itself remains unchanged and is protected against output overwrite.

## Tested version profile

- Official Stable Audio 3 source commit `77e0398c26f7d22d801ec1bb036307e8b231f741`.
- Dedicated Python 3.11; torch/torchaudio 2.7.1+cu128.
- Tested transitive runtime versions are pinned in packaged `constraints-windows.txt`.
- Flash Attention 2.8.2, wheel tagged `cu128torch2.7.1`, `cp311`, `win_amd64`.
- Post-trained Medium revision `d4d3579395a8ac0df4ab13b5b1d7138ce7290f81`.
- Official loading APIs are used with an in-memory local encoder path to prevent
  newer remote encoder weights from being selected. Checkpoint contents are unchanged.

## Raw audio, measurements and limits

Outputs default to `<workspace>/outputs/`. Each generation writes a float32 WAV and
a JSON sidecar with prompt, seed, all generation parameters, upstream commit,
checkpoint revisions, runtime, model-load time, peak allocated/reserved CUDA memory,
source hash (if present), output hash and audio signal checks. Existing outputs are
not replaced unless `--overwrite` is explicit; overwriting the source is always refused.
Use `--output-dir` and `--prompt-file` for your own paths and longer prompts.

The version-specific hook captures official decoder samples before upstream output
peak scaling/clipping. **Raw float files may contain peaks above 0 dBFS**; they are
retained without normalization, clipping, EQ, denoising or mastering. Some players
can clip at unity playback gain. The official UI retains its normal upstream output
behavior; the raw export policy applies to the CLI generation commands.

```powershell
stable-audio-local validate 'D:\Music\output.wav'
```

Validation is read-only and reports JSON to stdout. It checks finite samples, silence,
duration, sample rate, RMS, DC offset, samples at/above 0 dBFS and noise-related spectral
statistics. **It cannot certify musical fidelity, instrument realism, harmonic coherence,
rhythmic alignment or freedom from every audible artifact.** Listen before choosing a
model or treating a stem as successful. Seeds are recorded and Python/NumPy/Torch RNGs
are initialized, but bitwise GPU determinism across hardware/software is not guaranteed.
Run one large loaded model/UI at a time on a 16 GB GPU.

## Development and distribution

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e . build
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m build
```

Tests cover dry-run isolation, argument handling, refusal of unofficial/dirty checkouts,
source protection and overwrite behavior. CI runs these lightweight tests and builds on
Windows/Linux with Python 3.11/3.12; it does not download weights or claim GPU coverage.
Wheel and source distributions are attached to the GitHub `v0.1.1` release. The package
is not published to PyPI. This repository is independent of the other audio wrapper.

MIT applies to this wrapper code only. Model access and use remain subject to the
upstream model terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
