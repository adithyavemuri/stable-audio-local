# Third-party notices

This independent wrapper calls [Stable Audio 3 Medium Local upstream code](https://github.com/Stability-AI/stable-audio-3); it does not
redistribute that source code, model weights or binaries in its wheel or repository.
The upstream code is MIT licensed. Model terms are separate: review the
[official model card](https://huggingface.co/stabilityai/stable-audio-3-medium) before downloading or using checkpoints.
Hugging Face authentication and any gated access acceptance are performed by the user.
No upstream affiliation or endorsement is claimed.

PyTorch, CUDA runtime libraries and other dependencies retain their own licenses and
are installed into a separate runtime from their official package sources. NVIDIA
drivers are not included, installed or modified by this project.

The optional runtime Flash Attention binary comes from the community
[sdbds Windows build](https://github.com/sdbds/flash-attention-for-windows), with the exact
wheel URL recorded in `spec.json`. That URL is referenced by the official
[ACE-Step dependency metadata](https://github.com/ace-step/ACE-Step-1.5/blob/dce621408bee8c31b4fcf4811682eb9359e1bc94/acestep/third_parts/nano-vllm/pyproject.toml).
It is not redistributed in this package. Flash Attention's upstream source is at
https://github.com/Dao-AILab/flash-attention; its code and wheel dependencies retain
their respective licenses. Stable Audio model weights remain subject to their
Stability AI model license and access terms, not this wrapper's MIT license.
