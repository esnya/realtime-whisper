# Dependency compatibility and security

The October 2026 update uses Transformers 4.48.3, the patch release in the
minor series that fixed the old model deserialization advisories in 4.48.0:
[GHSA-qxrp-vhvm-j765](https://github.com/advisories/GHSA-qxrp-vhvm-j765),
[GHSA-wrfc-pvp9-mr9g](https://github.com/advisories/GHSA-wrfc-pvp9-mr9g), and
[GHSA-hxxf-235m-72v3](https://github.com/advisories/GHSA-hxxf-235m-72v3).
Python 3.11 is the new minimum for the supported dependency environment.

The application loads `WhisperForConditionalGeneration` and
`Wav2Vec2ForSequenceClassification` directly, reads a Whisper tokenizer, and
never saves a downloaded tokenizer/processor. It does not load LightGlue or
custom generation code. The 4.48.3 wheel validates attention implementation
names against built-in implementations and contains no remote-kernel loader.
The newer version-only Transformers alerts
[GHSA-29pf-2h5f-8g72](https://github.com/advisories/GHSA-29pf-2h5f-8g72),
[GHSA-fgcw-684q-jj6r](https://github.com/advisories/GHSA-fgcw-684q-jj6r), and
[GHSA-xrqw-3rrv-vx5w](https://github.com/advisories/GHSA-xrqw-3rrv-vx5w)
therefore remain visible but their reported paths are not used here. Reassess
before allowing other model families, custom code, or tokenizer saving.

PyTorch 2.6.0 fixed the `weights_only=True` loading bypass
([GHSA-53q9-r3pm-6pq6](https://github.com/advisories/GHSA-53q9-r3pm-6pq6)).
Pipenv and the Docker image now use the matching Torch/TorchAudio 2.7.1 CUDA
11.8 pair. CUDA 11.8 has official 2.7.1 wheels and avoids requiring a newer
CUDA driver simply to retire Torch 2.2.1. The old FlashAttention URL was tied
to Torch 2.2, CUDA 12.2, and Python 3.10; it cannot be reused with the new
environment. It is removed from automatic installation. Built-in attention
and SDPA remain available. External FlashAttention, bitsandbytes, GPU kernel
behavior, and live audio quality require validation on the intended GPU.

The Docker base tag and digest were checked against official Docker Hub
metadata. No image was pulled or built during the local CPU checks. Deploying
the new image requires a separate GPU smoke test; the tests do not establish
live CUDA compatibility.

The optional UI requirements use `gradio==6.16.0+esnya.pillow13.1` with Pillow
13.0.0. This local maintenance build retains Gradio 6.16.0's frontend protocol,
compatibility with Transformers 4.48.3's Hub dependency, and its cookie
injection and FileExplorer traversal fixes. Official Gradio 6.16.0 and 6.30.0
require `pillow<13.0`; the [reviewable recipe](../dependency_patches/gradio/README.md)
changes one PNG helper to preserve legacy integer clipping explicitly, declares
`pillow<14.0`, and identifies the distribution with a local version. The UI
selects Pillow 13.0.0 for the current decoder fixes. There is no resolver
override, and this maintenance build is not an official release.

The stdlib-only PEP 517 backend checks the exact official wheel URL, size,
SHA-256, patch contexts and prepared metadata, preserves license notices, and
rebuilds RECORD. Metadata preparation is offline. Building downloads only the
fixed public Gradio wheel, or uses `ESNYA_COMPAT_WHEELHOUSE` for an offline
wheelhouse, without executing the original wheel's Python/build code.
`requirements-ui.txt` remains optional; Pipfile, Pipfile.lock and the plain
package's audio dependencies are unchanged. Docker copies the recipe before
installing its existing UI-enabled requirements.

The real UI constructor exposed an existing Gradio 6 API incompatibility:
`Audio(source="microphone")` raised `TypeError`. It now uses
`sources=["microphone"]`, and the existing CSS is passed to `launch`, where
Gradio 6 applies it. Streaming, queues, transcription callbacks and configured
launch options are retained.

The optional UI tests construct the real component tree with no server launch,
exercise generated WAV and transcription-queue callbacks, and round-trip PNG,
integer PNG, JPEG, TIFF and multi-frame GIF fixtures. They load no pretrained
models. These checks do not establish live microphone or GPU behavior, or the
applicability or absence of every Gradio/Transformers advisory.

CI constructs tiny random Whisper and language-ID models, writes local
safetensors and tokenizer fixtures, invokes the real application loaders,
and runs CPU forward passes. No Hub weights, datasets, or license acceptance
are required. The existing pretrained-model transcription test remains a
separate integration test and is not invoked in this offline workflow.
