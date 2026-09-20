# syntax=docker/dockerfile:1
# comfyui-gfx1151 runtime image — pure assembly from prebuilt artifacts, no
# native compilation of torch/ROCm.
#
# Sibling of sebt3/vllm-gfx1151 (see that repo's Dockerfile / README for the
# full history of why this project stopped building torch/triton from
# source on gfx1151). This image reuses exactly the same building blocks —
# the prebuilt ROCm wheel set from wheels.vllm.ai/rocm/ (mutually
# ABI-pinned torch/torchvision/torchaudio/triton, gfx1151-capable,
# hardware-qualified via lemonade-sdk/vllm-rocm) + the ROCm 7.14 userspace
# tarball — but drops everything that's specific to LLM serving: no vllm
# wheel, no amd-aiter, no flash-attn, none of vllm-gfx1151's FLA/GDN/MoE
# site-packages patches. On top: ComfyUI itself (official
# comfyanonymous/ComfyUI, GPL-3.0, see LICENSE.upstream) installed with its
# own requirements.txt, constrained so pip can't swap our gfx1151 torch
# wheel for a vanilla CUDA/CPU one from PyPI.
#
# Wheels (from the same pinned wheels.vllm.ai/rocm/ commit vllm-gfx1151
# uses — keeps every gfx1151-validated component on one known-good build):
#   torch        2.12.0+git6bbd260 — libtorch_hip.so fat multi-arch,
#                                    gfx1151/1150/1152/1153 all present.
#   triton       3.7.1+gitf0b55c07 — needed by ComfyUI's own model compiler
#                                    (torch.compile) and comfy-kitchen's
#                                    optional --enable-triton-backend.
#   torchvision  0.27.1+df56172    — nms/roi_align etc. used by some node
#                                    packs (segmentation, upscalers).
#   torchaudio   2.11.0+34c52a6    — ComfyUI core imports it unconditionally
#                                    (audio nodes: ACE-Step, Stable Audio).
#   amdsmi       26.2.2+c2d9476115 — optional, cheap, useful for GPU
#                                    telemetry from custom nodes/monitoring.
#
# NOT carried over from vllm-gfx1151: amd-aiter, flash-attn. Both are tuned
# for LLM decode attention shapes; ComfyUI's cross/self-attention in
# UNet/DiT blocks is a different access pattern and neither has been
# validated here. Default attention path is torch SDPA via
# --use-pytorch-cross-attention, backed by ROCm's AOTriton SDPA backend
# (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 below) — unlike vLLM's
# text-only decode path where that flag was a no-op (AOTriton SDPA is only
# ever hit by the ViT encoder there), ComfyUI's whole attention path runs
# through SDPA, so this flag is actually load-bearing here. Revisit
# aiter/flash-attn only after a real-hardware A/B.
#
# ROCm 7.14 runtime still comes from AMD's stable repo (repo.amd.com) as a
# tarball into /opt/rocm — same reasoning as vllm-gfx1151: the torch wheel
# bundles no ROCm userspace, so libamdhip64 / librocblas / libhipBLASLt and
# the llvm/clang+lld some custom nodes' JIT-compiled ops need at runtime
# all come from here.
#
# ⚠️ Unvalidated on real hardware as of first commit — this Dockerfile is
# the ComfyUI counterpart of vllm-gfx1151's assembly, not yet booted on the
# Strix Halo gfx1151 node. See think/apps/comfyui/DEBUG.md in kydah/home
# for validation status before trusting any of the above as fact.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG ROCM_DIST_URL=https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.0.tar.gz
ARG VLLM_WHEELS_COMMIT=2cf0a6915ce544dc493a0990f2ea38d81601128a
ARG VLLM_WHEELS_BASE=https://wheels.vllm.ai/rocm/${VLLM_WHEELS_COMMIT}
ARG COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI
ARG COMFYUI_TAG=v0.36.0

# 1. Runtime system deps.
# build-essential + python3.12-dev: several popular custom node packs
# (insightface, some controlnet-aux preprocessors) JIT-compile small C/C++
# extensions on first install via ComfyUI-Manager — same reasoning as
# vllm-gfx1151's aiter JIT requirement. git: ComfyUI-Manager installs node
# packs by cloning their repos at runtime into custom_nodes/, so it must be
# on PATH inside the running container, not just at build time.
# libgl1 + libglib2.0-0: opencv-python (pulled in by many node packs for
# preprocessing) needs libGL.so.1 / libglib at import time — absent from a
# bare Ubuntu base, invisible until a workflow using cv2 actually runs.
# ffmpeg: ComfyUI's own video nodes (VHS-style combine/load) shell out to
# it; "prêt à l'emploi" means it shouldn't be a first-run surprise.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      python3.12 python3.12-venv python3.12-dev \
      build-essential \
      libatomic1 libnuma-dev libgomp1 libelf1t64 \
      libdrm-dev zlib1g-dev libssl-dev \
      libgoogle-perftools4 libprotobuf32t64 libsleef3 \
      libgl1 libglib2.0-0 ffmpeg \
      procps \
    && rm -rf /var/lib/apt/lists/*

# 2. ROCm 7.14 runtime for gfx1151 (see vllm-gfx1151 Dockerfile step 2 for
# provenance — same tarball, same rationale).
WORKDIR /tmp
RUN mkdir -p /opt/rocm && \
    curl -fsSL "${ROCM_DIST_URL}" | tar xz -C /opt/rocm

# 3. Python 3.12 venv via uv — the wheels.vllm.ai ROCm set is cp312 only.
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
ENV UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=4 UV_NO_CACHE=1
RUN uv venv /opt/venv --python 3.12 && \
    uv pip install \
      pip==26.1.1 \
      wheel==0.47.0 \
      packaging==26.2 \
      setuptools==79.0.1

# 4. Strip dev/test/bench cruft from the ROCm tarball (identical list to
# vllm-gfx1151 — same tarball, same dead weight).
RUN rm -rf \
      /opt/rocm/bin/rocshmem_info /opt/rocm/bin/hipify-clang \
      /opt/rocm/bin/rocprof-sys-* /opt/rocm/bin/rocgdb-py* \
      /opt/rocm/bin/rdcd /opt/rocm/bin/rdci \
      /opt/rocm/bin/hipdnn_integration_tests /opt/rocm/bin/MIOpenDriver \
      /opt/rocm/bin/flatc \
      /opt/rocm/share/doc /opt/rocm/share/man \
      /opt/rocm/lib/*.a \
      /opt/rocm/lib/rdc /opt/rocm/lib/librocprof-sys* /opt/rocm/lib/rocprofiler-systems

# 5. The prebuilt ROCm wheel set from wheels.vllm.ai/rocm/ — torch stack
# only, no vllm/aiter/flash-attn. --no-deps: the rest of ComfyUI's runtime
# deps come from step 6b via its own requirements.txt.
RUN mkdir -p /tmp/wheels && cd /tmp/wheels && \
    for w in \
      "torch-2.12.0%2Bgit6bbd260-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "torchvision-0.27.1%2Bdf56172-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "torchaudio-2.11.0%2B34c52a6-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "triton-3.7.1%2Bgitf0b55c07-cp312-cp312-manylinux_2_35_x86_64.whl" \
      "amdsmi-26.2.2%2Bc2d9476115-py3-none-any.whl" \
    ; do \
      echo "fetch ${w}" && \
      curl -fsSL -o "$(python3 -c "import urllib.parse,sys;print(urllib.parse.unquote(sys.argv[1]))" "${w}")" \
        "${VLLM_WHEELS_BASE}/${w}" ; \
    done && \
    uv pip install --no-deps /tmp/wheels/*.whl && \
    cd / && rm -rf /tmp/wheels

# 5b. MPI stub — this torch build is WITH MPI regardless of what consumes
# it (vllm-gfx1151 or this image); libtorch_cpu.so / libtorch_python.so
# need ~178 MPI symbols to *resolve* at import time even though ComfyUI
# never calls into MPI. Identical fix to vllm-gfx1151 step 5b: enumerate
# every undefined MPI symbol and emit a no-op definition for each.
RUN set -eu; \
    cd /opt/venv/lib/python3.12/site-packages/torch/lib; \
    nm -D --undefined-only *.so 2>/dev/null \
      | awk '$1=="U"{print $2}' \
      | grep -E '^(MPI_|MPIX_|PMPI_|_ZN3MPI|_ZNK3MPI|ompi_)' \
      | sort -u \
      | awk '{print "void " $0 "(void){}"}' > /tmp/mpi_stub.c; \
    echo "MPI stub: $(grep -c . /tmp/mpi_stub.c) symbols"; \
    for s in libmpi.so.40 libmpi_cxx.so.40; do \
      gcc -x c -shared -fPIC -Wl,-soname,"$s" -o "/usr/local/lib/$s" /tmp/mpi_stub.c; \
    done; \
    ldconfig; \
    rm -f /tmp/mpi_stub.c; \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
      python -c "import ctypes, os; ctypes.CDLL('/opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so', mode=os.RTLD_NOW|os.RTLD_GLOBAL); print('libtorch_cpu.so: all symbols resolve (MPI stub complete)')"

# 6. ComfyUI itself, pinned tag, shallow clone.
RUN git clone --branch "${COMFYUI_TAG}" --depth 1 "${COMFYUI_REPO}" /opt/ComfyUI

# 6b. ComfyUI's own requirements.txt — constrained so the resolver can't
# swap our gfx1151-capable torch/torchvision/torchaudio/triton for vanilla
# PyPI CUDA/CPU wheels (requirements.txt itself leaves torch unpinned, same
# situation vllm-gfx1151 handles for its requirements/{common,rocm}.txt).
RUN TORCH_VER=$(python -c "from importlib.metadata import version; print(version('torch'))") && \
    TRITON_VER=$(python -c "from importlib.metadata import version; print(version('triton'))") && \
    TVIS_VER=$(python -c "from importlib.metadata import version; print(version('torchvision'))") && \
    TAUD_VER=$(python -c "from importlib.metadata import version; print(version('torchaudio'))") && \
    printf "torch==%s\ntriton==%s\ntorchvision==%s\ntorchaudio==%s\n" \
      "$TORCH_VER" "$TRITON_VER" "$TVIS_VER" "$TAUD_VER" > /tmp/constraints.txt && \
    echo "Pinning: torch==$TORCH_VER triton==$TRITON_VER torchvision==$TVIS_VER torchaudio==$TAUD_VER" && \
    uv pip install -r /opt/ComfyUI/requirements.txt --constraint /tmp/constraints.txt && \
    rm -f /tmp/constraints.txt

# 6c. Import gate — fail the CI build here, not on the Strix Halo node, if
# the native lib graph doesn't load. No GPU at build time (device init is
# lazy in torch), so this only catches missing .so / unresolved symbols —
# same class of bug that broke every vllm-gfx1151 assembly attempt.
RUN LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    python -c "import torch; print('torch', torch.__version__, torch.version.hip); \
import triton; print('triton', triton.__version__); \
import torchvision; print('torchvision', torchvision.__version__)" && \
    cd /opt/ComfyUI && LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    python -c "import comfy.options; import comfy.cli_args; print('comfy: cli_args import OK')"

# 7. Runtime env.
# ROCM_HOME/ROCM_PATH: custom nodes' _find_rocm_home()-style checks.
# CC/CXX: node packs that JIT-build C++ extensions default to bare cc/c++,
# which don't exist here — point at the clang toolchain from ROCm.
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1: unlike vllm-gfx1151 (where
# this flag was a no-op — its text-only decode path never calls SDPA),
# ComfyUI's whole attention path goes through torch SDPA via
# --use-pytorch-cross-attention, so this is the actual attention backend
# here, not a dead flag.
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++ \
    ROCBLAS_USE_HIPBLASLT=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    MIOPEN_FIND_MODE=FAST \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/ComfyUI
EXPOSE 8188
ENTRYPOINT ["/opt/venv/bin/python", "main.py"]
CMD ["--listen", "0.0.0.0", "--port", "8188"]
