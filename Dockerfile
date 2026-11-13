# syntax=docker/dockerfile:1
FROM mcr.microsoft.com/dotnet/sdk:11.0.100@sha256:65f3c7c6a25a239ab3e6719f2896147e7b68242ff2d8818dd70498bde6ef8bea AS build
WORKDIR /src
ENV NUGET_PACKAGES=/root/.nuget/packages

# Restore dependencies before copying source so dependency layers remain cacheable.
COPY global.json Directory.Build.props Directory.Build.targets Directory.Packages.props NuGet.config BannedSymbols.txt ./
COPY build ./build
COPY tools/olive/prune_attention_outputs.py ./tools/olive/prune_attention_outputs.py
COPY runtime/trt-rtx-ep.manifest.json ./runtime/trt-rtx-ep.manifest.json
COPY src/Trackdub.Cli/Trackdub.Cli.csproj src/Trackdub.Cli/
COPY src/Trackdub.Sdk/Trackdub.Sdk.csproj src/Trackdub.Sdk/
COPY src/Trackdub.Application/Trackdub.Application.csproj src/Trackdub.Application/
COPY src/Trackdub.Contracts/Trackdub.Contracts.csproj src/Trackdub.Contracts/
COPY src/Trackdub.Domain/Trackdub.Domain.csproj src/Trackdub.Domain/
COPY src/Trackdub.Composition/Trackdub.Composition.csproj src/Trackdub.Composition/
COPY src/Trackdub.Infrastructure/Trackdub.Infrastructure.csproj src/Trackdub.Infrastructure/
COPY src/Trackdub.Inference/Trackdub.Inference.csproj src/Trackdub.Inference/
COPY src/Trackdub.Inference.Onnx/Trackdub.Inference.Onnx.csproj src/Trackdub.Inference.Onnx/
COPY src/Trackdub.Licensing/Trackdub.Licensing.csproj src/Trackdub.Licensing/
COPY src/Trackdub.Media/Trackdub.Media.csproj src/Trackdub.Media/
COPY src/Trackdub.Media.Playback/Trackdub.Media.Playback.csproj src/Trackdub.Media.Playback/
COPY src/Trackdub.Analyzers/Trackdub.Analyzers.csproj src/Trackdub.Analyzers/
COPY src/Trackdub.OnnxRuntime.Dnnl.Native/Trackdub.OnnxRuntime.Dnnl.Native.csproj src/Trackdub.OnnxRuntime.Dnnl.Native/

RUN \
    dotnet restore src/Trackdub.Cli/Trackdub.Cli.csproj -p:TargetFramework=net11.0 -r linux-x64 \
    && dotnet restore src/Trackdub.Analyzers/Trackdub.Analyzers.csproj -p:TargetFramework=netstandard2.0

COPY src ./src
RUN \
    dotnet publish src/Trackdub.Cli/Trackdub.Cli.csproj \
    --configuration Release \
    --framework net11.0 \
    --runtime linux-x64 \
    --self-contained false \
    --no-restore \
    --output /app/publish \
    /p:PublishTrimmed=false \
    /p:PublishReadyToRun=false \
    /p:DebugType=None \
    /p:DebugSymbols=false

# NuGet publish asset selection puts the CPU Microsoft.ML.OnnxRuntime core at
# /app/publish/libonnxruntime.so even though the GPU provider library is present.
# That CPU core cannot register CUDAExecutionProvider. Overwrite it, in the Linux
# container publish only, with the CUDA-enabled core from the transitive
# Microsoft.ML.OnnxRuntime.Gpu.Linux package (version = OnnxRuntimeVersion in
# Directory.Packages.props). Desktop/Windows publishes are unaffected.
RUN \
    GPU_CORE="$HOME/.nuget/packages/microsoft.ml.onnxruntime.gpu.linux/1.31.0/runtimes/linux-x64/native/libonnxruntime.so" \
    && test -f "$GPU_CORE" \
    && cp "$GPU_CORE" /app/publish/libonnxruntime.so

# Publish also drops the base ORT GenAI native library for linux-x64: only the
# CUDA flavor (libonnxruntime-genai-cuda.so, a provider helper with a different
# ABI) lands in /app/publish, so the GenAI loader cannot find
# libonnxruntime-genai.so and every GenAI model fails to load in the container.
# Copy the base library from the Microsoft.ML.OnnxRuntimeGenAI package, same pattern as the
# GPU core overwrite above. The build fails if this pin drifts from the CUDA GenAI provider
# (pinned through OnnxRuntimeGenAiCudaVersion in Directory.Packages.props), so the base library
# and provider helper stay one version.
RUN \
    GENAI_VERSION="$(sed -n 's/.*Include="Microsoft.ML.OnnxRuntimeGenAI" Version="\([^"]*\)".*/\1/p' Directory.Packages.props)" \
    && GENAI_CUDA_VERSION="$(sed -n 's|.*<OnnxRuntimeGenAiCudaVersion>\([^<]*\)</OnnxRuntimeGenAiCudaVersion>.*|\1|p' Directory.Packages.props)" \
    && test -n "$GENAI_VERSION" \
    && test "$GENAI_VERSION" = "$GENAI_CUDA_VERSION" \
    && grep -qF 'Microsoft.ML.OnnxRuntimeGenAI.Cuda" Version="$(OnnxRuntimeGenAiCudaVersion)"' Directory.Packages.props \
    && GENAI_LIB="$HOME/.nuget/packages/microsoft.ml.onnxruntimegenai/${GENAI_VERSION}/runtimes/linux-x64/native/libonnxruntime-genai.so" \
    && test -f "$GENAI_LIB" \
    && cp "$GENAI_LIB" /app/publish/libonnxruntime-genai.so

FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04@sha256:f2c12914cf4751e61073843724275aa35d4817e01dbdc87eac03905971628c6e AS cuda-runtime
FROM mcr.microsoft.com/dotnet/runtime:11.0.0@sha256:3fe5f5495141c97986a49d3a4e76061f6225c3f93f94e8a7e522781376193863 AS dotnet-runtime

# Ubuntu 24.04 (noble) matches the ubuntu2404 CUDA/TensorRT package repo used below.
# MCR's default 11.0.0 runtime tag is Ubuntu 26.04 (resolute), so keep the noble
# base OS image and layer the .NET 11 runtime over /usr/share/dotnet.
FROM mcr.microsoft.com/dotnet/runtime:10.0-noble@sha256:a3f9f81954b9c810af2ed18849716c8dc74a4707bd0907fdbd69be81554c0f30 AS runtime-base
# Docker COPY merges directories; remove the .NET 10 host/framework first.
RUN rm -rf /usr/share/dotnet
COPY --from=dotnet-runtime /usr/share/dotnet /usr/share/dotnet
WORKDIR /app

# TensorRT 10 runtime for the native TensorRT EP: ORT 1.31's
# libonnxruntime_providers_tensorrt.so links libnvinfer.so.10, libcudnn.so.9 and
# libcudart.so.13. TensorRT 10.13 is the first 10.x line with CUDA 13.0 support
# (NVIDIA TensorRT 10.13.2 release notes). Network-repo install is NVIDIA's
# documented method for containers. Do not install TensorRT 11: wrong soname.
RUN apt-get update \
    && apt-get install --no-install-recommends --only-upgrade --yes apt libapt-pkg6.0t64 \
    && apt-get install --no-install-recommends --yes ca-certificates wget \
    && wget --quiet https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg --install cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install --no-install-recommends --yes \
        espeak-ng ffmpeg libgomp1 libstdc++6 util-linux \
        libnvinfer10=10.13.3.9-1+cuda13.0 \
        libnvinfer-plugin10=10.13.3.9-1+cuda13.0 \
        libnvonnxparsers10=10.13.3.9-1+cuda13.0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir --parents /usr/local/cuda/lib \
    && groupadd --system --gid 10001 trackdub \
    && useradd --system --uid 10001 --gid trackdub --create-home --home-dir /home/trackdub trackdub \
    && mkdir --parents /data/models /data/projects /data/engine-cache /work/input /work/output /work/tmp /logs /home/trackdub/.local/linml-deps \
    && chown --recursive trackdub:trackdub /app /data /work /logs /home/trackdub

ENV DOTNET_EnableDiagnostics=0 \
    DOTNET_TieredCompilation=1 \
    DOTNET_TieredCompilationQuickJit=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    TRACKDUB_ENGINE_CACHE_ROOT=/data/engine-cache \
    ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu \
    TMPDIR=/work/tmp

COPY --from=cuda-runtime /usr/local/cuda-13.0/targets/x86_64-linux/lib/ /usr/local/cuda/lib/
COPY --from=cuda-runtime /usr/lib/x86_64-linux-gnu/libcudnn*.so* /usr/local/cuda/lib/
RUN printf '/usr/local/cuda/lib\n' > /etc/ld.so.conf.d/cuda.conf && ldconfig

COPY --from=build --chown=trackdub:trackdub /app/publish ./
COPY --from=build --chown=trackdub:trackdub /src/src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json ./src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json

USER trackdub
ENTRYPOINT ["dotnet", "Trackdub.Cli.dll", "--model-directory", "/data/models"]
CMD ["doctor"]

# Two published profiles share runtime-base; the only difference is the profile
# marker consumed by provisioning (seed-and-warm-gpu-worker.sh). Neither bakes a
# VRAM budget: the session pool auto-scales from detected GPU VRAM.
#
# rtx is the TensorRT-RTX route (plugin installed during provisioning).
# server is the AWS datacenter-GPU route (G6/L4, G6e/L40S): classic TensorRT
# plus CUDA, no TensorRT-RTX plugin or license, and the poc-server-v1 preset
# pins the accepted CUDA baseline.
FROM runtime-base AS rtx
ENV TRACKDUB_GPU_PROFILE=rtx

FROM runtime-base AS server
ENV TRACKDUB_GPU_PROFILE=server
