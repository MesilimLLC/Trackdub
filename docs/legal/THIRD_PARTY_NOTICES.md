# Third-Party Notices

Track all third-party dependencies here.

This includes:

- NuGet packages
- native binaries
- FFmpeg builds
- ONNX Runtime / Windows ML dependencies
- model weights
- ONNX exports
- tokenizer files
- TTS voices
- sample media
- icons and branding assets

Do not add a dependency or model unless its license is recorded.

## Video Cleanup workflow adaptation (agent documentation)

- name: Video Cleanup by Mehdi Ksibi (`Mehdi-Ks/video-cleanup`), adapted as `trackdub-media-cleanup`
- version / revision: supplied package `Video Cleanup-1.0.1-v1`; no upstream commit or independently verified release is claimed
- source URL: https://github.com/Mehdi-Ks/video-cleanup
- license: MIT; Copyright (c) 2026 Mehdi Ksibi
- commercial use allowed? yes, under the retained MIT terms
- redistribution allowed? yes, with the upstream copyright and permission notice retained
- attribution required? yes; complete notice retained in [.claude/skills/trackdub-media-cleanup/LICENSE](../../.claude/skills/trackdub-media-cleanup/LICENSE)
- notes: Agent workflow documentation adapted from the supplied `SKILL.md` and README, not an Anthropic-authored release or endorsement. Trackdub-specific readiness, artifact, routing, consent, and verification rules replace the standalone Python recipe. No upstream processing scripts or additional runtime/model dependency is introduced by this adaptation. See [adaptation record](../../.claude/skills/trackdub-media-cleanup/ATTRIBUTION.md) and [operational entry point](../operations/media-cleanup.md). The adaptation is MIT; unrelated Trackdub code retains its existing license.

Each entry should include:

- name
- version / revision
- source URL
- license
- commercial use allowed?
- redistribution allowed?
- attribution required?
- notes

## FFmpeg / ffprobe binaries

- name: FFmpeg / ffprobe Windows x64 binaries (GyanD `codexffmpeg`, `essentials_build`)
- version / revision: FFmpeg 8.1.2, asset `ffmpeg-8.1.2-essentials_build.zip`
- source URL: https://github.com/GyanD/codexffmpeg/releases
- license: GNU General Public License, version 3 (GPLv3) — confirmed directly against the shipped `LICENSE` file and `ffmpeg -version`'s own `--enable-gpl --enable-version3` build configuration, not assumed from the filename
- commercial use allowed? yes
- redistribution allowed? yes, provided the corresponding-source obligation is met when the binary is conveyed — see notes; Trackdub itself does not convey this binary
- attribution required? yes
- notes: Used only for the automatic FFmpeg bootstrap path (`FfmpegAutoDownloader`). The downloader fetches this at runtime on the end user's own machine; Trackdub does not bundle, ship, or otherwise convey the binary itself, so the GPLv3 source-conveyance obligation stays with GyanD as the actual distributor, not with Trackdub. Pinned to the exact asset URL and a locally-computed SHA-256 (GyanD does not publish its own checksums); the downloader rejects any download whose bytes don't hash to that recorded value, so a changed asset is a hard failure rather than a silent swap. GyanD (gyan.dev) publishes versioned releases (this entry targets `8.1.2` specifically) rather than a rolling/"latest" tag — an observation about how GyanD structures releases, not a guarantee that the asset at this URL can never change; the pinned-hash check is what actually enforces that, not the URL's versioned appearance. Chosen specifically because this build variant includes `libx264` (verified: extracted the binary, ran `-encoders`), which the software H.264 encoder fallback in `Trackdub.Media` depends on.

- name: FFmpeg / ffprobe Windows arm64, Linux x64, and Linux arm64 binaries (BtbN FFmpeg-Builds, `gpl-shared`)
- version / revision: BtbN's literal `latest` tag (their continuously-republished current-pointer release, distinct from their dated `autobuild-*` tags), assets `ffmpeg-master-latest-winarm64-gpl-shared.zip`, `ffmpeg-master-latest-linux64-gpl-shared.tar.xz`, `ffmpeg-master-latest-linuxarm64-gpl-shared.tar.xz`
- source URL: https://github.com/BtbN/FFmpeg-Builds/releases
- license: GPLv3 for the selected `gpl-shared` build variant (BtbN does not publish a `-buildconf` per asset ahead of download the way this notice would ideally cite; treat as GPLv3 pending a build-time `-buildconf` capture, consistent with the Windows x64 entry above)
- commercial use allowed? yes
- redistribution allowed? yes, same corresponding-source reasoning as the Windows x64 entry — Trackdub does not convey this binary, it is fetched by the end user's own machine
- attribution required? yes
- notes: GyanD (used for Windows x64 above) does not publish arm64 or Linux builds, so these three RIDs use BtbN's `gpl-shared` variant of the rolling `latest` tag instead. Because `latest` is a moving target, there is no fixed asset to publish a hash for ahead of time (`Sha256` is `null` on these three packages — see the doc comment on `FfmpegDownloadPackage`), so `FfmpegAutoDownloader` falls back to trust-on-first-use: it persists the first successful download's hash and compares against it on any later re-download, which doesn't authenticate that first download but does catch the rolling release silently changing underneath an already-trusted install. This is weaker than the Windows x64 entry's real pinned-hash verification, accepted specifically because BtbN's dated tags were found to be unreliably retained (see issue #23) and GyanD does not cover these platforms.

## Lucene.NET analyzer package family

- name: Lucene.NET analyzer package family (`Lucene.Net.Analysis.Common`, `Lucene.Net.Analysis.Kuromoji`, `Lucene.Net.Analysis.SmartCn`, plus transitive Lucene.NET packages)
- version / revision: 4.8.0-beta00017
- source URL: https://www.nuget.org/packages/Lucene.Net.Analysis.Common/4.8.0-beta00017
- license: Apache-2.0
- commercial use allowed? yes
- redistribution allowed? yes, provided Apache-2.0 license notice is retained
- attribution required? yes
- notes: Managed NuGet analyzers used for project glossary matching in Arabic, Japanese, and Chinese. Lockfiles pin the Lucene.NET/ICU4N transitive closure used by this beta package family, and composition tests instantiate/tokenize through the registered analyzers as a runtime-load smoke test. Korean glossary matching remains on Trackdub's managed fallback scanner; no native dictionaries, Java runtime, Python runtime, or sidecar tokenizer is added in this slice.

## LibVLCSharp (.NET bindings for LibVLC)

- name: LibVLCSharp
- version / revision: centrally pinned in Directory.Packages.props
- source URL: https://www.nuget.org/packages/LibVLCSharp
- license: LGPL-2.1-or-later
- commercial use allowed? yes
- redistribution allowed? yes, provided LGPL-2.1 obligations are met (dynamic linking, no source modifications)
- attribution required? yes
- notes: .NET bindings for the LibVLC media framework. Trackdub links to LibVLC dynamically (LGPL compliance). No modifications are made to the LibVLC or LibVLCSharp source. Used in `Trackdub.Media.Playback` as one of two composited playback backends (libmpv is the primary compositor; LibVLC is the fallback). Bundling the native LibVLC runtime and any Avalonia-specific video-rendering control is a packaging concern of the consuming desktop product, not the public core, and is documented in that product's own third-party notices.

## LibVLC on Linux (system-installed)

- name: LibVLC (system package)
- version / revision: system-provided (distro package manager)
- source URL: https://www.videolan.org/vlc/
- license: LGPL-2.1-or-later
- commercial use allowed? yes
- redistribution allowed? n/a — not bundled; users install via their system package manager
- attribution required? yes (license notice in documentation)
- notes: On Linux, no NuGet runtime package is available. `Trackdub.Media.Playback`'s `LibVlcRuntimeLocator` falls back to the system-installed `libvlc.so`. End users must install VLC (e.g., `sudo apt install vlc` or equivalent) for the LibVLC playback backend to function.

## Trackdub.OnnxRuntime.Dnnl.Native (generated package)

- name: ONNX Runtime with oneDNN/DNNL execution provider
- version / revision: matches the managed `Microsoft.ML.OnnxRuntime` package version used by Trackdub; package skeleton currently targets 1.24.4
- source URL: https://github.com/microsoft/onnxruntime
- license: MIT for ONNX Runtime; oneDNN dependency is Apache-2.0
- commercial use allowed? yes
- redistribution allowed? yes, provided upstream license notices are retained
- attribution required? yes
- notes: Trackdub-owned native runtime flavor for manifest token `onnxruntime-dnnl`. Binaries are built by `tools/onnxruntime-dnnl/Build-OnnxRuntimeDnnlNativePackage.ps1` and packaged under `src/Trackdub.OnnxRuntime.Dnnl.Native/runtimes/<rid>/native/`; no placeholder binaries are checked in.

## Windows ML .NET APIs

- name: Microsoft.Windows.AI.MachineLearning (Windows ML APIs)
- version / revision: centrally pinned in Directory.Packages.props
- source URL: https://www.nuget.org/packages/Microsoft.Windows.AI.MachineLearning
- license: see NuGet license information (Windows ML license; separate from ONNX Runtime MIT license)
- commercial use allowed? yes, subject to Windows ML license terms
- redistribution allowed? yes, via NuGet / Windows App SDK runtime; Trackdub does not redistribute Windows ML binaries outside normal NuGet/runtime channels
- attribution required? yes (link to NuGet page and Windows ML docs in this notice and/or app documentation)
- notes: Provides the Windows ML managed API surface (`ExecutionProviderCatalog`, session creation, etc.) used by Trackdub’s inference layer on Windows 11.

## WindowsAppSDK.ML (Windows ML integration with Windows App SDK)

- name: Microsoft.WindowsAppSDK.ML
- version / revision: centrally pinned in Directory.Packages.props
- source URL: https://www.nuget.org/packages/Microsoft.WindowsAppSDK.ML
- license: see NuGet license information (Windows App SDK ML extension)
- commercial use allowed? yes, subject to Windows App SDK license terms
- redistribution allowed? yes, via Windows App SDK runtime; Trackdub does not redistribute the runtime outside normal Windows App SDK servicing
- attribution required? yes (link to NuGet page and Windows App SDK ML docs in this notice and/or app documentation)
- notes: Provides Windows App SDK integration and servicing for Windows ML execution providers on Windows 11 24H2+ devices. Trackdub uses it for catalog-delivered EPs (MIGraphX, NvTensorRtRtx, OpenVINO, QNN, VitisAI).

## ONNX Runtime and DirectML (included in Windows ML)

- name: onnxruntime.dll (ONNX Runtime engine) and DirectML.dll (included GPU execution provider)
- version / revision: version bundled with Microsoft.Windows.AI.MachineLearning / Microsoft.WindowsAppSDK.ML 2.x at time of writing
- source URL: https://onnxruntime.ai/ and https://learn.microsoft.com/windows/ai/new-windows-ml/distributing-your-app#whats-in-windows-ml
- license: ONNX Runtime is MIT-licensed; DirectML.dll is a Windows component covered by Windows licensing terms
- commercial use allowed? yes, subject to ONNX Runtime MIT license and Windows licensing terms
- redistribution allowed? yes, when using the recommended framework-dependent deployment; Trackdub does not bundle these binaries directly and relies on the Windows ML / Windows App SDK runtime
- attribution required? yes (link to ONNX Runtime site and Windows ML distribution docs in this notice and/or app documentation)
- notes: Windows ML is composed of `Microsoft.Windows.AI.MachineLearning.dll`, `onnxruntime.dll`, and `DirectML.dll` (~41 MB total). Trackdub uses the ONNX Runtime instance embedded in Windows ML; DirectML.dll serves as the included legacy GPU EP.

## MIGraphX (AMD Windows ML execution provider)

- name: MIGraphXExecutionProvider (AMD GPU execution provider via Windows ML catalog)
- version / revision: catalog-delivered; current MSIX `1.8.55.0` on Windows ML 1.8.x / 2.x at time of writing
- source URL: https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html
- license: AMD Ryzen AI Licensing Information (see https://ryzenai.docs.amd.com/en/latest/licenses.html)
- commercial use allowed? yes, subject to AMD Ryzen AI license terms
- redistribution allowed? yes, via Windows ML catalog; Trackdub does not bundle MIGraphX binaries directly and relies on Windows Update for deployment
- attribution required? yes (link to AMD MIGraphX / Ryzen AI docs and license in this notice and/or app documentation)
- notes: MIGraphX EP accelerates inference on supported AMD GPUs. It is delivered as a Windows ML catalog EP on Windows 11 24H2+ systems and is marked as “not supported for GenAI scenarios today” in Microsoft’s documentation. Trackdub exposes a Model Manager “Install provider” flow that presents the AMD Ryzen AI license link before installation.

## NvTensorRTRTXExecutionProvider (NVIDIA TensorRT RTX EP ABI plugin)

- name: NvTensorRTRTXExecutionProvider (NVIDIA TensorRT RTX execution provider via ONNX Runtime EP ABI plugin bundle)
- version / revision: pinned bundle `0.5.0` (win-x64 and linux-x64, vendoring TensorRT-RTX `1.7.1`; Linux is installed through LinML.Catalog), CUDA `cu13`, per `runtime/trt-rtx-ep.manifest.json` (NVIDIA TensorRT-RTX-EP-ABI release v0.5.0)
- source URL: https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html (upstream EP docs); bundle archives from https://github.com/NVIDIA/TensorRT-RTX-EP-ABI/releases/tag/v0.5.0 (win-x64 and linux-x64)
- license: NVIDIA SOFTWARE LICENSE AGREEMENT (TensorRT-RTX) and NVIDIA CUDA EULA (see https://docs.nvidia.com/deeplearning/tensorrt-rtx/latest/reference/sla.html and https://docs.nvidia.com/cuda/eula/index.html)
- commercial use allowed? yes, subject to NVIDIA TensorRT-RTX and CUDA license terms
- redistribution allowed? yes, via Trackdub’s manifest-driven download/install of the EP ABI plugin bundle into the user data directory; not via Windows ML catalog
- attribution required? yes (link to NVIDIA TensorRT-RTX docs and license pages in this notice and/or app documentation)
- notes: Trackdub registers `onnxruntime_providers_nv_tensorrt_rtx` / `libonnxruntime_providers_nv_tensorrt_rtx.so` from the installed bundle directory. Model Manager requires `NvidiaTensorRtRtxLicenseAccepted` before download/install. Requires NVIDIA GeForce RTX 30xx or newer with a supported driver/CUDA stack per NVIDIA documentation.

## NVIDIA Audio Effects (Maxine AFX) SDK and hosted Studio Voice runtime

- name: NVIDIA Audio Effects (AFX / Maxine) SDK, core package and per-effect feature models, plus the Trackdub-hosted Studio Voice High Quality runtime
- version / revision: verified with SDK `3.0.0.51` and `2.1.0.9` (NGC org `nvidia`, team `maxine`: resource `maxine_windows_audio_effects_sdk`, models `afx_win_*`); feature versions must match the SDK version. The hosted Studio Voice runtime is AFX `3.0.0` for Windows x86, built per GPU architecture (Turing, Ampere, Ada, Blackwell) and pinned by SHA-256 in `src/Trackdub.Composition/nvidiaafx-runtime.manifest.json`
- source URL: https://docs.nvidia.com/maxine/afx/latest/index.html ; the user-installed SDK and non-Studio-Voice features come from the NVIDIA NGC catalog with the user's own NGC account. The Studio Voice runtime is downloaded from `https://downloads.trackdub.com/afx/3.0.0/`
- license: the user-installed SDK and its feature packages are under the NVIDIA Software License Agreement (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the NVIDIA Product-Specific Terms for AI Products, as shipped in each package's `license` folder; the Early Access Speaker Focus effect is under the NVIDIA Software and Model Evaluation License Agreement. The hosted Studio Voice High Quality runtime and its models are under the NVIDIA Open Model License (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/), and NVIDIA's license documents ship inside each hosted archive
- commercial use allowed? yes for the hosted Studio Voice High Quality runtime under the NVIDIA Open Model License; the user-installed SDK and other AFX profiles remain governed by the user's own agreement with NVIDIA and are not treated as commercial-safe until reviewed
- redistribution allowed? Trackdub hosts and redistributes the Studio Voice High Quality runtime package only, under the NVIDIA Open Model License. Trackdub does not host or redistribute the core AFX SDK or any other AFX feature package; users install those themselves
- attribution required? yes; NVIDIA's branding guidelines apply when AFX is integrated (https://www.nvidia.com/maxine-sdk-guidelines)
- notes: Optional and off by default. Readiness requires the user to accept the NVIDIA AFX license in Settings (`NvidiaAfxLicenseAccepted`). Studio Voice can be installed from Settings without NGC credentials; all other profiles require the user to point Trackdub at a local SDK folder (`NvidiaAfxRuntimeDirectory`). Early Access effects (Speaker Focus) are licensed by NVIDIA for internal testing and evaluation only, not for production use, and must not be shipped to end users unless NVIDIA grants that in writing. NVIDIA's 3.0.0 documentation marks Acoustic Echo Cancellation and Voice Font as deprecated. The commercial-safe mode impact of the optional hosted Studio Voice runtime is limited to the NVIDIA Open Model License terms; the user-installed SDK remains an open review item.

## OpenVINO (Intel Windows ML execution provider)

- name: OpenVINOExecutionProvider (Intel OpenVINO execution provider via Windows ML catalog)
- version / revision: catalog-delivered; current MSIX `1.8.69.0` (OpenVINO 2026.0) on Windows ML 1.8.x / 2.x at time of writing
- source URL: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html
- license: Intel OBL Distribution Commercial Use License Agreement v2025.02.12 (see https://cdrdv2.intel.com/v1/dl/getContent/849090?explicitVersion=true)
- commercial use allowed? yes, subject to Intel OBL license terms
- redistribution allowed? yes, via Windows ML catalog; Trackdub does not bundle OpenVINO binaries directly and relies on Windows Update for deployment
- attribution required? yes (link to Intel OpenVINO documentation and license in app documentation and this notice)
- notes: OpenVINO EP is installed and updated by Windows ML’s `ExecutionProviderCatalog` on compatible Intel hardware (Tiger Lake/11th Gen+ CPU, Alder Lake/12th Gen+ GPU, Arrow Lake/15th Gen+ NPU). Trackdub surfaces an explicit “Install provider” action and shows the Intel OBL license link before calling `EnsureReadyAsync` for this EP.

## QNN (Qualcomm Windows ML execution provider)

- name: QNNExecutionProvider (Qualcomm QNN execution provider via Windows ML catalog)
- version / revision: catalog-delivered; current MSIX `2.2420.43.0` (QAIRT 2.42) on Windows ML 2.x at time of writing
- source URL: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
- license: Qualcomm QNN license (contained in the Qualcomm® Neural Processing SDK ZIP as `LICENSE.pdf`)
- commercial use allowed? yes, subject to Qualcomm QNN license terms
- redistribution allowed? yes, via Windows ML catalog; Trackdub does not bundle QNN binaries directly and relies on Windows Update for deployment
- attribution required? yes (link to Qualcomm QNN documentation and SDK in app documentation and this notice)
- notes: QNN EP targets Snapdragon X Elite / X Plus devices with a Hexagon NPU and minimum driver version 30.0.140.0+. Trackdub surfaces an explicit “Install provider” action and requires the user to acknowledge the QNN license (by linking to the Qualcomm Neural Processing SDK download page and its LICENSE.pdf) before calling `EnsureReadyAsync` for this EP.

## VitisAI (AMD Windows ML execution provider)

- name: VitisAIExecutionProvider (AMD Ryzen AI / XDNA NPU execution provider via Windows ML catalog)
- version / revision: catalog-delivered; current MSIX `1.8.59.0` on Windows ML 1.8.x / 2.x at time of writing
- source URL: https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html
- license: AMD Ryzen AI Licensing Information (see https://ryzenai.docs.amd.com/en/latest/licenses.html)
- commercial use allowed? yes, subject to AMD Ryzen AI license terms
- redistribution allowed? yes, via Windows ML catalog; Trackdub does not bundle VitisAI binaries directly and relies on Windows Update for deployment
- attribution required? yes (link to AMD Ryzen AI documentation and license in this notice and/or app documentation)
- notes: VitisAI EP accelerates inference on AMD Ryzen AI (XDNA) NPUs. It is supported only within a bounded driver range (Adrenalin Edition 25.6.3–25.9.1, NPU driver 32.00.0203.280–32.00.0203.297). Trackdub treats VitisAI under the same AMD Ryzen AI licensing umbrella as MIGraphX and requires a one-time AMD Ryzen AI license acknowledgment in the Model Manager before installation.

## Bundled inference models (ONNX)

Entries mirror `src/Trackdub.Inference/Runtime/ModelManifest/bundled-models.manifest.json`. Each `model_id` below is listed when `requires_attribution=true`.

### cgus/diar_streaming_sortformer_4spk-v2.1-onnx (sortformer diarization)

- name: NVIDIA Streaming Sortformer diarization ONNX export (`cgus/diar_streaming_sortformer_4spk-v2.1-onnx`)
- version / revision: `2be05a08b477e8a526fd26963802845069c02c7c` (Trackdub download pin)
- source URL: https://huggingface.co/tonythethompson/diar-streaming-sortformer-4spk-v2.1-onnx
- license: NVIDIA Open Model License
- commercial use allowed? yes (per manifest `commercial_allowed` and `commercial_use_verified`)
- redistribution allowed? yes (per manifest)
- attribution required? yes
- notes: Speaker diarization (`engine_family`: `sortformer`). Upstream NVIDIA Sortformer weights; ONNX packaging on Hugging Face.

### tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx (Nemotron 3.5 ASR)

- name: NVIDIA Nemotron 3.5 ASR Streaming Multilingual 0.6B ONNX bundle (`tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx`)
- version / revision: `b3ea33d792e4edd1ea9ffe222d250bc3239ee4ae` (Trackdub download pin)
- source URL: https://huggingface.co/tonythethompson/nemotron-3.5-asr-streaming-0.6b-onnx
- upstream model URL: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- ONNX artifact source used for this packaging pass: https://huggingface.co/altunenes/parakeet-rs/tree/main/nemotron-3.5-asr-streaming-0.6b-onnx
- license: OpenMDW-1.1
- commercial use allowed? yes (per manifest `commercial_allowed` and `commercial_use_verified`)
- redistribution allowed? yes (per manifest)
- attribution required? yes
- notes: Multilingual ASR (`engine_family`: `nemotron-asr`). The bundle includes `encoder.onnx`, `encoder.onnx.data`, `decoder_joint.onnx`, `tokenizer.model`, `config.json`, `LICENSE.OpenMDW-1.1`, and `NOTICE.md`. NVIDIA lists the training-data stack as NVIDIA Riva multilingual ASR training data, NVIDIA Granary, Multilingual LibriSpeech, Mozilla Common Voice, FLEURS, VoxPopuli, and Europarl-ASR. Relevant upstream dataset license notices include Granary under CC-BY-3.0 with some listed source components under CC-BY-4.0, Multilingual LibriSpeech under CC-BY-4.0, Common Voice under CC0, FLEURS under CC-BY-4.0, VoxPopuli under CC0 with European Parliament raw-data notice, and Europarl-ASR under CC-BY-4.0. The ONNX runtime wrapper/export reference used during implementation is from `parakeet-rs`, licensed MIT OR Apache-2.0.

### tonythethompson/parakeet-tdt-0.6b-v3-onnx (Parakeet TDT 0.6B v3)

- name: NVIDIA Parakeet TDT 0.6B v3 ONNX bundle (`tonythethompson/parakeet-tdt-0.6b-v3-onnx`)
- version / revision: `107353b64181b47e250dfa846cf3de6a0b084c73` (Trackdub download pin)
- source URL: https://huggingface.co/tonythethompson/parakeet-tdt-0.6b-v3-onnx
- upstream model URL: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- ONNX artifact source used for this packaging pass: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx (revision `8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce`, redistributed unmodified)
- license: CC-BY-4.0
- commercial use allowed? yes (per manifest `commercial_allowed` and `commercial_use_verified`)
- redistribution allowed? yes (per manifest)
- attribution required? yes
- notes: European-language ASR (`engine_family`: `parakeet-tdt`, 25 languages). The bundle includes `nemo128.onnx`, `encoder-model.onnx`, `encoder-model.onnx.data`, `decoder_joint-model.onnx`, `vocab.txt`, `config.json`, `LICENSE.CC-BY-4.0`, and `NOTICE.md`. NVIDIA trained the model primarily on the NVIDIA Granary multilingual corpus. The export was produced with NVIDIA NeMo for `onnx-asr` (https://github.com/istupakov/onnx-asr), licensed MIT; the TDT greedy decoding reference used during implementation is from `parakeet-rs`, licensed MIT OR Apache-2.0.

### csukuangfj/sherpa-onnx-spleeter-2stems (Spleeter separation)

- name: Sherpa-ONNX Spleeter 2-stem separation (`csukuangfj/sherpa-onnx-spleeter-2stems`)
- version / revision: `3e5a4dd68259675ab0ddc38269fdae1d0c41f20c` (Trackdub manifest pin)
- source URL: https://huggingface.co/csukuangfj/sherpa-onnx-spleeter-2stems
- file hashes (sha256): `vocals.onnx` `932fff35af1ded8eedc501031bfd6dc12cb0024a3dfa8e2c4537fd0aee9eb149`; `accompaniment.onnx` `79ee6096c9fdac9f6df781fea97035f0ffda48f73441d053b131cf220d8a7d73`
- license: MIT (upstream Deezer Spleeter model; the HF repository itself ships no license file or model card)
- upstream notices preserved:
  - Spleeter: MIT License, Copyright (c) 2019-present, Deezer SA. https://github.com/deezer/spleeter/blob/master/LICENSE
  - sherpa-onnx (ONNX export and `scripts/spleeter` reference implementation): Apache License 2.0, Copyright Xiaomi Corp. https://github.com/k2-fsa/sherpa-onnx/blob/master/LICENSE
- commercial use allowed? yes
- redistribution allowed? yes (ONNX conversion is a modification; retain both notices)
- attribution required? yes
- notes: Optional vocals/accompaniment separation (`engine_family`: `spleeter`). Revision `3e5a4dd` replaced the previously shipped `main` snapshot (`7001ba3`); the weight bytes differ.

### tonythethompson/sepformer-whamr16k-onnx (SepFormer + overlap speech detection)

- name: SepFormer WHAMR! separation + pyannote OSD ONNX bundle (`tonythethompson/sepformer-whamr16k-onnx`)
- version / revision: `58bdc0d470478f6ad6da352c5852df3f146fb431` (Trackdub download pin)
- source URL: https://huggingface.co/tonythethompson/sepformer-whamr16k-onnx
- license: composite — `sepformer.onnx` derived from SpeechBrain (Apache-2.0); `osd.onnx` derived from pyannote (MIT)
- commercial use allowed? yes (per manifest `commercial_allowed` and `commercial_use_verified`)
- redistribution allowed? yes (per manifest)
- attribution required? yes
- notes: Speech-speech separation (`engine_family`: `sepformer`). Trackdub ships two ONNX files from one Hugging Face repo: (1) `sepformer.onnx` — exported from https://huggingface.co/speechbrain/sepformer-whamr16k (Apache-2.0; WHAMR! training set). (2) `osd.onnx` — overlap-speech detector exported from https://huggingface.co/pyannote/segmentation-3.0 (MIT; HF gated for weight download only; end users receive the pre-exported ONNX, not pyannote weights). Runtime uses OSD to find overlap regions, then runs SepFormer on overlapping chunks only. SHA-256 pins: `sepformer.onnx` = `d1e5ed49b7cc09e2c5946ce710fec33692b13a69c9c40441b0b14120cb649553`; `osd.onnx` = `52e073b4b6ae20b55c7cfd59c73be3388e62115c49c27d3e2257c1c0601f7143`. Export script: `scripts/export-sepformer-onnx.py`. Cite SpeechBrain, SepFormer (Subakan et al., ICASSP 2021), and pyannote segmentation (Plaquet & Bredin, INTERSPEECH 2023) where attribution is shown.

### tonythethompson/deepfilternet3-onnx (DeepFilterNet3 speech enhancement)

- name: DeepFilterNet3 ONNX (`tonythethompson/deepfilternet3-onnx`)
- version / revision: `dcbbe520263d1061693c4c4a56a6d6a917f30b25`
- source URL: https://huggingface.co/tonythethompson/deepfilternet3-onnx
- license: MIT
- commercial use allowed? yes (per manifest `commercial_use_verified`)
- redistribution allowed? yes
- attribution required? yes
- notes: Experimental lane speech enhancement (`engine_family`: `deepfilternet3`).

### onnx-community/Kokoro-82M-v1.0-ONNX (Kokoro TTS)

- name: Kokoro 82M v1.0 ONNX (`onnx-community/Kokoro-82M-v1.0-ONNX`)
- version / revision: `1939ad2a8e416c0acfeecc08a694d14ef25f2231`
- source URL: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
- license: Apache-2.0
- commercial use allowed? yes
- redistribution allowed? yes
- attribution required? yes
- notes: English TTS with bundled voice `.bin` files (`engine_family`: `kokoro`). Starter-pack variant ONNX files (`onnx/model_q8f16.onnx`, `onnx/model_fp16.onnx`) may download from Trackdub mirrors under the same Apache-2.0 terms; see **Trackdub premade HF variant mirrors** below.

### Chatterbox TTS (engine_family: chatterbox)

- name: Chatterbox / Chatterbox Turbo / Chatterbox Multilingual ONNX TTS
- version / revision: see per-model pins in bundled manifest
- source URL: https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX, https://huggingface.co/onnx-community/chatterbox-ONNX, https://huggingface.co/onnx-community/chatterbox-multilingual-ONNX
- license: MIT
- commercial use allowed? yes (per manifest `commercial_use_verified` where flipped; see `docs/internal/model-audits/chatterbox.md`)
- redistribution allowed? yes
- attribution required? yes
- notes: Bundled `model_id` values: `ResembleAI/chatterbox-turbo-ONNX`, `onnx-community/chatterbox-ONNX`, `onnx-community/chatterbox-multilingual-ONNX`. Voice-cloning capable; `requires_user_consent: true` unchanged by verification (ADR-0006).

### CosyVoice 300M ONNX TTS (engine_family: cosyvoice)

- name: CosyVoice 300M ONNX voice-cloning bundle
- version / revision: see `tonythethompson/CosyVoice-300M-ONNX` pin in bundled manifest (`1dc153e98d3ef267fc2d71dac0feab9ee35bdb03` at integration)
- source URL: https://huggingface.co/tonythethompson/CosyVoice-300M-ONNX (derived from https://huggingface.co/FunAudioLLM/CosyVoice-300M)
- license: Apache-2.0 (upstream FunAudioLLM CosyVoice-300M; conversion disclaimer on ONNX repo)
- commercial use allowed? yes (`commercial_use_verified: true` after CPU FP32 + int8 integration smokes; see `docs/internal/model-audits/cosyvoice-300m-onnx.md`)
- redistribution allowed? yes (subject to Apache-2.0 upstream terms)
- attribution required? yes
- notes: Multilingual zero-shot voice cloning (`voice_cloning: true`, `requires_user_consent: true`). Premade ModelOpt `int8`/`int4` variants on HF; local Olive optimization not enabled in v1.

### ByteDance/LatentSync-1.6 ONNX (engine_family: latentsync-diffusion)

- name: LatentSync 1.6 ONNX lip synthesis bundle (`ByteDance/LatentSync-1.6` / alias `latentsync`)
- version / revision: `8e1dd855e910df770732bb1be7d77666dd28ee45` (Trackdub ONNX mirror pin)
- source URL: https://huggingface.co/ByteDance/LatentSync-1.6 (weights); ONNX mirror https://huggingface.co/tonythethompson/latentsync-1.6-onnx
- license: openrail++ (upstream ByteDance weights; behavioral restrictions apply)
- commercial use allowed? yes, pending integration smoke (`commercial_allowed: true`; `commercial_use_verified: false` until real-model smoke passes — see `docs/internal/model-audits/latentsync-1-6-approved.md`)
- redistribution allowed? yes (per manifest)
- attribution required? yes
- notes: M23 original-footage lip repair (`task`: `lip-synthesis`). ONNX bundle includes `unet.onnx`, `vae_encoder.onnx`, `vae_decoder.onnx`, and `whisper_encoder.onnx` with per-file sha256 pins in bundled manifest. Companion face models for quality gating: `InsightFace/scrfd-500m` (MIT) and `InsightFace/2d106det` (MIT). MuseTalk remains experimental and is not the shipping lane. Cite ByteDance LatentSync where attribution is shown.
### Helsinki OPUS-MT ONNX pairs (engine_family: opus-mt)

- name: OPUS-MT Marian translation ONNX exports
- version / revision: see per-model `revision` in bundled manifest
- source URL: https://huggingface.co/onnx-community and https://huggingface.co/Xenova (opus-mt-* repos)
- license: CC-BY-4.0 (onnx-community) / MIT export + CC-BY upstream (Xenova)
- commercial use allowed? yes (per manifest `commercial_use_verified` for audited pairs)
- redistribution allowed? yes
- attribution required? yes
- notes: Bundled `model_id` values: `onnx-community/opus-mt-en-es`, `onnx-community/opus-mt-es-en`, `onnx-community/opus-mt-en-fr`, `onnx-community/opus-mt-en-de`, `onnx-community/opus-mt-en-it`, `onnx-community/opus-mt-en-ROMANCE`, `Xenova/opus-mt-es-fr`, `Xenova/opus-mt-es-de`, `Xenova/opus-mt-es-it`. Derived from Helsinki-NLP OPUS-MT checkpoints; ONNX community / Xenova packaging.

### Microsoft Phi ONNX GenAI translation (engine_family: phi-genai)

- name: Microsoft Phi instruct ONNX (ORT GenAI)
- version / revision: see per-model `revision` in bundled manifest
- source URL: https://huggingface.co/microsoft/Phi-3.5-mini-instruct-onnx, https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx, https://huggingface.co/microsoft/phi-4-onnx
- license: MIT
- commercial use allowed? yes (per manifest `commercial_use_verified` for all three Phi GenAI bundles)
- redistribution allowed? yes
- attribution required? yes
- notes: Bundled `model_id` values: `microsoft/Phi-3.5-mini-instruct-onnx`, `microsoft/Phi-4-mini-instruct-onnx`, `microsoft/phi-4-onnx`. Pivot / multi-language translation via ORT GenAI. For `microsoft/Phi-4-mini-instruct-onnx`, `cpu-int4` and `gpu-int4` variant bundles may download from Trackdub mirrors; see **Trackdub premade HF variant mirrors** below.

### Trackdub premade HF variant mirrors (starter packs)

Trackdub-hosted Hugging Face repos that mirror upstream commercial ONNX variant files for low-spec starter-pack download. Same licenses and attribution as upstream; files are byte-identical or upstream-published quants republished under `tonythethompson/trackdub-*` for reliable resolve URLs.

| Mirror repo | Upstream | Variant | License |
|-------------|----------|---------|---------|
| [tonythethompson/trackdub-silero-vad-int8](https://huggingface.co/tonythethompson/trackdub-silero-vad-int8) | [onnx-community/silero-vad](https://huggingface.co/onnx-community/silero-vad) | `int8` | MIT |
| [tonythethompson/trackdub-silero-vad-fp16](https://huggingface.co/tonythethompson/trackdub-silero-vad-fp16) | [onnx-community/silero-vad](https://huggingface.co/onnx-community/silero-vad) | `fp16` | MIT |
| [tonythethompson/trackdub-kokoro-q8f16](https://huggingface.co/tonythethompson/trackdub-kokoro-q8f16) | [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) | `q8f16` | Apache-2.0 |
| [tonythethompson/trackdub-kokoro-fp16](https://huggingface.co/tonythethompson/trackdub-kokoro-fp16) | [onnx-community/Kokoro-82M-v1.0-ONNX](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX) | `fp16` | Apache-2.0 |
| [tonythethompson/trackdub-phi-4-mini-cpu-int4](https://huggingface.co/tonythethompson/trackdub-phi-4-mini-cpu-int4) | [microsoft/Phi-4-mini-instruct-onnx](https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx) | `cpu-int4` | MIT |
| [tonythethompson/trackdub-phi-4-mini-gpu-int4](https://huggingface.co/tonythethompson/trackdub-phi-4-mini-gpu-int4) | [microsoft/Phi-4-mini-instruct-onnx](https://huggingface.co/microsoft/Phi-4-mini-instruct-onnx) | `gpu-int4` | MIT |

Manifest wiring: `download_file_sources` overrides on the upstream `model_id` entries in `bundled-models.manifest.json`. Voice/tokenizer assets for Kokoro still resolve from upstream unless overridden.

### google/madlad400-3b-mt (MADLAD-400 translation)

- name: MADLAD-400 3B machine translation ONNX (`google/madlad400-3b-mt`)
- version / revision: `67037ad42f58d6c0fc3dafaa45f3ec97a46e7eb9` (Trackdub ONNX export pin)
- source URL: https://huggingface.co/tonythethompson/madlad400-3b-mt-onnx (weights derived from https://huggingface.co/google/madlad400-3b-mt)
- license: Apache-2.0
- commercial use allowed? yes (per manifest `commercial_use_verified`)
- redistribution allowed? yes
- attribution required? yes
- notes: Multi-language translation (`engine_family`: `madlad`).

### Qwen3 TTS ONNX (engine_family: qwen3-tts)

- name: Qwen3-TTS 0.6B CustomVoice ONNX
- version / revision: `747ac4c3f6c6a317e83f6303148f28585a8bcadf`
- source URL: https://huggingface.co/tonythethompson/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX
- license: Apache-2.0
- commercial use allowed? yes (per manifest `commercial_use_verified`)
- redistribution allowed? yes
- attribution required? yes
- notes: Preset voices only (`voice_cloning: false`). Upstream: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

- name: Qwen3-TTS 1.7B CustomVoice ONNX
- version / revision: `a2b73f56ffec086c75f154024d5fc7f391f228af`
- source URL: https://huggingface.co/tonythethompson/Qwen3-TTS-12Hz-1.7B-CustomVoice-ONNX
- license: Apache-2.0
- commercial use allowed? yes (per manifest `commercial_use_verified`)
- redistribution allowed? yes
- attribution required? yes
- notes: Preset voices only (`voice_cloning: false`). Upstream: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

- name: Qwen3-TTS 0.6B Base ONNX (voice clone)
- version / revision: `17a2fccf89a5391005f9ff163b07e13f7814dddf`
- source URL: https://huggingface.co/tonythethompson/Qwen3-TTS-12Hz-0.6B-Base-ONNX
- license: Apache-2.0
- commercial use allowed? yes (per manifest `commercial_use_verified`)
- redistribution allowed? yes
- attribution required? yes
- notes: Voice clone (`voice_cloning: true`, `requires_user_consent: true`). Upstream: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base

- name: Qwen3-TTS 1.7B Base ONNX (voice clone)
- version / revision: `ab09194b7e07e645f2165fda2e95ac97c2446b38`
- source URL: https://huggingface.co/tonythethompson/Qwen3-TTS-12Hz-1.7B-Base-ONNX
- license: Apache-2.0
- commercial use allowed? yes (per manifest `commercial_use_verified`)
- redistribution allowed? yes
- attribution required? yes
- notes: Voice clone (`voice_cloning: true`, `requires_user_consent: true`). Upstream: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base

See `docs/internal/model-audits/qwen3-tts-experimental.md` for CPU smoke evidence.

### Qwen3 forced aligner ONNX (engine_family: onnx-qwen-forced-aligner)

- name: Qwen3 Forced Aligner 0.6B q4 ONNX
- version / revision: `4b904d4e0eb18c8dd1726ff5e830f72c06dd665b`
- source URL: https://huggingface.co/tonythethompson/Qwen3-ForcedAligner-0.6B-ONNX
- license: Apache-2.0
- commercial use allowed? yes (per manifest `commercial_use_verified`)
- redistribution allowed? yes
- attribution required? no
- notes: Bundled `model_id`: `qwen3-forced-aligner-0.6b-q4-onnx`. Word-level forced alignment.

### MADLAD-400 training dataset (model training data)

- name: MADLAD-400 Dataset
- version / revision: 3T token monolingual dataset, 419 languages (based on CommonCrawl, September 2023 snapshot)
- source URL: https://huggingface.co/datasets/google/madlad-400 (paper: https://arxiv.org/abs/2309.04662)
- license: ODC-BY (Open Data Commons Attribution License 1.0)
- commercial use allowed? yes (per odc-by terms)
- redistribution allowed? yes (per odc-by terms)
- attribution required? yes
- notes: Training data for the MADLAD-400 3B MT model (`google/madlad400-3b-mt`). Curated and audited by Google Research. The odc-by license requires attribution when the dataset or its derived works (including models trained on it) are used. This entry covers the dataset only; the model weights have their own Apache-2.0 license entry above.
