# RefForge · 开发笔记（PROJECT_NOTES.md）

> 本文件记录本项目的**实际**架构、关键数值、踩坑与解决办法、以及环境/开发/验收 Checklist。
>
> **维护规则（强制）**：开发中凡遇到新的坑、新约定、解决办法，必须即时写入本文件；需求文档 `REQUIREMENTS.md` 为初步定义，实际实现为准，若与文档不一致需同步回改 `REQUIREMENTS.md`。
>
> **版本说明（当前）**：本工具**已移除运镜识别**，不再输出 `motion_meta.json`；`pipeline/motion.py` 仅保留尺度轨迹累积供 `--debug` 诊断，故事图选帧不再依赖它（改为 3x3 九宫格、均匀时间间隔就近取清晰帧）。第五节的“历史记录（已移除）”列出与运镜识别相关的旧条目，仅供追溯。

---

## 1. 项目架构

> 记录**实际实现**的模块划分、数据流与关键接口。

### 1.1 目录结构（实际）

```text
ref_forge/
    启动.bat                  # 双击入口：设置包内环境并调用 app/main.py
    dev.bat                   # 开发入口：在包内环境执行任意 python 命令
    清理生成内容.bat           # 一键清理生成物（调用 scripts/clean_generated.py --yes）
    下载模型.bat               # 一键下载除 llama.cpp 外的全部模型（ASR+分离）
    一键环境搭建.bat            # 克隆后首次环境搭建：引导 Python + 依赖 + 运行期资产 + 全部模型
    config.json               # 阈值配置（与 app/config.py 默认值一致）
    requirements.txt          # 包内依赖固定版本
    AGENTS.md                 # 面向 AI/开发者的项目约定
    PROJECT_NOTES.md          # 本文件
    readme.md                 # 用户使用说明（面向使用者）
    REQUIREMENTS.md           # 需求与设计文档（以代码为准，按执行顺序）
    python/                   # 包内独立 Python 运行时（可移植，3.10.11）
    app/
        main.py               # CLI 入口与流程编排（分割 -> 故事图 -> 分离 -> ASR -> 描述）
        config.py             # 配置加载 + 内置默认值
        io_utils.py           # 路径解析 / JSON / JPEG / 视频枚举 / safe_stem / shot_detail_dir
        logging_utils.py      # 控制台 + output/run.log 日志
        timing.py             # StageTimer：线程安全阶段计时
        pipeline/
            __init__.py
            segmentation.py   # 镜头分割：PySceneDetect + 迭代时长修正
            segment_export.py # 片段导出：ffmpeg copy/reencode + ffprobe
            video_io.py       # 元信息、灰度抽帧（支持帧区间）、彩色回读
            quality.py        # 帧质量评估
            trim.py           # 有效区间修剪
            sampling.py       # 多帧采样
            features.py       # SIFT（回落 ORB）提取与匹配
            motion.py         # 单对估计 + 尺度轨迹累积（仅供 --debug 诊断）
        reference/
            __init__.py
            generator.py      # 宫格故事图（3x3，均匀时间间隔就近取清晰帧）
            storyboard.py     # 网格拼贴
        speech/               # 音频子系统（人声分离 + Qwen3-ASR + FunASR 声纹）
            __init__.py
            audio.py          # ffmpeg 抽音 / WAV 读写
            separation.py     # 人声/背景音乐分离（audio-separator，GPU/CPU，本地离线）
            models.py         # 设备/精度选择 + 模型惰性加载
            diarize.py        # fsmn-vad + CAM++ / ERes2NetV2 嵌入与聚类
            transcribe.py     # Qwen3-ASR 批量转写 + 词级时间戳
            voiceprint.py     # 全局说话人声纹中心库（JSON）
            export.py         # SRT / 结构化转录导出
            runner.py         # 识别阶段编排与按镜头切分
        describe/             # 视频描述子系统（llama.cpp + Qwen3.5 多模态）
            __init__.py
            frames.py         # ffmpeg 2fps 抽帧（每格取清晰帧，PNG，≤1080P）
            prompt.py         # H3 Ref2VA 中文 prompt 组装与输出清理
            client.py         # llama-server 生命周期管理 + 多模态 chat
            runner.py         # 描述阶段编排与落盘
            prompts/          # storyboard_* / frames_* 两套中文模板
        assets/               # 可选字体（本期未使用）
    bin/                      # ffmpeg.exe / ffprobe.exe / ffplay.exe
    llama_cpp/                # 视频描述推理框架（llama.cpp）
        llama_bin/            # llama-server.exe + DLL
    models/                   # 全部离线模型目录
        asr/                  # Qwen3-ASR-1.7B / ForcedAligner / fsmn-vad / campplus / eres2netv2
        separator/            # 人声分离模型（fast/balanced/best/dereverb + UVR 元数据）
        llm/                  # llm_model.gguf / mmproj_model.gguf
    scripts/
        acceptance_common.py       # 验收公共工具（从 input/ 发现并复制真实视频素材）
        make_test_videos.py        # 可选合成素材工具（C1-C11 / S1-S4；验收不再依赖）
        run_segmentation_test.py   # 分割时长约束、故事图与目录结构校验（素材取 input/）
        run_asr_test.py            # 语音识别产物核对（SRT / JSON / 归档 / 声纹中心；素材取 input/）
        run_separation_test.py     # 人声分离三轨核对（原始 / 人声 / 伴奏 + 元数据；素材取 input/）
        run_describe_test.py       # 视频描述抽帧 + prompt 自检（--llm 端到端；素材取 input/）
        download_models.py         # ModelScope 下载全部语音模型
        download_separation_models.py  # 下载全部人声分离模型（多镜像加速）
        bootstrap_python.ps1       # 引导 embeddable Python 3.10.11 + get-pip（一键环境搭建调用）
        setup_env.py               # 一键安装依赖 + 运行期资产 + 全部模型（可 --skip-* / --force）
        download_utils.py          # 多源测速选最快源（probe_speed / pick_fastest / download_smart）
        download_runtime.py        # 下载 ffmpeg / llama.cpp / GGUF（含多源测速）
        clean_generated.py         # 预览/清理生成物（默认 dry-run）
        setup_runtime.bat          # 调用 setup_env.py（已引导 Python 后的环境安装入口）
    input/                    # 待处理视频（默认多镜头长视频）
    output/                   # 输出根
        run.log
        llama_server.log
        voiceprint_center.json # 全局说话人声纹中心库
        segments/<源文件名去扩展>/
            segments.json
            transcript.json
            audio/            # 人声分离三轨
                original.wav / vocals.wav / instrumental.wav / separation.json
            speakers/<speaker>/  # 按人归档的 1~30s 语音（wav+json+global）
            shot_0001/
                seg_0001.mp4
                scene_reference.jpg   # 3x3 宫格故事图
                seg_0001.srt          # 该片段字幕（时间相对片段起点）
                original.wav          # 该片段原始音轨
                frames/frame_00001.png ...   # 2fps 采样帧（每格取清晰帧，≤1080P）
                video_prompt_zh.txt   # 中文 H3 描述 prompt
                detailed/             # 其余详细信息
                    asr.json / video_prompt.json / vocals.wav / instrumental.wav
                    debug/            # --debug 中间产物
            shot_0002/ ...
    skill/                    # H3 prompt 写作规范（SKILL.md / ref-en.txt / base-en.txt）
    demo/                     # 参考项目（只读）
```

> 验收脚本不写项目根目录，改用系统临时目录 `<temp>/ref_forge_<name>/`，运行后可直接查看或删除。

### 1.2 数据流（实际执行顺序）

```text
input/*.mp4（多镜头长视频）
  -> --interactive（双击 启动.bat 默认）：一次性收集 分割时长 / 是否分割 /
       输入图像 / 描述片段；预枚举片段仅做分割，结果 precomputed 复用，
       按 (源视频, shot_id) 过滤 describe_jobs
  -> segmentation.detect_shot_cuts          # PySceneDetect AdaptiveDetector（尺寸归一化流）
  -> segmentation.build_segments            # 长镜头再分割 + 短镜头合并 + 迭代校验
  -> segment_export.export_segment          # ffmpeg copy/reencode -> shot_XXXX/seg_XXXX.mp4
  -> 写 <源目录>/segments.json
  -> 每个片段（worker 池并行）：
       video_io.read_metadata(seg) / decode_gray_frames
       -> quality.assess_frames -> trim.find_valid_range
       -> sampling.sample_frames
       -> features.extract/match -> motion.estimate_pair/accumulate_base
            （只算尺度轨迹，仅供 --debug 诊断；不做分类/主体/光流）
       -> reference.generator.generate（3x3 宫格故事图，均匀时间就近取清晰帧）
       -> shot_XXXX/scene_reference.jpg（不再输出 motion_meta.json / 首尾帧）
  -> 人声分离阶段（对所有源视频，模型只加载一次）：
       speech.separation.extract_original_audio（ffmpeg 原始音轨 original.wav）
       -> audio-separator 分离 -> vocals.wav / instrumental.wav + separation.json
       -> 片段级：shot_XXXX/original.wav + shot_XXXX/detailed/{vocals,instrumental}.wav
       -> 返回 {video: vocals}，回填为 ASR job 的 audio_source
  -> 语音识别阶段（对所有源视频，模型只加载一次）：
       转写/声纹/归档：speech.audio.extract_audio(优先 vocals.wav，回退原始音轨 -> 16kHz 单声道)
       VAD 分句：speech.audio.extract_audio(始终用源视频原始音轨 -> 16kHz 单声道)
       -> diarize.detect_speech(fsmn-vad，输入原始音轨) -> cut_clip(人声轨)
       -> transcribe.transcribe_clips(Qwen3-ASR + ForcedAligner) -> 只保留有文本的语音段
       -> diarize.make_windows -> extract_embedding(CAM++ / ERes2NetV2) -> cluster_embeddings
       -> voiceprint.match_or_create -> 全局 speaker_N
       -> 写 <输入目录>/<源名>.srt、transcript.json、speakers/<speaker>/、
          各 shot_XXXX/detailed/asr.json、output/voiceprint_center.json
       -> shutdown() 卸载模型并释放显存
  -> 视频描述阶段（对所有片段，llama-server 只加载一次）：
       按 describe.input_mode 准备图像：storyboard（默认）= 直接用 3x3 故事图
       scene_reference.jpg；frames = ffmpeg 2fps 抽帧（每格取最清晰帧）到
       shot_XXXX/frames/frame_XXXXX.png
       -> 输入「图像 + 片段时长 + 本片段字幕 detailed/asr.json」
       -> llama.cpp（llm_model.gguf + mmproj_model.gguf，-ngl 999 -fa on --jinja --reasoning off）
          按 H3 full-reference 六段式直接生成中文 prompt
       -> 写 shot_XXXX/video_prompt_zh.txt + shot_XXXX/detailed/video_prompt.json
       -> 结束关闭 llama-server
  -> --clean-segments：待 ASR / 描述读取完毕后删除 seg_XXXX.mp4
```

### 1.3 关键接口（实际签名）

```python
# pipeline/segmentation.py
scenedetect_available() -> bool
sensitivity_threshold(cfg, sensitivity) -> float
detect_shot_cuts(path, fps, sensitivity, min_scene_len_sec, cfg) -> (cuts_frames | None, meta)
make_subdetector(path, sensitivity, min_scene_len_sec, cfg) -> Callable | None
build_segments(total_frames, fps, cuts, cfg, subdetect_fn=None) -> (list[Segment], meta)
# Segment: index,start_frame,end_frame,start_sec,end_sec,duration_sec,method,
#          below_min,shot_id,file,shot_dir
# meta: detected_shot_count / segment_count / split_iterations / fallback

# pipeline/segment_export.py
probe_duration(path) -> float
export_segment(src_path, start_sec, end_sec, dest_path, encode_mode, cfg) -> str | None

# pipeline/video_io.py
read_metadata(path) -> VideoMeta(fps,width,height,frame_count,duration_sec)
decode_gray_frames(path, long_side=960, start_frame=0, end_frame=None) -> list[np.ndarray]
read_color_frame(path, frame_index, max_side=None) -> np.ndarray
make_frame_reader(path, max_side=None, frame_offset=0, frame_size=None) -> Callable[[int], np.ndarray]

# pipeline/quality.py: assess_frames(gray_frames, cfg) -> FrameQuality
# pipeline/trim.py:    find_valid_range(quality, cfg) -> (start, end, status) / invalid_reason(quality)
# pipeline/sampling.py: choose_k(duration, cfg) / sample_frames(start, end, duration, cfg) -> list[int]
# pipeline/features.py: extract_features(gray, cfg) -> Features / match_pair(fi, fj, cfg) -> (pts_i,pts_j,count)|None

# pipeline/motion.py（仅保留尺度轨迹累积，仅供 --debug 诊断）
estimate_pair(pts_i, pts_j, image_shape, sharpness, cfg) -> PairEvidence
accumulate_base(pairs, sample_indices, image_shape, cfg,
                sharpness_median, low_sharpness_ratio, duration_sec) -> AggregateEvidence
    # AggregateEvidence 关键字段：sample_indices / cum_scale（故事图已不再依赖）
# reference/generator.py
reference_indices(frame_quality, valid_start, valid_end, fps, cfg) -> list[int]   # 均匀时间选帧（时间升序）
generate(frame_quality, frame_reader, fallback_frames, valid_start, valid_end, fps, cfg) -> np.ndarray | None   # 3x3 宫格

# speech/（人声分离 + 语音识别）
separation_available(cfg) -> (bool, list[str])   # 分离模型 / audio-separator 是否就绪
run_separation_phase(jobs, cfg, logger, stats=None) -> {video: info}
    # 加载分离模型一次，逐视频：抽原始音轨 -> 分离 -> audio/{original,vocals,instrumental}.wav + separation.json
available(cfg) -> (bool, list[str])      # 模型目录 / python 包是否就绪（find_spec，不触发重导入）
run_phase(jobs, output_root, cfg, logger, debug=False) -> dict
    # 加载模型一次，逐视频处理，结束 shutdown()；jobs: [{video, source_dir, shots:[...]}]
shutdown()                               # 卸载模型并释放显存
# speech/separation.py: TIERS / resolve_device(cfg) / model_path(cfg, tier) / extract_original_audio / cut_wav
# speech/audio.py: extract_audio / has_audio_stream / read_wav_mono / write_wav / cut_clip / wav_levels
# speech/models.py: configure_language(cfg); resolve_device(cfg); torch_dtype(cfg, use_cuda); class SpeechModels
#     SpeechModels: vad() / speaker_embedder() / verify_embedder() / asr() / unload()
# speech/diarize.py: detect_speech(vad, wav) / extract_embedding(model, clip, sr)
#     cluster_embeddings(embs, threshold) / make_windows(regions, win, hop) / smooth_labels(labels, radius)
# speech/transcribe.py: transcribe_clips(asr_model, clips, language, sample_rate)
#     / split_sentences_with_words(text, words) / token_count(text) / join_text(left, right)
# speech/voiceprint.py: class VoiceprintCenter(cfg, output_root)
#     match_or_create(cam_emb, eres_emb, video_name) -> speaker_id
# speech/export.py: write_srt(segments, out_path) / safe_speaker_name(id) / segment_payload(...)

# describe/（视频描述：llama.cpp + Qwen3.5 多模态）
available(cfg) -> (bool, list[str])          # llama-server / 模型是否就绪
run_phase(jobs, output_root, cfg, logger, debug=False) -> dict
    # 加载模型一次，按 input_mode 准备图像（3x3 故事图 / 采样帧）-> 组装 prompt -> 生成中文 -> 落盘；结束关闭服务
    # jobs: [{video, source_file, source_dir, shots:[{shot_id, shot_dir, file, start_sec, end_sec}]}]
# describe/frames.py: fit_long_side(w,h,max_side) / extract_frames(video,out,fps,max_side,...) / probe_video
# describe/prompt.py: normalize_mode() / system_prompt(mode) / reference_mapping_text() / build_user_prompt(mode,...)
#     / clean_output() / missing_sections() / SECTION_HEADINGS
# describe/client.py: class LlamaServer(start/chat/close) / model_path / mmproj_path

# app/interactive.py（--interactive 交互配置）
class Choices: min_sec / max_sec / split / input_mode /
               describe_all / describe_none / describe_shots / precomputed
collect_choices(videos, cfg, logger, enumerate_fn) -> Choices   # 交互收集并写回 cfg
# main._enumerate_source(path, cfg, logger) -> {meta, segments, build_meta}  # 仅分割
# main.process_source(..., precomputed=None) 复用交互枚举结果，不再重复分割
```

### 1.4 环境说明（实际）

- 包内 Python 版本：**3.10.11**（python.org embeddable）
- 基础依赖：OpenCV **4.10.0.84**（`opencv-python-headless`）、numpy **2.2.6**、Pillow **12.3.0**、PySceneDetect **0.6.4**（`--no-deps` 安装，复用已有 opencv/numpy）、click **8.5.0**、platformdirs **4.11.9**
- ffmpeg / ffprobe：`bin/`（全功能构建，含 libx264）
- 打包方式：embeddable 运行时 + get-pip（未用 micromamba / conda-pack）
- 语音识别：**Qwen3-ASR-1.7B**（转写，标点原生）+ **Qwen3-ForcedAligner-0.6B**（词级时间戳）+ **FunASR**（fsmn-vad 分句 / CAM++ 嵌入与聚类 / ERes2NetV2 声纹验证）。依赖全部装入包内 `python/`：`torch==2.9.1+cu128`、`torchaudio==2.9.1+cu128`、`funasr==1.4.15`、`qwen-asr==0.0.6`、`modelscope==1.40.1`、`transformers==4.57.6`、`librosa==0.11.0`、`soundfile==0.14.0`；模型在 `models/asr/`（Qwen3-ASR ≈4.4GB + ForcedAligner ≈1.7GB + 三个声纹/VAD 小模型）。默认 GPU（`cuda:0`，`bfloat16`），`asr.device=cpu` 时可切 CPU（`float32`）。本机 GPU：RTX 3080 Laptop 16GB（compute 8.6）。
- 人声分离：**audio-separator 0.47.0** + **onnxruntime-gpu 1.23.2**（MDX 走 GPU/CPU）+ torchvision 0.24.1+cu128（`onnx2torch` 依赖）。依赖装入包内 `python/`：`beartype 0.18.5` / `diffq-fixed 0.2.4` / `einops 0.8.2` / `julius 0.2.8` / `ml_collections 1.1.0` / `resampy 0.4.3` / `rotary-embedding-torch 0.6.5` / `samplerate 0.1.0` / `absl-py 2.5.0` / `ml_dtypes 0.6.0` / `onnx-weekly 1.24.0.dev20260914` / `onnx2torch-py313 1.6.0`。**全部为 torch 生态，MDX 经 onnxruntime（复用 torch 自带 CUDA/cuDNN DLL）**；默认 GPU，`separation.device=cpu` 时切 CPU。模型在 `models/separator/`（四档，约 1.8GB）。
- 视频描述：**llama.cpp**（`llama_cpp/llama_bin/llama-server.exe` + DLL）+ `llm_model.gguf` 与 `mmproj_model.gguf`（`models/llm/`）。

`python/python310._pth`：`python310.zip` / `.` / `Lib\site-packages` / `..\app` / `import site`。

---

## 2. 关键数值

> 与 `config.json` / `app/config.py` 保持一致。
> **当前实际配置块**：`frame_quality` / `trim` / `sampling` / `performance` / `matching` / `output` / `reference` / `segmentation` / `separation` / `asr` / `describe`。`classification` / `subject` / `enhancement` / `keyframe` 已随运镜识别移除。

| 参数 | 默认值 | 位置 | 说明 |
|---|---|---|---|
| `frame_quality.black_mean` / `black_std` | 15 / 10 | config.json | 黑场判定 |
| `frame_quality.white_mean` / `white_std` | 240 / 10 | config.json | 白场判定 |
| `frame_quality.low_info_std` | 8 | config.json | 低信息量判定 |
| `frame_quality.blur_ratio_of_median` | 0.4 | config.json | 模糊阈值（相对清晰度中位数） |
| `trim.trim_tolerance_frames` / `min_valid_frames` | 2 / 2 | config.json | 坏帧容忍 / 最短有效 |
| `trim.invalid_black_ratio` | 0.9 | config.json | 坏帧比例超过判 invalid |
| `sampling.k_short` / `k_normal` / `k_long` | 7 / 10 / 14 | config.json | `<1s` / `<=3s` / `>3s` 采样帧数 |
| `performance.workers` | 0 | config.json | 逐片段并行；0=自动 `min(8, CPU)`，`--workers` 覆盖 |
| `performance.opencv_threads` | 0 | config.json | OpenCV 线程数；0=默认，`--opencv-threads` 覆盖 |
| `matching.max_features` | 2000 | config.json | SIFT/ORB 特征上限 |
| `matching.ratio_test` | 0.75 | config.json | Lowe ratio |
| `matching.ransac_reproj_threshold` | 3.0 | config.json | RANSAC 阈值（像素） |
| `matching.min_match_count` / `min_inlier_count` | 8 / 8 | config.json | 失败对阈值 |
| `matching.low_inlier_ratio` | 0.3 | config.json | 弱对剔除阈值 |
| `output.jpeg_quality` | 92 | config.json | JPEG 质量 |
| `output.max_side_frame` | 1920 | config.json | 彩色帧长边上限 |
| `output.max_side_reference` | 4096 | config.json | 故事图长边上限 |
| `reference.grid_rows` / `grid_cols` | 3 / 3 | config.json | 统一九宫格 |
| `reference.grid_gap_px` / `grid_margin_px` | 10 / 10 | config.json | 单元格间距 / 画布留白 |
| `reference.select_tolerance_ms` | 50 | config.json | 选帧相对均匀目标时刻的最大时间偏差（ms） |
| `segmentation.enabled` | true | config.json | `--no-split` 覆盖为 false |
| `segmentation.sensitivity` | medium | config.json | low/medium/high |
| `segmentation.adaptive_thresholds` | 4.5/3.0/2.0 | config.json | 三档阈值 |
| `segmentation.min_scene_len_sec` | 0.5 | config.json | 送检测的最短场景 |
| `segmentation.min_segment_sec` / `max_segment_sec` | 1.0 / 15.0 | config.json | 片段时长区间 |
| `segmentation.max_split_iterations` | 5 | config.json | 迭代上限 |
| `segmentation.long_shot_subdetect` / `..._sensitivity` | true / high | config.json | 超长镜头二次检测 |
| `segmentation.segment_encode` | auto | config.json | copy/reencode/auto |
| `segmentation.copy_duration_tolerance_sec` | 0.1 | config.json | auto 判定 |
| `segmentation.keep_segments` | true | config.json | `--clean-segments` 覆盖 |
| `separation.enabled` | true | config.json | `--no-separate` 覆盖 |
| `separation.model` | fast | config.json | fast/balanced/best/dereverb |
| `separation.models` | models/separator | config.json | 分离模型目录 |
| `separation.device` | gpu | config.json | gpu/cpu（无 CUDA 自动回退） |
| `separation.sample_rate` | 44100 | config.json | 分离输出采样率 |
| 档位→模型 | — | `app/speech/separation.py:TIERS` | fast=UVR-MDX-NET-Voc_FT.onnx / balanced=bs_roformer_ep_317 / best=mel_band_roformer_ep_3005 / dereverb=Reverb_HQ_By_FoxJoy.onnx |
| `asr.enabled` | true | config.json | `--no-asr` 覆盖 |
| `asr.engine` | qwen3-asr | config.json | 推理引擎标识 |
| `asr.device` | gpu | config.json | gpu(cuda:0)/cpu；无 CUDA 自动回退 |
| `asr.dtype` | bfloat16 | config.json | GPU 精度；CPU 恒 float32 |
| `asr.models` | models/asr | config.json | 模型根目录 |
| `asr.asr_model` | Qwen3-ASR-1.7B | config.json | 转写模型 |
| `asr.forced_aligner` | Qwen3-ForcedAligner-0.6B | config.json | 词级时间戳 |
| `asr.vad_model` | fsmn-vad | config.json | 语音活动检测 |
| `asr.speaker_model` | campplus | config.json | 本地说话人聚类 |
| `asr.verify_model` | eres2netv2 | config.json | 声纹验证 |
| `asr.language` | auto | config.json | auto/Chinese/English/Japanese |
| `asr.max_new_tokens` | 2048 | config.json | 单段生成上限 |
| `asr.max_inference_batch_size` | 4 | config.json | 转写批大小 |
| `asr.vad_max_segment_sec` | 60 | config.json | VAD 单段最长 |
| `asr.min_speech_sec` | 0.6 | config.json | 过短语音过滤 |
| `asr.min_archive_sec` / `max_archive_sec` | 1.0 / 30.0 | config.json | 按人归档区间 |
| `asr.local_speaker_threshold` | 0.55 | config.json | 本地 CAM++ 窗口聚类阈值 |
| `asr.diarize_window_sec` / `diarize_hop_sec` | 1.2 / 0.4 | config.json | 说话人区分滑窗 |
| `asr.diarize_smooth_radius` | 1 | config.json | 窗口标签多数滤波 |
| `asr.sentence_merge_gap_sec` | 0.8 | config.json | 相邻同人句合并间隔 |
| `asr.voiceprint_sim_threshold` | 0.30 | config.json | 全局 CAM++ 一致性阈值 |
| `asr.verify_sim_threshold` | 0.55 | config.json | ERes2NetV2 主匹配阈值 |
| `asr.voiceprint_center` | "" | config.json | 空 = `<output>/voiceprint_center.json` |
| `asr.sample_rate` | 16000 | config.json | 抽音采样率 |
| `asr.debug` | false | config.json | 保留位 |
| `describe.enabled` | true | config.json | `--no-describe` 覆盖 |
| `describe.engine` | llama.cpp | config.json | 推理引擎 |
| `describe.bin` / `server` | llama_cpp/llama_bin / llama-server.exe | config.json | 可执行文件目录/名 |
| `describe.models` | models/llm | config.json | 模型目录 |
| `describe.model` / `mmproj` | llm_model.gguf / mmproj_model.gguf | config.json | 主模型 / 多模态投影 |
| `describe.host` / `port` | 127.0.0.1 / 8080 | config.json | 被占用时自动向后找端口 |
| `describe.context` / `gpu_layers` | 50000 / 999 | config.json | `-c` / `-ngl` |
| `describe.flash_attn` / `jinja` / `reasoning` | true / true / off | config.json | `-fa on` / `--jinja` / `--reasoning off` |
| `describe.enable_thinking` | false | config.json | 附带 `chat_template_kwargs.enable_thinking=false` |
| `describe.threads` | 0 | config.json | `-t`；0=llama.cpp 默认 |
| `describe.frame_fps` | 2.0 | config.json | 采样帧率 |
| `describe.frame_max_side` | 1920 | config.json | 1080P 画框长边上限（短边按 9/16） |
| `describe.frame_dir` / `frame_format` | frames / png | config.json | 采样帧目录与格式 |
| `describe.frame_select` | true | config.json | 每个时间格取最清晰帧 |
| `describe.frame_candidates` | 3 | config.json | 候选倍数（`fps×该值` 抽候选） |
| `describe.max_frames` / `keep_frames` | 0 / true | config.json | 单片段帧数上限 / 是否保留 |
| `describe.temperature` / `top_p` / `max_tokens` | 0.2 / 0.9 / 4096 | config.json | 采样参数 |
| `describe.repeat_penalty` | 1.1 | config.json | 重复惩罚（抑制逐帧枚举退化） |
| `describe.load_timeout_sec` / `request_timeout_sec` | 900 / 1800 | config.json | 加载 / 单次推理超时 |
| `describe.limit` | 0 | config.json | 仅处理前 N 个片段；`--describe-limit` 覆盖 |

补充常量：

| 常量 | 值 | 位置 | 说明 |
|---|---|---|---|
| 平移定义 | 画面中心点位移 | `pipeline/motion.py:center_displacement` | 避免缩放/旋转污染平移 |
| 有效相似变换尺度 | `[0.5, 2.0]` | `pipeline/motion.py:estimate_pair` | 越界视为退化对 |
| 尺度离群剔除 | `abs(log(scale/median)) >= 0.25` | `pipeline/motion.py:accumulate_base` | 鲁棒累积 |
| 方向变化累加门槛 | 单步位移 `> 0.02*diag` | `pipeline/motion.py:accumulate_base` | 避免噪声放大 |
| 等长切分公式 | `n = ceil(length / max_f)` | `pipeline/segmentation.py:_split_over_max` | 长镜头兜底 |
| 采样分档 | `<1s`=7 / `<=3s`=10 / `>3s`=14 | `pipeline/sampling.py` | K 值 |

---

## 3. 问题与解决

> 格式：**现象 / 原因 / 解决办法 / 影响范围 / 是否已回改文档**。
> 仅保留与**当前实现**相关的条目；运镜识别相关的旧条目见 3.4（历史）。

### 3.1 环境与打包

#### [P1] 包内运行时打包方式
- 现象：需求曾要求 micromamba + conda-pack；开发机无 micromamba/conda。
- 原因：工具链不可用且体积大。
- 解决办法：改用 python.org embeddable + get-pip；`._pth` 开启 site 并加入 `..\app`。
- 影响：`python/`、`启动.bat`、`dev.bat`。已同步：是。

#### [P8] PySceneDetect 安装会拉入 GUI 版 OpenCV
- 现象：`pip install scenedetect` 依赖 `opencv-python`，与 `opencv-python-headless` 冲突。
- 原因：两发行版都提供 `cv2` 模块。
- 解决办法：`pip install --no-deps scenedetect==0.6.4`，复用已有 opencv/numpy；实测 `detect` / `SceneManager` 正常。
- 影响：`python/Lib/site-packages`。已同步：是。

#### [P23] PowerShell `Set-Content` 破坏 UTF-8 文件（事故与恢复）
- 现象：用 `Get-Content -Raw | ... | Set-Content -Encoding UTF8` 批量改阈值后，`readme.md` / `config.json` / `app/config.py` 全部变成乱码（UTF-8 被按 CP936 读入，且不可逆）。
- 原因：PS 5.1 `Get-Content` 默认按 ANSI(CP936) 解码；`Set-Content` 又加 BOM。
- 解决办法：改用编辑工具改文件；从 git 取回后再重新应用改动。**禁止用 PowerShell 文本 cmdlet 读写含中文的 UTF-8 文件。**
- 影响：文档与配置文件。已同步：不适用。

### 3.2 分割与故事图

#### [P2] 缩放/旋转污染“平移”量
- 现象：纯中心推镜的平移量被高估。
- 原因：相似变换把“绕中心缩放”表示为“缩放+平移”，平移项 `c(1-s)` 非内容平移。
- 解决办法：平移取画面中心点位移 `A·c - c`（`center_displacement`）。
- 影响：`pipeline/motion.py`。已同步：是。

#### [P3] 退化相似变换污染尺度累积
- 现象：少量错配点上 `estimateAffinePartial2D` 给出病态解。
- 解决办法：`inlier_count < min_inlier_count`、尺度不在 `[0.5,2.0]`、非有限值记为失败对。
- 影响：`pipeline/motion.py:estimate_pair`。已同步：是。

#### [P9] 合成噪声素材导致 medium 灵敏度漏检切点
- 现象：S1（3 个镜头）在 medium 下只检出 1~2 个切点。
- 原因：均匀高频噪声使帧间内容差本就很大，硬切峰值不显著。
- 解决办法：合成测试视频每个镜头加不同色调（`_tint`），切点变显著；真实视频用默认 medium。
- 影响：`scripts/make_test_videos.py`。已同步：说明性。

#### [P10] ffmpeg `-c copy` 因关键帧对齐导致片段时长偏差
- 现象：请求 4.0s，copy 得到 4.12s。
- 原因：`-ss` 快进到最近关键帧。
- 解决办法：`auto` 模式用 ffprobe 校验时长，偏差 `> copy_duration_tolerance_sec(0.1s)` 则改 reencode（libx264，精确 4.000s）。
- 影响：`pipeline/segment_export.py`。已同步：是。

#### [P11] `--no-split` 时产物写到了源目录根而非 shot_0001
- 原因：`Segment.shot_dir` 默认空串，拼接后等于源目录。
- 解决办法：无分割分支显式设置 `shot_id/shot_dir = "shot_0001"`。
- 影响：`app/main.py:_build_segments`。已同步：不适用（实现细节）。

#### [P12] 片段导出失败时的内存兜底
- 解决办法：`decode_gray_frames` / `make_frame_reader` 支持帧区间与 `frame_offset`，按源视频的 `[start_frame,end_frame)` 处理，`segment.file=""`（写 `null`）。
- 影响：`pipeline/video_io.py`、`app/main.py`。已同步：是。

#### [P19] 随机 seek 取帧不准
- 现象：`read_color_frame` 用 `cap.set(CAP_PROP_POS_FRAMES)` 随机定位，H.264 下会返回错误帧。
- 解决办法：`make_frame_reader` 持有一个 `VideoCapture` **顺序解码**并按目标帧缓存；故事图调用前按帧号升序预读去重。
- 影响：`pipeline/video_io.py`、`reference/generator.py`、`app/main.py`。已同步：是。

#### [P21] 移除长图拼接，统一 2x3 六宫格
- 现象：内置全景拼接与 video2panorama 在真实素材（运动主体/视差）上常出现重影、拉伸涂抹或人物重复。
- 解决办法：彻底移除全景/长图策略，所有片段统一 `2x3` 六宫格（6 帧、间距 10px、留白 10px），按尺度变化选帧；删除 `reference/panorama.py` 等与相关依赖。
- 影响：`reference/*`、`config.json`、`requirements.txt`。已同步：是。

#### [P24] 六宫格未按时间先后排列
- 现象：多张 `scene_reference.jpg` 的 6 格不是时间顺序。
- 原因：按 `log(scale)` 进度取“最接近”的采样帧，当累积尺度非单调时会前后跳跃。
- 解决办法：`generator._scale_progress_frames` 对选出的帧号**排序去重（升序）**，不足 6 个用时间均匀帧补齐并再次排序。
- 影响：`reference/generator.py`。已同步：是。

#### [P34] 分辨率切换视频导致 PySceneDetect 刷屏报错并漏检
- 现象：`input/video_cn.mp4` 在同一文件内切换分辨率，`run.log` 刷 `incorrect size`，该区间漏检。
- 根因：OpenCV 按各段实际尺寸返回帧；PySceneDetect 用首帧尺寸做一致性校验，尺寸变化即视为损坏并跳过。
- 修复：
  1. `segmentation.py` 新增 `_NormalizingStream`，把 `frame_size` 之外的帧重采样回声明分辨率；`detect_shot_cuts` / `make_subdetector` 改用 `SceneManager` + `AdaptiveDetector` 配合该包装流。
  2. `video_io.py`：`decode_gray_frames` 统一到首帧尺寸；`make_frame_reader` 新增 `frame_size` 参数，把彩色帧统一到声明分辨率（`main.py` 传 `segment_meta.width/height`）。
- 影响：`pipeline/segmentation.py`、`pipeline/video_io.py`、`app/main.py`。已同步：是。

### 3.3 音频 / ASR / 视频描述

#### [P29] 语音识别：Qwen3-ASR + FunASR 声纹
- 背景：要求 Qwen3-ASR-1.7B（转写，标点原生）+ Qwen3-ForcedAligner-0.6B（词级时间戳）+ FunASR（fsmn-vad + CAM++）+ ERes2NetV2（声纹验证），支持中/英/日，GPU/CPU 可切换，不使用 ct-punc，且满足“独立一键包”。
- 环境：依赖装进包内 `python/`（见 1.4）；模型放 `models/asr/`。
- 代码：新增 `app/speech/` 包；**重依赖全部惰性导入**（`import speech` 不加载 torch）。
- API：`Qwen3ASRModel.from_pretrained(..., forced_aligner=..., forced_aligner_kwargs=...)`；**forced_aligner 必须在初始化时传入**；`audio` 只接受 `str` 或 `(np.ndarray, sr)`。
- 关键坑 1：ERes2NetV2 本地目录无 `config.yaml`，FunASR `AutoModel(model=<dir>)` 报 “not registered”。解决：显式 `AutoModel(model="ERes2NetV2", model_conf={}, model_path=<dir>, frontend="WavFrontend", frontend_conf={"fs":16000})`。
- 关键坑 2：VAD 会在纯音乐/滚动文档上产生 ~0.5s 假语音并被 Qwen 幻觉成 `嗯嗯`，污染声纹中心。解决：**先转写再聚类**，只保留 `text` 非空的语音段，并用 `asr.min_speech_sec=0.6` 过滤过短区间。
- 验证：`run_asr_test.py`（真实 6 条：5 条有语音、`Screenrecording.mp4` 无语音，说话人跨 4 条视频合并为同一 ID）。已同步：是。

#### [P30] 修复同一镜头内两人对白被识别为一人
- 现象：同一 VAD 段内紧邻的两人对白被识别为同一说话人。
- 根因：fsmn-vad 把两人紧邻无停顿的对白并入一个 VAD 段，而初版把「一个 VAD 段」当作「一个说话人」。
- 修复：
  1. `diarize.make_windows`：语音区间上重叠滑窗（`diarize_window_sec=1.2` / `diarize_hop_sec=0.4`），窗口级 CAM++ 聚类（`local_speaker_threshold=0.55`）+ `smooth_labels` 多数滤波。
  2. `transcribe.split_sentences_with_words`：按句末标点分句并映射词级时间戳；`_speaker_for_span` 用与句时间重叠的窗口投票决定说话人；`sentence_merge_gap_sec` 合并相邻同人句。
  3. 本地簇代表向量用**该簇全部窗口拼接音频**再提一次声纹；全局声纹中心**只登记真正被句子使用的簇**。
  4. 短窗下 ERes2NetV2 比 CAM++ 更稳：全局匹配改为 **ERes 主判据**（`verify_sim_threshold=0.55`）+ CAM++ 一致性校验（`voiceprint_sim_threshold=0.30`）。
- 已知局限：跨视频同一真人声纹仍可能被拆成两个 ID（<2s 语音区分度有限），但不再出现不同人共用同一 ID。
- 影响：`app/speech/diarize.py`、`transcribe.py`、`voiceprint.py`、`runner.py`、配置。已同步：是。

#### [P32] 人声/背景音乐分离，ASR 与归档改用人声轨
- 选型：`audio-separator==0.47.0`。
  - 关键坑 1：其 `pip install` 会把 torch 升级（PyPI 默认 CPU 版）→ 必须 `--no-deps`，再手动装依赖并固定版本。
  - 关键坑 2：`torchvision` 是 `onnx2torch` 硬依赖；须用 `torchvision==0.24.1+cu128`（`--no-deps`）。
  - 关键坑 3：0.47.0 的 `Separator.__init__` 已无 `use_cuda`，强制 CPU 需构造后覆写 `torch_device` / `torch_device_cpu` / `onnx_execution_provider`。
  - 关键坑 4：`download_model_files` 依赖 GitHub 上的元数据文件，须预置到模型目录方可离线。
  - 关键坑 5（时间轴漂移）：分离后的人声轨残留背景音乐，`fsmn-vad` 检测不到静音边界，把整段并成一个 region，ForcedAligner 词级时间戳严重漂移。解决：**VAD 分句用原始音轨**，转写/声纹/归档用人声轨。
- 实现：`app/speech/separation.py`（模型加载一次，四档可配，本地离线）；`main.py` 在 ASR 前新增分离阶段；片段级 `original.wav`（根）+ `detailed/{vocals,instrumental}.wav`。
- 验证：`run_separation_test.py`（best+gpu）通过；四档 GPU 与 fast CPU 实测可用；`run_asr_test.py` 通过且日志显示 `source=vocals`、`separation_ok=6`。已同步：是。

#### [P33] 片段级字幕归属 + 片段三轨音频 + 片段 SRT
- 现象：跨切点的语音段同时出现在相邻两个镜头的 `asr.json`。
- 解决办法（`app/speech/runner.py`）：
  1. `_assign_segments_to_shots`：每个语音段按“与镜头**时间重叠最多**”唯一归属到一个镜头。
  2. 每段输出 `shot_XXXX/seg_XXXX.srt`（与片段同名）：内容与 `asr.json` 一致，时间相对片段起点并裁剪到片段范围；无语音写空 SRT。
  3. 分离阶段 `cut_wav` 按片段区间从整视频三轨无损切出片段音频。
- 验证：`C4_pan_C2_push_C6_crane` 的 shot_0001/0002 各只含本片内容；`run_asr_test.py` 已加校验并通过。已同步：是。

#### [P36] 逐片段视频描述（llama.cpp + Qwen3.5，H3 full-reference）
- 实现：新增 `app/describe/`（`frames` / `prompt` / `client` / `runner` + `prompts/`）；`main.py` 在**语音识别之后、`--clean-segments` 删除片段之前**新增独立阶段（`--no-describe` 关闭，`--describe-limit N` 调试）。
  1. **抽帧**：以 `fps × frame_candidates`（默认 2×3=6fps）抽候选帧到 `frames/_candidates/`，用 `cv2.Laplacian` 方差在每个时间格内取最清晰的一帧，输出 `frame_%05d.png`；候选目录用后删除；按「1080P 画框」算好尺寸再传 `scale`。
  2. **prompt**：系统/用户模板放 `app/describe/prompts/*.txt`（中文说明），运行期替换占位符。
  3. **推理**：常驻 `llama-server.exe -m llm_model.gguf --mmproj mmproj_model.gguf -ngl 999 -c 50000 -fa on -np 1 --jinja --reasoning off`，`/health` 轮询就绪，`/v1/chat/completions` 发多模态消息；仅用标准库 urllib；端口占用自动顺延。
  4. **落盘**：`video_prompt_zh.txt`（中文六段式）/ `detailed/video_prompt.json`。
- 关键坑（逐帧枚举退化）：首版模型把采样帧当参考图，`subject_definitions` 编到 `<Subject 141>`、token 耗尽。修复：系统提示明确“帧不是参考资产”、硬约束最多 6 行 subject / `<Picture N>` N≤6、给出六段骨架、`repeat_penalty=1.1`、`enable_thinking=false`。
- 第二轮优化：移除翻译、改为**直接输出中文**；抽帧 3fps→2fps 且每格取清晰帧；`<Picture 1>` 语义改为**2x3 宫格故事图**；`(Sx)` 片段内按首次发声重编号。
- 第三轮优化：片段可能含多个镜头，要求依据采样帧实际硬切判断镜头数；明确**不得让视频生成模型生成字幕/画面文字**。
- 已知小偏差：9B 对「`[Shot 1]` 不带时间戳」执行不稳定，可能写出 `00:00.000`；六段结构、参考标签、说话人与对白格式正确。
- 验证：`run_describe_test.py` 默认校验抽帧与六段模板；`--llm` 端到端通过。已同步：是。

#### [P37] 片段时长下限与归档下限调整
- `segmentation.min_segment_sec` 由 `2.0` → `5.0` → 最终 **`3.0`**（随 P38 落地）。
- 按人归档语音由 `3~30s` 改为 **`1~30s`**（`asr.min_archive_sec=1.0`）。
- 验收脚本统一显式传 `--min-seg-sec 2.0`（S1–S4 等合成/短素材按 2s 片段设计），产品默认 3.0。已同步：是。

#### [P38] 移除运镜识别、精简配置、整理 shot 目录（当前形态）
- 需求：大模型重新生成视频不需要运镜内容；移除运镜识别、不再输出 `motion_meta.json`，但 `scene_reference.jpg` 必须与移除前**逐字节一致**；整理 shot 目录；片段下限 5s→3s。
- 代码改动：
  1. **删除** `app/classifiers/`、`pipeline/keyframes.py`、`pipeline/subject.py`、`pipeline/geometry.py`；`pipeline/motion.py` 精简为 `estimate_pair` + `accumulate_base`（保留 `cum_scale`）。
  2. **`main.py` 重写**：每片段只做「解码 → 质量 → 修剪 → 采样 → 匹配 → `accumulate_base` → `reference.generator.generate`」；不再有首尾帧 / `motion_meta.json` / 分类 / 两池流水。并行改为单个 worker 池 `map`。invalid 片段不写元数据、跳过描述。
  3. `reference/generator.py` 只依赖 `agg.sample_indices` / `agg.cum_scale`，选帧逻辑不变。
  4. `io_utils.py` 新增 `SHOT_DETAIL_DIR="detailed"` 与 `shot_detail_dir`。
  5. `speech/runner.py`：`asr.json` → `detailed/asr.json`；`seg_XXXX.srt` 仍在 shot 根。
  6. `speech/separation.py`：`original.wav` 放 shot 根，`vocals/instrumental` 放 `detailed/`。
  7. `describe/runner.py`：字幕读 `detailed/asr.json`；`video_prompt.json` 写 `detailed/`；`frames/` 与 `video_prompt_zh.txt` 在 shot 根。
  8. 配置删除 `classification` / `subject` / `enhancement` / `keyframe` 块与 `output.overlay_text` / `output.font_path`。
  9. 测试脚本删除运镜回归 `run_regression.py` / `run_combined_test.py` / `run_real_test.py`。
- 一致性验证：同一 `--min-seg-sec 5.0` 下 `shot_0001/0002` 的 `scene_reference.jpg` SHA-256 与移除前完全相同。
- 验证：`run_segmentation_test.py` 全通过；`run_describe_test.py` 通过、`--llm` 通过；`compileall` 通过、config 与 `DEFAULT_CONFIG` 一致。已同步：是。

#### [P39] 视频描述参考标签只保留 `<Subject N>` 与 `<Audio N>`（已被 P42 取代）
- 需求：移除 `<Picture N>`（以及仅作为来源的 `<Video N>`），视频描述 prompt 的参考资产只使用 `<Subject N>` 与 `<Audio N>`。
- 映射：`<Subject 1>` = 本片段 3x3 宫格故事图（storyboard，定义镜头顺序 / 视角 / 主体位置 / 关键情节）；第 n 个主要人物 = `<Subject n+1>`（外貌来自第 n 张人物参考图）；`<Audio n>` = 对应人物的声音音色参考；`(Sx)` 仍在片段内按首次发声重编号。
- 改动：
  1. `app/describe/prompts/ref2va_system.txt`：参考资产映射、硬性约束、输出骨架全部改为 Subject/Audio；明确禁止 `<Picture N>` / `<Video N>`。
  2. `app/describe/prompts/ref2va_user.txt`：补充“参考资产只使用 `<Subject N>` 与 `<Audio N>`”约束。
  3. `app/describe/prompt.py:reference_mapping_text()`：`<Subject 1>`=故事图，`<Subject n+1>`/`<Audio n>`=第 n 个人物。
  4. `scripts/run_describe_test.py`：references 校验 token 改为 `<Subject 1>` / `<Subject 2>` / `<Audio 1>` / 故事图，并断言不含 `<Picture` / `<Video`。
  5. `skill/ref-en.txt`：标签表只保留 Subject/Audio；原 `<Picture N>`（帧锚点/故事板）与 `<Video N>`（整片结构）合并进 `<Subject N>` 说明；示例同步。
  6. `skill/base-en.txt`：首/尾帧指令把 `<Picture N>` 改为 `<Subject N>`（参考图即 `<Subject N>`）。
  7. `skill/SKILL.md`：新增“Reference Labels”约定（仅 Subject/Audio）。
  8. `AGENTS.md` / `readme.md` / `REQUIREMENTS.md`：参考标签映射同步。
- 验证：`run_describe_test.py`（新增 references 标签校验）与 `--llm` 端到端；`compileall` 通过。已同步：是。

#### [P40] 故事图改为 3x3 九宫格 + 均匀时间就近取清晰帧；片段下限降到 1s
- 需求：故事图由 `2x3` 六宫格改为 `3x3` 九宫格；选帧由「按累积尺度进度」改为「**均匀时间间隔**选帧，并在每个目标时刻附近（时间偏差 ≤ `select_tolerance_ms`，默认 50ms）取**最清晰**的一帧」，避免故事图无法完整描述视频经过；片段时长下限由 `3s` 降为 `1s`。
- 代码改动：
  1. `config.json` / `app/config.py`：`reference.grid_rows=3` / `grid_cols=3`，新增 `reference.select_tolerance_ms=50`，删除 `reference.scale_progress`；`segmentation.min_segment_sec` 3.0→1.0。
  2. `reference/generator.py` 重写：`reference_indices(frame_quality, valid_start, valid_end, fps, cfg)` 在 `[valid_start, valid_end]` 上 `np.linspace` 取 9 个目标帧号，逐点在 `±select_tolerance_ms` 帧内优先取未判坏帧、其次 `sharpness` 最大者；结果非递减（时间升序）。`generate(frame_quality, frame_reader, fallback_frames, valid_start, valid_end, fps, cfg)` 签名同步。
  3. `app/main.py`：`_finalize_segment` 预读与 `generate` 改为传 `frame_quality / valid_start / valid_end / fps`。
  4. `pipeline/motion.py`：故事图不再依赖 `cum_scale`，模块仅保留供 `--debug` 诊断。
  5. prompt（`ref2va_system.txt` / `ref2va_user.txt` / `prompt.py`）：`<Subject 1>` 描述由 2x3/6 画面改为 3x3/9 画面。
  6. 文档同步：`AGENTS.md` / `REQUIREMENTS.md` / `readme.md` / `input_info.md` / `PROJECT_NOTES.md`。
- 验证：`run_segmentation_test.py` 通过；`scene_reference.jpg` 为 3x3 九宫格、9 格时间升序、每格偏差 ≤ 50ms。
- 影响：`reference/generator.py`、`app/main.py`、`config.json`、`app/config.py`、`describe/prompts/*`、文档。已同步：是。

#### [P41] 验收素材改为直接使用 `input/` 中的真实视频
- 需求：测试与验收不再依赖固定命名的合成素材，**`input/` 里有什么视频就用什么视频**验收。
- 改动：
  1. 新增 `scripts/acceptance_common.py`：`input_videos()` 动态发现 `input/` 视频，`stage_input()` 复制到临时目录（整视频 SRT 写源视频旁，避免污染），`run_pipeline()` 统一调用主程序。
  2. `run_segmentation_test.py`：删除对 `make_test_videos.make_segmentation_videos` 与 S1–S4 的依赖；对每个输入视频校验每段时长落在 `[min_segment_sec, max_segment_sec]`（`below_min` 除外）、`scene_reference.jpg` 可解码、`detailed/` 与片段视频齐全。
  3. `run_separation_test.py`：改为遍历 `input/` 中含音轨的视频（`speech.audio.has_audio_stream` 过滤），逐视频核对三轨 + 元数据。
  4. `run_asr_test.py`：删除硬编码视频名 / `NO_SPEECH` / 关键词 / 跨视频合并等素材相关断言，改为对每个输入视频做通用结构校验（SRT / `transcript.json` / 说话人格式 / `detailed/asr.json` / 片段 SRT / 三轨 / 按人归档）。
  5. `run_describe_test.py`：去掉 `PREFERRED` 视频，遍历 `input/` 全部视频做抽帧 + prompt 校验；`--llm` 仅对第一个视频端到端。
- 说明：`make_test_videos.py` 保留为可选合成素材工具，验收流程不再调用。
- 验证：`input/` 仅 `video_cn.mp4`（44.9s，1280x720，含音轨）时四个验收脚本均通过；`app/` 与 `scripts/` `compileall` 通过。已同步：是。

#### [P42] 参考标签改回 `<Picture N>` / `<Audio N>`；描述推理输入可选故事图或采样帧
- 需求：1) 视频描述 prompt 的参考标签只允许 `<Picture N>` 与 `<Audio N>`（不再使用 `<Subject N>` / `<Video N>`，即反转 P39）；2) 逐片段描述由「固定用采样帧推理」改为配置可选：`describe.input_mode` = `storyboard`（默认，直接把片段的 3x3 宫格故事图作为推理图像）或 `frames`（ffmpeg 采样帧）；两种输入含义不同，需用**不同的提示词文件**区分。
- 映射：`<Picture 1>` = 本片段 3x3 宫格故事图（storyboard）；第 n 个主要人物 = `<Picture n+1>`（外貌来自第 n 张人物参考图）；`<Audio n>` = 对应人物的声音音色参考；`(Sx)` 仍在片段内按首次发声重编号。
- 改动：
  1. `config.json` / `app/config.py`：`describe` 新增 `input_mode`（默认 `"storyboard"`）。
  2. `app/describe/prompts/`：删除 `ref2va_system.txt` / `ref2va_user.txt`，改为两套模板——`storyboard_system.txt` / `storyboard_user.txt`（随附图像即参考资产 `<Picture 1>`）与 `frames_system.txt` / `frames_user.txt`（随附图像是采样帧、不是参考资产）；两套统一只使用 `<Picture N>` / `<Audio N>`。
  3. `app/describe/prompt.py`：新增 `normalize_mode` / `SYSTEM_FILES` / `USER_FILES`；`system_prompt(mode)` 与 `build_user_prompt(mode, ...)` 按模式选模板；`reference_mapping_text()` 改回 `<Picture 1>`=故事图、`<Picture n+1>`/`<Audio n>`=第 n 个人物。
  4. `app/describe/runner.py`：按 `input_mode` 选输入——`storyboard` 直接读 `scene_reference.jpg` 不抽帧（缺失则回退 `frames` 并告警）；`frames` 走原抽帧逻辑；`detailed/video_prompt.json` 新增 `input_mode` / `storyboard` / `grid_rows` / `grid_cols`，帧字段在故事图模式下为空。
  5. `app/main.py`：引擎就绪日志增加 `input=` 模式。
  6. `scripts/run_describe_test.py`：`references` 与两套 system/user 模板校验 `<Picture 1>` / `<Picture 2>` / `<Audio 1>` / 故事图，断言不含 `<Subject N>` / `<Video N>`；`--llm` 按 `input_mode` 校验对应输入资产与 payload。
  7. 文档同步：`AGENTS.md` / `REQUIREMENTS.md` / `readme.md` / `PROJECT_NOTES.md`。
- 验证：`run_describe_test.py`（两套模板 + 故事图/采样帧 payload）通过；`compileall` 通过；config 与 `DEFAULT_CONFIG` 键一致；故事图/采样帧两种模式经 stub server 验证输入图像与 payload 正确。已同步：是。

#### [P43] `storyboard` 模式把故事图按 1080P 缓存到 `detailed/` 再推理
- 需求：用 3x3 故事图推理时，故事图原图（`output.max_side_reference=4096`，实测 4096x2317）过大导致推理变慢；需先缩放到最大 1080P 画框并缓存到 `output/segments/<源名>/shot_XXXX/detailed/` 再喂模型；原图已 ≤1080P 则直接用原图（不放大）。
- 改动：
  1. `config.json` / `app/config.py`：`describe` 新增 `storyboard_max_side`（默认 `1920`，即 1080P 画框长边）。
  2. `app/describe/runner.py`：新增 `_cache_storyboard()`——读 `scene_reference.jpg`，用 `frames.fit_long_side` 计算 1080P 画框尺寸（长边 ≤ `storyboard_max_side`、短边 ≤ 长边×9/16、偶数）；超出时按 `output.jpeg_quality` 写 `detailed/storyboard_1080p.jpg` 并返回缓存路径，未超出时返回原图；`process_shot` 用返回路径推理，`video_prompt.json` 记录 `storyboard`（实际使用路径）/ `storyboard_source` / `storyboard_cached`。
  3. `scripts/run_describe_test.py`：`--llm` 校验实际故事图路径存在，且 `storyboard_cached` 时 `detailed/storyboard_1080p.jpg` 存在。
  4. 文档同步：`AGENTS.md` / `REQUIREMENTS.md` / `readme.md` / `PROJECT_NOTES.md`。
- 验证：stub server 下 4096x2317 → 1908x1080 缓存并用于推理、`storyboard_cached=true`；1280x720 → 不缓存、直接用原图、`storyboard_cached=false`；`run_describe_test.py` 通过；`compileall` 通过；config 与 `DEFAULT_CONFIG` 键一致。已同步：是。

#### [P44] 双击启动改为交互式选择（`--interactive`）
- 需求：双击 `启动.bat` 不再直接跑全流程，改为**一次性**交互选择要执行的内容，全部选完再开始：1) 分割时长（默认 1~15s）；2) 是否执行「镜头分割 -> 故事图 -> 人声分离 -> 语音识别」（选否则原视频整体当单个片段，故事图 / 分离 / ASR 照常）；3) 描述用故事图还是抽帧（默认故事图）；4) 按编号选择要生成描述的片段（默认全部，可 none）。
- 设计：
  1. 新增 `app/interactive.py`：`Choices` + `collect_choices(videos, cfg, logger, enumerate_fn)`；非法输入重问、EOF 回落默认值；选择结果写回 `cfg`。
  2. 第四步需知道片段编号，故先**预枚举**：`main._enumerate_source` 每个源视频只做一次镜头分割（不导出、不生成故事图），结果存入 `Choices.precomputed`；`process_source(..., precomputed=...)` 直接复用，**不重复分割**。
  3. 描述过滤：选择以 `(源视频路径, shot_id)` 记录，登记完 `describe_jobs` 后按 `shot_id` 过滤；`none` 关闭 `describe`，`all` 不过滤。
  4. 触发：新增 `--interactive`；`启动.bat` 无参数时自动追加，带参数仍走原 CLI（交互参数不影响验收脚本）。
- 关键坑：`_prompt_selection` 对 `_prompt_raw` 返回的 `None`（EOF）直接 `.lower()` 会崩，已改为先判 `None` 再回落默认（全部）。
- 影响：`app/interactive.py`（新）、`app/main.py`、`启动.bat`、文档。已同步：是。
- 验证：`run_segmentation_test.py` / `run_describe_test.py` 通过；直接调用 `python app/main.py --interactive` 分别测「全部 / none / 编号 2 / 不分割」四类选择，日志确认 `describe=none` 不加载模型、选 `2` 时仅 `describe shot_0002`、不分割时 `shots=1`，且 stage timing 无 `source.segmentation`（说明枚举结果被复用）。

#### [P45] 视频人物提取与归类（人脸主键，阶段 4）
- 需求：见 `PERSON_REQUIREMENTS.md`。整片按人脸身份归类，每人一个文件夹存多张原分辨率矩形截图（`max_per_person=300`）；全程无脸 track 入 `unknown/`。
- 实现：新增 `app/persons/`（`models`/`sampler`/`detect`/`face`/`associate`/`cluster`/`crop`/`export`）；`main` 中作为**阶段 4**（统一流程之后、描述之前）登记 `persons_jobs` 并调用 `run_phase`；新增 `config.persons` 块、`--no-persons`/`--persons-device`/`--persons-max`、交互第 3 项、`persons.available` 引擎检查、`done` 统计与 `phase.persons` 计时。
- 关键决策/坑：
  1. **不装 `insightface`**：人脸检测/嵌入用 `onnxruntime` 直驱 buffalo_l 的 `det_10g.onnx`(SCRFD) + `w600k_r50.onnx`(ArcFace)，自实现 SCRFD 3 尺度解码、NMS、5 点相似变换对齐，避免引入 `albumentations`/`scikit-image` 等重依赖。
  2. **Ultralytics 兼容**：`ultralytics --no-deps` 安装，另装 `lap`（BoT-SORT 线性分配）；不触碰 `opencv-python`(headless) / `torch` / `numpy`；`scripts/verify_person_env.py` 安装后校验。
  3. onnxruntime CUDA EP 需要 cuDNN9；torch 自带 `torch/lib/cudnn64_9.dll` 但不在 PATH，`FaceEngine` 初始化时 `os.add_dll_directory(torch/lib)` 才能启用 GPU（否则刷屏报错并回退 CPU）。
  4. 1fps 下 BoT-SORT 易丢 id：对无 id 的检测按中心距离就近继承最近 track，避免同一人碎成多段；随后人脸聚类再做身份合并。
  5. SCRFD 关键点需 `reshape(-1,2)`，否则 `estimateAffinePartial2D` 断言失败（`checkVector` 不匹配）。
  6. 同帧互斥：同一代表帧出现的不同 track 禁止合并；聚类用**约束感知凝聚式合并**（非朴素并查集）避免传递性错误。
  7. **识别效果优化（实测 `video_cn.mp4` 后）**：动画人物同人帧被拆进 `unknown` 的三个原因与修复——
     a. 只在人体裁剪图上跑 SCRFD 时，部分帧检测分偏低（如 1229 帧裁剪图 0.42、整帧 0.65）；改为**裁剪图 + 整帧双路检测合并取最高分**（`_detect_raw` 增加 threshold 参数）。
     b. 去重 bug：无脸记录统一用所属 track 的均值人脸当 key，导致同一 track 的所有无脸帧互相判为 >0.92 近重复而**全部被删**；改为**双方都有本人嵌入才用余弦，无嵌入改用 32×32 缩略图近像素抑制，一有一无不比较**（`cluster.dedup_records`）。
     c. 同一 BoT-SORT track 跨硬切把不同角色串成一条（白毛角色的人脸均值被带到蒙眼女/bird 帧），且曾用 `merge_unknown_single` 把无脸 track 盲目并入唯一人物，导致**不同人物混进同一文件夹**。修复：**按 PySceneDetect 硬切把 track 强制断开**（`_scene_cuts`）+ **双阈值人脸**（`face_det_score=0.55` 高质量；`face_embed_score=0.10` 低分回退，仅当 track 无高质量人脸时用于生成特征），并**彻底撤销 `merge_unknown_single`**。
  - 优化结果：`video_cn.mp4` 现为 `person_001`=白毛角色（14 图，覆盖 44~1507 帧）、`person_002`=蒙眼女角色（4 图：862/914/974/990）、`unknown`=其余无脸（bird 帧 1042 与白毛背影 119/149/180/359/420）；蒙眼女与 bird 不再混入白毛角色。
  - 数据限制：`input/video_cn.mp4` 为 2D 动画，真实人脸模型对侧脸/风格化脸部检出仍有限；上述优化已尽量兜底，但对真实人脸为主的视频效果更佳。
- 影响：`app/persons/`（新）、`app/main.py`、`app/config.py`、`config.json`、`app/interactive.py`、`requirements.txt`、`scripts/`（`run_persons_test.py`/`download_person_models.py`/`verify_person_env.py`/既有脚本加 `--no-persons`）、`下载模型.bat`、`setup_runtime.bat`、`REQUIREMENTS.md`、`AGENTS.md`、`readme.md`。已同步：是。
- 验证：`run_persons_test.py` 全通过（产出 `person_001`（5 图 / 3 track）+ `unknown` 7 track；metadata / 命名 / 原分辨率 / 去重 / 环境校验通过）；`run_segmentation_test.py`、`run_describe_test.py` 回归通过。

#### [P46] 人物提取新增动漫模型选型（`persons.domain`，真人 / 动漫双后端）
- 需求：真人模型在 2D 动漫上系统性失效（YOLO/SCRFD/ArcFace 训练分布全是真人），需在交互与 CLI 中新增**真人 / 动漫**模型选型；动漫模型覆盖 2D 与 3D，参考 `new_requirements.md` 的修正方案。真人沿用原完整流程，配置与产物不变。
- 选型核实：文档原推荐的 `hysts/anime_instance_segmentation`、`hysts/anime_face_recognition` 在 HF 上**并不存在**；`hysts/anime-face-detector` 依赖 `mmdet/mmpose/mmcv`，与包内可移植环境冲突。最终采用**可离线、零新依赖**的 `deepghs` 动漫专用 ONNX 检测器（`onnxruntime` 直驱）：
  1. **人物框**：`deepghs/anime_person_detection` 的 `person_detect_v1.1_m`（YOLO ONNX，F1≈0.87，`det_conf=0.30`）＋**人脸锚点扩展**（`expand_face_to_person`，补救遮挡/远景漏检）；
  2. **人脸**：`deepghs/anime_face_detection` 的 `face_detect_v1.4_s`（YOLO ONNX，F1≈0.95，替代 LBP 级联）；
  3. **身份嵌入**：**CLIP 人物裁剪图嵌入**（`transformers` 的 `CLIPModel.get_image_features`，`openai/clip-vit-large-patch14`）——始终嵌入人物裁剪图，**背影/侧脸帧也参与聚类**（关键修复）；**不使用 DeepDanbooru 标签向量**（用户明确要求）；
  4. **跟踪**：track_id 置空，由 `associate_track_id` 按中心距离 + 尺寸比帧间关联（硬切处断开）；不走 `ultralytics` 的 ONNX 跟踪（其会把 `onnx-weekly` 误判为缺 `onnx` 并尝试联网 pip 安装）。
- **效果修复（首版很差的根因与改法）**：首版用「COCO YOLO 低阈值 + 锚点扩展 + OpenCV `lbpcascade_animeface` 级联 + 优先人脸区域 CLIP」实测在 2D 番剧上很差：级联在报纸插画上大量误检、YOLO 漏检、且只有检测到正脸才有嵌入 → 大量背影/侧脸帧成为 `unknown`（3 person / 4 图 / 16 unknown）。改为：① 动漫专用人物检测替代 COCO YOLO；② 动漫专用人脸检测替代 LBP 级联；③ **CLIP 始终嵌入人物裁剪图**（不再依赖人脸），同一素材由 3 person/4 图/16 unknown 改善为 **6 person/15 图/0 unknown**，同一角色正/侧/背影正确归并。
- 实现：`app/persons/models.py` 增加 `domain_of`/`effective_persons_cfg`/`AnimeModels`（`person_detector`/`face`/`make_tracker`）与动漫模型路径、`build_models`、按领域分派的 `available`；`PersonModels`/`AnimeModels` 增加 `embed_mode`（`face`/`person`）；`PersonTracker` 增加 `tracker_file` 与 `whole_faces` 参数、改用 `models.persons_params`；`process_video` 泛化（人脸检测前移、`det["level"]` 覆盖、按 `embed_mode` 选择人脸/人物裁剪图嵌入、`embed` 增加可选 `face_box`）；`cluster_tracks` 改为显式阈值入参；`export.write_outputs` 接收生效参数并在 `metadata.json` 增 `domain`；新增 `app/persons/anime/`（`onnx_detector`/`face`/`embed`/`detect`）；新增 `scripts/download_anime_person_models.py`（deepghs + CLIP 均走 hf-mirror）；交互第 4 项改为「真人 / 动漫 / 不执行」；新增 `--persons-domain real|anime`。
- 配置：`persons.domain`（默认 `anime`）＋ `persons.anime` 覆盖组（独立阈值：`cluster_threshold=0.80`、`cluster_ambiguous_low=0.72`、`dedup_threshold=0.88`、`det_conf=0.30`、`min_face_px=32`、`face_det_score=0.50`、`face_embed_score=0.30`、`head_body_ratio=7.0`、`input_size=640`、`nms_iou=0.5`、`person_onnx`/`face_onnx` 等）；`config.json` 与 `DEFAULT_CONFIG` 键必须一致。
- 边界与待校准：CLIP 嵌入是「语义相似」非「身份相同」，**同款校服误合并/同角色换装分裂**在动漫上更重，`persons.anime.cluster_threshold`（默认 0.80）**必须按片源重扫校准**；人物检测在多人紧贴/拥抱时可能一框含两人（轴对齐矩形的固有局限）；动漫人脸检测只检较正面的脸，背影帧靠人物裁剪图嵌入兜底；`head_body_ratio` 仅在锚点扩展兜底时生效（Q 版 3~4，番剧 6.5~7.5，写实 3D 7.5~8.5）。
- 影响：`app/persons/`（含新 `anime/`）、`app/main.py`、`app/config.py`、`config.json`、`app/interactive.py`、`scripts/run_persons_test.py`（新增 `--domain` 与 metadata `domain` 校验）、`scripts/download_anime_person_models.py`（新）、`下载模型.bat`、`AGENTS.md`、`readme.md`、`PERSON_REQUIREMENTS.md`、`REQUIREMENTS.md`。无新增第三方依赖（复用 `transformers`/`onnxruntime`）。
- 默认与验证：`persons.domain` 默认 **`anime`**（交互第 4 项默认第 1 项、`--persons-domain` 默认 anime、`run_persons_test.py` 默认 `--domain anime`）。本地已下载人物/人脸 ONNX + CLIP（经 hf-mirror）并跑通：`run_persons_test.py`（anime，`123.mp4` → 6 person / 15 图 / 0 unknown，`metadata.domain=anime`）与 `--domain real` 回归均通过；`run_segmentation_test.py`、`run_describe_test.py` 回归通过。未备模型时 `persons.available` 会关闭人物阶段并告警，不影响其余阶段。

#### [P47] 一键环境搭建（克隆后可自举）+ 多源测速 + 描述模型换源
- 需求：1) 为仓库补全 `.gitignore`，把体积庞大的运行环境与离线资产排除出仓库；2) 新增「一键环境搭建」`.bat`，从 GitHub 克隆后运行一次即可让 `启动.bat` 正常工作；3) 搭建过程按实测网速自动切换最快下载源；4) 视频描述主模型默认改用 ModelScope `unsloth/Qwen3.5-4B-GGUF`。
- `.gitignore`：忽略 `/python/`（可移植运行时，约 6GB）、`/models/`（全部模型，约 15GB）、`/bin/`（ffmpeg 约 650MB）、`/llama_cpp/`（llama.cpp 运行时）、`/input/`、`/output/`、`*.log`、`*.part`、Python 缓存与编辑器临时文件；仓库只保留源码、脚本、配置与文档。
- 自举链路：`一键环境搭建.bat` → `scripts/bootstrap_python.ps1`（下载 python.org embeddable 3.10.11，国内镜像兜底；写 `python310._pth`；get-pip 安装 pip）→ `scripts/setup_env.py`（分阶段 pip 安装：基础 / torch cu128 / 语音 / 人声分离 / 人物，并跑 `verify_person_env.py`）→ `scripts/download_runtime.py`（ffmpeg / llama.cpp / GGUF）→ 复用既有人物/分离/ASR 下载脚本。已存在文件按大小校验跳过，可重复运行续传。
- 多源测速：新增 `scripts/download_utils.py`（仅标准库）。`probe_speed` 对候选源做 1MB Range 探测，`pick_fastest` 并发测速选最快，`download_smart` 先选最快源、失败自动回退其余源并在进程内缓存同组选择；`download_runtime.py`、`download_separation_models.py`、`download_person_models.py` 及 `download_anime_person_models.py`（HF 端点）均已接入。实测：llama 的 GitHub 镜像会在 ghproxy.net / gh-proxy.com / 直连之间切换；ModelScope 的 API 端点快于 resolve 与 hf-mirror。
- 描述模型换源：`models/llm/llm_model.gguf` 改为 `unsloth/Qwen3.5-4B-GGUF` 的 `Qwen3.5-4B-UD-Q4_K_XL.gguf`（2,912,109,728 字节）、`mmproj_model.gguf` 改为 `mmproj-F16.gguf`（672,423,616 字节）；候选源为 ModelScope（resolve / API）→ hf-mirror → HF 官方，落盘文件名不变，`config.json` 的 `describe.model` / `describe.mmproj` 无需改动。
- 关键修正（实测踩坑）：① llama.cpp 的 `llama-server.exe` 只是约 9KB 的启动器，不能用「文件 >1MB」判断是否就绪，改为校验 `llama-server.exe` + `llama-server-impl.dll` + `ggml-base.dll`；② GitHub 代理可能返回 200 但内容被截断，必须为每个资产带上期望字节数并校验完整性（否则会把坏 zip 当成下载完成）；③ 固定文件名（如 `llm_model.gguf`）的模型按大小区间 `[0.98, 1.02]×` 校验，避免旧的更大/更小模型被误判为已就绪。
- 影响：新增 `一键环境搭建.bat`、`scripts/bootstrap_python.ps1`、`scripts/setup_env.py`、`scripts/download_utils.py`、`scripts/download_runtime.py`；重写 `.gitignore`；`scripts/setup_runtime.bat` 改为调用 `setup_env.py`；`scripts/download_separation_models.py` / `download_person_models.py` / `download_anime_person_models.py` 接入多源测速；`readme.md` / `AGENTS.md` / `REQUIREMENTS.md` / 本文件同步。`config.json` 与 `app/config.py` 未改动（模型名不变）。
- 验证：`python -m compileall app scripts` 通过；本机确认 `download_runtime.py --only ffmpeg/llama` 正确识别就绪并跳过；测速探测确认 ffmpeg（BtbN vs gyan）、llama（GitHub 镜像）、GGUF（ModelScope / hf-mirror）均能选出最快源；强制重下 llama b10991（CPU + CUDA）后 `llama-server.exe --version` 正常（build 10991）。全量模型下载耗时较长，未在本次完整重跑。

#### [P48] Windows 脚本编码与换行（一键环境搭建报错修复）
- 现象：双击 `一键环境搭建.bat` 出现 `'导包内可移植' is not recognized as an internal or external command`、`'（ASR' is not recognized` 等；随后 `scripts/bootstrap_python.ps1` 报 `Array index expression is missing or not valid` / `The string is missing the terminator`，中文全成乱码（如 `閿欒锛歱ip 涓嶅彲鐢ㄣ€?`），搭建中止并提示重试。
- 原因：
  1. `一键环境搭建.bat`、`scripts/setup_runtime.bat`、`scripts/bootstrap_python.ps1` 为 **LF 换行**；cmd.exe 只能可靠解析 CRLF 批处理，LF 会使行被错误连接/拆断（`@echo off` 失效、命令回显、`echo` 行错位）。
  2. 批处理为 **UTF-8 无 BOM**，但 cmd.exe 读取批处理时按**当前控制台代码页**解码，中文 Windows 为 CP936/GBK；`chcp 65001` 在文件开始被读取之后才执行，故含中文的 `echo` 行被按 GBK 误解码而错位。给 `.bat` 加 UTF-8 BOM 也**无效**（实测 cmd 不剥离 BOM，首行变成 `?@echo off`，`@echo off` 不生效）。`下载模型.bat` 等同样存在此隐患（仅 banner 报错、功能未受影响而长期未被发现）。
  3. `.ps1` 为 UTF-8 无 BOM，PowerShell 5.1 默认按 ANSI(CP936) 解码脚本，中文乱码会破坏引号配对，导致解析期报错。
- 二次踩坑（改用 GBK 后 banner 仍是乱码）：把 `.bat` 存成 GBK 后，`@echo off` 与解析都正常了，但 `chcp 65001` 下 banner 仍乱码（显示为 `һ�������`，即 GBK 字节被按 UTF-8 渲染）。实测结论：**cmd.exe 始终按系统 ANSI 代码页（本机 936）读取/解析 `.bat`，`chcp` 不改变该行为；而 `echo` 输出的是文件里的原始字节，由当前控制台代码页渲染**。因此 `.bat` 内任何非 ASCII 都无解：UTF-8 → 解析错乱；GBK → `chcp 65001` 下显示乱码。此外给 `.bat` 加 UTF-8 BOM 也无效（cmd 不剥离 BOM，首行变 `?@echo off`）。
- 解决办法（约定，最终）：
  1. **`.bat` 只允许 ASCII 字符（不得含中文），CRLF 换行、无 BOM**；面向用户的中文提示一律交给 `.py` / `.ps1` 打印（它们能正确处理编码）。保留 `chcp 65001`，只为让后续 `PYTHONUTF8=1` 的 Python 与 PowerShell 的中文输出正确显示。
  2. **`.ps1` 存为 UTF-8 **带 BOM**、CRLF 换行**，PowerShell 5.1 据此正确解码中文。
  3. `.gitattributes` 固定 `*.bat` / `*.ps1` 为 CRLF，避免换机器 clone 后被改成 LF 再次失效。
  4. 改脚本用编辑工具或 `[System.IO.File]::WriteAllBytes` + 显式编码，禁止用会改编码的文本 cmdlet（见 P23）。
- 影响：`一键环境搭建.bat`、`下载模型.bat`、`启动.bat`、`dev.bat`、`清理生成内容.bat`、`scripts/setup_runtime.bat`、`scripts/bootstrap_python.ps1`、新增 `.gitattributes`；`AGENTS.md` / `REQUIREMENTS.md` 同步。
- 验证：确认 6 个 `.bat` 均无 >127 字节（纯 ASCII）、CRLF、无 BOM；提取各 bat 头部 `cmd /c` 运行均无 `not recognized` 且 `@echo off` 生效；`[System.Management.Automation.Language.Parser]::ParseFile` 解析 `bootstrap_python.ps1` 无错误；实际运行 `bootstrap_python.ps1` 退出码 0、中文提示正常。

#### [P49] PyTorch cu128 下载加速：新增南大 / 上交镜像并按真实 wheel 测速选源
- 现象：安装 torch cu128 时从官方 `download.pytorch.org` 下载很慢（实测 torch-2.9.1+cu128-cp310-win_amd64 约 0.3~0.8MB/s），而 `setup_env.py` 原先固定只从官方索引取 torch/torchaudio、torchvision 只挂阿里云 `-f`。
- 候选源核实（同一 `torch-2.9.1+cu128-cp310-cp310-win_amd64.whl`）：
  1. 可用且快：`mirrors.nju.edu.cn/pytorch/whl/cu128`（PEP503，约 4~10MB/s）、`mirror.sjtu.edu.cn/pytorch-wheels/cu128`（PEP503，约 1~10MB/s，会 301 跳转到 s3.jcloud.sjtu）；二者均含 torch/torchaudio/torchvision 的 cp310 win_amd64 wheel。
  2. 可用但慢：`download.pytorch.org/whl/cu128`（官方，约 0.3~0.8MB/s）；`mirrors.aliyun.com/pytorch-wheels/cu128` 是**扁平 find-links 目录**（子包页 404），只能作 `-f` 兜底，实测约 1MB/s。
  3. 不可用：`mirrors.aliyun.com/pytorch-wheels/nightly/cu128`（仅 nightly，无稳定 2.9.1）；`mirror.sjtu.edu.cn/astral-wheels/cu128`（仅 flash-attn / vllm 等第三方包，无 torch）；`mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch`（conda 频道，pip 不可用）。
- 关键坑：**不能用索引导航页测速**——官方 `/torch/` 索引页只有 ~86KB、各镜像都秒开，会把官方误判为最快；必须探测**真实 wheel 的前 1MB**（大文件吞吐才能反映差异）。`probe_speed` 要求样本 ≥64KB，wheel 天然满足。
- 实现：`scripts/setup_env.py` 新增常量 `NJU_TORCH` / `SJTU_TORCH` 与候选列表 `TORCH_MIRRORS`（官方 + 南大 + 上交，均为 PEP503）、新增 `pick_torch_index()`（按 `sys.version_info` 拼出当前解释器的 wheel 名，对三个索引上同一 wheel 做 1MB Range 探测，`pick_fastest(group="torch")` 选最快）；`install_pip_deps()` 用其替换固定的 `TORCH_INDEX`：torch/torchaudio → `--index-url torch_index --extra-index-url pypi`；torchvision → `--index-url torch_index --extra-index-url pypi -f ALIYUN_TORCH`（阿里云仅兜底）。
- 影响：`scripts/setup_env.py`、`requirements.txt`（注释补充可选镜像）、`REQUIREMENTS.md` 12.2、本文件；无新增依赖。
- 验证：实测 nju / sjtu wheel 探测均返回 206 且速度显著高于官方，官方与阿里云可用、其余三源按预期排除；`python -m compileall scripts` 通过。

#### [P50] ffmpeg 下载加速：新增 npmmirror 源并支持 tar.xz 解压
- 现象：`download_runtime.py` 下载 ffmpeg 时只有 BtbN 直连 GitHub 与 gyan.dev 两源，国内直连很慢。
- 候选源核实：
  1. `https://registry.npmmirror.com/-/binary/ffmpeg-builds/`：**可用但用法特殊**——它是目录索引而非文件；**没有 `latest/` 别名**（404）；资产名为 `ffmpeg-<ver>-win32-x64-gpl.tar.xz`（不是 BtbN 的 `ffmpeg-master-latest-win64-gpl.zip`）。实测 `v8.1.3/ffmpeg-8.1.3-win32-x64-gpl.tar.xz`：140,604,648 字节、sha256 与官方 `.sha256.txt` 一致、下载 16.7MB/s；解压后 `bin/{ffmpeg,ffprobe,ffplay}.exe` 齐全、静态（无 DLL）、`ffmpeg -version` 正常、含 `libx264`。
  2. `https://github.com/BtbN/FFmpeg-Builds/releases/latest/download/ffmpeg-master-latest-win64-gpl.zip`：可用，与代码原用的 `releases/download/latest/...` **等价**，均 302 → `release-assets.githubusercontent.com`，最终 zip 196,019,840 字节。
- 关键点：npmmirror 与 GitHub 源的**归档格式不同**（tar.xz vs zip），且 npmmirror 无固定“最新”URL，必须先解析索引。故不能简单把 URL 加进列表，需：① `_npmmirror_ffmpeg_url()` 读索引 JSON、正则出 `v<major>.<minor>[.<patch>]` 目录取版本最大者、再校验该目录确有 `win32-x64-gpl.tar.xz` 资产后拼 URL；② `_extract_flat()` 改为按内容用 `tarfile.is_tarfile` 区分 zip / tar.xz（`_extract_zip_flat` / `_extract_tar_flat`），仍平铺取 `ffmpeg.exe`/`ffprobe.exe`/`ffplay.exe`；③ `_ffmpeg_urls()` 把 npmmirror 放在候选首位，交 `download_smart(group="ffmpeg")` 测速。
- 实现：`scripts/download_runtime.py` 新增 `import json/re/tarfile/urllib.*`、常量 `FFMPEG_NPM_INDEX`、`_npmmirror_ffmpeg_url()`、`_ffmpeg_urls()`；`_extract_flat()` 拆分并支持 tar.xz；`ensure_ffmpeg()` 改用 `_ffmpeg_urls()`。归档临时名仍为 `ffmpeg.zip`，靠内容识别格式，与来源无关。
- **踩坑（首版即现）**：BtbN zip 下载中途断流，只拿到 114,768,076 / 196,019,840 字节即被 `download_smart` 判为成功（`download_smart` 传 `expected=0` 时 `download()` 不做完整性校验），随后 `_extract_flat` 按内容识别为 zip 却在 `archive.open()` 抛 `BadZipFile: Bad magic number`；且坏包留在 `%TEMP%\ref_forge_runtime\ffmpeg.zip`，下次运行因 `size_ok(archive,1)` 命中而**跳过重下、反复失败**。
- 修复：
  1. `download_utils.download_smart()` 新增可选 `verify` 回调：`download()` 完成后调用 `verify(dest)`，返回 False 视为该源失败，删除 `dest` 与 `.part` 并切换下一源。
  2. `download_runtime.py` 新增 `_archive_ok(path)`：zip 用 `ZipFile.testzip()`、tar.xz 用完整遍历成员读取，能识别截断；`ensure_ffmpeg()` 用它作①缓存判定（`force or not _archive_ok(archive)`，不再用 `size_ok` 误信坏包）与②`download_smart(..., verify=_archive_ok)`；解压再包一层异常兜底并删除坏包。
  3. 抽 `_is_tar(path)` 统一 `tarfile.is_tarfile` 的异常处理。
- 影响：`scripts/download_runtime.py`、`scripts/download_utils.py`、`REQUIREMENTS.md` 12.2、本文件；无新增依赖（xz 由标准库 `lzma`/`tarfile` 支持）。
- 验证：`_archive_ok(截断的 114MB BtbN zip)=False`；`download_runtime.py --only ffmpeg --force` 实测选 npmmirror（2.58MB/s）、下载 140.6MB、`_archive_ok` 通过、解出 `bin/{ffmpeg,ffprobe,ffplay}.exe`，`ffmpeg/ffprobe -version` 正常（n8.1.3）；`_npmmirror_ffmpeg_url()` 解析出 `.../v8.1.3/ffmpeg-8.1.3-win32-x64-gpl.tar.xz`；三源按预期返回；`python -m compileall app scripts` 通过。

### 3.4 历史记录（已移除的运镜识别，仅供追溯）

以下条目对应的功能已随 P38 移除，不再实现；此处仅保留一句话结论：

- P4/P16：yaw 摇镜透视弱 → 曾用“弱拟合”pan 判据；真实素材误伤严重后移除。
- P5：`center_displacement` 的 `+EPS` 固定偏差（已修，逻辑保留）。
- P6/P17/P22/P25/P26：跟拍 / 弧拍 / 淡出拉镜难例与主体检测、本质矩阵 / LK / 光流增强引擎。
- P7/P13/P14/P15/P18/P20：首尾帧差异、甩镜头时长冲突、全景长图与降级链。
- P27/P28：CrispASR / VibeASR.cpp（后被 P29 的 Qwen3-ASR 方案取代）。
- P31/P35：两池流水 + 并行主体分析等运镜处理提速（主体/流水线已随 P38 移除；保留 `StageTimer` 阶段计时与并行导出的思路）。

---

## 4. Checklist

### 4.1 环境

- [x] 包内独立 Python 环境就绪（embeddable 3.10.11）
- [x] OpenCV / numpy / Pillow / scenedetect / click / platformdirs 安装完成
- [x] `bin/ffmpeg.exe`、`bin/ffprobe.exe` 就位
- [x] 语音运行时入驻包内 `python/`：torch/torchaudio cu128 + funasr + qwen-asr + modelscope + librosa + soundfile，模型在 `models/asr/`（见 3.3 P29）
- [x] 人声分离运行时入驻包内 `python/`：audio-separator 0.47.0 + onnxruntime-gpu 1.23.2 + torchvision 0.24.1+cu128，四档模型在 `models/separator/`（见 3.3 P32）
- [x] 视频描述运行时：`llama_cpp/llama_bin/llama-server.exe` + `models/llm/` 模型（默认 `unsloth/Qwen3.5-4B-GGUF`，见 P47）
- [x] `启动.bat` / `dev.bat` 可用，且使用包内 Python
- [x] 一键环境搭建链路就位：`一键环境搭建.bat` + `bootstrap_python.ps1` + `setup_env.py` + `download_runtime.py` + `download_utils.py`（多源测速，见 P47）
- [ ] 更换目录 / 盘符后仍可运行（待人工复验）
- [ ] 断网状态下仍可运行（待人工复验）

### 4.2 开发

- [x] 帧质量评估 / 有效区间修剪 / 多帧采样
- [x] 特征提取与匹配 / 单对估计 / 尺度轨迹累积
- [x] 3x3 宫格故事图生成（均匀时间间隔就近取清晰帧、严格时间顺序）
- [x] 镜头分割（PySceneDetect 硬切 + 迭代时长修正 + 分辨率归一化）
- [x] 片段导出（ffmpeg copy/reencode + ffprobe 校验）
- [x] 输出目录结构（`output/segments/<源名>/shot_XXXX/` + `detailed/`）
- [x] `--no-split` / `--clean-segments` / 跳过与覆盖语义
- [x] 语音识别子系统（Qwen3-ASR 转写 + ForcedAligner 时间戳 + fsmn-vad + CAM++/ERes2NetV2 声纹；SRT / transcript.json / 按人归档 / 声纹中心 / 各 shot `detailed/asr.json`；`--no-asr`）
- [x] 人声/背景音乐分离（四档可配、GPU/CPU、本地离线；三轨 + separation.json；`--no-separate`）
- [x] 片段级字幕归属（按多数重叠唯一归属）+ 片段 SRT + 片段三轨音频
- [x] 视频描述（llama.cpp + Qwen3.5，中文六段式；`--no-describe` / `--describe-limit`）
- [x] 阶段计时日志（`stage timing`）

### 4.3 验收

- [x] 分割验收：对 `input/` 真实视频校验时长约束、故事图与目录结构（素材改为直接用 `input/`，见 P41）
- [x] 统一 3x3 九宫格故事图（均匀时间就近取清晰帧、严格时间顺序；每格偏差 ≤ 50ms）
- [x] 语音识别验证：`run_asr_test.py` 全通过（SRT / transcript / 归档 / 声纹中心跨视频合并；Screenrecording 正确判无语音）
- [x] 同镜头多说话人对白拆分（窗口聚类 + 句级投票）
- [x] 语音识别设备验证：`device=gpu` 正常；`device=cpu` 正常；无 CUDA 自动回退
- [x] 人声分离验证：`run_separation_test.py`（best+gpu）通过；四档模型 GPU 与 fast CPU 均可用；`run_asr_test.py` 校验 `source=vocals`
- [x] 片段级字幕/音频验证：`C4` 片段归属正确、每镜头 `seg_XXXX.srt` 与三轨 wav 齐全
- [x] 视频描述验证：`run_describe_test.py` 抽帧 + prompt 通过；`--llm` 中文六段完整
- [x] 人物提取验证：`run_persons_test.py` 通过（产物结构 / 命名可溯源 / 原分辨率矩形 / 去重与 Top-300 / `unknown` / 环境未变；见 P45）
- [ ] 打包验收（无 Python / 无网络机器，待人工复验）

### 4.4 复现命令

```bat
:: 主程序
dev.bat app\main.py --input input --output output --overwrite

:: 验收（素材直接取 input/；可选 dev.bat scripts\make_test_videos.py 生成合成素材）
dev.bat scripts\run_segmentation_test.py
dev.bat scripts\run_asr_test.py
dev.bat scripts\run_separation_test.py
dev.bat scripts\run_describe_test.py
dev.bat scripts\run_describe_test.py --llm
dev.bat scripts\run_persons_test.py

:: 清理生成物
dev.bat scripts\clean_generated.py --yes

:: 克隆后首次一键搭建环境（联网；引导 Python + 依赖 + 资产 + 模型）
一键环境搭建.bat

:: 已引导 Python 后重新安装环境 + 模型
scripts\setup_runtime.bat
```
