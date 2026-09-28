# RefForge · 需求与设计文档（REQUIREMENTS.md）

> 版本：2.1（代码为唯一基准；本文件按 `app/main.py` 的执行顺序组织）
> 状态：已实现并验收
> 本期新增：**视频人物提取与归类**（阶段 4，见第十四节；详细需求另见 `PERSON_REQUIREMENTS.md`）。新增内容不改变原有阶段的产物与环境（`cv2`/`torch`/`numpy` 版本不变）。
>
> **阅读约定**
> 1. 本文件描述的是**当前代码的实际行为**。任何实现变更必须同步回改本文件；本文件与 `PROJECT_NOTES.md`（踩坑与实测）互为补充。
> 2. 本工具**不做运镜识别**，不输出 `motion_meta.json`；`pipeline/motion.py` 仅保留尺度轨迹累积供 `--debug` 诊断，故事图选帧不再依赖它。
> 3. 阈值/常量集中在 `config.json`，内置默认值在 `app/config.py:DEFAULT_CONFIG`，两者键名与默认值必须一致。
> 4. 环境强制使用包内 Python（`python\python.exe` / `启动.bat` / `dev.bat`），不联网、不写死开发机路径。

## 目录

1. [项目概述](#一项目概述)
2. [总体流程（按 `main.py` 执行顺序）](#二总体流程按-mainpy-执行顺序)
3. [阶段 0：CLI、配置与引擎就绪检查](#三阶段-0cliconfig-与引擎就绪检查)
4. [阶段 1：逐源视频处理（分割 + 故事图）](#四阶段-1逐源视频处理分割--故事图)
5. [阶段 2：人声/背景音乐分离](#五阶段-2人声背景音乐分离)
6. [阶段 3：语音识别 + 按人归档](#六阶段-3语音识别--按人归档)
7. [阶段 4：逐片段视频描述](#七阶段-4逐片段视频描述)
8. [阶段 5：清理、统计与阶段计时](#八阶段-5清理统计与阶段计时)
9. [输出目录与 Schema 汇总](#九输出目录与-schema-汇总)
10. [配置与阈值（`config.json`）](#十配置与阈值configjson)
11. [工程结构与模块接口](#十一工程结构与模块接口)
12. [依赖与独立一键包](#十二依赖与独立一键包)
13. [验收标准与测试](#十三验收标准与测试)
14. [人物提取与归类（新增）](#十四人物提取与归类新增)
15. [非目标（本期不实现）](#十五非目标本期不实现)

---

## 一、项目概述

**RefForge** 是一个 Python 3.10 CLI 一键离线包：把输入视频自动加工成视频生成模型的 full-reference 参考资产（镜头片段、3x3 宫格故事图、人声/背景音乐分离、按人归档语音与声纹、人物归类截图、中文 H3 视频描述 prompt），**免安装、免联网、可整包拷贝**。

### 1.1 输入

一个或多个视频文件（`input/` 批处理，或 `--single` 单文件）。**默认输入是多镜头长视频 / 多镜头合集**。

支持的扩展名：`.mp4` / `.mov` / `.mkv` / `.avi` / `.webm`（大小写不敏感）。按文件名自然排序处理，同一源文件内按时间顺序编号。

### 1.2 输出

对每个源视频，产出一个以源文件名（去扩展名）命名的目录 `output/segments/<源文件名去扩展>/`，其中：

1. **镜头分割清单** `segments.json`；
2. 每个片段一个 `shot_XXXX/` 目录，含：
   - `seg_XXXX.mp4`（片段视频，序号与 shot 一致）；
   - `scene_reference.jpg`（**3x3 宫格故事图**，作为视频生成模型的参考图 1）；
   - `seg_XXXX.srt`（本片段字幕，时间相对片段起点）；
   - `original.wav`（本片段原始音轨，有音轨时）；
   - `frames/`（2fps 采样帧，PNG，每个时间格取最清晰帧）；
   - `video_prompt_zh.txt`（中文 H3 full-reference 视频描述 prompt）；
   - `detailed/`（`asr.json`、`video_prompt.json`、`vocals.wav`、`instrumental.wav`、`--debug` 时的 `debug/`）。

音频侧另产出（见阶段 2/3）：

- 源级 `audio/`：`original.wav` / `vocals.wav` / `instrumental.wav` / `separation.json`；
- `transcript.json`（整视频结构化转录）；
- `speakers/<speaker>/`（按人归档的 1~30s 语音：`.wav` + `.json` + `<speaker>_global.json`）；
- `<输入视频同目录>/<源文件名>.srt`（整视频字幕，带全局说话人前缀）；
- `output/voiceprint_center.json`（全局说话人声纹中心库，跨视频合并）。

### 1.3 核心原则

1. 输入默认多镜头长视频，**必须先分割**为单一镜头片段；每段时长落在 `[min_segment_sec, max_segment_sec]`（默认 `[1s, 15s]`），不满足则迭代再分割 / 合并（除非输入总时长本身不足 1s）。
2. 所有片段统一生成 `3x3` 九宫格故事图，**按均匀时间间隔选帧、每点就近取清晰帧**（时间偏差 ≤ `reference.select_tolerance_ms`，默认 50ms）、结果严格按时间升序；不使用全景 / 长图拼接。
3. 音频侧先做**人声 / 背景音乐分离**，ASR 与按人归档使用分离后的人声轨；**VAD 分句固定使用原始音轨**（原因见 6.1）。
4. 语音识别按人归档，全局说话人 ID 跨视频一致。
5. 视频描述由本地多模态大模型（llama.cpp + Qwen3.5）**直接生成中文**六段式 prompt，不做翻译。
6. 任一阶段失败（模型/依赖缺失、单片段异常）都不得导致进程崩溃；缺失则对应阶段跳过并告警。

### 1.4 文档与实现双向同步

- 实现变更必须回改本文件；踩的坑、新约定、解决办法写入 `PROJECT_NOTES.md`。
- 冲突处理顺序：先修正实现 → 再更新文档；或先确认新方案 → 改文档 → 再改代码。禁止长期不一致。

---

## 二、总体流程（按 `main.py` 执行顺序）

```text
main(argv):
  1. parse_args -> resolve(_input/_output/_config) -> load_config -> _apply_overrides
  2. 设置 OpenCV 线程（performance.opencv_threads > 0 时 cv2.setNumThreads）
  3. ensure_dir(output) -> setup_logger(output/run.log)
  4. 收集视频：args.single 或 list_videos(input)
  4b. --interactive：interactive.collect_choices(...) 一次性收集分割时长 / 是否分割 /
        输入图像 / 描述片段（枚举结果 precomputed 供后续复用），写回 cfg
  5. 引擎就绪检查（check.engines）：
       speech.available(cfg)            -> 关闭 asr（缺依赖/模型）
       speech.separation_available(cfg) -> 关闭 separation
       describe.available(cfg)          -> 关闭 describe
  6. 初始化 stats / speech_jobs / separation_jobs / describe_jobs / cleanup
     workers = performance.workers(0=min(8,CPU)) -> ThreadPoolExecutor
  7. phase.scene_reference：for 每个源视频 process_source(...)
        元信息 -> 跳过/覆盖 -> 镜头分割（有 precomputed 则复用交互枚举结果）
        -> 并行导出片段 -> 写 segments.json
        -> 逐片段并行生成 scene_reference.jpg
        -> 登记 separation / asr / describe 任务 + 待清理片段
        -> 交互选择了描述片段时按 (源视频, shot_id) 过滤 describe_jobs

  8. phase.separation（模型只加载一次）：
       抽原始音轨 -> 分离 -> audio/{original,vocals,instrumental}.wav + separation.json
       -> 片段级音频；{video: vocals} 回填给 asr jobs（audio_source）
  9. phase.asr（模型只加载一次，结束卸载）：
        VAD（原始音轨）-> 转写/声纹（人声轨）-> 分配全局说话人 -> 导出产物
 10. phase.persons（模型只加载一次，结束卸载；见第十四节）：
        逐源视频整片：智能抽帧(1fps 取最清晰) -> YOLO11-pose + BoT-SORT
        -> 人体裁剪图上 SCRFD 人脸 + ArcFace 嵌入 -> 人脸归属与 track 均值特征
        -> 跨 track 聚类(约束感知) -> 裁剪(union+5%)/层级/去重/Top-300
        -> persons/{person_XXX, unknown}/ + metadata.json
 11. phase.describe（llama-server 只加载一次，结束关闭）：
        逐片段抽帧 -> 组装 prompt -> 生成中文 -> 落盘
 12. phase.cleanup：--clean-segments 时删除 seg_XXXX.mp4
 13. 打印汇总 + stage timing
```

> 引擎检查在第 5 步统一进行；缺失的模型/依赖只会关闭对应阶段，其余阶段照常。跳过/覆盖判定在第 7 步按源视频进行。

---

## 三、阶段 0：CLI、配置与引擎就绪检查

### 3.1 命令行接口（CLI）

```text
python main.py [--input DIR] [--output DIR] [--config FILE] [--single FILE]
               [--debug] [--overwrite]
               [--no-split] [--split-sensitivity low|medium|high]
               [--min-seg-sec SEC] [--max-seg-sec SEC]
               [--segment-encode auto|copy|reencode] [--clean-segments]
               [--no-asr] [--no-describe] [--describe-limit N]
               [--no-separate] [--separate-model fast|balanced|best|dereverb]
               [--separate-device gpu|cpu]
               [--no-persons] [--persons-device gpu|cpu] [--persons-max N]
               [--workers N] [--opencv-threads N] [--interactive]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input` | `./input` | 批处理输入目录 |
| `--output` | `./output` | 输出根目录 |
| `--config` | `./config.json` | 阈值配置，缺失则用内置默认值 |
| `--single` | 无 | 只处理单个视频文件（与 `--input` 二选一） |
| `--debug` | 关闭 | 输出中间产物（故事图 `detailed/debug/`） |
| `--overwrite` | 关闭 | 覆盖已存在的输出目录；默认跳过已处理源文件 |
| `--no-split` | 关闭 | 关闭镜头分割，输入整体按单镜头处理 |
| `--split-sensitivity` | `None`（配置 `medium`） | 分割灵敏度：`low` / `medium` / `high` |
| `--min-seg-sec` | `None`（配置 `1.0`） | 片段最短时长（秒） |
| `--max-seg-sec` | `None`（配置 `15.0`） | 片段最长时长（秒） |
| `--segment-encode` | `None`（配置 `auto`） | 片段导出编码：`copy` / `reencode` / `auto` |
| `--clean-segments` | 关闭 | 置 `segmentation.keep_segments=false`，处理完成后删除片段视频 |
| `--no-asr` | 关闭 | 关闭语音识别（默认开启） |
| `--no-describe` | 关闭 | 关闭逐片段视频描述（默认开启） |
| `--describe-limit` | `None`（配置 `0`=全部） | 仅处理前 N 个片段的视频描述（调试用） |
| `--no-separate` | 关闭 | 关闭人声/背景音乐分离（默认开启） |
| `--separate-model` | `None`（配置 `fast`） | 人声分离档位 |
| `--separate-device` | `None`（配置 `gpu`） | 分离设备 `gpu` / `cpu` |
| `--no-persons` | 关闭 | 关闭人物提取与归类（默认开启） |
| `--persons-device` | `None`（配置 `gpu`） | 人物提取设备 `gpu` / `cpu` |
| `--persons-domain` | `None`（配置 `anime`） | 人物模型领域 `anime`（动漫，2D/3D，默认）/ `real`（真人） |
| `--persons-max` | `None`（配置 `0`=全部） | 仅处理前 N 个源视频的人物提取（调试用） |
| `--workers` | `None`（配置 `0`=自动） | 逐片段处理的并行线程数；0 = `min(8, CPU 核数)` |
| `--opencv-threads` | `None`（配置 `0`=默认） | OpenCV 内部线程数；仅影响性能，不影响产物 |
| `--interactive` | 关闭 | 启动时交互式选择要执行的内容（见 3.2）；双击 `启动.bat` 默认启用 |

所有相对路径基于**包根目录**解析（`_resolve`：绝对路径取 `normpath`，否则相对 `package_root()`）。

### 3.2 交互模式（`--interactive`）

双击 `启动.bat`（无参数）时自动追加 `--interactive`；带参数调用则走原 CLI 路径。交互在**引擎检查之前**、收集到视频列表之后进入，**一次性**收集七项选择，全部确认后才开始流程：

1. **运行设备**：GPU（默认）或 CPU，写回 `separation.device` / `asr.device` / `persons.device`，并把 `describe.gpu_layers` 设为 `999`（GPU）或 `0`（CPU）；
2. **分割时长**：片段最短 / 最长时长（秒），默认 `1` / `15`，写回 `segmentation.min_segment_sec` / `max_segment_sec`；
3. **是否执行前半段流程**：`镜头分割 -> 故事图 -> 人声分离 -> 语音识别`，默认执行；选否时 `segmentation.enabled=false`，原视频整体视为单个片段，故事图 / 人声分离 / 语音识别照常在该单片段上执行（等价 `--no-split`）；
4. **人物提取与归类**：选择模型领域——`1` 动漫（默认，2D/3D）、`2` 真人、`3` 不执行；写回 `persons.enabled` 与 `persons.domain`（等价 `--no-persons` / `--persons-domain`）；
5. **是否生成视频描述**：默认生成，写回 `describe.enabled`（等价 `--no-describe` 的开关）；选否时**跳过**后续"输入图像 / 描述片段"两步，且不加载描述模型，直接开始流程；
6. **视频描述输入图像**：故事图 `storyboard`（默认）或抽帧 `frames`，写回 `describe.input_mode`；
7. **需要生成描述的片段**：通过 `interactive.enumerate_fn`（即 `main._enumerate_source`）预先枚举每个源视频的片段并列出编号，默认全部（回车），可输入编号 / 范围（如 `1,3,5-8`）或 `none` 表示不生成描述。

其中第 7 步的枚举只做镜头分割（不导出片段、不生成故事图），结果缓存在 `Choices.precomputed`；`process_source` 通过 `precomputed` 参数直接复用，**不会重复分割**。选择的片段以 `(源视频路径, shot_id)` 形式记录，在登记完 `describe_jobs` 后按 `shot_id` 过滤；全部 / 不选择分别关闭 / 保留描述阶段。

实现位于 `app/interactive.py`：`Choices` 保存选择结果，`collect_choices(videos, cfg, logger, enumerate_fn)` 负责交互与校验（非法输入重问，EOF 时回落默认值）。

### 3.3 配置加载与覆盖

- `load_config(path)`：以 `DEFAULT_CONFIG` 深拷贝为基础，与用户 JSON 做**深合并**；文件缺失或解析失败时回落到默认值。
- `_apply_overrides(cfg, args)` 将 CLI 覆盖写入配置：
  - `--no-split` → `segmentation.enabled=false`；`--split-sensitivity` / `--min-seg-sec` / `--max-seg-sec` / `--segment-encode` 分别覆盖对应键；
  - `--clean-segments` → `segmentation.keep_segments=false`；
  - `--no-asr` → `asr.enabled=false`；`--no-describe` → `describe.enabled=false`；`--describe-limit` → `describe.limit`；
  - `--no-separate` → `separation.enabled=false`；`--separate-model` / `--separate-device` 覆盖对应键；
  - `--no-persons` → `persons.enabled=false`；`--persons-device` / `--persons-domain` / `--persons-max` 覆盖对应键；
  - `--opencv-threads` → `performance.opencv_threads`。
- `performance.workers` 由 `_resolve_workers`：CLI 覆盖优先，否则取配置值（>0），否则 `min(8, CPU 核数)`。

### 3.4 引擎就绪检查（缺则关闭对应阶段）

| 阶段 | 检查函数 | 检查内容 | 缺失行为 |
|---|---|---|---|
| ASR | `speech.available(cfg)` | `asr.models` 下 5 个模型目录 + `torch`/`funasr`/`qwen_asr` 可导入 | `asr.enabled=false` + warning |
| 分离 | `speech.separation_available(cfg)` | `audio_separator` 可导入 + 当前档位模型文件存在 | `separation.enabled=false` + warning |
| 描述 | `describe.available(cfg)` | `llama-server.exe` + `describe.model` + `describe.mmproj` | `describe.enabled=false` + warning |
| 人物 | `persons.available(cfg)` | 按 `persons.domain` 分派：`real` 查 `ultralytics` / `onnxruntime` + YOLO 权重 + buffalo_l ONNX；`anime` 查 `torch` / `transformers` / `onnxruntime` + 动漫人物/人脸 ONNX + CLIP 权重 | `persons.enabled=false` + warning |

---

## 四、阶段 1：逐源视频处理（分割 + 故事图）

`main.process_source(path, ...)` 对每个源视频执行以下步骤。

### 4.1 元信息与跳过 / 覆盖

- `video_io.read_metadata(path)` → `VideoMeta(fps, width, height, frame_count, duration_sec)`。若 OpenCV 读不到有效元信息，回落包内 `ffprobe`；两者都失败抛 `IOError`。
- 源目录 `output/segments/<safe_stem>/`；若 `segments.json` 已存在且未加 `--overwrite` → 记录 `skip` 并返回；`--overwrite` 时先 `rmtree` 源目录再重建。

### 4.2 镜头分割（`pipeline/segmentation.py`）

`_build_segments(path, meta, cfg, logger)`：

1. **关闭分割**（`segmentation.enabled=false`，即 `--no-split`）：整个输入作为单段 `Segment(index=1, 0..frame_count, method="none", shot_id/shot_dir="shot_0001")`，`meta.method="none"`。
2. **硬切检测** `detect_shot_cuts`：
   - `scenedetect` 不可用 → `(None, meta)`，交由等长切分；
   - 用 `SceneManager` + `AdaptiveDetector(adaptive_threshold=阈值, min_scene_len=round(min_scene_len_sec*fps))`；视频流用 `_NormalizingStream` 包装，把与声明分辨率不一致的帧 `cv2.resize` 回声明尺寸（解决同文件内分辨率切换导致 PySceneDetect 判损坏并漏检、刷屏，见 P34）；
   - 灵敏度阈值来自 `adaptive_thresholds`（`low/medium/high` = 4.5 / 3.0 / 2.0）；
   - 无场景时返回 `[0]`（method 仍为 `pyscenedetect_adaptive`，scene_count=0）；否则切点 = 各场景起始帧 + 末尾结束帧，`sorted(set(...))`。
3. **长镜头二次检测**（`long_shot_subdetect=true`）：`make_subdetector` 返回 `subdetect(start,end)`，对超过 `max_segment_sec` 的区间用 `long_shot_subdetect_sensitivity`（默认 `high`）再检测子切点。
4. **`build_segments(total_frames, fps, cuts, cfg, subdetect_fn)`**：
   - `min_f=round(min_segment_sec*fps)`、`max_f=round(max_segment_sec*fps)`；
   - `total <= 0` → 空列表 + `fallback=true`；
   - `total < min_f` → 单段 `method="below_min"`、`below_min=true`、`fallback=true`；
   - 检测到切点（`len(cuts)>=2`）时以切点划分区间；否则整段为 `[0,total]`；
   - 二次检测结果并入切点集合；
   - **迭代修正**：循环执行「超长等分 `_split_over_max`（`ceil(length/max_f)` 段）+ 过短合并 `_merge_under_min`（优先并入合并后不超上限的邻居，其次较短一侧）」，直到全部合规或达到 `max_split_iterations`；
   - 仍不合规 → `_force_fix` 强制修正并置 `fallback=true`；
   - 生成 `Segment` 列表，`method` 由 `_method_for` 判定：`equal_split`（无检测）、`merge`（区间内含原始切点）、`pyscenedetect_adaptive`（首尾均为原始切点）；
   - `meta`：`detected_shot_count`（迭代前区间数）、`segment_count`、`split_iterations`、`fallback`。
   - `Segment` 字段：`index, start_frame, end_frame, start_sec, end_sec, duration_sec, method, below_min, shot_id, file, shot_dir`；`shot_id = shot_dir = "shot_%04d"`。

### 4.3 片段导出（`pipeline/segment_export.py`）

- `method=="none"` 的段不导出（`file=""`，后续按内存帧区间处理）。
- `export_segment(src, start_sec, end_sec, dest, encode_mode, cfg)`：
  - `copy`/`auto`：`ffmpeg -ss START -i src -t DURATION -c copy ...`；`copy` 成功即返回；`auto` 用 `probe_duration` 校验实际时长，偏差 `<= copy_duration_tolerance_sec`（默认 0.1s）才返回，否则删除副本转 `reencode`（关键帧对齐导致 copy 时长偏差，见 P10）；
  - `reencode`：`ffmpeg -ss START -i src -t DURATION -c:v libx264 -crf 18 -preset veryfast -pix_fmt yuv420p -c:a aac -movflags +faststart`；
  - 失败返回 `None`，调用方置 `file=""` 并按源视频内存帧区间处理（`frame_offset`）。
- 导出在 worker 池中并行执行（`pool.map(_export_one, segments)`）。

### 4.4 分割清单 `segments.json`

每个源文件一份，`schema_version="1.0"`：

```json
{
  "schema_version": "1.0",
  "source": {"file": "clipA.mp4", "fps": 30.0, "frame_count": 900, "duration_sec": 30.0},
  "sensitivity": "medium",
  "min_segment_sec": 1.0,
  "max_segment_sec": 15.0,
  "detected_shot_count": 2,
  "segment_count": 4,
  "split_iterations": 2,
  "fallback": false,
  "segments": [
    {
      "index": 1, "shot_id": "shot_0001",
      "start_frame": 0, "end_frame": 224,
      "start_sec": 0.0, "end_sec": 7.5, "duration_sec": 7.5,
      "method": "pyscenedetect_adaptive", "below_min": false,
      "file": "seg_0001.mp4", "shot_dir": "shot_0001"
    }
  ]
}
```

### 4.5 逐片段故事图

对每个片段构造 `_SegmentJob`：优先用导出的片段视频（`frame_offset=0`），导出失败时用源视频并按 `[start_frame, end_frame)` 处理（`frame_offset=start_frame`）。每个 job 经 `_prepare_segment` + `_finalize_segment`，在 worker 池中并行。

#### 4.5.1 解码（`pipeline/video_io.py`）

`decode_gray_frames(path, long_side=960, start_frame, end_frame)`：灰度顺序解码 `[start_frame, end_frame)`；以**首帧尺寸**为统一目标，长边超过 `long_side` 时等比缩小；尺寸不一致时逐帧重采样。空帧抛 `IOError`。

#### 4.5.2 帧质量评估（`pipeline/quality.py`）

逐帧计算 `mean`、`std`、`sharpness = Laplacian(gray).var()`，并取清晰度中位数：

```text
is_black    = mean < 15  and std < 10
is_white    = mean > 240 and std < 10
is_low_info = std < 8
is_blur     = sharpness < 0.4 * median_sharpness
bad         = is_black | is_white | is_low_info | is_blur

sharpness_score = clip(sharpness / (1.5 * median_sharpness), 0, 1)   # median=0 时为 0
exposure_score  = clip(1 - abs(mean - 128) / 128, 0, 1)
contrast_score  = clip(std / 60, 0, 1)
quality_score   = sharpness_score * exposure_score * contrast_score
low_sharpness_ratio = mean(is_blur)
```

#### 4.5.3 有效区间修剪（`pipeline/trim.py`）

`find_valid_range(quality, cfg)` → `(valid_start, valid_end, status)`：

- 帧数为 0 → `(0,0,"invalid")`；坏帧比例 `> invalid_black_ratio`（默认 0.9）→ `(0,n-1,"invalid")`；
- 从两端扫描，跳过连续坏帧，要求窗口内好帧数 `>= min_valid_frames + 1`，容忍连续坏帧 `trim_tolerance_frames`（默认 2）；
- `valid_end < valid_start` 时交换；
- 有效长度 `< min_valid_frames` 时退化为全局 `quality_score` 最高帧（`valid_start = valid_end`）；
- `status=="invalid"` 时 job 标记 invalid，仅记日志，不生成故事图、不参与描述；
- `invalid_reason` 返回 `no_frames` / `all_frames_black` / `all_frames_white` / `all_frames_low_info` / `invalid_quality`。

#### 4.5.4 多帧采样（`pipeline/sampling.py`）

`choose_k`：时长 `<1s` → `k_short(7)`；`<=3s` → `k_normal(10)`；否则 `k_long(14)`。`sample_frames` 在 `[valid_start, valid_end]` 上 `linspace` 取 K 个帧号（`K=min(K, span+1)`，至少 2），去重升序；有效跨度为 0 时取单帧。

#### 4.5.5 特征提取与匹配（`pipeline/features.py`）

- 检测器：`SIFT_create(max_features)`（不可用时回落 `ORB`），按 `max_features` 线程本地缓存。
- `match_pair`：`BFMatcher` + `knnMatch(k=2)` + Lowe ratio（`ratio_test` 默认 0.75）；无足够描述子或异常返回 `None`。返回 `(pts_i, pts_j, count)`。

#### 4.5.6 单对运动估计（`pipeline/motion.py:estimate_pair`）

对每对相邻采样帧：

- 匹配数 `< min_match_count(8)` → 失败对；
- `findHomography`（仅用于透视残差诊断）与 `estimateAffinePartial2D`（主模型），RANSAC 阈值 `ransac_reproj_threshold=3.0`；
- `scale_step` / `rotation_step_deg` 取自仿射矩阵；`inlier_count < min_inlier_count(8)` 或尺度非有限、越界 `[0.5, 2.0]` → 失败对（避免退化矩阵污染累积，见 P3）；
- `translation_step = center_displacement(affine)`：平移取**画面中心点位移**，避免绕中心缩放/旋转的平移项污染（见 P2）；
- `perspective_error = clip((err_affine - err_homo)/diag, 0, 1)`（内点上的 RMS 残差之差）；
- `flow_magnitude = median(‖pts_j-pts_i‖)/diag`；
- `h_inlier_ratio`（单应内点率）、`parallax_cv`（匹配点位移的标准差/均值）、`center_inlier_ratio`（中心 25%~75% 区域的内点率，供稳定性诊断）。
- 失败对由 `failed_pair` 构造（各项为 0/默认）。

#### 4.5.7 尺度轨迹累积（`pipeline/motion.py:accumulate_base`）

输入 `pairs / sample_indices`，产出 `AggregateEvidence`（`cum_scale` 等仅供 `--debug` 诊断，故事图选帧不再依赖）：

- 统计：`avg_match_count`（所有对）、`avg_inlier_count/ratio`、`low_inlier_rate`、`perspective_avg`、`flow_magnitude_avg`、`h_inlier_ratio_avg/min`、`parallax_cv_avg`（后几项基于成功对），`match_fail_rate`；
- **鲁棒累积**：优先使用 `inlier_ratio >= low_inlier_ratio(0.3)` 的对；不足 2 对时退回全部成功对；成功对 >=3 时剔除 `abs(log(scale/median_scale)) >= 0.25` 的尺度离群对（见 P16/P26）；
- 按 `pairs` 时间顺序连乘可用对仿射变换得到 `transform`，累积 `zoom_path`、`rotation_path`、`translation_path`；`direction_change_deg` 仅在相邻平移步长度均 `> 0.02*diag` 时累加夹角（避免噪声放大，见 P16）；
- 每处理一对（无论是否可用）就追加一次 `cum_scale / cum_rotation_deg / cum_translation / cum_homography`，故 `cum_scale` 与 `sample_indices` 等长；
- 总尺度 `scale_total`、总旋转 `rotation_deg_total`、总平移 `translation_total`（画面中心位移）、`translation_norm`、`translation_path_ratio`、`direction_change_deg`；
- 归一化得分：`score_zoom = |log(scale_total)|/0.10`、`score_rot = |rotation_deg_total|/3.0`、`score_trans = translation_norm/0.03`；
- `max_translation_step_norm`、`max_flow_magnitude`、`center_inlier_ratio_mean/std`、`diag`、`proc_long_side`。

> 本模块不做分类 / 主体 / 光流 / 几何增强；这些能力已随运镜识别移除。现仅用于 `--debug` 诊断输出，故事图改为均匀时间选帧后不再依赖 `cum_scale`。

#### 4.5.8 选帧与拼贴（`reference/generator.py` + `reference/storyboard.py`）

- `reference_indices(frame_quality, valid_start, valid_end, fps, cfg)`：`grid_rows(3) × grid_cols(3) = 9` 格；
  - 在 `[valid_start, valid_end]` 上按**帧号等间隔**取 9 个目标时刻（`np.linspace`），即均匀时间间隔；
  - 对每个目标时刻，在时间偏差 `≤ select_tolerance_ms(50)/1000*fps` 帧的候选帧中取**最清晰**的一帧：优先未判为坏帧（黑/白/低信息/模糊）的候选，其次清晰度 `sharpness` 最大者；候选全为坏帧时取清晰度最高者；无清晰度数据时取最近帧；
  - 结果按目标顺序**非递减**输出（片段过短导致目标重叠时重复取帧），保证 9 格严格按时间先后（见 P24）。
- `generate(frame_quality, frame_reader, fallback_frames, valid_start, valid_end, fps, cfg)`：按上述帧号读取彩色帧（调用方先按升序预读去重）；全部读取失败时用 `fallback_frames`（起点/终点区间）补齐；`make_grid` 排成 `3x3`，单元格间距 `grid_gap_px(10)`、画布留白 `grid_margin_px(10)`，长边上限 `output.max_side_reference(4096)`，超限等比缩小；输出 `scene_reference.jpg`（JPEG 质量 `output.jpeg_quality=92`）。
- `storyboard.make_grid` 用 Pillow 实现，单元格取 9 帧最大宽高为基准、`LANCZOS` 缩放并居中；不叠加文字。
- 参考图生成失败时置 `fallback=true` 并告警，不写文件。

#### 4.5.9 调试产物（`--debug`）

写入 `shot_XXXX/detailed/debug/`：`sampled_frames/frame_%06d.jpg`、`pairs.json`（各对证据字段）、`quality.csv`（逐帧 mean/std/sharpness/quality_score/bad）。

#### 4.6 任务登记与待清理

- 收集 `shots`（`shot_id / shot_dir / file / start_sec / end_sec`）；`describe_shots` 排除 invalid 片段。
- `separation.enabled` 时登记 `separation_jobs`；`asr.enabled` 时登记 `speech_jobs`；`describe.enabled` 时登记 `describe_jobs`。
- `segmentation.keep_segments=false`（`--clean-segments`）时把片段视频加入 `cleanup`，待 ASR / 描述读取完毕后再删除。

---

## 五、阶段 2：人声/背景音乐分离

独立阶段，先于 ASR；模型只加载一次、逐视频复用，结束卸载（`speech.separation.run_phase`）。

### 5.1 前置与设备

- `separation_available(cfg)`：`audio_separator` 可导入 + 当前档位模型文件存在。
- `resolve_device(cfg)`：`separation.device=cpu` → CPU；否则 `torch.cuda.is_available()` 决定 GPU/CPU；无 torch 时 CPU。
- 过滤掉无音轨的源视频（`has_audio_stream is False`）。

### 5.2 模型档位（`speech/separation.py:TIERS`）

| 档位 | 模型文件 | 架构 | 体积(约) |
|---|---|---|---|
| `fast`（默认） | `UVR-MDX-NET-Voc_FT.onnx` | MDX-Net | 67MB |
| `balanced` | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | BS-Roformer | 639MB |
| `best` | `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` | Mel-Band Roformer | 1.0GB |
| `dereverb` | `Reverb_HQ_By_FoxJoy.onnx` | MDX-Net（去混响） | 67MB |

所有模型本地加载，运行期禁止联网；UVR 元数据（`download_checks.json` / `vr_model_data.json` / `mdx_model_data.json`）随模型预置。`dereverb` 档输出 “No Reverb / Reverb” 两轨，映射为 `vocals`（干净人声）/ `instrumental`（混响残留）。

### 5.3 产物

对每个源视频：

1. `extract_original_audio` 抽原始音轨（保留原始采样率/声道，PCM 16-bit）→ `audio/original.wav`；
2. `audio-separator` 分离 → `audio/vocals.wav` + `audio/instrumental.wav`（`_pick_stems` 归类词干，兼容 Vocals / Instrumental / No Vocals / Inst / No Reverb / Reverb）；
3. 写 `audio/separation.json`；
4. 片段级 `cut_wav`：`shot_XXXX/original.wav`（根目录）+ `shot_XXXX/detailed/{vocals,instrumental}.wav`；
5. 返回 `{video: {"vocals": path, ...}}`，`main` 回填为 ASR job 的 `audio_source`（供 ASR 优先使用人声轨）。

`separation.json`（`schema_version="1.0"`）：

```json
{
  "schema_version": "1.0",
  "source": "样例.mp4",
  "tier": "fast",
  "model_file": "UVR-MDX-NET-Voc_FT.onnx",
  "device": "gpu",
  "sample_rate": 44100,
  "elapsed_sec": 7.0,
  "tracks": {
    "original": "original.wav",
    "vocals": "vocals.wav",
    "instrumental": "instrumental.wav"
  }
}
```

- 无音轨跳过、不创建 `audio/`；分离失败不影响故事图，ASR 自动回退源视频原始音轨。

---

## 六、阶段 3：语音识别 + 按人归档

独立阶段；`speech.models.get_models` 惰性加载 `torch` / `funasr` / `qwen_asr`，模型只加载一次，全部视频完成后 `shutdown()` 卸载并释放显存（`speech.runner.run_phase`）。

### 6.1 音轨分工（关键约定）

- **VAD 分句固定使用原始音轨**：分离后人声轨常残留背景音乐，会把整段并成一句（无静音边界），导致 ForcedAligner 词级时间戳整体漂移（P32 关键坑 5）。因此当 `audio_source != video` 时，另抽源视频原始音轨做 VAD。
- **转写 / 声纹聚类 / 按人归档使用分离后人声轨**（无分离结果时回退源视频原始音轨）。
- 两者时间轴一致，可直接按原始音轨区间从人声轨取片段。

### 6.2 流程（`speech/runner.py:process_video`）

1. 对 `audio_source` 抽 16kHz 单声道 WAV；无人声分离时回退视频；视频本身无音轨 → 写空产物 `reason=no_audio`。
2. `detect_speech(vad, vad_input)`（fsmn-vad，输入原始音轨）→ 语音区间列表（毫秒区间转秒，已排序、剔除无效）；按 `min_speech_sec(0.6)` 过滤过短区间；无区间 → `reason=no_speech`。
3. 从人声轨 `cut_clip` 出各区间音频，`transcribe_clips`（Qwen3-ASR + Qwen3-ForcedAligner）批量转写；只保留 `text` 非空的区间；无文本 → `reason=empty_transcript`。
4. **细粒度说话人区分**：
   - `make_windows`：在保留区间上做重叠滑窗（`diarize_window_sec=1.2` / `diarize_hop_sec=0.4`），区间不长于窗口则整段一个窗，末尾补覆盖结尾的窗；
   - 对每个窗口提 CAM++ 与 ERes2NetV2 声纹（192 维）；
   - `cluster_embeddings`（CAM++ 贪心余弦聚类，阈值 `local_speaker_threshold=0.55`）→ `smooth_labels` 多数滤波（半径 `diarize_smooth_radius=1`）；
   - 本地簇代表向量用**该簇全部窗口拼接音频**重新提取（比窗口均值稳）；
   - `split_sentences_with_words`（按标点分句并映射词级时间戳）+ `_speaker_for_span`（句时间重叠窗口投票）+ `sentence_merge_gap_sec=0.8` 合并相邻同人句；
   - 只为**真正被句子使用**的本地簇登记全局说话人（避免边界簇污染中心库）。
5. `VoiceprintCenter.match_or_create(rep_cam, rep_eres, source_file)` 分配全局 `speaker_N`（详见 6.4）。
6. 导出产物：`transcript.json`、整视频 SRT、按人归档、各 `detailed/asr.json`、片段 SRT。

### 6.3 产物 Schema

**整视频 SRT**（写到输入视频同目录 `<源文件名>.srt`）：每条带全局说话人前缀 `speaker_N: 文本`。

**`transcript.json`**：

```json
{
  "schema_version": "1.0",
  "source": {"file": "样例.mp4", "duration_sec": 15.08},
  "speaker_count": 2,
  "speakers": ["speaker_1", "speaker_2"],
  "segments": [
    {
      "speaker": "speaker_1",
      "start_sec": 0.14, "end_sec": 6.14,
      "text": "...", "language": "English",
      "words": [{"text": "Smell", "start": 1.9, "end": 2.38}],
      "index": 0
    }
  ]
}
```

**按镜头 `detailed/asr.json`**：`segments[]` 只包含**该镜头内确实能听到**的语音段；每个语音段按「与镜头时间**重叠最多**」唯一归属到一个镜头（跨切点连续说话按多数归属，不同时出现在相邻两个镜头，见 P33）。`start_sec/end_sec` 为源视频内绝对秒；`words[]` 为词级时间戳。无语音 / 无音轨时 `text=""`、`segments=[]` 并带 `reason`。

**片段字幕 `seg_XXXX.srt`**（shot 根目录）：内容与 `detailed/asr.json` 一致，时间改为相对片段起点并裁剪到片段范围。

**按人归档 `speakers/<speaker>/`**：仅归档时长 `1~30s`（`min_archive_sec` / `max_archive_sec`）的语音，`seg_XXXX.wav`（16kHz 单声道）+ `seg_XXXX.json` + `<speaker>_global.json`（该人本视频内全部归档的合并）。

### 6.4 全局声纹中心（`speech/voiceprint.py`）

- 路径：`asr.voiceprint_center` 为空 → `<output>/voiceprint_center.json`；可配置绝对路径或相对包根。
- `match_or_create`：以 **ERes2NetV2 相似度为主判据**（`verify_sim_threshold=0.55`），**CAM++ 一致性校验**（`voiceprint_sim_threshold=0.30`）；两者都达阈值才合并，否则新建 `speaker_<next_id>`。质心用在线均值更新并按 `samples` 计数。
- 结构：

```json
{
  "schema_version": "1.0",
  "engine": "campplus+eres2netv2",
  "thresholds": {"campplus": 0.30, "eres2netv2": 0.55},
  "next_id": 4,
  "speakers": {
    "speaker_1": {
      "campplus_center": [192 维向量],
      "eres2netv2_center": [192 维向量],
      "samples": 3,
      "videos": ["a.mp4", "b.mp4"]
    }
  }
}
```

- 同一 ID 在整视频 JSON / 镜头 JSON / 按人归档中完全一致；跨视频合并只登记真正被使用的簇。

### 6.5 设备与语言

- `resolve_device`：`asr.device=gpu` → `cuda:0`（不可用回退 CPU）；`torch_dtype`：GPU 用 `asr.dtype`（默认 `bfloat16`），CPU 恒 `float32`。
- `configure_language`：`auto`→`None`；`Chinese/English/Japanese`（含 `zh/en/ja`）归一化；其他透传首字母大写。
- ERes2NetV2 本地目录无 `config.yaml`，需显式 `AutoModel(model="ERes2NetV2", model_conf={}, model_path=..., frontend="WavFrontend", frontend_conf={"fs":16000})`（P29）；CAM++ 直接传目录。
- 离线环境变量：`MODELSCOPE_NO_NETWORK` / `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` / `TOKENIZERS_PARALLELISM=false`。

---

## 七、阶段 4：逐片段视频描述

独立阶段，在 ASR 之后、`--clean-segments` 删除片段之前执行；`llama-server` 只加载一次，结束关闭（`describe.runner.run_phase`）。按 `describe.limit`（`--describe-limit`）限制处理片段数。输入图像由 `describe.input_mode` 决定：`storyboard`（默认）直接用片段的 3x3 宫格故事图进行推理；`frames` 使用 ffmpeg 采样帧。两种模式使用不同模板文件。

### 7.1 输入图像（`describe/runner.py` 选择）

- `storyboard`（默认）：读取 `shot_XXXX/scene_reference.jpg`（3x3 宫格故事图）作为推理图像；若其超出 1080P 画框（长边 > `describe.storyboard_max_side`，默认 1920；短边 > 长边×9/16），先缩放到画框内并缓存为 `detailed/storyboard_1080p.jpg`（JPEG）再推理；原图已 ≤1080P 时直接用原图（不放大、不重复编码）；故事图缺失时回退到 `frames` 模式并告警。
- `frames`：用包内 ffmpeg 对片段按 `describe.frame_fps`（默认 **2.0fps**）均匀采样；
  - `frame_select=true` 时以 `frame_fps × frame_candidates(3) = 6fps` 抽候选帧到 `frames/_candidates/`，在每个时间格内取 `Laplacian` 方差最大（最清晰）的一帧，输出 `frame_%05d.png`，候选目录用后删除；
  - 尺寸：`fit_long_side` 限制在 1080P 画框内（横屏 ≤1920×1080、竖屏 ≤1080×1920，不放大），并取偶数尺寸；
  - 抽帧目录为 `shot_XXXX/frames/`（独立子目录）；片段视频缺失时对源视频用 `-ss/-t` 回退；
  - 返回 `{frames:[{name,path,time_sec}], width, height, fps, count, selected, ...}`。

### 7.2 Prompt 组装（`describe/prompt.py` + `prompts/`）

- 两套系统 / 用户模板，按模式选取：
  - `prompts/storyboard_system.txt` / `prompts/storyboard_user.txt`（`input_mode=storyboard`，随附图像即故事图，也就是参考资产 `<Picture 1>`）；
  - `prompts/frames_system.txt` / `prompts/frames_user.txt`（`input_mode=frames`，随附图像是采样帧，不是参考资产）。
- 用户消息内容：片段时长、输入图像信息（`frames` 模式为帧时间线；`storyboard` 模式为宫格数 / 时间顺序）、本片段字幕（读 `detailed/asr.json`，换算为相对片段起点，附源视频全局 `speaker_N` 并说明不得直接用作 `(Sx)`）、参考资产映射、全局说话人列表。
- 参考资产映射：
  - `<Picture 1>`：本片段的 **3x3 宫格故事图**（`scene_reference.jpg`）对应的内容单元，9 格按从左到右、从上到下的故事顺序定义镜头顺序、视角、主体位置与关键情节；**不是场景参考图**；
  - `<Picture 2..6>`：最多 5 个人物主体，其外貌来自按首次清晰出现排序的人物参考图（第 n 个人物 = `<Picture n+1>`）；
  - `<Audio 1..5>`：与人物一一对应的声音参考；`(S1)/(S2)` 在片段内按首次发声重编号。
- 硬性约束：**参考标签只允许 `<Picture N>` 与 `<Audio N>`（不使用 `<Subject N>` / `<Video N>`）**；`subject_definitions` 最多 6 行（1 行故事图 + ≤5 人物）；禁止在 prompt 中要求生成字幕 / 画面文字；片段可能含一个或多个镜头，**依据输入图像中实际可见的硬切**判断镜头数（无硬切只写 `[Shot 1]`）。
- `build_user_prompt(mode, duration, fps, frames, subtitles, speakers, grid)` 返回填充后的用户文本；`system_prompt(mode)` 返回对应系统模板；`missing_sections(text)` 检测六段标题是否齐全。

### 7.3 推理（`describe/client.py`）

- 启动一次常驻 `llama-server.exe`：

```text
llama-server.exe -m llm_model.gguf --mmproj mmproj_model.gguf
  -ngl <gpu_layers> -c <context> -fa on -np 1 --host 127.0.0.1 --port <port>
  [--jinja] [--reasoning off] [-t <threads>]
```

- 端口被占用时从 `describe.port` 起向后寻找可用端口；`/health` 轮询至就绪（`load_timeout_sec`）；
- `/v1/chat/completions` 发送多模态消息（文本 + 每帧 `image_url` data URI）；采样参数取 `temperature/top_p/max_tokens/repeat_penalty`；`enable_thinking=false` 时附带 `chat_template_kwargs`；
- 仅用标准库 `urllib`，无额外 pip 依赖；`finish_reason=="length"` 时告警截断；结束 `terminate` 进程并关闭日志文件。

### 7.4 落盘（`describe/runner.py`）

- `video_prompt_zh.txt`（shot 根目录）：模型输出的中文六段式 prompt（`clean_output` 去除思考块 / 代码围栏 / 特殊 token）；
- `detailed/video_prompt.json`：

```json
{
  "schema_version": "1.0",
  "source": "样例.mp4",
  "shot_id": "shot_0001",
  "duration_sec": 5.0,
  "input_mode": "storyboard",
  "storyboard": "detailed/storyboard_1080p.jpg",
  "storyboard_source": "scene_reference.jpg",
  "storyboard_cached": true,
  "grid_rows": 3,
  "grid_cols": 3,
  "frame_fps": 0.0,
  "frame_count": 0,
  "frame_select": false,
  "frame_dir": "",
  "frames": [],
  "speakers": ["speaker_1"],
  "subtitles": [{"speaker": "speaker_1", "start_sec": 0.2, "end_sec": 1.4, "text": "..."}],
  "language": "zh-CN",
  "prompt": "...",
  "engine": "llama.cpp",
  "model": "llm_model.gguf",
  "missing_sections": [],
  "elapsed_sec": 63.3
}
```

- `input_mode` 记录实际使用的模式（`storyboard` / `frames`；故事图缺失回退时记 `frames`）；`storyboard` 记录实际喂给模型的图像相对路径（故事图被缩放时为 `detailed/storyboard_1080p.jpg`，否则为 `scene_reference.jpg`），`storyboard_source` / `storyboard_cached` 记录原图与是否走了缓存；`grid_rows` / `grid_cols` 仅 `storyboard` 模式有意义；
- `frames` 模式下 `frame_fps` / `frame_count` / `frame_select` / `frame_dir` / `frames` 记录采样帧；`frame_select` 记录是否走了清晰帧选择；`missing_sections` 记录缺失的六段标题；`keep_frames=false` 且 `frames` 模式时写出后删除 `frames/`；
- 单片段失败只计数并告警，不影响其他片段；引擎就绪由阶段 0 检查。

---

## 八、阶段 5：清理、统计与阶段计时

- **清理**：`--clean-segments` 时在 ASR / 描述读取完毕后删除 `cleanup` 中登记的 `seg_XXXX.mp4`。
- **统计汇总**：`done:` 日志输出 `ok/invalid/error/skipped/fallback`、`separation_ok/skip/error`、`asr_ok/error/segments`、`describe_ok/error/skipped`。
- **阶段计时**：`app/timing.py:StageTimer` 线程安全累加，结束时按耗时降序打印 `stage timing`，包含 `phase.*`（scene_reference / separation / asr / describe / cleanup 的墙钟耗时）与片段级 `source.*` / `prepare.*` / `finalize.*` 等。

---

## 九、输出目录与 Schema 汇总

```text
output/
    run.log
    llama_server.log
    voiceprint_center.json
    segments/
        <源文件名去扩展>/
            segments.json
            transcript.json
            audio/
                original.wav / vocals.wav / instrumental.wav / separation.json
            speakers/<speaker>/
                seg_0001.wav / seg_0001.json / <speaker>_global.json
            persons/                    # 人物提取与归类（见第十四节）
                metadata.json
                person_001/ person_001_f0000123.jpg ...
                unknown/track_0001/ unknown_f0000200.jpg ...
            shot_0001/
                seg_0001.mp4
                scene_reference.jpg
                seg_0001.srt
                original.wav
                frames/frame_00001.png ...   # 仅 describe.input_mode=frames 时
                video_prompt_zh.txt
                detailed/
                    asr.json
                    video_prompt.json
                    storyboard_1080p.jpg   # 仅 storyboard 模式且故事图需缩放时
                    vocals.wav
                    instrumental.wav
                    debug/            # 仅 --debug
            shot_0002/ ...
```

- 同一源文件内 `shot_XXXX` 与 `seg_XXXX` 序号一致，从 `0001` 起。
- 无效片段（全黑 / 全白 / 低信息量等）仍创建 `shot_XXXX/` 与 `detailed/`，但不生成故事图，也不参与描述。
- 除 `output/` 外，整视频 SRT 写到**输入视频同目录**。

Schema 版本号：`segments.json` / `transcript.json` / `asr.json` / `separation.json` / `video_prompt.json` / `voiceprint_center.json` 均为 `"1.0"`。

---

## 十、配置与阈值（`config.json`）

所有阈值集中在包根 `config.json`，并与 `app/config.py:DEFAULT_CONFIG` 保持一致。以下为当前实际配置块（已移除 `classification` / `subject` / `enhancement` / `keyframe`）：

```json
{
  "frame_quality": {
    "black_mean": 15, "black_std": 10,
    "white_mean": 240, "white_std": 10,
    "low_info_std": 8, "blur_ratio_of_median": 0.4,
    "exposure_ideal_min": 80, "exposure_ideal_max": 200
  },
  "trim": { "trim_tolerance_frames": 2, "min_valid_frames": 2, "invalid_black_ratio": 0.9 },
  "sampling": { "k_short": 7, "k_normal": 10, "k_long": 14 },
  "performance": { "workers": 0, "opencv_threads": 0 },
  "matching": {
    "max_features": 2000, "ratio_test": 0.75, "ransac_reproj_threshold": 3.0,
    "min_match_count": 8, "min_inlier_count": 8, "low_inlier_ratio": 0.3
  },
  "output": { "jpeg_quality": 92, "max_side_frame": 1920, "max_side_reference": 4096 },
  "reference": {
    "grid_rows": 3, "grid_cols": 3, "grid_gap_px": 10, "grid_margin_px": 10,
    "select_tolerance_ms": 50
  },
  "segmentation": {
    "enabled": true, "sensitivity": "medium",
    "adaptive_thresholds": { "low": 4.5, "medium": 3.0, "high": 2.0 },
    "min_scene_len_sec": 0.5, "min_segment_sec": 1.0, "max_segment_sec": 15.0,
    "max_split_iterations": 5, "long_shot_subdetect": true,
    "long_shot_subdetect_sensitivity": "high",
    "segment_encode": "auto", "copy_duration_tolerance_sec": 0.1,
    "keep_segments": true
  },
  "separation": {
    "enabled": true, "model": "fast", "models": "models/separator",
    "device": "gpu", "sample_rate": 44100
  },
  "asr": {
    "enabled": true, "engine": "qwen3-asr", "device": "gpu", "dtype": "bfloat16",
    "models": "models/asr",
    "asr_model": "Qwen3-ASR-1.7B", "forced_aligner": "Qwen3-ForcedAligner-0.6B",
    "vad_model": "fsmn-vad", "speaker_model": "campplus", "verify_model": "eres2netv2",
    "language": "auto", "max_new_tokens": 2048, "max_inference_batch_size": 4,
    "vad_max_segment_sec": 60, "min_speech_sec": 0.6,
    "min_archive_sec": 1.0, "max_archive_sec": 30.0,
    "local_speaker_threshold": 0.55,
    "diarize_window_sec": 1.2, "diarize_hop_sec": 0.4, "diarize_smooth_radius": 1,
    "sentence_merge_gap_sec": 0.8,
    "voiceprint_sim_threshold": 0.30, "verify_sim_threshold": 0.55,
    "voiceprint_center": "", "sample_rate": 16000, "debug": false
  },
  "persons": {
    "enabled": true, "models": "models/person",
    "pose_model": "yolo11x-pose.pt", "face_pack": "buffalo_l", "device": "gpu",
    "fps_target": 1.0, "candidates_per_window": 5,
    "det_conf": 0.45, "kp_conf": 0.3,
    "min_face_px": 48, "face_det_score": 0.55, "face_embed_score": 0.1,
    "cluster_threshold": 0.45, "cluster_ambiguous_low": 0.35,
    "dedup_threshold": 0.92, "max_per_person": 300,
    "crop_margin_ratio": 0.05, "min_crop_width": 32, "min_crop_height": 64,
    "jpeg_quality": 92, "keep_unknown": true,
    "debug": false
  },
  "describe": {
    "enabled": true, "engine": "llama.cpp",
    "bin": "llama_cpp/llama_bin", "server": "llama-server.exe",
    "models": "models/llm",
    "model": "llm_model.gguf", "mmproj": "mmproj_model.gguf",
    "host": "127.0.0.1", "port": 8080,
    "context": 50000, "gpu_layers": 999,
    "flash_attn": true, "jinja": true, "reasoning": "off", "enable_thinking": false,
    "threads": 0, "input_mode": "storyboard", "storyboard_max_side": 1920,
    "frame_fps": 2.0, "frame_max_side": 1920, "frame_dir": "frames",
    "frame_format": "png", "frame_select": true, "frame_candidates": 3,
    "max_frames": 0, "keep_frames": true,
    "temperature": 0.2, "top_p": 0.9, "max_tokens": 4096, "repeat_penalty": 1.1,
    "load_timeout_sec": 900, "request_timeout_sec": 1800, "limit": 0
  }
}
```

### 10.1 关键字段说明

**`segmentation`**

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | `--no-split` 覆盖为关闭 |
| `sensitivity` | `medium` | 阈值档位 |
| `adaptive_thresholds` | 4.5 / 3.0 / 2.0 | 三档 AdaptiveDetector 阈值 |
| `min_scene_len_sec` | `0.5` | 送检测的最短场景长度（秒） |
| `min_segment_sec` / `max_segment_sec` | `1.0` / `15.0` | 最终片段时长区间（秒） |
| `max_split_iterations` | `5` | 迭代修正上限，超过走 `_force_fix` |
| `long_shot_subdetect` / `..._sensitivity` | `true` / `high` | 超长镜头二次检测 |
| `segment_encode` | `auto` | `copy` / `reencode` / `auto` |
| `copy_duration_tolerance_sec` | `0.1` | `auto` 下 copy 时长误差容忍 |
| `keep_segments` | `true` | `--clean-segments` 覆盖为删除 |

**`separation`**：`enabled`（`--no-separate`）、`model`（fast/balanced/best/dereverb）、`models`（模型目录）、`device`（gpu/cpu）、`sample_rate`（44100）。

**`asr`**：`enabled`（`--no-asr`）、`engine`、`device`（gpu/cpu）、`dtype`、`models`、各模型目录、`language`、`max_new_tokens`、`max_inference_batch_size`、`vad_max_segment_sec`、`min_speech_sec`、`min_archive_sec`/`max_archive_sec`（1~30s）、`local_speaker_threshold`、`diarize_window_sec`/`diarize_hop_sec`/`diarize_smooth_radius`、`sentence_merge_gap_sec`、`voiceprint_sim_threshold`、`verify_sim_threshold`、`voiceprint_center`、`sample_rate`、`debug`。

**`describe`**：`enabled`（`--no-describe`）、`engine`、`bin`/`server`、`models`、`model`/`mmproj`、`host`/`port`、`context`/`gpu_layers`、`flash_attn`/`jinja`/`reasoning`/`enable_thinking`/`threads`、`input_mode`（`storyboard`（默认，直接用 3x3 故事图）/ `frames`（采样帧））、`storyboard_max_side`（故事图推理缓存长边上限，默认 1920=1080P）、`frame_fps`/`frame_max_side`/`frame_dir`/`frame_format`/`frame_select`/`frame_candidates`/`max_frames`/`keep_frames`、`temperature`/`top_p`/`max_tokens`/`repeat_penalty`、`load_timeout_sec`/`request_timeout_sec`、`limit`。

**`persons`**（见第十四节）：`enabled`（`--no-persons`）、`models`（模型根目录）、`pose_model`、`face_pack`、`device`（`--persons-device`）、`fps_target`/`candidates_per_window`、`det_conf`/`kp_conf`、`min_face_px`/`face_det_score`/`face_embed_score`、`cluster_threshold`/`cluster_ambiguous_low`、`dedup_threshold`/`max_per_person`、`crop_margin_ratio`/`min_crop_width`/`min_crop_height`、`jpeg_quality`、`keep_unknown`、`debug`、`limit`（`--persons-max`）。

**`reference`**：`grid_rows`/`grid_cols`（3×3）、`grid_gap_px`/`grid_margin_px`（10）、`select_tolerance_ms`（50，选帧相对均匀目标时刻的最大时间偏差）。

**`performance`**：`workers`（0=自动 `min(8,CPU)`，`--workers` 覆盖）、`opencv_threads`（0=默认，`--opencv-threads` 覆盖；仅影响性能）。

---

## 十一、工程结构与模块接口

### 11.1 目录结构

```text
ref_forge/
    启动.bat / dev.bat / 清理生成内容.bat / 下载模型.bat
    config.json / requirements.txt
    AGENTS.md / readme.md / REQUIREMENTS.md / PROJECT_NOTES.md
    app/
        main.py              # CLI 入口与流程编排
        config.py            # 配置加载 + DEFAULT_CONFIG
        interactive.py       # --interactive 交互配置 + 片段枚举（供描述选择）
        io_utils.py          # 路径 / JSON / JPEG / 视频枚举 / shot_detail_dir
        logging_utils.py     # 控制台 + output/run.log 日志
        timing.py            # StageTimer 阶段计时
        pipeline/
            segmentation.py  # PySceneDetect + 迭代时长修正
            segment_export.py# ffmpeg copy/reencode + ffprobe
            video_io.py      # 元信息 / 灰度抽帧 / 彩色回读
            quality.py / trim.py / sampling.py / features.py
            motion.py        # 单对估计 + 尺度轨迹累积（仅供 --debug 诊断）
        reference/
            generator.py / storyboard.py
        speech/
            audio.py / separation.py / models.py / diarize.py
            transcribe.py / voiceprint.py / export.py / runner.py
        describe/
            frames.py / prompt.py / client.py / runner.py
            prompts/         # storyboard_* / frames_* 两套六段式模板
        persons/             # 人物提取与归类（阶段 4；见第十四节）
            __init__.py / models.py / sampler.py / detect.py / face.py
            associate.py / cluster.py / crop.py / export.py
        assets/              # 预留（未使用）
    bin/                     # ffmpeg.exe / ffprobe.exe / ffplay.exe
    llama_cpp/               # llama_bin/（可执行文件）
    models/                  # asr/ 语音 + separator/ 分离 + person/ 人物 + llm/ llama.cpp
    python/                  # 包内可移植 Python 运行时
    scripts/                 # 开发与验收脚本
    input/ / output/         # 输入 / 输出
    skill/                   # H3 prompt 写作规范
    demo/                    # 参考项目（只读）
```

### 11.2 关键接口（实际签名）

```python
# pipeline/segmentation.py
scenedetect_available() -> bool
sensitivity_threshold(cfg, sensitivity) -> float
detect_shot_cuts(path, fps, sensitivity, min_scene_len_sec, cfg) -> (cuts | None, meta)
make_subdetector(path, sensitivity, min_scene_len_sec, cfg) -> Callable | None
build_segments(total_frames, fps, cuts, cfg, subdetect_fn=None) -> (list[Segment], meta)
# Segment: index,start_frame,end_frame,start_sec,end_sec,duration_sec,method,
#          below_min,shot_id,file,shot_dir

# pipeline/segment_export.py
probe_duration(path) -> float
export_segment(src, start_sec, end_sec, dest, encode_mode, cfg) -> str | None

# pipeline/video_io.py
read_metadata(path) -> VideoMeta(fps,width,height,frame_count,duration_sec)
decode_gray_frames(path, long_side=960, start_frame=0, end_frame=None) -> list[np.ndarray]
read_color_frame(path, frame_index, max_side=None) -> np.ndarray
make_frame_reader(path, max_side=None, frame_offset=0, frame_size=None) -> Callable[[int], np.ndarray]

# pipeline/quality.py: assess_frames(gray_frames, cfg) -> FrameQuality
# pipeline/trim.py:    find_valid_range(quality, cfg) -> (start, end, status) / invalid_reason(quality)
# pipeline/sampling.py: choose_k(duration, cfg) / sample_frames(start, end, duration, cfg) -> list[int]
# pipeline/features.py: extract_features(gray, cfg) -> Features / match_pair(fi, fj, cfg) -> (pts_i,pts_j,count)|None
# pipeline/motion.py:   estimate_pair(pts_i, pts_j, shape, sharpness, cfg) -> PairEvidence
#                       accumulate_base(pairs, sample_indices, shape, cfg, sharpness_median,
#                                       low_sharpness_ratio, duration_sec) -> AggregateEvidence
# reference/generator.py: reference_indices(agg, cfg) -> list[int]
#                         generate(agg, frame_reader, fallback_frames, cfg) -> np.ndarray | None

# speech/
separation_available(cfg) -> (bool, list[str])
run_separation_phase(jobs, cfg, logger, stats=None) -> {video: info}
available(cfg) -> (bool, list[str])
run_phase(jobs, output_root, cfg, logger, debug=False) -> dict
shutdown()
# speech/separation.py: TIERS / resolve_device / model_path / extract_original_audio / cut_wav
# speech/audio.py:      extract_audio / has_audio_stream / read_wav_mono / write_wav / cut_clip / wav_levels
# speech/models.py:     configure_language / resolve_device / torch_dtype / class SpeechModels
# speech/diarize.py:    detect_speech / extract_embedding / cluster_embeddings
#                       make_windows / smooth_labels
# speech/transcribe.py: transcribe_clips / split_sentences_with_words / token_count / join_text
# speech/voiceprint.py: class VoiceprintCenter(cfg, output_root).match_or_create(cam, eres, video)
# speech/export.py:     write_srt / safe_speaker_name / segment_payload

# describe/
available(cfg) -> (bool, list[str])
run_phase(jobs, output_root, cfg, logger, debug=False) -> dict
# describe/frames.py: fit_long_side / extract_frames / probe_video / frame_paths
# describe/prompt.py: normalize_mode / system_prompt(mode) / reference_mapping_text /
#                     build_user_prompt(mode, duration, fps, frames, subtitles, speakers, grid)
#                     clean_output / missing_sections / SECTION_HEADINGS
# describe/client.py: model_path / mmproj_path / class LlamaServer(start/chat/close)
```

### 11.3 并行调度

- 每个源视频的片段通过**单个 `ThreadPoolExecutor`** 并行处理（池大小 = `performance.workers`）；片段导出与故事图生成共用该池；镜头分割（PySceneDetect）按源文件串行。
- 线程安全：SIFT/ORB 检测器为**线程本地**缓存（`features.py`）；`StageTimer` 累加加锁；片段统计在 `pool.map` 返回后于主线程汇总。
- 产物与 worker 数、OpenCV 线程数、调度顺序无关：故事图选帧只取决于片段帧质量与帧号（均匀时间选帧、就近取清晰帧），不再依赖 `agg.cum_scale`。
- 阶段与片段级耗时通过 `StageTimer` 汇总打印。

---

## 十二、依赖与独立一键包

### 12.1 硬性交付要求

1. 自带独立 Python 运行时，不依赖目标机 Python / conda；
2. 双击 `.bat` 启动；
3. 整目录拷贝即可迁移，路径基于包根动态解析；
4. 免安装、免配置、免联网。

### 12.2 运行时与依赖

- 打包方式：python.org embeddable 运行时（`python/`，3.10.11）+ get-pip；`python310._pth` 开启 `import site` 并加入 `Lib\site-packages` 与 `..\app`。
- 基础依赖：`opencv-python-headless` / `numpy` / `Pillow` / `scenedetect` / `click` / `platformdirs`。
- 语音识别：`torch==2.9.1+cu128` / `torchaudio==2.9.1+cu128` / `funasr` / `qwen-asr` / `modelscope` / `transformers` / `librosa` / `soundfile`；模型在 `models/asr/`。
- 人声分离：`audio-separator==0.47.0`（`--no-deps`）+ `torchvision==0.24.1+cu128` + `onnxruntime-gpu` + 若干固定版本依赖；模型在 `models/separator/`。
- 视频描述：`llama_cpp/llama_bin/`（`llama-server.exe` + DLL）+ `models/llm/`（`llm_model.gguf` + `mmproj_model.gguf`）。
- 人物提取：`ultralytics==8.3.253`（`--no-deps` 安装，禁止其改动 `opencv-python`/`torch`）+ `lap==0.5.13`；人脸侧用已装 `onnxruntime-gpu` 直驱 `models/person/face/buffalo_l/`（不引入 `insightface`）；YOLO 权重 `models/person/detect/yolo11x-pose.pt`。
- 完整版本见 `requirements.txt`；安装脚本 `scripts/setup_env.py`（经根目录 `一键环境搭建.bat` 或 `scripts/setup_runtime.bat` 调用）；一键下载模型 `下载模型.bat`（内部调用 `scripts/download_models.py`、`scripts/download_separation_models.py`、`scripts/download_person_models.py`、`scripts/download_anime_person_models.py`）。
- **从 GitHub 克隆后的首次环境搭建**：仓库不含运行时与离线资产（见 `.gitignore`：`python/`、`models/`、`bin/`、`llama_cpp/` 均忽略），需联网双击 `一键环境搭建.bat`，依次：`scripts/bootstrap_python.ps1` 引导 embeddable Python 3.10.11 + get-pip → `scripts/setup_env.py` 安装依赖 → `scripts/download_runtime.py` 下载 ffmpeg / llama.cpp / GGUF → 复用 `download_*_models.py` 下载全部模型。
- **多源测速**：`scripts/download_utils.py` 提供 `probe_speed` / `pick_fastest` / `download_smart`，对每个资产的多个候选源（GitHub 加速镜像、ModelScope、hf-mirror 等）**按实测网速自动选择最快源**，失败自动切换；同组选源在进程内缓存。已存在文件按大小校验跳过，支持 `.part` 断点续传。
- **PyTorch 源测速**：`setup_env.py` 的 `pick_torch_index()` 在 官方 `download.pytorch.org/whl/cu128`、南大 `mirrors.nju.edu.cn/pytorch/whl/cu128`、上交 `mirror.sjtu.edu.cn/pytorch-wheels/cu128` 三个 PEP503 索引间，**探测同一 torch wheel 的真实下载速度**（而非索引导航页）选最快者作为 `torch`/`torchaudio`/`torchvision` 的 `--index-url`；阿里云 `mirrors.aliyun.com/pytorch-wheels/cu128` 为扁平 find-links，仅作 torchvision 的 `-f` 兜底。实测南大 / 上交约为官方源的数倍。
- **ffmpeg 源测速与格式**：`download_runtime.py` 在 `https://registry.npmmirror.com/-/binary/ffmpeg-builds/`（国内高速，实测 >15MB/s）、BtbN `releases/latest/download/ffmpeg-master-latest-win64-gpl.zip`、gyan.dev `ffmpeg-release-essentials.zip` 三源间按实测网速选最快。npmmirror **无 `latest/` 别名**，其资产为 `ffmpeg-<ver>-win32-x64-gpl.tar.xz`，由 `_npmmirror_ffmpeg_url()` 先读索引 JSON 解析最新版本目录再拼资产名；`_extract_flat()` 按实际内容自动识别 zip / tar.xz（`tarfile.is_tarfile`）并平铺解压出 `ffmpeg.exe` / `ffprobe.exe` / `ffplay.exe`（均为静态 GPL，含 libx264）。
- **视频描述主模型**：默认 `unsloth/Qwen3.5-4B-GGUF` 的 `Qwen3.5-4B-UD-Q4_K_XL.gguf` + `mmproj-F16.gguf`（ModelScope 优先，自动切换 HF 镜像），落盘为 `models/llm/llm_model.gguf` 与 `mmproj_model.gguf`。

### 12.3 启动脚本

```bat
@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%~dp0python;%~dp0python\Scripts;%~dp0bin;%PATH%"
if "%~1"=="" (
    "%~dp0python\python.exe" "%~dp0app\main.py" --interactive
) else (
    "%~dp0python\python.exe" "%~dp0app\main.py" %*
)
if errorlevel 1 pause
pause
```

无参数双击 `启动.bat` 时进入交互模式（`--interactive`）；带参数时按原 CLI 执行。`dev.bat` 同样设置包内环境后执行任意 python 命令。embeddable 通过 `._pth` 已把 `app/` 与 site-packages 加入 `sys.path`，故不设置 `PYTHONHOME` / `PYTHONPATH`。

**脚本编码/换行（强制，见 `PROJECT_NOTES.md` P48）**：所有 `.bat` **只允许 ASCII 字符（不含中文）**，保存为 **CRLF、无 BOM**。cmd.exe 按系统 ANSI 代码页读取 `.bat`（与 `chcp` 无关），文件内的非 ASCII 会解析错乱或乱码（UTF-8 解析失败；GBK 在 `chcp 65001` 下显示为乱码），故 `.bat` 内一律用英文提示；`chcp 65001` 仅用于让后续 Python（`PYTHONUTF8=1`）与 PowerShell 的中文输出正确显示。`scripts/bootstrap_python.ps1` 保存为 **UTF-8 带 BOM、CRLF**，供 PowerShell 5.1 正确解码中文。修改脚本时须保持该编码与换行。

---

## 十三、验收标准与测试

### 13.1 功能验收

1. 每个片段目录按第九节结构生成文件；无效片段只建目录、不生成故事图；不再生成 `motion_meta.json`。
2. 故事图 `scene_reference.jpg` 为有效 3x3 九宫格，9 格按时间升序、每格相对均匀目标时刻的时间偏差 ≤ `reference.select_tolerance_ms`（默认 50ms）。
3. 多镜头长视频被正确切分；迭代后每段时长落在 `[1s, 15s]`（除非输入总时长不足 1s）。
4. 分割依赖缺失 / 检测异常 / 无切点时，仍按 `max_segment_sec` 等长切分且不崩溃。
5. 有语音的源视频生成整视频 SRT、`transcript.json`、按人归档、各 `detailed/asr.json`、`voiceprint_center.json`；跨视频同一声纹合并为同一 `speaker_N`。
6. 每个 `detailed/asr.json` 只含本片段可听语音；跨切点语音按多数重叠唯一归属。
7. 有音轨的源视频生成 `audio/` 三轨 + `separation.json`；ASR 使用 `vocals.wav`；无音轨跳过不报错。
8. 视频描述生成 `frames/`、`video_prompt_zh.txt`（六段式中文）、`detailed/video_prompt.json`；引擎缺失时整段跳过并告警。
9. 处理过程不崩溃；单片段异常不影响其余片段。
10. `--interactive`：一次性收集分割时长 / 是否分割 / 是否人物提取 / 输入图像 / 描述片段；选否分割时原视频整体作为单片段且故事图 / 分离 / ASR 照常；按编号选择时仅对选中片段生成描述，选 `none` 时不加载描述模型；交互枚举的片段被复用、不重复分割。
11. 人物提取（见第十四节）：每源视频产出 `persons/metadata.json` + `person_XXX/` + `unknown/`；图片为原帧矩形、原分辨率、`person_XXX_f%07d.jpg` 可溯源；`max_per_person` 截断、近重复已去重；引擎缺失 / 单源异常跳过并告警；新增依赖不改变原有阶段产物与环境。

### 13.2 验收脚本

| 脚本 | 覆盖内容 | 命令 |
|---|---|---|
| `run_segmentation_test.py` | 对 `input/` 真实视频校验时长约束、`scene_reference.jpg`、`detailed/` 结构 | `dev.bat scripts\run_segmentation_test.py` |
| `run_separation_test.py` | `input/` 中含音轨视频的三轨 + 元数据（默认 best+gpu） | `dev.bat scripts\run_separation_test.py` |
| `run_asr_test.py` | `input/` 真实视频：SRT / transcript / 按人归档 / 声纹中心 / `source=vocals` | `dev.bat scripts\run_asr_test.py` |
| `run_describe_test.py` | 抽帧（2fps / 清晰帧 / ≤1080P / 独立目录）+ 六段模板；`--llm` 端到端 | `dev.bat scripts\run_describe_test.py [--llm]` |
| `run_persons_test.py` | 人物产物结构、命名可溯源、原分辨率矩形、去重/Top-300、`unknown/`、环境未变 | `dev.bat scripts\run_persons_test.py` |
| `clean_generated.py` | 预览 / 清理生成物（默认 dry-run） | `dev.bat scripts\clean_generated.py [--yes]` |

- 验收素材统一取自 `input/`：**有什么视频就用什么视频**（`acceptance_common.py` 负责发现并复制素材），不再生成或依赖固定命名的合成视频。
- 验收脚本写入系统临时目录，不污染 `input/` / `output/`；既有脚本一般传 `--no-describe`（避免加载 4B 模型）。
- `make_test_videos.py` 仅作为可选合成素材工具保留，验收流程不再调用。

### 13.3 打包验收

在未安装 Python、无网络的 Windows 机器上双击 `.bat` 完成一次完整处理；拷贝到任意路径（含其他盘符）后仍可运行；错误时窗口保留并打印可读信息。

---

## 十四、人物提取与归类（新增）

> 面向开发的完整需求另见 `PERSON_REQUIREMENTS.md`；本节为已实现行为的汇总。

### 14.1 定位与流程

**阶段 4**，按源视频**整片**独立处理，位于统一流程 `镜头分割 → 故事图 → 人声分离 → 语音识别` **之后**、逐片段描述之前（见第二节顺序图）；`--interactive` 第 4 项可开关，默认执行。该功能**不服务**视频描述，`describe` 行为不变；`persons` 与 ASR/描述阶段按顺序执行，模型用完即卸，不同时占用显存。按源视频独立，不跨视频合并身份。

按 `persons.domain` 路由两套后端：`real`（真人，下述流程与阈值）与 `anime`（动漫，2D/3D 通用，见 14.8）；两套共享 `sampler`/`crop`/`cluster`/`export` 框架，仅替换检测/人脸/嵌入/跟踪实现并使用各自的阈值组。

```text
① 智能抽帧(OpenCV + Laplacian)              # 每 1s 窗口取最清晰帧 -> 代表帧序列
② YOLO11-pose 检测 + BoT-SORT 跟踪           # person 框 / track_id / COCO 关键点
③ 人体裁剪图上 SCRFD 人脸检测 + ArcFace 512 维嵌入
④ 人脸归属到 person 框 -> 每个 track 的人脸均值特征
⑤ 层级标注(关键点) + 矩形裁剪 + 质量过滤
⑥ 跨 track 聚类(余弦 ≥ 0.45，约束感知合并；0.35~0.45 二次复核)
⑦ 簇内去重(> 0.92) + Top-300
⑧ 输出 persons/person_XXX/ + unknown/ + metadata.json
```

### 14.2 输出目录

```text
output/segments/<源文件名去扩展>/
    persons/
        metadata.json                     # schema_version="1.0"
        person_001/                       # 一个真实人物一个文件夹（多张截图）
            person_001_f0000123.jpg       # 身份 + 源全局帧号，可溯源
        person_002/ ...
        unknown/                          # 全程无脸、无法确定身份的 track
            track_0001/ unknown_f0000200.jpg ...
            track_0002/ ...
```

- `person_XXX` **按图片数量降序**编号（并列按首次出现）；图片保留 ≤ `max_per_person`，JPEG 质量 `jpeg_quality`。
- `metadata.json`：`schema_version` / `source` / 阈值 / `person_count` / `unknown_track_count` / `persons`（`num_images`、`tracks_merged`、`time_range`（秒）、`frame_range`（源帧号）、`representative`、`levels`（`full_body`/`upper_body`/`torso`/`chest`/`head`/`unknown` 计数））/ `unknown`（`source_track_id`、`num_images`、`time_range`）。

### 14.3 功能需求摘要

- **FR-1 抽帧**：每 1s 窗口内按时间均匀取最多 `candidates_per_window=5` 个候选帧（原生帧率 <5 时取全部），取灰度 Laplacian 方差最大者为代表帧；全为黑/白/低信息则跳过该窗口；帧号记源全局帧号。
- **FR-2 检测跟踪**：仅 `person`，`det_conf=0.45`，BoT-SORT 输出 `track_id`；track 仅用于帧间锚定。BoT-SORT 未给 id 时按中心就近继承最近 track，避免 1fps 断链。
- **FR-3 身份判定**：以**人体裁剪图**为主做 SCRFD 人脸检测，并在**整帧**补充检测、合并两侧候选（裁剪图对贴合人物更稳，整帧对远景/特殊角度召回更高，二者取检测分最高者）；`min_face_px=48` 门控，人脸中心须落在该 person 检测框内，重叠框归属"包含中心面积最小"者；**双阈值**：检测分 ≥ `face_det_score=0.55` 为高质量人脸；`[face_embed_score=0.10, 0.55)` 为低分人脸，**仅当某 track 无高质量人脸时才用其生成 track 特征**（恢复风格化/遮挡人脸，又不过度污染正常视频）；**track 在硬切处强制断开**（复用 PySceneDetect 切点），避免不同角色被串成一条 track；ArcFace 512 维嵌入取每个 track 的**均值**为身份特征；跨 track 余弦 ≥ `cluster_threshold=0.45` 合并，歧义区间 `[0.35, 0.45)` 用各自最高质量单张人脸二次复核（阈值 0.45）；**同帧不同 track 互斥不得合并**；聚类用**约束感知的凝聚式合并**（非朴素并查集）。
- **FR-4 裁剪保存**：裁剪矩形 = `union(检测框, 可见关键点外接框)` 外扩 `crop_margin_ratio=5%`，越界截断；原分辨率、不缩放；层级 = 身体链上可见的最低关节（任一侧可见，`kp_conf=0.3`）。
- **FR-5 过滤去重**：丢弃宽 <`min_crop_width=32` 或高 <`min_crop_height=64` 者；`score = 簇内归一化清晰度 × 检测置信度 × 人脸因子`；簇内去重只对**双方都有本人脸嵌入**的图按余弦 > `dedup_threshold=0.92` 判近重复，无嵌入的图（背影/侧脸）改用 32×32 缩略图做近像素重复抑制（避免用 track 均值把不同姿态误判为重复）；每人按 `score` 截断至 `max_per_person=300`。
- **FR-6 无脸兜底**：track 其他帧有脸则背影/侧脸照常入库；全程无脸入 `unknown/track_XXXX/`（按首次出现顺序编号），不参与聚类；仅同 track 内近似抑制。**绝不把无脸 track 盲目并入人物**（否则会把不同角色混入同一文件夹）；不同角色的隔离由"**track 在硬切处断开** + 人脸聚类"共同保证。
- **FR-7**：不服务视频描述，不与 `speakers` 对齐。

### 14.4 配置（`persons` 块）

```json
"persons": {
  "enabled": true, "domain": "anime", "models": "models/person",
  "pose_model": "yolo11x-pose.pt", "face_pack": "buffalo_l", "device": "gpu",
  "fps_target": 1.0, "candidates_per_window": 5,
  "det_conf": 0.45, "kp_conf": 0.3,
  "min_face_px": 48, "face_det_score": 0.55, "face_embed_score": 0.1,
  "cluster_threshold": 0.45, "cluster_ambiguous_low": 0.35,
  "dedup_threshold": 0.92, "max_per_person": 300,
  "crop_margin_ratio": 0.05, "min_crop_width": 32, "min_crop_height": 64,
  "jpeg_quality": 92, "keep_unknown": true,
  "debug": false, "limit": 0,
  "anime": {
    "models": "models/person/anime",
    "person_onnx": "person/person_detect_v1.1_m/model.onnx",
    "face_onnx": "face/face_detect_v1.4_s/model.onnx",
    "clip_model": "clip/clip-vit-large-patch14",
    "input_size": 640, "nms_iou": 0.5, "head_body_ratio": 7.0,
    "det_conf": 0.3, "min_face_px": 32,
    "face_det_score": 0.5, "face_embed_score": 0.3,
    "cluster_threshold": 0.8, "cluster_ambiguous_low": 0.72,
    "dedup_threshold": 0.88, "crop_margin_ratio": 0.1
  }
}
```

### 14.5 模块与接口

```python
# app/persons/__init__.py
available(cfg) -> (ok, missing)
run_phase(jobs, output_root, cfg, logger, stats=None, debug=False) -> dict
# jobs: [{"video": <源视频路径>, "source_dir": <output/segments/<stem>>}]
# app/persons/models.py:  domain_of / effective_persons_cfg / build_models / available
#                         PersonModels: yolo()/face()/make_tracker()；AnimeModels: person_detector()/face()/make_tracker()
#                         embed_mode: "face"（真人）/ "person"（动漫，始终嵌入人物裁剪图）
# app/persons/sampler.py: select_representative_frames(path, cfg) -> [(frame_idx, sec)]
# app/persons/detect.py:  PersonTracker.update(frame, whole_faces=None) -> [det...]  # tracker_file 可换
# app/persons/face.py:    FaceEngine.detect / embed / cosine_similarity
# app/persons/anime/onnx_detector.py: OnnxYoloDetector.detect（YOLO ONNX 直驱）
# app/persons/anime/face.py:   AnimeFaceEngine.detect / embed（动漫人脸 ONNX + CLIP）
# app/persons/anime/embed.py:  ClipEmbedder.embed
# app/persons/anime/detect.py: AnimePersonTracker.update（动漫人物 ONNX + 人脸锚点扩展）
# app/persons/crop.py:    crop_rect / classify_level / score_of
# app/persons/cluster.py: build_tracks / conflict_pairs / cluster_tracks / dedup_records
# app/persons/export.py:  write_outputs
```

### 14.6 依赖与模型

- `ultralytics==8.3.253`（`--no-deps` 安装）+ `lap==0.5.13`；**禁止**安装 `opencv-python`、升级 `torch`/`torchvision`/`numpy`。
- 人脸侧**不引入 `insightface`**：用已装 `onnxruntime-gpu` 直驱 buffalo_l 的 `det_10g.onnx`（SCRFD）+ `w600k_r50.onnx`（ArcFace），自实现解码、NMS 与 5 点对齐；显存不可用时回退 CPU。
- 模型：`models/person/detect/yolo11x-pose.pt`、`models/person/face/buffalo_l/{det_10g.onnx,w600k_r50.onnx}`；下载优先级 GitHub → hf-mirror，运行期全离线。
- 动漫模式（14.8）**不新增第三方依赖**：复用既有 `transformers`（CLIP）与 `onnxruntime`（直驱动漫人物/人脸检测 ONNX）；模型 `models/person/anime/{person/person_detect_v1.1_m, face/face_detect_v1.4_s, clip/clip-vit-large-patch14}`，由 `scripts/download_anime_person_models.py` 下载（deepghs + CLIP 均走 hf-mirror）。
- 安装后校验 `scripts/verify_person_env.py`（`cv2`/`torch`/`numpy` 版本与 headless 不被破坏）。

### 14.7 验收

1. 同一人物（含换装、跨镜头）图片落在同一 `person_XXX/`；不同人物不混入。
2. 每张图为该人物在原帧中的完整矩形、原分辨率。
3. 清晰（无运动模糊主导）、近重复已去除、按 `score` 截断 ≤ 300。
4. `unknown/` 仅含全程无脸 track，按 track 子目录依次递增。
5. `metadata.json` 字段完整、可溯源到源帧号（`f%07d`）。
6. 引擎缺失 / 单源异常跳过并告警，进程不崩溃；新增依赖不改变原环境。
7. 脚本 `scripts/run_persons_test.py` 通过；动漫模式 `scripts/run_persons_test.py --domain anime` 通过（需先备好动漫模型），并校验 `metadata.json` 的 `domain` 字段。

> 说明：人脸身份依赖真实人脸可检测性；`input/` 中的测试素材为 2D 动画，侧脸/风格化脸部可能检不出而落入 `unknown/`，属预期数据限制，不影响管线正确性。动漫素材请用 `--persons-domain anime`。

### 14.8 动漫模型选型（`persons.domain`）

真人模型在 2D 动漫上系统性失效（YOLO/SCRFD/ArcFace 训练分布均为真人），故新增**动漫后端**，覆盖 2D 与 3D；`persons.domain=anime`（**默认**）时启用，共享 `sampler`/`crop`/`cluster`/`export` 框架，仅替换实现与阈值：

```text
① 智能抽帧（不变）
② 人物框：deepghs 动漫人物检测 ONNX(det_conf=0.30) + 人脸锚点扩展(head_body_ratio)
          -> associate_track_id 帧间关联（硬切处断开）
③ 人脸：deepghs 动漫人脸检测 ONNX（F1≈0.95，替代 LBP 级联）
④ 嵌入：CLIP 人物裁剪图嵌入（背影/侧脸帧也参与聚类；不使用 DeepDanbooru 标签向量）
⑤ 裁剪 + 层级（锚点框记 head，其余 unknown）
⑥ 聚类 / 去重（复用框架，用 persons.anime 独立阈值）
⑦ 输出含 domain 的 metadata.json
```

- 阈值必须**按片源重校准**：CLIP 相似度尺度与 ArcFace 不同，`cluster_threshold=0.8`（经验区间 0.75~0.85）、`dedup_threshold=0.88` 仅为起点。
- `head_body_ratio` 按画风调（Q 版 3~4，番剧 6.5~7.5，写实 3D 7.5~8.5）。
- 交互第 4 项「真人 / 动漫 / 不执行」；CLI `--persons-domain real|anime`；引擎检查按领域分派。
- 原文档推荐的 `anime_instance_segmentation` / `anime_face_recognition` 经核验在 HF 上不存在；`hysts/anime-face-detector` 因 `mmdet/mmpose/mmcv` 依赖与可移植环境冲突而不采用。最终采用 `deepghs/anime_person_detection`（F1≈0.87）与 `deepghs/anime_face_detection`（F1≈0.95）的 YOLO ONNX 模型，`onnxruntime` 直驱。
- 首版曾用「COCO YOLO 低阈值 + 锚点扩展 + OpenCV `lbpcascade_animeface` 级联 + 优先人脸区域 CLIP」，在 2D 番剧上效果很差（级联在插画上误检、背影/侧脸无嵌入导致大量 `unknown`）；改为动漫专用 ONNX 检测 + **CLIP 始终嵌入人物裁剪图**后，同素材由 3 person/4 图/16 unknown 改善为 6 person/15 图/0 unknown。

---

## 十五、非目标（本期不实现）

- 运镜识别 / 运镜分类 / 首尾帧 / `motion_meta.json`（已移除，`pipeline/motion.py` 仅保留尺度轨迹）；
- 3D 场景重建、深度估计、点云；
- 转场特效类型识别（淡入淡出 / 叠化）与跨镜头内容拼接（仅硬切分割）；
- 其他音频分析（仅人声 / 背景音乐分离）；
- 全景 / 长图拼接（统一改为 3x3 宫格）；
- 训练新的分类模型、SuperPoint / LightGlue 集成；
- 人物：跨视频人物身份（ReID）全局库、为视频描述提供人物参考图 / 与 `speakers` 对齐、人体分割 / 抠图 / 背景去除、表情 / 年龄 / 性别 / 服饰属性识别、训练或微调检测 / 人脸模型；
- Web / GUI 界面（本期为 CLI + 批处理）。
