# RefForge · AGENTS.md

**RefForge** 是一个 **Python 3.10 CLI 工具**（一键离线包）：把输入视频加工成视频生成模型的 full-reference 参考资产。输入视频按以下顺序处理：**镜头分割 → 逐片段 3x3 宫格故事图 → 人声/背景音乐分离（三轨）→ 语音识别 + 按人归档 → 视频人物提取与归类（整片）→ 逐片段中文视频描述**。音频侧做「人声/背景音乐分离（双轨）」与「语音识别 + 按人归档」；人物侧以**人脸为主键**把整片人物归类到每人一个文件夹（多张截图）；最后用多模态大模型（llama.cpp + Qwen3.5）逐片段生成**中文** H3 full-reference 视频描述 prompt。

**本工具不做运镜识别，不输出 `motion_meta.json`；`pipeline/motion.py` 仅保留尺度轨迹累积供 `--debug` 诊断，故事图选帧不再依赖它。** 最终产物是**可整包拷贝、双击即用、免安装免联网的独立一键包**。

## 执行顺序（与 `app/main.py` 一致，改动须保持）

1. 解析 CLI / 加载 `config.json`（与 `app/config.py` 的 `DEFAULT_CONFIG` 深合并）→ 应用覆盖 → 设置 OpenCV 线程 → 初始化日志。
2. 收集输入视频（`input/` 批处理或 `--single`）。
3. 引擎就绪检查（缺依赖/模型则各自阶段关闭并告警）：`speech.available`（ASR）、`speech.separation_available`（分离）、`describe.available`（描述）、`persons.available`（人物）。
4. **逐源视频阶段**：读元信息 → 跳过/覆盖判定 → 镜头分割 → 并行导出片段 → 写 `segments.json` → 逐片段生成 `scene_reference.jpg`（并行）→ 登记分离 / ASR / 描述任务 → 登记待清理片段。
5. **人声分离阶段**（独立阶段，模型只加载一次）：抽原始音轨 → 分离出人声/伴奏 → 写源级 `audio/` 三轨 + `separation.json` + 片段级音频；返回 `{video: vocals}` 供 ASR 优先使用。
6. **语音识别阶段**（独立阶段，模型只加载一次，结束卸载）：VAD（原始音轨）→ 转写 / 声纹 / 归档（人声轨）→ 写整视频 SRT / `transcript.json` / `speakers/` / 各 `detailed/asr.json` / 更新声纹中心。
7. **人物提取与归类阶段**（独立阶段，模型只加载一次，结束卸载；在统一流程之后）：按源视频整片智能抽帧（1fps 取最清晰），按 `persons.domain`（默认 `anime`）路由两套后端——`real`（真人：YOLO11-pose + BoT-SORT → 人体裁剪图上 SCRFD 人脸 + ArcFace）与 `anime`（动漫，2D/3D 通用：动漫人物检测 ONNX + 人脸锚点扩展 + 中心关联 → 动漫人脸检测 ONNX + CLIP 人物裁剪图嵌入）→ track 均值特征 → 约束感知跨 track 聚类（动漫用 `persons.anime` 独立阈值）→ 裁剪/层级/去重/Top-300 → 写 `persons/metadata.json`（含 `domain`）+ `person_XXX/` + `unknown/`（不服务描述）。
8. **视频描述阶段**（独立阶段，llama-server 只加载一次，结束关闭）：按 `describe.input_mode` 准备输入图像（默认 `storyboard` 直接用 3x3 故事图；`frames` 逐片段抽帧）→ 组装 prompt → 直接生成中文 → 写 `video_prompt_zh.txt` / `detailed/video_prompt.json`。
9. 清理（`--clean-segments`）→ 统计汇总 → `stage timing`。

## 项目结构

- `app/` — 业务代码
  - `main.py` — CLI 入口与流程编排（按上节执行顺序）
  - `config.py` — 内置默认配置 `DEFAULT_CONFIG`（必须与 `config.json` 一致）
  - `io_utils.py` / `logging_utils.py` / `timing.py` — 路径解析、JSON/JPEG 写出、`shot_detail_dir`、日志、阶段计时
  - `pipeline/` — `segmentation`（PySceneDetect + 迭代时长修正）、`segment_export`、`video_io`、`quality`、`trim`、`sampling`、`features`、`motion`（仅尺度轨迹累积）
  - `reference/` — `generator`（宫格故事图，按均匀时间间隔选帧、每点就近取清晰帧）、`storyboard`（网格拼贴）
  - `speech/` — `audio` `separation`（人声/背景音乐分离）`models` `diarize` `transcribe` `voiceprint` `export` `runner`
  - `describe/` — `frames`（ffmpeg 2fps 抽帧、每格取清晰帧）、`prompt` / `client` / `runner`、`prompts/`（`storyboard_*` / `frames_*` 两套六段式模板，对应两种输入模式）
  - `persons/` — `models`（real/anime 两套模型容器，按 `persons.domain` 路由）`sampler`（1fps 取最清晰）`detect`（BoT-SORT）`face`（SCRFD+ArcFace）`associate` `cluster` `crop` `export`；`anime/`（动漫后端：`embed`=CLIP、`onnx_detector`=YOLO ONNX 直驱、`face`=动漫人脸检测、`detect`=动漫人物检测+人脸锚点扩展+中心关联）；`__init__.py` 暴露 `available` / `run_phase`
- `scripts/` — 开发、环境搭建与验收脚本（`acceptance_common`（验收公共工具：以 `input/` 视频为素材）`run_segmentation_test` `run_asr_test` `run_separation_test` `run_describe_test` `run_persons_test`（`--domain real|anime`）`make_test_videos`（可选合成素材工具，验收不再依赖）`bootstrap_python.ps1`（引导 embeddable Python）`setup_env.py`（一键安装依赖 + 运行期资产 + 全部模型）`download_utils.py`（多源测速选最快源）`download_runtime.py`（ffmpeg / llama.cpp / GGUF）`download_models` `download_separation_models` `download_person_models` `download_anime_person_models` `verify_person_env` `clean_generated` `setup_runtime.bat`）
- `bin/` — `ffmpeg.exe` / `ffprobe.exe` / `ffplay.exe`
- `llama_cpp/` — `llama_bin/`（`llama-server.exe` + DLL 及其他可执行文件）
- `python/` — 包内可移植 Python 运行时（含 site-packages）
- `models/` — 全部离线模型：`models/asr/`（语音识别 / 声纹 / VAD）、`models/separator/`（人声分离四档）、`models/person/`（`detect/yolo11x-pose.pt` + `face/buffalo_l/`）、`models/llm/`（llama.cpp 的 `llm_model.gguf` + `mmproj_model.gguf`）
- `下载模型.bat` — 一键下载除 llama.cpp 外的全部模型（ASR 走 ModelScope，分离走 GitHub，人物走 GitHub / hf-mirror）
- `一键环境搭建.bat` — 从 GitHub 克隆后的首次环境搭建（引导包内 Python → 安装依赖 → 下载运行期资产 → 下载全部模型）；下载统一按实测网速自动选择最快源，可重复运行续传
- `input/` — 待处理视频；`output/` — 输出资产
- `config.json` — 阈值配置；`REQUIREMENTS.md` — 需求与设计（以代码为准）；`readme.md` — 用户使用说明；`PROJECT_NOTES.md` — 开发笔记
- `skill/` — H3 prompt 写作规范（`SKILL.md` / `ref-en.txt` / `base-en.txt`）
- `demo/` — 参考项目，**只读参考，不要修改**

## 开发环境（强制）

- **只用包内 Python**：`python\python.exe`；通过 `启动.bat` 或 `dev.bat` 运行，禁止使用系统 Python / conda / 全局包。
- 路径全部基于包根目录动态解析，禁止写死开发机绝对路径；运行时不联网。
- 常用命令：

  | 目的 | 命令 |
  |---|---|
  | 跑主程序 | `dev.bat app\main.py --input input --output output --overwrite` |
  | 分割 / 故事图验收 | `dev.bat scripts\run_segmentation_test.py` |
  | 语音识别验收 | `dev.bat scripts\run_asr_test.py` |
  | 人声分离验收 | `dev.bat scripts\run_separation_test.py` |
  | 视频描述验收 | `dev.bat scripts\run_describe_test.py`（加 `--llm` 跑端到端大模型） |
  | 人物提取验收 | `dev.bat scripts\run_persons_test.py` |
  | 一键搭建环境（首次，联网） | 双击 `一键环境搭建.bat` |
  | 重新安装环境 + 模型（已引导 Python） | `scripts\setup_runtime.bat` |
  | 下载运行期资产（ffmpeg / llama.cpp / GGUF） | `dev.bat scripts\download_runtime.py` |
  | 下载模型（除 llama.cpp） | 双击 `下载模型.bat` |
  | 清理生成物 | `dev.bat scripts\clean_generated.py --yes` |
  | 任意 Python | `dev.bat -c "..."` 或 `dev.bat <script.py>` |

- 验收脚本写入系统临时目录，**不要污染** `input/` / `output/`。

## 代码规范

- Python 3.10；`app/` 已加入 `sys.path`，使用顶层模块导入（如 `from pipeline import motion`）。
- 阈值/常量集中在 `config.json`，任何改动必须同步 `app/config.py` 的 `DEFAULT_CONFIG`，两者键名与默认值保持一致。
- 依赖已固定（基础：`opencv-python-headless` / `numpy` / `Pillow` / `scenedetect` / `click` / `platformdirs`；语音：`torch cu128` / `funasr` / `qwen-asr` / `modelscope` / `librosa` / `soundfile`；人声分离：`audio-separator` / `onnxruntime-gpu` / `torchvision cu128`；人物：`ultralytics --no-deps` / `lap`，人脸侧用 `onnxruntime` 直驱 buffalo_l，**不装 `insightface`**；动漫侧用既有 `transformers` 的 CLIP + `onnxruntime` 直驱动漫人物/人脸检测 ONNX（deepghs），**不新增依赖**，也**不使用 DeepDanbooru 标签向量**）；不要引入其他重量级或需联网下载的依赖，也不要安装 `opencv-python`（会与 headless 冲突）。新增依赖必须同步 `requirements.txt` 与 `scripts/setup_env.py`（及 `scripts/setup_runtime.bat`）。
- 默认不添加多余注释；中文注释/文档字符串使用 UTF-8。
- **禁止用 PowerShell 文本 cmdlet（`Get-Content` / `Set-Content` / `-replace` 等）读写含中文的 UTF-8 文件**（会破坏编码，见 `PROJECT_NOTES.md` P23）；改文件请用编辑工具。
- **Windows 脚本编码/换行（强制，见 `PROJECT_NOTES.md` P48）**：`.bat` **只允许 ASCII 字符（不得出现中文/非 ASCII）**，保存为 **CRLF、无 BOM**——cmd 按系统 ANSI 代码页读取 `.bat`（与 `chcp` 无关），非 ASCII 会解析错乱或乱码（UTF-8 解析失败；GBK 在 `chcp 65001` 下显示为乱码）；面向用户的中文提示统一由 `.py`/`.ps1` 打印。`.ps1` 保存为 **UTF-8 带 BOM、CRLF**（PowerShell 5.1 才正确解码中文）。保存时确保换行与编码正确。

## 项目约定

- **文档双向同步**：改实现必须回改 `REQUIREMENTS.md` 对应章节；踩的坑/新约定/解决办法写入 `PROJECT_NOTES.md`。
- **分割约束**：输入默认多镜头长视频，每段时长必须落在 `[1s, 15s]`（除非输入总时长不足 1s）。
- **ASR 归档**：按人归档的语音时长为 `1~30s`（`asr.min_archive_sec=1.0`）。
- **shot 输出结构**：`output/segments/<源文件名去扩展>/segments.json` + 每个 `shot_XXXX/`：
  - 根目录：`seg_XXXX.mp4`、`scene_reference.jpg`（3x3 宫格故事图）、`seg_XXXX.srt`、`original.wav`、`frames/`（仅 `describe.input_mode=frames` 时）、`video_prompt_zh.txt`；
  - `detailed/`：其余详细信息（`asr.json`、`video_prompt.json`、`vocals.wav`、`instrumental.wav`、storyboard 模式下的 `storyboard_1080p.jpg` 缓存、`--debug` 时的 `debug/`）。
  - 同一源文件内 `shot_XXXX` 与 `seg_XXXX` 序号一致；源级另有 `audio/`（`original.wav` / `vocals.wav` / `instrumental.wav` / `separation.json`）与 `persons/`（`metadata.json` + `person_XXX/` + `unknown/track_XXXX/`，人物截图）。
- **故事图**：所有片段统一 `3x3` 九宫格（按均匀时间间隔选帧、每点就近取清晰帧、帧间时间偏差不超过 `reference.select_tolerance_ms`（默认 50ms），间距/留白 10px），作为视频生成模型的参考图 1「故事图」，**不是场景参考图**（故事顺序从左到右、从上到下）。
- **视频描述**：中文六段式；参考标签仅 `<Picture N>` 与 `<Audio N>`——`<Picture 1>`=3x3 故事图、人物从 `<Picture 2>` 起，`<Audio 1..5>`=对应声音参考，`(Sx)` 在片段内重编号；**不使用** `<Subject N>` / `<Video N>`。输入图像由 `describe.input_mode` 选择（默认 `storyboard` 直接用故事图，`frames` 用采样帧），两种模式使用不同模板（`prompts/storyboard_*` / `prompts/frames_*`）。`storyboard` 模式会先按 `describe.storyboard_max_side`（默认 1920）把故事图缩放到 `detailed/storyboard_1080p.jpg` 再推理（原图更小则直接用原图），避免图片过大拖慢推理。

## 验证要求

- 改动后至少跑通：`run_segmentation_test.py`（全通过）、`run_describe_test.py`（抽帧 + prompt；`--llm` 端到端需 GPU/模型）。
- 涉及分割 / 故事图改动时：核对 `scene_reference.jpg` 为 `3x3` 九宫格、9 格按时间升序、每格时间偏差 ≤ `select_tolerance_ms`（默认 50ms），并跑 `run_segmentation_test.py`。
- 涉及音频 / ASR / 归档改动时另跑：`run_separation_test.py`（人声分离三轨）、`run_asr_test.py`（语音识别产物与 `source=vocals` 校验通过）；详见 `PROJECT_NOTES.md` 问题与解决 P32/P33。
- 涉及视频描述改动时另跑：`run_describe_test.py`；详见 `PROJECT_NOTES.md` P36。既有验收脚本均传 `--no-describe`，避免加载 4B 模型。
- 涉及人物提取改动时另跑：`run_persons_test.py`（产物结构 / 命名可溯源 / 原分辨率矩形 / 去重与 Top-300 / `unknown` / 环境未变）；动漫后端另跑 `run_persons_test.py --domain anime`（需先 `download_anime_person_models.py`）；详见 `PROJECT_NOTES.md` P39/P46。既有回归脚本均传 `--no-persons`，避免加载 YOLO 模型。
- 提交前确保 `config.json` 与 `app/config.py` 键一致、`app/` 可 `compileall` 通过。
