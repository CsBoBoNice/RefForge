# RefForge · 使用说明

**RefForge** 是一键、离线、免安装的视频参考资产流水线：把一段（或多段）视频自动**切分成镜头**，
为每个片段生成 **3x3 宫格故事图**（`scene_reference.jpg`）；同时做**人声分离**、**语音识别（按人归档）**
与**人物提取与归类**，并用多模态大模型为每个片段生成**中文视频描述 prompt（H3 full-reference）**。

- 双击即用：包内自带 Python、ffmpeg、模型，**免安装、免配置、免联网**。
- 输入默认是**多镜头长视频**，程序会自动切分并保证每段时长在 `[1s, 15s]`。
- 本工具**不做运镜识别**，不输出 `motion_meta.json`；片段的具体内容由多模态大模型描述。
- 程序按顺序执行：**镜头分割 → 故事图 → 人声分离 → 语音识别 → 人物提取与归类 → 视频描述**。
- 双击 `启动.bat` 会先进入**交互模式**，一次性选好要执行的内容后再开始处理（见 [第二节](#二快速开始)）。
- 开发/需求细节见 `REQUIREMENTS.md`，踩坑与实测数据见 `PROJECT_NOTES.md`。

---

## 目录

1. [运行环境](#一运行环境)
2. [快速开始](#二快速开始)
3. [输入要求](#三输入要求)
4. [输出说明](#四输出说明)
5. [结果怎么用（给视频生成模型）](#五结果怎么用给视频生成模型)
6. [常用命令行参数](#六常用命令行参数)
7. [常用配置（config.json）](#七常用配置configjson)
8. [常用脚本](#八常用脚本)
9. [常见问题](#九常见问题)

---

## 一、运行环境

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 64 位 |
| 运行方式 | 包内自带 Python 与依赖，**无需安装 Python / conda**，**无需联网** |
| 显卡（推荐） | NVIDIA 独显（RTX 30 / 40 / 50 系），驱动建议 **570+**，显存建议 **8GB 以上** |
| 无独显 | 自动回退 CPU，功能可用但**显著变慢**（可接受） |
| 磁盘 | 整包（含运行时与模型）约 20GB 左右；输出另计 |

> **从 GitHub 克隆**得到的仓库**不含**包内 Python、ffmpeg、llama.cpp 与全部模型（见 `.gitignore`）。
> 首次使用请联网双击根目录 **`一键环境搭建.bat`**：自动引导包内 Python、安装依赖、下载运行期资产与全部模型；
> 下载会**按实测网速自动切换最快源**，已存在的文件自动跳过，可重复运行续传。完成后双击 `启动.bat` 即可。
> 若模型或运行时不完整，也可见 [第八节](#八常用脚本) 单独补齐。

---

## 二、快速开始

1. 把待处理视频放进包根目录的 `input\` 文件夹（可放多个）。
2. 双击根目录的 **`启动.bat`**，按提示依次完成 7 项选择（直接在黑窗口回车即用默认值）：
   1. **运行设备**：`1` GPU（默认，更快）或 `2` CPU（兼容性更好）；
   2. **分割时长**：片段最短 / 最长时长（秒），默认 `1` / `15`；
   3. **是否执行**「镜头分割 → 故事图 → 人声分离 → 语音识别」：默认执行；选 `n` 则整段视频当 1 个片段，故事图 / 分离 / ASR 照常；
   4. **人物提取与归类**：默认**动漫模型**（2D/3D 动画，动漫人物/人脸检测 ONNX + CLIP 人物裁剪图嵌入，需先下载动漫模型）；选 `2` 用**真人模型**（写实视频，把人按人脸归类到 `persons/`）；选 `3` 不执行；
   5. **是否生成视频描述**：默认生成；选 `n` 则不生成描述，直接跳过后续两步；
   6. **视频描述输入**：`1` 故事图识别（默认）或 `2` 帧识别；
   7. **要生成描述的片段**：默认全部（回车）；可输入编号 / 范围（如 `1,3,5-8`），或输入 `none` 不生成描述。
3. 选择完成后程序自动开始，等待黑窗口输出日志（首次加载模型较慢，之后会快很多）。
4. 处理完成后，到 `output\segments\<视频文件名>\` 查看结果。
5. 需要清空结果重新跑时，双击 **`清理生成内容.bat`**（只清理 `output\`、输入目录里的 `.srt` 和临时目录，不动模型与视频）。

> 想指定别的输入输出目录或调整参数，可用命令行（见 [第六节](#六常用命令行参数)），例如：
> `启动.bat --input D:\videos --output D:\result --overwrite`；带参数调用不会进入交互模式。

---

## 三、输入要求

- **支持的扩展名**：`.mp4` / `.mov` / `.mkv` / `.avi` / `.webm`（大小写不敏感）。
- **推荐多镜头长视频**：程序会自动做镜头分割，再把每个镜头切成 `3s~15s` 的合规片段。
- 已切好的单镜头短视频也可以（通常会得到 1 段）；若不想自动分割，用 `--no-split`。
- 有音轨会额外做人声分离与语音识别；**无音轨的视频会跳过音频处理**，不影响故事图与描述。
- 视频按文件名自然顺序处理；同一视频内按时间顺序编号（`shot_0001`、`shot_0002`…）。

---

## 四、输出说明

以源视频 `demo.mp4` 为例：

```text
output/
    run.log                             # 本次运行日志
    llama_server.log                    # 视频描述阶段的 llama-server 日志
    voiceprint_center.json              # 全局说话人声纹库（跨视频合并）
    segments/
        demo/
            segments.json               # 镜头分割清单（切点、时长、方法）
            transcript.json             # 整视频转录（谁 / 何时 / 说了什么）
            audio/                      # 人声分离三轨（整视频）
                original.wav            #   原始音轨
                vocals.wav              #   人声（语音识别用）
                instrumental.wav        #   背景音乐 / 伴奏
                separation.json         #   分离元数据（档位 / 设备 / 耗时）
            speakers/                   # 按人归档的 1~30s 语音
                speaker_1/
                    seg_0001.wav
                    seg_0001.json
                    speaker_1_global.json
            persons/                    # 人物提取与归类（每人一个文件夹，多张截图）
                metadata.json           #   归类元信息（人数 / track / 帧号 / 层级）
                person_001/             #   一个人物 = 一个文件夹
                    person_001_f0000123.jpg
                unknown/                #   全程未露脸、无法确定身份的 track
                    track_0001/
                        unknown_f0000200.jpg
            shot_0001/                  # 第 1 个片段
                seg_0001.mp4            #   片段视频（序号与 shot 一致）
                scene_reference.jpg     #   3x3 宫格故事图（参考图 1）
                seg_0001.srt            #   本片段字幕（时间相对片段起点）
                original.wav            #   本片段原始音轨
                frames/                 #   2fps 采样帧（仅 input_mode=frames）
                video_prompt_zh.txt     #   中文 H3 视频描述 prompt
                detailed/               #   其余详细信息
                    asr.json            #     本片段语音识别结果
                    video_prompt.json   #     描述阶段结构化记录
                    storyboard_1080p.jpg#     故事图 1080P 缓存（仅 storyboard 且需缩放）
                    vocals.wav          #     本片片段人声
                    instrumental.wav    #     本片段伴奏
                    debug/              #     --debug 时的中间产物
            shot_0002/
                ...
```

另外，整视频字幕会写到**输入视频同目录**：`<源视频目录>\<源文件名>.srt`。

> **shot 根目录只放常用文件**：`seg_XXXX.mp4` / `scene_reference.jpg` / `seg_XXXX.srt` / `original.wav` / `frames/` / `video_prompt_zh.txt`；其余（`asr.json`、`video_prompt.json`、`vocals.wav`、`instrumental.wav`、`debug/`）统一放在 `detailed/` 子目录，避免目录杂乱。

---

## 五、结果怎么用（给视频生成模型）

**推荐**：把 `video_prompt_zh.txt`（中文六段式描述）作为文本提示交给视频生成模型，并按 prompt 内的标签配合参考资产：

| 标签 | 对应资产 | 说明 |
|---|---|---|
| `<Picture 1>` | `scene_reference.jpg` | **3x3 宫格故事图**（storyboard），故事顺序从左到右、从上到下（不是场景参考图） |
| `<Picture 2..6>` | 人物参考图（最多 5 个） | 第 n 个人物 = `<Picture n+1>`，按 prompt 描述的顺序 |
| `<Audio 1..5>` | 对应人物的声音参考 | 可用 `speakers/<speaker>/` 下归档的语音 |

> 视频描述 prompt 由本地多模态大模型（llama.cpp + Qwen3.5-4B）**直接生成中文**六段式
> （`subject_definitions` / `summary` / `retention_analysis` / `detailed_description` /
> `overall_soundscape` / `non_diegetic_music`），规范见 `skill\ref-en.txt`。
> 参考标签仅 `<Picture N>` 与 `<Audio N>`（不使用 `<Subject N>` / `<Video N>`）：其中 `<Picture 1>` 为 3x3 故事图，人物从 `<Picture 2>` 起；说话人 `(S1)/(S2)` 在片段内按首次发声重编号。
> 推理输入由 `describe.input_mode` 选择：默认 `storyboard` 直接把 3x3 故事图喂给模型；`frames` 则改用逐片段采样帧（两种模式使用不同提示词模板）。

---

## 六、常用命令行参数

`启动.bat` 支持直接透传参数：

```bat
启动.bat [--input DIR] [--output DIR] [--config FILE] [--single FILE]
         [--overwrite] [--debug]
         [--no-split] [--split-sensitivity low^|medium^|high]
         [--min-seg-sec SEC] [--max-seg-sec SEC]
         [--segment-encode auto^|copy^|reencode] [--clean-segments]
         [--no-asr] [--no-separate]
         [--separate-model fast^|balanced^|best^|dereverb]
         [--separate-device gpu^|cpu]
         [--no-persons] [--persons-device gpu^|cpu] [--persons-domain real^|anime]
         [--persons-max N]
         [--workers N] [--opencv-threads N]
         [--no-describe] [--describe-limit N]
         [--interactive]
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--input` | `./input` | 批处理输入目录 |
| `--output` | `./output` | 输出根目录 |
| `--config` | `./config.json` | 阈值配置，缺失则用内置默认值 |
| `--single` | 无 | 只处理单个视频文件 |
| `--overwrite` | 关闭 | 覆盖已存在的输出；默认跳过已处理过的视频 |
| `--debug` | 关闭 | 在 `detailed/debug/` 额外输出中间产物 |
| `--no-split` | 关闭 | 不做镜头分割，整个视频按单镜头处理 |
| `--split-sensitivity` | `medium` | 分割灵敏度 `low` / `medium` / `high`（阈值 4.5 / 3.0 / 2.0） |
| `--min-seg-sec` / `--max-seg-sec` | `1.0` / `15.0` | 片段时长范围（秒） |
| `--segment-encode` | `auto` | 片段导出编码：`copy` / `reencode` / `auto` |
| `--clean-segments` | 关闭 | 处理完删除 `seg_XXXX.mp4`，只留清单与产物 |
| `--no-asr` | 关闭 | 关闭语音识别 |
| `--no-separate` | 关闭 | 关闭人声/背景音乐分离 |
| `--separate-model` | `fast` | 分离档位：`fast`（快）/ `balanced` / `best`（最好）/ `dereverb`（去混响） |
| `--separate-device` | `gpu` | 分离设备 `gpu` / `cpu`（无 CUDA 自动回退 CPU） |
| `--no-persons` | 关闭 | 关闭人物提取与归类（默认开启） |
| `--persons-device` | `gpu` | 人物提取设备 `gpu` / `cpu`（无 CUDA 自动回退 CPU） |
| `--persons-domain` | `anime` | 人物模型领域：`anime`=动漫（deepghs 人物/人脸检测 ONNX + CLIP 人物裁剪图嵌入，2D/3D 通用，默认），`real`=真人（YOLO-pose + SCRFD/ArcFace） |
| `--persons-max` | 全部 | 只对前 N 个源视频做人物提取（调试用） |
| `--workers` | 自动 | 逐片段处理的并行线程数（`0` = 自动，`min(8, CPU 核数)`） |
| `--opencv-threads` | 自动 | OpenCV 内部线程数（`0` = 保持默认）；仅影响速度，不改产物 |
| `--no-describe` | 关闭 | 关闭逐片段视频描述生成（默认开启；llama.cpp 或模型缺失时自动跳过） |
| `--describe-limit` | 全部 | 只对前 N 个片段生成视频描述（调试用）；`0` = 全部 |
| `--interactive` | 关闭 | 启动时交互式选择要执行的内容（分割时长 / 是否分割 / 输入图像 / 描述片段）；双击 `启动.bat` 无参数时自动启用 |

---

## 七、常用配置（`config.json`）

一般不用改；常用项如下（改后即时生效，键名与默认值需与 `app/config.py` 保持一致）：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `segmentation.min_segment_sec` / `max_segment_sec` | `1.0` / `15.0` | 片段时长范围（秒） |
| `segmentation.sensitivity` | `medium` | 镜头分割灵敏度（合成噪声素材可试 `high`） |
| `separation.model` | `fast` | 人声分离档位。追求质量可改 `best`，追求速度用 `fast` |
| `separation.device` | `gpu` | `gpu` / `cpu`；无 CUDA 自动回退 |
| `asr.enabled` | `true` | 是否做语音识别（也可用 `--no-asr`） |
| `asr.device` | `gpu` | 识别设备 `gpu` / `cpu` |
| `asr.language` | `auto` | `auto` / `Chinese` / `English` / `Japanese` |
| `asr.min_archive_sec` / `max_archive_sec` | `1.0` / `30.0` | 按人归档语音时长范围（秒） |
| `describe.enabled` | `true` | 是否生成逐片段视频描述（也可用 `--no-describe`） |
| `describe.input_mode` | `storyboard` | 推理输入图像：`storyboard`=直接用 3x3 故事图，`frames`=用采样帧（两者提示词不同） |
| `describe.storyboard_max_side` | `1920` | 故事图推理缓存长边上限（1080P）；原图更小时直接用原图 |
| `describe.frame_fps` | `2.0` | 采样帧率（仅 `input_mode=frames`） |
| `describe.frame_max_side` | `1920` | 采样帧 1080P 画框长边上限（不放大） |
| `describe.frame_select` | `true` | 是否在每个时间格内取最清晰帧（避免模糊帧） |
| `describe.frame_candidates` | `3` | 清晰帧选择的候选倍数（`fps×该值` 抽候选帧） |
| `describe.limit` | `0` | 仅处理前 N 个片段（`0` = 全部，也可用 `--describe-limit`） |
| `persons.enabled` | `true` | 是否做人物提取与归类（也可用 `--no-persons`） |
| `persons.domain` | `anime` | 人物模型领域 `anime` / `real`（也可用 `--persons-domain`） |
| `persons.device` | `gpu` | 人物提取设备 `gpu` / `cpu` |
| `persons.cluster_threshold` | `0.45` | 真人：人脸跨 track 聚类阈值（越高越不合并） |
| `persons.max_per_person` | `300` | 每个人物最多保留的截图数 |
| `persons.face_embed_score` | `0.10` | 低分人脸回退阈值（仅用于没检测到高质量人脸的 track，恢复遮挡/风格化人脸） |
| `persons.anime.cluster_threshold` | `0.80` | 动漫：CLIP 嵌入聚类阈值（尺度与真人不同，需按片源校准） |
| `persons.anime.dedup_threshold` | `0.88` | 动漫：近重复去重阈值（比真人 `0.92` 收紧，抑制重复帧） |
| `persons.anime.head_body_ratio` | `7.0` | 动漫：人脸锚点扩展头身比（Q 版约 3~4，写实 3D 约 7.5~8.5） |

---

## 八、常用脚本

包内脚本用 `dev.bat <脚本>` 在包内 Python 环境运行（`dev.bat` 与 `启动.bat` 一样自带环境）：

| 目的 | 命令 |
|---|---|
| **一键搭建环境**（引导 Python + 依赖 + 资产 + 全部模型；需联网，首次） | 双击 `一键环境搭建.bat` |
| 重新安装运行环境 + 下载模型（已引导 Python 后等价） | `scripts\setup_runtime.bat` |
| 仅下载运行期资产（ffmpeg / llama.cpp / GGUF，按网速选最快源） | `dev.bat scripts\download_runtime.py` |
| 一键下载全部模型（除 llama.cpp 外；需联网） | 双击 `下载模型.bat` |
| 下载/补齐语音识别模型 | `dev.bat scripts\download_models.py` |
| 下载/补齐人声分离模型 | `dev.bat scripts\download_separation_models.py` |
| 镜头分割 / 故事图自检 | `dev.bat scripts\run_segmentation_test.py` |
| 人声分离自检 | `dev.bat scripts\run_separation_test.py` |
| 语音识别自检 | `dev.bat scripts\run_asr_test.py` |
| 视频描述自检（快：抽帧 + prompt 组装） | `dev.bat scripts\run_describe_test.py` |
| 视频描述自检（端到端，加载 4B 模型，较慢） | `dev.bat scripts\run_describe_test.py --llm` |
| 下载/补齐动漫人物模型（人物/人脸检测 ONNX + CLIP） | `dev.bat scripts\download_anime_person_models.py` |
| 人物提取自检（真人） | `dev.bat scripts\run_persons_test.py` |
| 人物提取自检（动漫，需先备好动漫模型） | `dev.bat scripts\run_persons_test.py --domain anime` |
| 清理生成物（预览 / 执行） | `dev.bat scripts\clean_generated.py [--yes]` |

> 模型缺失时程序会**自动跳过**对应阶段并告警，不会崩溃；补齐模型后重跑即可。

---

## 九、常见问题

**Q：处理很慢 / 显存不够怎么办？**
优先用 `fast` 档分离、必要时 `--separate-device cpu` 或 `--no-separate`、`--no-asr` 只跑分割与故事图；
视频描述（4B 多模态模型）较慢，可用 `--no-describe` 关闭，或 `--describe-limit N` 只跑前几段；
显存不足时把 `asr.device` 设为 `cpu`（功能不变，速度更慢）、或把 `describe.gpu_layers` 调小。

**Q：想让人声分离质量更高？**
把 `separation.model` 改为 `best`（或 `balanced`）。

**Q：重复运行会不会重算？**
默认会跳过已有 `output\segments\<视频名>\segments.json` 的视频；要重跑加 `--overwrite`。

**Q：只想输出一个视频？**
用 `--single "完整路径\xxx.mp4"`。

**Q：想省磁盘空间？**
加 `--clean-segments`，处理完删除片段视频 `seg_XXXX.mp4`。

**Q：路径有中文或空格可以吗？**
可以，正常使用中文/空格路径；建议给包放在**非系统盘、路径不含特殊符号**的目录。

**Q：视频没有声音？**
音频相关阶段会跳过并写出空产物，故事图与描述不受影响。

**Q：机器没有 NVIDIA 显卡？**
程序会自动用 CPU，可直接使用；若想强制，把 `asr.device` / `separation.device` 设为 `cpu`。

**Q：输出乱了想清空？**
双击 `清理生成内容.bat`（保护模型、Python 环境与输入视频，不会误删）。

---

## 相关文档

- `REQUIREMENTS.md`：需求与设计（权威，面向开发）。
- `PROJECT_NOTES.md`：开发笔记（实际架构、关键数值、踩坑与验证结果）。
- `input_info.md`：`input/` 测试素材信息。
