# 视频人物提取与归类功能 · 需求与设计（PERSON_REQUIREMENTS.md）

> 版本：0.6（新增 `persons.domain`：真人 / 动漫双后端；真人沿用原流程，动漫覆盖 2D/3D，见第十七节。0.5：已实现并通过验收；含按实测的识别优化：整帧补充检测、去重修正、track 硬切断开、双阈值人脸；已撤销会混入不同人物的"单人无脸并入"）
> 依据：`new_requirements.md` + 需求方对 Q1~Q21 的答复 + 后续修订（人物阶段位置、人脸检测位置、`max_per_person`），按本项目（`REQUIREMENTS.md` / `config.json` / `app/` / `scripts/`）的目录、配置、阶段与验收约定重新整理。
> 定位：本文件是主文档 `REQUIREMENTS.md` 的**待合并新增章节**；实现落地后把稳定结论回并 `REQUIREMENTS.md`，踩坑写入 `PROJECT_NOTES.md`。
> 阅读约定：
> 1. 阈值/常量集中在包根 `config.json` 的新增 `persons` 块，并与 `app/config.py:DEFAULT_CONFIG` 保持一致；两者键名与默认值必须一致。
> 2. 路径基于包根动态解析，不写死开发机绝对路径；运行时不联网；只用包内 Python（`python\python.exe` / `启动.bat` / `dev.bat`）。
> 3. 本文件第 14 节为决策冻结表；标 ⚙ 的为我方据"自行判断"作出的确定选择，开发中如实测需调整须回改本表与配置。

## 目录

1. [功能概述](#一功能概述)
2. [在现有流程中的定位](#二在现有流程中的定位已冻结)
3. [输入](#三输入)
4. [输出目录结构](#四输出目录结构已冻结)
5. [功能需求（FR）](#五功能需求fr)
6. [技术架构与模块设计](#六技术架构与模块设计)
7. [配置（`config.json` 新增 `persons` 块）](#七配置configjson-新增-persons-块已冻结)
8. [CLI 与交互模式](#八cli-与交互模式已冻结)
9. [依赖与模型](#九依赖与模型已冻结)
10. [关键参数表](#十关键参数表默认值已冻结)
11. [边界场景行为](#十一边界场景行为)
12. [验收标准](#十二验收标准)
13. [已验证的设计决策](#十三已验证的设计决策)
14. [决策冻结表（Q1~Q23）](#十四决策冻结表q1q23)
15. [开发准备建议（里程碑与验证）](#十五开发准备建议里程碑与验证)
16. [非目标](#十六非目标)

---

## 一、功能概述

从**单个源视频整片**中自动提取所有出现的人物，按**人物身份**（以**人脸相似度**为主键）归类，每人输出一个独立文件夹，文件夹内保存该人物在各时间点、画面中的**完整矩形区域**图片（从原帧直接裁出、原分辨率、不缩放）。

- **输出目标是"截图分类"**：一个人物会有**多张**截图（不同时间 / 不同姿态 / 不同景别），供后续人工或下游选择；本期**不挑选唯一代表图**，只按清晰度×置信度排序保留至多 `max_per_person` 张。
- 核心主张（已验证）：身份主键是**人脸**，不是衣着、不是人体外观嵌入；换装 / 镜头切换 / 离开再返回须归为同一人，同款衣服的不同人不得合并。
- 人体检测 + 跟踪（BoT-SORT）只用于**帧间锚定**：把背影 / 侧脸 / 无脸帧关联到已有身份，track 本身不是最终身份。
- 人脸检测在**人体裁剪图**上进行（聚焦该人物以提升准确性；仅保留中心落在人体检测框内的人脸），见 FR-3/Q22。

---

## 二、在现有流程中的定位（已冻结）

**决策（Q1）**：作为主流程**新阶段（阶段 4：人物提取与归类）**，按源视频**整片独立处理**。由于 `镜头分割 → 故事图 → 人声分离 → 语音识别` 是一个**统一流程**，人物提取与归类**在该统一流程全部完成之后**执行（位于视频描述之前）。`--interactive` 增加"是否使用该功能"，**默认使用**。

理由：人物身份需要整段源视频的连续性（跨镜头 / 跨换装），故不复用逐片段 `shot_XXXX/` 切分，直接读原始视频；该阶段不依赖音频，放在统一流程之后，符合 Q17"流程依次进行、不同时占用显存"。

主流程顺序（新增阶段 4 后）：

```text
阶段0 CLI/配置/引擎检查
   -> 阶段1 逐源视频：镜头分割 -> 片段导出 -> 3x3 故事图
   -> 阶段2 人声/背景音乐分离
   -> 阶段3 语音识别 + 按人归档
   -> 阶段4 人物提取与归类（整片）        # 新增，在统一流程之后
   -> 阶段5 逐片段视频描述
   -> 阶段6 清理/统计/计时
```

- 阶段 4 引擎检查：`persons.available(cfg)` 检查依赖与模型；缺失则 `persons.enabled=false` + warning，其余阶段照常，进程不崩溃。
- 阶段 4 模型只加载一次、全部视频处理完卸载后再进入下一阶段（Q17）。
- 人物提取不服务描述（Q3），故其相对视频描述的顺序不影响产物；当前冻结为"统一流程之后、视频描述之前"。

流程示意：

```text
① 智能抽帧(OpenCV + Laplacian)              # 每 1s 取最清晰 1 帧 -> 代表帧序列
② YOLO11-pose 检测 + BoT-SORT 跟踪           # person 框 / track_id / 关键点
③ 人体裁剪图 + 整帧 SCRFD 人脸检测 + ArcFace 512 维嵌入
④ 人脸归属到 person 框 -> 每个 track 的人脸特征（均值）
⑤ 层级标注(关键点) + 矩形裁剪 + 质量过滤
⑥ 跨 track 聚类(余弦 ≥ 0.45，约束感知合并；0.35~0.45 二次复核)
⑦ 簇内去重(双方有嵌入用余弦、否则缩略图) + Top-300；单人视频无脸 track 并入唯一人物
⑧ 输出 persons/person_XXX/ + unknown/ + metadata.json
```

---

## 三、输入

- 视频文件：与本项目一致，支持 `.mp4` / `.mov` / `.mkv` / `.avi` / `.webm`（大小写不敏感），任意分辨率；`input/` 批处理或 `--single`。
- 复用 `io_utils.list_videos` / `pipeline.video_io.read_metadata`。
- 运行参数：输出目录、阈值、设备等，均带默认值（见第 7、8 节）。
- **按源视频独立**（Q4）：人物编号、聚类、声纹无关，均在同一视频内独立；不跨视频合并。

---

## 四、输出目录结构（已冻结）

**决策（Q2）**：采用方案 A，人物资产作为源视频产物放在既有 `output/segments/<源文件名去扩展>/` 之下，与 `audio/`、`speakers/` 同级：

```text
output/segments/<源文件名去扩展>/
    segments.json
    transcript.json
    audio/
    speakers/
    persons/                              # 新增：人物提取与归类产物
        metadata.json                     # 归类元信息 + schema_version
        person_001/                       # 一个真实人物 = 一个文件夹（含多张截图）
            person_001_f0000123.jpg       # 身份 + 源全局帧号，可溯源
            person_001_f0000456.jpg
            ...
        person_002/
            ...
        unknown/                          # 全程未露脸、无法确定身份的 track
            track_0001/                   # 每个无脸 track 一个子目录，编号依次递增
                unknown_f0000200.jpg
                ...
            track_0002/
                ...
        debug/                            # 仅 --debug（可选）
```

- `person_XXX` 在同一源视频内唯一，从 `001` 起；**按图片数量降序编号**（Q11；并列时按首次出现时间，早者在前）。
- 帧号编码 `f%07d` = **源视频全局帧号**（Q6），便于溯源。
- `unknown/track_XXXX` 按视频内**首次出现顺序依次递增**编号（Q12），与 BoT-SORT 原始 `track_id` 不要求一致。
- `persons/` 与既有 `--overwrite` / `clean_generated.py`（清空 `output/`）行为一致；重跑时随源目录一起覆盖。

### 4.1 `metadata.json`（建议 schema）

```json
{
  "schema_version": "1.0",
  "source": {"file": "样例.mp4", "fps": 30.0, "frame_count": 900, "duration_sec": 30.0},
  "cluster_threshold": 0.45,
  "ambig_low": 0.35,
  "dedup_threshold": 0.92,
  "max_per_person": 300,
  "person_count": 2,
  "unknown_track_count": 1,
  "persons": {
    "person_001": {
      "num_images": 42,
      "tracks_merged": [1, 5, 17],
      "time_range": [12.0, 3451.0],
      "frame_range": [360, 103530],
      "representative": "person_001_f0012345.jpg",
      "levels": {"full_body": 18, "upper_body": 20, "torso": 3, "chest": 0, "head": 1}
    }
  },
  "unknown": {
    "track_0001": {"source_track_id": 7, "num_images": 5, "time_range": [6.0, 42.0]}
  }
}
```

- `time_range` 单位**秒**；`frame_range` 单位**源帧号**（Q6）。
- `representative` 为**按尺寸选取**的代表图（Q11），仅作索引；不删除其余截图。

---

## 五、功能需求（FR）

### FR-1 抽帧：每秒一帧、选最清晰的

- 以每 1 秒为一个窗口，窗口内取最多 **5 帧候选**，按 **Laplacian 方差**评分，仅保留最清晰的 1 帧作为该窗口的**代表帧**。
- 目的：规避运动模糊与虚焦，保证入库图片清晰。
- ⚙ 实现细节（Q6，我方确定）：
  1. 对整片做**单次顺序解码**；按 1s 窗口（`[k, k+1)` 秒）在窗口内**按时间均匀取最多 5 个候选帧**；
  2. 原生帧率 < 5 时取窗口内**全部可用帧**；窗口内无帧（极短片尾）则跳过；
  3. 灰度 Laplacian 方差最大者为代表帧；全部候选判为黑/白/低信息（复用 `pipeline/quality.py` 阈值）时跳过该窗口；
  4. 代表帧的**源全局帧号**用于文件名与 metadata；检测/跟踪/人脸均在该代表帧序列上运行（Q5）；
  5. 裁剪阶段按需**回读**代表帧的原分辨率彩色帧（不整片驻留内存），保证"原图分辨率裁剪"。
- 窗口之间的候选帧不去重；若同一原生帧被多窗口选中，照常各记一次。

### FR-2 人物检测与跟踪

- 检测类别仅 `person`，置信度阈值 **0.45**（`det_conf`）。
- 使用 **BoT-SORT** 跟踪，输出 `track_id`；track 仅用于**帧间锚定**，不作为最终人物身份。
- 关键点：使用 YOLO11-pose 的 COCO 17 关键点，可见性阈值 `kp_conf = 0.3`（用于 FR-4）。
- 档位建议 `yolo11x-pose`（精度优先）；如速度不足可切 `yolo11m-pose`（配置可换）。

### FR-3 身份判定：以人脸为主键（核心需求）

- **人脸检测以人体裁剪图为主、整帧补充**（Q22 + Q24 优化）：对每个 person 框，在其人体区域（`union(检测框, 可见关键点外接框)` 外扩 5%）的**原分辨率裁剪图**上做人脸检测；同时对该帧做一次**整帧**人脸检测，两侧候选合并、取检测分最高者（裁剪图对贴合人物更稳，整帧对远景/特殊角度召回更高）。仅保留中心落在人体检测框内的人脸。（原"只在整帧检测"结论作废，见第 13 节。）
- **人脸质量门控（双阈值）**：人脸短边 ≥ **48px**（`min_face_px`）；检测分 ≥ **0.55**（`face_det_score`）为高质量人脸；`[face_embed_score=0.10, 0.55)` 为低分人脸，**仅当某 track 无高质量人脸时才用其生成 track 特征**（恢复风格化/遮挡人脸，又不过度污染正常视频）。
- 每个 track 收集所有合格人脸嵌入，**取均值**作为该 track 的身份特征。
- **人脸 ↔ 人体归属**（Q8/Q22，⚙ 我方确定）：
  - 人脸在裁剪图上检测，天然属于该 person；**仅保留人脸框中心落在该 person 检测框（未外扩原框）内的人脸**，以排除裁剪边缘混入的他人人脸；
  - 若同一张人脸同时出现在多个重叠 person 的裁剪图中，归属**包含人脸中心的面积最小**的 person 框；
  - 一帧内同一 person 框取该帧全部合格人脸（通常 1 张，取检测分最高者参与均值）。
- **track 在硬切处断开**（Q24）：复用 PySceneDetect 切点，同一 track 不跨镜头，避免不同角色被串成一条 track 后携带错误身份（实测踩坑，见 PROJECT_NOTES P45）。
- **跨 track 聚类**：余弦相似度 **≥ 0.45** 合并（安全区间 0.45~0.50）。
- **歧义区间 0.35~0.45**：用各自**质量最高的单张人脸**（按 `face_det_score × 人脸尺寸` 排序，Q9）二次复核，**双过线 0.45** 才合并。
- **同帧互斥约束**（Q10，⚙ 我方确定）：任意两 track 若在**同一代表帧同时出现**，则二者"冲突、禁止合并"；冲突关系参与聚类判定。
- **聚类算法**（⚙ 我方确定）：采用**约束感知的凝聚式合并**替代朴素并查集——
  1. 计算所有 track 两两的均值人脸余弦相似度；
  2. 按相似度**降序**处理候选对；一对满足"相似度 ≥ 0.45（或歧义区间通过二次复核）**且**两 track 所属簇之间无任何冲突/同帧互斥"时合并；
  3. 合并后动态维护簇间冲突集，避免传递性违反互斥约束（A~B、B~C 合并但 A 与 C 同帧冲突的场景）。
- 换装、镜头切换、离开再返回的同一人 → 同一文件夹；不同人穿同款衣服 → 不合并。

### FR-4 裁剪与保存：画面中的完整矩形

- 保存内容 = 该人物在当前代表帧中的**矩形区域**，从原帧直接裁出。
- 原帧分辨率保存，**不缩放不放大**；JPEG 质量 **92**。
- **裁剪矩形**（Q8，⚙ 我方确定）= `union(person 检测框, 该 person 可见姿态关键点外接框)`，再**外扩 5%**（`crop_margin_ratio`），越界自动截断；无可用关键点时退化为检测框。目的：在"矩形区域"约束下**尽可能多包含该人物整体**。
- 允许矩形内出现他人少量肢体（轴对齐矩形固有局限，接受）。
- **不做全身过滤**，同时用姿态关键点标注层级（⚙ Q7 规则，取身体链上可见的**最低关节**，任一侧可见即算；可见 = `kp_conf ≥ 0.3`）：

  | 层级 `level` | 判定（按顺序，命中即止） |
  |---|---|
  | `full_body` | 踝关节（15/16）任一可见 |
  | `upper_body` | 膝关节（13/14）任一可见 |
  | `torso` | 髋关节（11/12）任一可见 |
  | `chest` | 肩关节（5/6）任一可见 |
  | `head` | 鼻/眼/耳（0~4）任一可见 |
  | `unknown` | 以上均不可见 |

- 层级计数写入 `metadata.persons.<person>.levels`。

### FR-5 过滤与去重

- 丢弃短边 < 64px 或宽 < 32px 的裁剪图（`min_crop_height` / `min_crop_width`）。
- **评分**（⚙ Q13 我方确定）：
  `score = norm_sharpness × det_conf × face_factor`
  - `norm_sharpness = 裁剪图灰度 Laplacian 方差 / 该人物候选集内的最大方差`（簇内归一化到 (0,1]）；
  - `det_conf` = 人体检测置信度；
  - `face_factor` = 该图所关联人脸的检测分（有关联人脸时），否则取 1.0。
- **簇内去重**（⚙ Q13 + Q24 优化）：
  - 两者都有**本人脸嵌入**：用 ArcFace 余弦 **> 0.92** 判近重复，仅保留 `score` 最高者；
  - 两者都无嵌入（背影/侧脸/无脸）：用 32×32 缩略图做**近像素重复抑制**，不跨身份；
  - 一有一无：不做嵌入比较（**避免用 track 均值把不同姿态误判为重复**——这是实测踩坑，见 PROJECT_NOTES P45）。
- 每人最多保留 `max_per_person = 300` 张（按 `score` 降序截断）；`unknown` 每个 track 同样按 `score` 截断（上限沿用 `max_per_person`）。

### FR-6 无脸内容兜底

- track 在其他帧有合格人脸 → 背影 / 侧脸帧照常入库（用 FR-5 的 track 均值嵌入参与去重）。
- track 全程无脸 → 图片入 `unknown/track_XXXX/`，不参与聚类；子目录按首次出现顺序依次递增（Q12）。
- `unknown` 不占用 `person_XXX` 编号。
- **绝不把无脸 track 盲目并入人物**（否则会把不同角色混入同一文件夹）；不同角色的隔离由"**track 在硬切处断开** + 人脸聚类 + 双阈值人脸"共同保证。

### FR-7 与视频描述的关系

- **决策（Q3）**：本期**不**服务于视频描述，不生成 `<Picture 2..6>` 人物参考图、不与 `speakers` 对齐。`describe` 阶段行为完全不变。

---

## 六、技术架构与模块设计

新增包 `app/persons/`，与 `app/speech/`、`app/describe/` 同级；`app/main.py` 负责登记 job、调用 `run_phase`、汇总统计：

```text
app/persons/
    __init__.py       # available(cfg) / run_phase(jobs, output_root, cfg, logger, stats, debug)
    models.py         # 惰性加载 yolov11-pose + (SCRFD/ArcFace ONNX)，设备解析，只加载一次
    sampler.py        # 智能抽帧：单次顺序解码，1s 窗口取最清晰代表帧；记录源帧号
    detect.py         # YOLO11-pose 检测 + BoT-SORT 跟踪；COCO 关键点
    face.py           # 人体裁剪图上 SCRFD 人脸检测 + ArcFace 512 维嵌入 + 质量门控 + 对齐
    associate.py      # 人脸归属到 person 框 -> track 均值特征 + 同帧冲突集
    cluster.py        # 约束感知凝聚式聚类 + 歧义二次复核
    crop.py           # union(检测框,关键点框)+5% 裁剪、层级标注、过滤、去重、Top-300
    export.py         # 目录组织 / 命名 / metadata.json
```

- 引擎就绪检查（`main.check.engines`）新增 `persons.available(cfg)`：检查 `ultralytics` / `onnxruntime` 可导入 + YOLO 权重 + buffalo_l ONNX 存在；缺失则关阶段并告警。
- 阶段计时用 `app/timing.py:StageTimer`，键名 `phase.persons` 及 `persons.*`。
- 统计字段建议：`persons_ok / persons_error / persons_skipped / persons_count / persons_unknown`，并入 `done:` 汇总。
- 失败处理遵循总原则：单源视频 / 单 track 异常不影响其余，不崩溃。

---

## 七、配置（`config.json` 新增 `persons` 块，已冻结）

新增配置块（须同步 `app/config.py:DEFAULT_CONFIG`）：

```json
"persons": {
  "enabled": true,
  "models": "models/person",
  "pose_model": "yolo11x-pose.pt",
  "face_pack": "buffalo_l",
  "device": "gpu",
  "fps_target": 1.0,
  "candidates_per_window": 5,
  "det_conf": 0.45,
  "kp_conf": 0.3,
  "min_face_px": 48,
  "face_det_score": 0.55,
  "face_embed_score": 0.1,
  "cluster_threshold": 0.45,
  "cluster_ambiguous_low": 0.35,
  "dedup_threshold": 0.92,
  "max_per_person": 300,
  "crop_margin_ratio": 0.05,
  "min_crop_width": 32,
  "min_crop_height": 64,
  "jpeg_quality": 92,
  "keep_unknown": true,
  "debug": false
}
```

- 键名用 `persons`（与 `asr` / `separation` / `describe` 平级）。
- 模型根目录 `models/person/`（Q14）：`detect/yolo11x-pose.pt`、`face/buffalo_l/{det_10g.onnx,w600k_r50.onnx}`。

---

## 八、CLI 与交互模式（已冻结）

新增 CLI 参数（沿用 `--no-*` 风格）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--no-persons` | 关闭 | 关闭人物提取与归类（默认开启） |
| `--persons-device` | `None`（配置 `gpu`） | `gpu` / `cpu` |
| `--persons-max` | `None`（配置 `0`=全部） | 仅处理前 N 个源视频（调试用） |

交互模式（`app/interactive.py`）**新增一步**"是否使用人物提取与归类"，默认使用（Q1）；选择结果写回 `persons.enabled`。交互当前为 4 步，加入后为 5 步（具体展示顺序实现时与既有步骤协调，保持"一次性收集"）。

`_apply_overrides` 追加：`--no-persons → persons.enabled=false`；`--persons-device` / `--persons-max` 覆盖对应键。

---

## 九、依赖与模型（已冻结）

| 项 | 版本/位置 | 说明 |
|---|---|---|
| `ultralytics` | 8.3.x | 复用已有 `torch 2.9.1+cu128` / `torchvision 0.24.1+cu128` |
| `onnxruntime-gpu` | 已有 | 直接驱动 buffalo_l 的 SCRFD / ArcFace ONNX |
| YOLO11-pose | `models/person/detect/yolo11x-pose.pt`（~100MB） | 速度不足可换 `yolo11m-pose.pt` |
| SCRFD+ArcFace | `models/person/face/buffalo_l/`（~330MB，`det_10g.onnx` + `w600k_r50.onnx`） | 离线预置 |

### 9.1 依赖安装策略（Q15，⚙ 我方确定，硬约束：不破坏原环境）

- **人脸侧不引入 `insightface` 包**：直接基于环境已装的 `onnxruntime-gpu` 调用 buffalo_l 的 `det_10g.onnx`（SCRFD）与 `w600k_r50.onnx`（ArcFace），自行实现 SCRFD 解码与 ArcFace 5 点相似变换对齐。这样避免 `insightface` 带入 `albumentations` / `scikit-image` / `easydict` 等重依赖，也不触碰 `opencv` / `torch` 版本。
- **`ultralytics` 以 `--no-deps` 安装**：显式补齐其运行必需的**不冲突**轻量依赖（按实际 `import` 报错最小集：`pyyaml` / `tqdm` / `psutil` / `py-cpuinfo` / `matplotlib` 等）；**严禁**其安装或升级 `opencv-python` / `torch` / `torchvision` / `numpy`。
- **安装后校验**（写入脚本）：确认 `cv2.__file__` 仍指向 `opencv-python-headless`、`torch.__version__ == 2.9.1+cu128`、`numpy` 版本未变；任一不符即中止并报错。
- 备选（仅当 ultralytics 无法脱离 `opencv-python`）：将 YOLO11-pose 导出为 ONNX，用 `onnxruntime` 推理 + 自实现 BoT-SORT；成本较高，非首选。
- 新增依赖写入 `requirements.txt` 独立小节，安装步骤追加 `scripts/setup_runtime.bat`。

### 9.2 模型下载与离线（Q16）

- 新增 `scripts/download_person_models.py`，并接入 `下载模型.bat`（与既有 `download_models.py` / `download_separation_models.py` 并列，不修改语音脚本逻辑）。
- 来源优先级：**ModelScope 或 GitHub Release** → 实在无资源再从**中国 HuggingFace 镜像**下载；下载后本地校验文件存在与大小。
- YOLO 权重为 `.pt` 本地加载；buffalo_l 为 ONNX 本地加载，**全程离线**。

---

## 十、关键参数表（默认值，已冻结）

| 参数 | 默认值 | 依据 |
|---|---|---|
| 抽帧率 `fps_target` | 1.0 | 需求 |
| 窗口候选帧数 `candidates_per_window` | 5 | 清晰度择优 |
| 人体检测置信度 `det_conf` | 0.45 | YOLO 常规 |
| 关键点可见性 `kp_conf` | 0.3 | ⚙ 层级判定 |
| 最小人脸 `min_face_px` | 48 | ArcFace 可靠下限 |
| 人脸检测分 `face_det_score` | 0.55 | 质量门控 |
| **聚类阈值** `cluster_threshold` | **0.45** | 沙箱验证 + ArcFace 推荐 |
| 歧义下界 `cluster_ambiguous_low` | 0.35 | 长相相似者防护 |
| 二次复核阈值 | 0.45 | Q9（后续按实测调整） |
| 去重阈值 `dedup_threshold` | 0.92 | 近重复判定 |
| `max_per_person` | 300 | Q23：多截图供选择 |
| `face_embed_score` | 0.10 | Q24：低分人脸回退阈值（仅用于无高质量人脸的 track） |
| 裁剪边距 `crop_margin_ratio` | 0.05 | 防贴边裁切 |
| 最小裁剪尺寸 | 32（宽）× 64（高）px | 无效图过滤 |
| JPEG 质量 `jpeg_quality` | 92 | 与项目一致 |

---

## 十一、边界场景行为

| 场景 | 行为 |
|---|---|
| 一帧多人 | 各自独立矩形、独立入库，人脸在各自人体裁剪图上检测并按中心归属；裁剪允许含他人少量肢体 |
| 人物部分出画 | 裁剪矩形贴边截断，保留画面内可见部分 |
| 人物重叠遮挡 | 各自矩形均保存；同一张人脸归属包含其中心的面积最小 person 框 |
| 换装后再次出现 | 脸相似 → 同一文件夹 |
| 同款衣服的不同人 | 脸不同 → 不合并 |
| 全程背影 / 极远景无脸 | 入 `unknown/track_XXXX/` |
| 无声 / 黑帧 / 转场帧 | Laplacian 评分为门控自然淘汰，无人物则跳过该窗口 |
| 无人物的视频 | 只写 `persons/metadata.json`（`persons: {}`），不报错 |
| 同帧两个 track | 触发同帧互斥，禁止合并（防误并与传递性错误） |

---

## 十二、验收标准

1. 同一人物（含换装、跨镜头）的图片 **100%** 落在同一文件夹；
2. 不同人物（含同款服装）**不出现**同一文件夹混入；
3. 每张保存图 = 该人物在原帧中的完整矩形，原分辨率；
4. 每人图片清晰（无运动模糊主导）、近重复帧已被去除、按 `score` 截断至 ≤ 300；
5. `unknown/` 存在且仅含全程无脸的 track 图片，按 track 子目录依次递增；
6. `metadata.json` 字段完整、可溯源到源帧号（`f%07d`）；
7. 引擎缺失 / 单源异常 → 阶段跳过或计数告警，进程不崩溃；
8. 产物与 `performance.workers`、调度顺序无关；
9. 新增依赖不改变原有阶段产物与环境（`cv2`/`torch`/`numpy` 校验通过）。

新增脚本 `scripts/run_persons_test.py`（素材取自 `input/`，写入系统临时目录，对齐 `scripts/acceptance_common.py`），校验第 1~6、9 项；**并在既有回归脚本默认参数中加入 `--no-persons`**，避免拖慢 `run_segmentation_test.py` / `run_asr_test.py` / `run_describe_test.py` 等。

---

## 十三、已验证的设计决策

| 决策 | 验证结论 |
|---|---|
| 全身判定用几何宽高比 | ❌ 不可靠 → 改用姿态关键点 |
| 身体嵌入(OSNet)作身份主键 | ❌ 换装失效、同款误合并 → 改人脸主键 |
| track 人脸特征取均值 | ✅ 同人 track 间 0.51~0.59 vs 异人 ≤0.44，分离度充足 |
| 聚类阈值 0.45~0.50 | ✅ 模拟扫描最优；0.40 以下过合并、0.55 以上过分裂 |
| 歧义二次复核 | ✅ 长相相似者可达 0.44，紧贴阈值，需防护 |
| ~~人脸检测在整帧而非裁剪图~~ | ⛔ **作废**：按需求方决定改为**在人体裁剪图上检测**（Q22），聚焦提升准确性；仅保留中心落在人体检测框内的人脸 |

---

## 十四、决策冻结表（Q1~Q23）

| # | 事项 | 冻结决策 |
|---|---|---|
| Q1 | 阶段位置/粒度 | 新阶段（阶段4），整片处理，位于统一流程`分割→故事图→分离→ASR`**之后**、描述之前；交互可开关，默认开 |
| Q2 | 输出位置 | 方案 A：`output/segments/<源文件名>/persons/` |
| Q3 | 是否服务描述 | 不服务；`describe` 不变 |
| Q4 | 多视频身份 | 每视频独立 |
| Q5 | 处理帧率 | 检测/跟踪/人脸均运行在 ~1fps 代表帧序列 |
| Q6 | 抽帧/帧号 | ⚙ 单次顺序解码；1s 窗口均匀 5 候选（fps<5 取全部）；Laplacian 取最清晰；文件名用源全局帧号；metadata 用秒 + 帧号 |
| Q7 | 层级规则 | ⚙ 取身体链上可见最低关节（任一即可），5 档 + unknown |
| Q8 | 人脸↔人体归属 | ⚙ 按"中心落在人体检测框内 + 多框取面积最小"归属；裁剪 = union(检测框, 可见关键点框)+5% |
| Q9 | 二次复核阈值/排序 | 阈值沿用 0.45；按 `face_det_score × 尺寸` 排序 |
| Q10 | 同帧互斥/聚类 | ⚙ 强制同帧不同 track 禁止合并；约束感知凝聚式合并（非朴素并查集） |
| Q11 | 编号/代表图 | 按图片数量降序编号；代表图按尺寸选取 |
| Q12 | unknown 组织 | 按 track 子目录，编号依次递增 |
| Q13 | 去重/评分 | ⚙ `score = norm(Laplacian) × det_conf × face_factor`；人脸用 ArcFace，无脸用 track 均值嵌入，unknown 仅同 track 内近似抑制 |
| Q14 | 模型路径 | `models/person/` |
| Q15 | 依赖冲突 | ⚙ 人脸用 `onnxruntime` 直驱 buffalo_l（不装 insightface）；ultralytics `--no-deps` + 最小轻依赖 + 安装后校验不破坏原环境 |
| Q16 | 模型来源/离线 | ModelScope/GitHub 优先，中国 HF 镜像兜底；`scripts/download_person_models.py` + `下载模型.bat`；全离线 |
| Q17 | 显存/顺序 | 阶段依次执行，模型用完即卸，不同时占用；默认 GPU |
| Q22 | 人脸检测位置 | 以**人体裁剪图**为主 + **整帧补充**（Q24 优化，提升召回）；仅保留中心落在人体检测框内的人脸 |
| Q23 | `max_per_person` | **300**（每人多张截图供后续选择） |
| Q24 | 识别效果优化 | 实测动画/遮挡场景后优化：① 裁剪图 + 整帧双路检测合并；② 去重仅对双方都有本人嵌入用余弦，无嵌入改用缩略图，杜绝 track 均值把不同姿态误判为重复；③ **track 在硬切处断开**（防不同角色被串成一条 track）；④ **双阈值人脸**（`face_det_score` 高质量 + `face_embed_score` 低分回退，仅供无高质量人脸的 track 用）；⑤ **不做无脸 track 盲目并入**（修复不同人物混入同一文件夹） |

> 开发中若实测需调整（尤其 Q9 二次复核阈值、Q10 互斥粒度为"代表帧"还是"原始帧"），须回改本表、`config.json` 与 `DEFAULT_CONFIG`。

---

## 十五、开发准备建议（里程碑与验证）

1. **M1 依赖打通**：`ultralytics --no-deps` + 最小依赖；`onnxruntime` 跑通 buffalo_l 的人脸检测 + 对齐 + 512 维嵌入；执行 9.1 的安装后校验。
2. **M2 核心算法**：抽帧 → 跟踪 → 裁剪图人脸检测与归属 → 均值特征 → 约束感知聚类 → 二次复核，在短视频上核对同人/异人距离分布。
3. **M3 产物与集成**：目录 / 命名 / metadata；接入 `app/main.py` 新阶段、`config.py`、CLI、交互、引擎检查、统计、计时。
4. **M4 验收与文档**：`scripts/run_persons_test.py`；既有脚本加 `--no-persons`；回并 `REQUIREMENTS.md`、更新 `readme.md` / `AGENTS.md`，`PROJECT_NOTES.md` 记录踩坑。
5. **验证命令**：
   - 主程序：`dev.bat app\main.py --input input --output output --overwrite`
   - 分割/故事图回归：`dev.bat scripts\run_segmentation_test.py`
   - 人物提取：`dev.bat scripts\run_persons_test.py`（新增后）
   - 描述回归：`dev.bat scripts\run_describe_test.py`

---

## 十六、非目标

- 跨视频人物身份（ReID）全局库（本期每视频独立）。
- 为视频描述提供人物参考图 / 与 `speakers` 对齐（Q3 明确不做）。
- 人体分割 / 抠图 / 背景去除（仅轴对齐矩形裁剪）。
- 表情 / 年龄 / 性别 / 服饰属性识别。
- Web / GUI 界面（本期为 CLI + 批处理）。
- 训练或微调检测 / 人脸模型（仅使用预置模型）。

---

## 附：与原文档的差异说明

| 项 | `new_requirements.md` | 本文件 |
|---|---|---|
| 输出位置 | `output_persons/` | `output/segments/<源文件名>/persons/`（Q2） |
| 模型路径 | `asr/models/...` | `models/person/...`（Q14） |
| 阶段位置 | 未涉及 | 统一流程（分割→故事图→分离→ASR）之后、描述之前（Q1） |
| 人脸检测位置 | 整帧 | 人体裁剪图（Q22） |
| `max_per_person` | 80 | 300（Q23） |
| 章节编号 | 缺第 8 节 | 已补齐并统一编号 |
| 层级数 | 文字 5 档、示例 4 档 | 统一 5 档 + unknown（FR-4/Q7） |
| 聚类 | 并查集 | 约束感知凝聚式合并（Q10） |
| 配置/CLI/依赖/描述关系 | 未涉及 | 新增第 6~9、FR-7 与依赖策略 |

---

## 十七、动漫模型选型（`persons.domain`，新增）

真人模型在 2D 动漫上系统性失效（YOLO/SCRFD/ArcFace 训练分布全是真人，动漫平面色块 + 夸张比例导致漏检、嵌入无判别力），故新增**真人 / 动漫**模型选型；动漫覆盖 2D 与 3D。真人路径**完全沿用**第 1~16 节，配置与产物不变。

### 17.1 选型修正（原 `new_requirements.md` 推荐已核验）

| 模块 | 文档原推荐 | 核验结果 | 落地实现 |
|---|---|---|---|
| 动漫人物检测 | `hysts/anime_instance_segmentation` | ❌ HF 上不存在 | **`deepghs/anime_person_detection`**（YOLO ONNX，F1≈0.87） |
| 动漫人脸识别 | `hysts/anime_face_recognition` | ❌ HF 上不存在 | **CLIP 人物裁剪图嵌入**（`transformers`） |
| 动漫人脸检测 | `hysts/anime-face-detector` | ✅ 存在，但依赖 `mmdet/mmpose/mmcv` | **`deepghs/anime_face_detection`**（YOLO ONNX，F1≈0.95） |
| 跟踪 | ByteTrack | ✅ | ONNX 检测 + `associate_track_id` 中心/尺寸关联（硬切处断开） |

> 首版曾用 COCO YOLO 低阈值 + 人脸锚点扩展 + OpenCV `lbpcascade_animeface` 级联，实测在 2D 番剧上**效果很差**（级联在报纸插画上误检、YOLO 漏检、背影/侧脸无嵌入导致大量 `unknown`）。现改为**动漫专用 ONNX 检测器**（onnxruntime 直驱，零新依赖）+ **CLIP 人物裁剪图嵌入**，同一素材上由 3 person/4 图/16 unknown 改善为 6 person/15 图/0 unknown，且同一角色的正/侧/背影被正确归并。

### 17.2 动漫后端流程（与真人版六步同构，仅换实现）

```text
① 智能抽帧（不变）
② 人物框：deepghs 动漫人物检测 ONNX（det_conf=0.30）+ 人脸锚点扩展（head_body_ratio）
          -> associate_track_id 帧间关联（硬切处断开）
③ 人脸：deepghs 动漫人脸检测 ONNX（替代 LBP 级联，F1≈0.95）
④ 身份嵌入：CLIP 人物裁剪图嵌入（背影/侧脸帧也参与聚类）  # 不使用 DeepDanbooru 标签向量
⑤ 裁剪 + 层级标注（锚点框记 head，其余 unknown）
⑥ 聚类 / 去重（复用框架，用 persons.anime 独立阈值）
⑦ 输出 persons/person_XXX/ + unknown/ + metadata.json（含 domain）
```

- **零新增依赖**：CLIP 用既有 `transformers`，两个检测器用既有 `onnxruntime` 直驱。
- **不使用 DeepDanbooru 标签向量**（用户明确要求）。
- **动漫独立阈值组** `persons.anime`（`config.json` 与 `DEFAULT_CONFIG` 一致）：
  `cluster_threshold=0.80`、`cluster_ambiguous_low=0.72`、`dedup_threshold=0.88`、`det_conf=0.30`、`min_face_px=32`、`face_det_score=0.50`、`face_embed_score=0.30`、`crop_margin_ratio=0.10`、`head_body_ratio=7.0`、`input_size=640`、`nms_iou=0.5`、`clip_model=clip/clip-vit-large-patch14`、`person_onnx=person/person_detect_v1.1_m/model.onnx`、`face_onnx=face/face_detect_v1.4_s/model.onnx`。
  CLIP 相似度尺度与真人 ArcFace 不同，**0.80 仅为起点，必须按片源阈值扫描校准**。
- **模型与下载**：`models/person/anime/{person/person_detect_v1.1_m, face/face_detect_v1.4_s, clip/clip-vit-large-patch14}`；`scripts/download_anime_person_models.py`（deepghs 走 hf-mirror、CLIP 走 hf-mirror），并接入 `下载模型.bat`；运行期离线。
- **交互 / CLI**：交互第 4 项「1 动漫（默认）/ 2 真人 / 3 不执行」，写回 `persons.domain`；`--persons-domain real|anime`，默认 `anime`。
- **引擎检查**：动漫模式检查 `torch` / `transformers` / `onnxruntime` 可导入 + 动漫人物/人脸 ONNX + CLIP 权重；缺失则关阶段并告警。
- **验收**：`scripts/run_persons_test.py --domain anime`（需先备好动漫模型），并校验 `metadata.json` 的 `domain` 字段。
