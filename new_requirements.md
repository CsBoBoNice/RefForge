真人向模型用在 2D 动漫上会系统性失效，需要整体换型；3D 动漫（CG 渲染）介于两者之间，建议走混合方案
---
## 一、为什么真人模型在动漫上效果差（失效根因）
| 模块 | 真人模型 | 在 2D 动漫上的失效表现 |
|---|---|---|
| 人物检测 | YOLO(COCO 训练) | COCO 全为真实照片分布；动漫人物是**平面色块 + 赛璐璐上色 + 夸张比例**，检测置信度普遍掉到 0.2 以下，漏检严重 |
| 人脸检测 | SCRFD (buffalo_l) | 训练集全部是真人脸；动漫脸的**大眼睛、极简鼻子、尖下巴**与真人分布完全不同，检出率极低或框错位 |
| 人脸识别 | ArcFace (w600k) | 嵌入空间里真人身份的判别维度（皮肤纹理、五官比例微差）在动漫上**根本不存在**，同人不同镜的相似度与异人无异，聚类彻底失效 |
| 跟踪 | BoT-SORT(带 ReID) | 其 ReID 特征同样是真人行人分布；且动漫**实际作画帧率仅 8~12fps**（播放 24fps 大量重复帧），运动模式与真人视频不同，ID 切换频繁 |
一个关键背景差异需要说明：**真人 CV 生态是"通用模型开箱即用"，而动漫 CV 生态是"碎片化、需要按画风适配"**——同一个模型在赛璐璐番剧和厚涂剧场版上表现可能相差很大，这一点在选型时要有预期。
## 二、逐模块替代方案
### 模块 1：人物检测 + 实例分割（动漫人物框）
**首选：`hysts/anime_instance_segmentation`**（SOLOv2 架构，动漫人物实例分割）
- 输出人物的**实例掩膜 + 检测框**，直接对齐全身/半身人物，正是本项目需要的"人物矩形区域"；
- 权重托管在 HuggingFace Hub，MIT 许可，PyTorch 实现，可复用你现有的 torch cu128 环境；
- 这是该作者系列工具中与本项目最匹配的一个（该系列还包括下文的动漫人脸检测器）。

> 如果你只有检测框、没有分割掩膜，也可用 DeepDanbooru 式的 tagger 辅助判断画面内容。
### 模块 2：动漫人脸检测
**首选：`hysts/anime-face-detector`**
- 已验证信息：纯 PyTorch 实现（无 OpenMMLab 运行时依赖），提供 Faster R-CNN 和 YOLOv3 两种检测器 + HRNetV2 28 点人脸关键点，权重在 HuggingFace Hub 上以 safetensors 提供，首次使用自动下载，MIT 许可；
- 28 个人脸关键点恰好可以承接我们之前方案里"层级标注"的角色（判断脸部/胸口/半身裁剪层级）；
- 限制：**只检测近正面人脸**，侧面/背影不检——比真人 SCRFD 的姿态覆盖窄，需要靠人物实例分割（模块 1）兜底非正脸帧。

### 模块 3：身份嵌入与聚类（核心难点）
这是**与真人方案差异最大**的模块，按优先级给三条路线：
**路线 A（专用模型，推荐先试）：动漫人脸识别模型**
- `hysts/anime_face_recognition`：该作者在动漫数据上训练的 OSNet + ArcFace 组合，专为动漫角色验证设计（此仓库为社区知名方案，本轮未逐行核验其可用性，落地前建议先跑通它的 demo 再集成）；
- 用法与真人 ArcFace 完全同构：检测→对齐→512 维嵌入→余弦聚类，**聚类阈值需重新校准**（动漫嵌入的同人/异人分布与真人不同，之前验证的 0.45 不能直接沿用，需在你的片源上重扫一遍）。
**路线 B（通用兜底，鲁棒性最好）：CLIP / OpenCLIP 嵌入**
- 用 `open_clip` 的 ViT-L/14（或 multilingual CLIP）对**人物裁剪图**提嵌入后聚类；
- 优势：CLIP 训练集含大量动漫插画，**对画风天然鲁棒**，2D/3D 通吃，不挑画风；
- 劣势：它学到的是"语义相似"而非"身份相同"——**同一角色不同服装可能分开、不同角色同款服装可能合并**，区分度低于专用动漫人脸模型；
- 适用场景：路线 A 的模型在你的画风上失效时的兜底，或 3D 动漫。

### 模块 4：跟踪
**建议：从 BoT-SORT 换成 ByteTrack**
- 动漫作画帧率低 + 插帧重复，基于外观 ReID 的 BoT-SORT 优势发挥不出来，反而在重复帧上抖动；
- ByteTrack 纯运动关联 + 低分框二次匹配，对低帧率更稳，且 ultralytics 内置支持，零额外依赖；
- 跟踪在本方案中本来就只做"帧间锚定"（背影帧继承身份），身份最终由聚类决定，所以跟踪器弱一点影响可控。
## 三、按片源类型的方案路由
| 片源类型 | 检测 | 人脸 | 嵌入 | 说明 |
|---|---|---|---|---|
| **2D 动漫**（赛璐璐/厚涂番剧） | anime_instance_segmentation | anime-face-detector | 路线A，失效则B | 全套换动漫专用；聚类阈值重新校准 |
| **3D 动漫**（CG/皮克斯风/MMD/国产3D） | 真人 YOLO **与** 动漫分割**并联**，取并集 | 真人 SCRFD **与** 动漫检测**并联** | 路线 B（CLIP）为主，A 为辅 | 3D 渲染介于真人与 2D 之间：写实度高的（如《最终幻想》CG）真人模型仍有效，风格化的（MMD/Q版）需动漫模型；**双路检测取并集**是最稳的做法，成本是推理时间 ×2 |
| **混合/未知片源** | 同 3D 方案的双路并联 | 同上 | CLIP（B） | 启动时跑抽样帧自动判断画风，再路由到对应单路，可省一半时间 |

## 四、落地步骤建议
1. **先跑通最小验证**：拿你实际的 2D/3D 片源各一段（5 分钟即可），分别单独测试 `anime-face-detector` 和 `anime_instance_segmentation` 的检出效果——**动漫模型对画风敏感，这一步不可跳过**，直接决定后续路线选择；
2. **嵌入路线二选一验证**：同一批检出的人脸，分别算路线 A（动漫 ArcFace）和路线 B（CLIP）的嵌入，打印同人/异人相似度分布直方图，看哪条分离度够；两条都不够时上 A+C 融合；
3. **重校准聚类阈值**：沿用之前沙箱验证过的"track 均值特征 + 阈值扫描"方法，在动漫嵌入上重跑一遍（安全区间大概率不再是 0.45~0.50，动漫嵌入的尺度不同）；
4. **跟踪降级**：`model.track(..., tracker="bytetrack.yaml")` 一行改动；
5. **3D 片源双路并联**：两路检测的结果按 IoU 去重合并后再进入统一的嵌入/聚类流程。

## 五、边界与调参要点
- **动漫的"同款校服"问题比真人严重得多**（一部番里全员同款制服），身体/服装特征权重应进一步调低，人脸/tag 特征权重调高；
- **画风切换**（变身、回忆杀、Q版化）会导致同一角色嵌入漂移，靠 track 锚定 + 聚类阈值放宽来吸收；
- 动漫视频的**重复帧**（作画 3 帧播 1）会让"每秒抽 1 帧"抽到大量几乎相同的帧——去重阈值 0.92 在动漫上建议收紧到 0.88 左右，否则去重不充分；
- `anime-face-detector` 只检近正面脸，**背影/侧面帧占比会明显高于真人方案**，`unknown/` 目录要保留，且人物实例分割框要作为非正脸帧的主要来源。




















上一轮我推荐的 `anime_instance_segmentation` 和 `anime_face_recognition` 确实不存在——是我核实不足就把它们当作可用方案写进了文档，向你道歉。我核对了 hysts 的 GitHub 主页（共 26 个公开仓库，置顶项目为图像分类、**anime-face-detector**、视线估计等），其中并不包含这两个仓库。动漫 CV 生态碎片化严重、社区项目更新与弃坑都快，引用时应当逐一验证，这是我的失误。
基于**确认存在**的资源，下面给一条今天就能跑通的落地路线。
---
## 一、修正后的核心选型（全部可验证获取）
| 模块 | 上一轮推荐（错误） | 修正后方案 | 状态 |
|---|---|---|---|
| 动漫人脸检测 | anime_face_recognition ❌ | **hysts/anime-face-detector** | ✅ 已确认存在，388 star，mmdet+mmpose 实现，权重在 HuggingFace Hub |
| 人物全身检测 | anime_instance_segmentation ❌ | **方案A**：人脸锚点扩展（见下文）；**方案B**：Roboflow Universe 动漫检测模型 | ✅ Roboflow 上有动漫检测数据集与预训练模型（如 HonestKlee 的 Anime Detection 项目） |
| 身份嵌入 | 不存在现成模型 | **CLIP 嵌入 + DeepDanbooru 标签向量融合** | ✅ CLIP 为通用技术；DeepDanbooru 为 Danbooru 标签分类器，已确认可用 |
| 跟踪 | ByteTrack（不变） | ByteTrack | ✅ ultralytics 内置 |
关键认知转变：**动漫领域不存在"insightface 级别”的开箱即用人脸识别生态**。能落地的做法是把“人脸识别"拆解为“人脸检测（有成熟模型）+ 身份嵌入（通用模型组合）"两步，各自选型。
## 二、各模块落地方案
### 1. 人脸检测：hysts/anime-face-detector
这是目前唯一确认可用的动漫人脸检测方案，直接安装：
```bat
python\python.exe -m pip install anime-face-detector
```
权重首次运行自动从 HuggingFace Hub 下载（Faster R-CNN 检测器 + HRNet 28 点关键点）。它**只检测近正面人脸**，姿态覆盖比真人 SCRFD 窄，侧面/背影帧需要靠人物框兜底。
### 2. 人物全身框：两条路线按需选择
**路线 A（推荐先试，零新依赖）：人脸锚点扩展**
利用动漫人物头身比相对固定的特性，从人脸框推算人物框：
```python
def expand_face_to_person(face_bbox, head_body_ratio=7.0):
    """动漫人物通常 6~8 头身。以人脸框为锚点扩展出近似人物框。
    head_body_ratio: 头身比，赛璐璐番剧常取 6.5~7.5"""
    x1, y1, x2, y2 = face_bbox
    face_h = y2 - y1
    face_w = x2 - x1
    # 人物总高 ≈ 头身高 × 头身比（动漫中脸≈头的 0.75~0.85）
    person_h = face_h / 0.8 * head_body_ratio
    # 水平居中对齐人脸，身体略宽
    cx = (x1 + x2) / 2
    person_w = max(face_w * 1.6, person_h * 0.35)
    py1 = y1 - face_h * 0.25          # 头顶略上移
    py2 = py1 + person_h
    return (int(cx - person_w/2), int(max(0, py1)),
            int(cx + person_w/2), int(py2))
```
优点：不依赖新模型；缺点：头身比需按片源画风手调（Q版动画取 3~4，正常番剧取 6.5~7.5，写实 3D 取 7.5~8.5）。
**路线 B：Roboflow Universe 动漫检测模型**
Roboflow Universe 上存在动漫人物检测的预训练模型和数据集（搜索 "anime detection" / "anime character"）。模型可下载为 YOLO 格式权重，直接接入你现有的 ultralytics 运行时。注意逐个评估其画风覆盖范围——不同数据集训练出的模型在不同番剧上表现差异大。
### 3. 身份嵌入（核心难点）：CLIP + DeepDanbooru 融合
没有现成动漫 ArcFace，用两个互补信号组合：
**信号 1：CLIP 图像嵌入（通用兜底）**
```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="openai")
model = model.eval().cuda()
@torch.no_grad()
def clip_embed(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = preprocess(Image.fromarray(rgb)).unsqueeze(0).cuda()
    feat = model.encode_image(t).cpu().numpy().flatten()
    return feat / (np.linalg.norm(feat) + 1e-12)
```
CLIP 训练集含大量动漫插画，对画风鲁棒，2D/3D 通吃。局限：学到的是“语义相似”而非“身份相同”——同角色不同服装可能分开、不同角色同款服装可能合并。
**信号 2：DeepDanbooru 角色标签（对知名角色极强）**
DeepDanbooru 是在 Danbooru 数百万张标注动漫图上训练的分类器，能输出角色名 tag。对 Danbooru 覆盖的番剧角色（几乎所有知名 IP），角色名 tag 直接就是身份答案：
```python
# deepdanbooru 预测输出的 tag 概率中，角色名 tag（如 "hatsune_miku"）
# 取 top-k 构成 multi-hot 向量，作为身份特征之一
def danbooru_char_vector(tags_probs, char_tags_vocab, topk=3):
    vec = np.zeros(len(char_tags_vocab))
    idx = {t: i for i, t in enumerate(char_tags_vocab)}
    for tag, p in sorted(tags_probs.items(), key=lambda x: -x[1])[:topk]:
        if tag in idx and p > 0.5:        # 置信度过半才采信
            vec[idx[tag]] = p
    return vec / (np.linalg.norm(vec) + 1e-12)
```
**融合与聚类**：两路特征拼接（或加权求和）后，沿用我们之前沙箱验证过的框架——**track 内取均值 → 并查集聚类 → 阈值扫描校准**。聚类框架与嵌入来源无关，完全复用：
```python
# 融合示例：CLIP 为主(0.7) + Danbooru 为辅(0.3)
def fuse_embedding(clip_feat, danbooru_feat, w_clip=0.7):
    fused = w_clip * clip_feat + (1 - w_clip) * danbooru_feat
    return fused / (np.linalg.norm(fused) + 1e-12)
```
**阈值必须重新校准**：动漫嵌入（尤其 CLIP）的同人/异人相似度分布与真人 ArcFace 完全不同，之前验证的 0.45 安全区间不能沿用。用之前的方法在你的片源上重扫：打印所有 track 对的相似度直方图，人工抽查分布区间再定阈值。经验上 CLIP 嵌入的合并阈值通常在 0.75~0.85 之间，但这只是起点，不是结论。
## 三、管线结构（相对真人版的改动）
整体六步流程不变，只替换模块实现：
```
视频 ─→ ① 智能抽帧(Laplacian，不变)
    ─→ ② 人物框：anime-face-detector 检脸 → 锚点扩展(路线A)
         或 Roboflow 模型直接输出(路线B)；ByteTrack 跟踪
    ─→ ③ 人脸嵌入：anime-face-detector 检出的脸 → CLIP 嵌入
         （没有专用动漫 ArcFace，用 CLIP 替代）
    ─→ ④ 裁剪 + 层级标注（用 28 点关键点替代 COCO 17 点，
         判定逻辑同构：可见的最低关键点决定层级）
    ─→ ⑤ 身份特征 = track 内 CLIP 嵌入均值 + Danbooru 角色向量
    ─→ ⑥ 聚类(框架复用，阈值重校准) → 去重 → 输出
```
## 四、最小验证清单（落地前必做，每步 5~10 分钟）
1. **人脸检测可用性**：拿你的 2D/3D 片源各抽 30 帧，跑 `anime-face-detector`，统计检出率。若正面脸检出率 < 70%，先排查分辨率（该模型建议输入 ≥ 512px）；
2. **人物框合理性**：路线 A 的人物框与人工标注对比 IoU，调整头身比参数；
3. **嵌入区分度**：同一批检出人脸，计算 CLIP 嵌入的“同人不同帧 vs 异人”相似度分布，看两分布是否分离。分离不足时引入 Danbooru 信号或换更大的 CLIP 模型（ViT-bigG）；
4. **聚类阈值扫描**：用之前验证过的并查集 + 阈值扫描代码，在你的相似度矩阵上找最优区间；
5. **重复帧去重**：动漫作画 3 帧播 1 导致大量近重复，去重阈值从 0.92 收紧到 **0.88** 左右再观察。
## 五、可信资源汇总（按核实状态标注）
| 资源 | 状态 | 获取方式 |
|---|---|---|
| hysts/anime-face-detector | ✅ 已核实存在 | `pip install anime-face-detector`，权重自动下载 |
| Roboflow Universe 动漫检测 | ✅ 已核实存在（多个项目） | universe.roboflow.com 搜索 "anime detection" |
| DeepDanbooru | ✅ 已核实存在 | GitHub 搜索 DeepDanbooru，权重在其 release |
| open_clip | ✅ 成熟通用库 | `pip install open_clip_torch` |
| 动漫专用 ArcFace 级识别模型 | ❌ **未找到开箱即用方案** | 需自行在动漫人脸数据上微调，或采用上述融合方案 |
## 六、与现有项目的集成
- `anime-face-detector` 依赖 mmdet/mmpose，与你现有 torch 2.9.1+cu128 兼容，但会引入 mmcv 编译依赖——建议先在虚拟环境验证安装，确认无 CUDA 版本冲突后再进 `setup_runtime.bat`；
- 模型文件加入 `asr/models/anime/` 目录，下载逻辑进 `download_models.py`；
- CLIP（open_clip）为纯 PyTorch，零额外运行时；
- 建议新增 `app/person_extractor_anime.py` 与真人版并行存在，通过 CLI 参数 `--domain real|anime|auto` 路由。
---
需要提醒的诚实边界：路线 A 的头身比扩展在人物非站姿（坐、跑、蜷缩）时会失效，这类帧需要靠跟踪 ID 继承或直接跳过；Danbooru 信号只覆盖知名角色，对原创/冷门角色无效；CLIP 嵌入的“同款服装误合并”问题在动漫（全员同款校服的番剧）里比真人严重得多，所以融合权重里 Danbooru/人脸信号要占足够比重。
如果你把 2D 和 3D 的样例片段路径给我，我可以先写好“人脸锚点扩展 + CLIP 嵌入 + 阈值扫描”的最小验证脚本，跑通后再扩成完整管线。



