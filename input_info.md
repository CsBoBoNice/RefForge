# input 实际测试素材说明

> **注意（当前版本，见 `PROJECT_NOTES.md` P38）**：本项目**已移除运镜识别**，不再输出 `motion_meta.json`，也不再做 C1–C11 运镜标签回归（`run_real_test.py` / `run_combined_test.py` / `run_regression.py` 已删除）。本文件保留作历史参考（各视频的镜头构成与内容）。

本文件记录 `input/` 下**实际测试视频**的具体信息（每条视频的镜头构成与内容），做实际测试时据此确认效果。

- 文件名中的 C1–C11 编号为历史运镜素材标记，编号含义见 `REQUIREMENTS.md` 2.3（该节为历史设计）。
- 核对方法：主程序默认 `--sensitivity medium`；分割 / 故事图验收用 `dev.bat scripts\run_segmentation_test.py`（输出到系统临时目录，不污染 `input/`、`output/`）。
- 单段时长默认落在 `[1s, 15s]`。

## 总览

| 文件 | 时长 | 镜头数 | 期望标签 | 备注 |
|---|---|---|---|---|
| `C11_C1_fixed_C4_pan_C2_push.mp4` | 15.10s | 3 | `fixed` / `pan` / `push` | 首部黑场淡入，`valid_range` 收紧 |
| `C1_fixed_C7_whip_C8_follow.mp4` | 15.08s | 3 | `fixed` / `whip` / `follow` | 跟拍为已知难例（主体检测修复） |
| `C3_pull_C9_orbit_C5_move.mp4` | 15.08s | 3 | `pull` / `orbit` / `move` | 弧拍靠本质矩阵旋转判据 |
| `C4_pan_C2_push_C6_crane.mp4` | 15.10s | 3 | `pan` / `push` / `crane` | — |
| `C9_orbit_C6_crane_C11_C3_pull.mp4` | 15.10s | 3 | `orbit` / `crane` / `pull` | 尾部黑场淡出，`valid_range` 收紧 |
| `Screenrecording.mp4` | 10.69s | 1 | `crane` | 竖排滚动文档，验证场景参考图 |

## 逐条详情

### input\C11_C1_fixed_C4_pan_C2_push.mp4

- 视频镜头：C11 首部黑场淡入 + valid_range 收紧 / C1 fixed / C4 pan / C2 push
- 期望标签：`fixed` / `pan` / `push`
- 提示词：

```
integrated_multimodal_description: [Shot 1] 2D-animated, warm anime style, the video fades in from complete black over the first second into a small bakery kitchen where a young woman in a coral apron pulls a tray of golden bread from the oven while a young man in a cream sweater dusts flour beside her and a corgi watches hopefully at their feet. The camera then holds a static shot with only a very slight handheld shake. The young man (S1) takes a deep breath and says: <d>[English] Smell that? First batch of our own shop.</d> The young woman (S2) beams: <d>[English] We really did it, didn't we?</d> [Shot 2] At 00:05.000, the camera cuts to a medium shot and pans left with medium amplitude at slow speed, sweeping across the steaming loaves, flour-dusted counter, and the corgi's eagerly tilted head. [Shot 3] At 00:10.000, the shot cuts to a close-up of the woman breaking a warm loaf in half, steam curling upward, and the camera pushes in with small amplitude at normal speed as she hands half to the man and the corgi barks once, both of them laughing as the frame settles on their happy faces.
overall_soundscape: The oven door clanks shut, bread crust crackles softly as it cools, and flour puffs with a dry hiss. The corgi's nails click on the tile floor, punctuated by one excited bark.
non_diegetic_music: A cozy acoustic-guitar pattern at a moderate tempo, joined by light glockenspiel notes and a warm, homey finish.
```

### input\C1_fixed_C7_whip_C8_follow.mp4

- 视频镜头：C1 fixed / C7 whip / C8 follow
- 期望标签：`fixed` / `whip` / `follow`
- 提示词：

```
integrated_multimodal_description: [Shot 1] 2D-animated, vibrant anime style, a static shot frames a sunny park where a young woman in a pink hoodie and a young man in a mint-green t-shirt stand on the grass, a golden retriever sitting between them wagging its tail. The camera holds a static shot with only a very slight handheld shake. The young woman (S1) turns to the man and says: <d>[English] Ready? Watch this!</d> The young man (S2) replies with a grin: <d>[English] The dog will beat you to it!</d> [Shot 2] At 00:05.000, the shot cuts to a medium shot of the woman hurling a yellow frisbee across the lawn. The camera whips left with large amplitude at fast speed, blurring heavily in mid-swing before snapping into sharp focus on the golden retriever leaping and catching the frisbee mid-air. [Shot 3] At 00:10.000, the camera cuts to a low tracking shot that follows the golden retriever at fast speed as it runs back across the grass with the frisbee, trees and flowers streaming past in the background. The two friends run after the dog laughing, and the woman (S1) shouts: <d>[English] Okay, okay, the dog wins!</d> while the man (S2) laughs: <d>[English] Told you so!</d>
overall_soundscape: Grass rustles under running paws, the frisbee cuts through the air with a soft whoosh, and the retriever's rapid panting and joyful barks carry across the park. Leaves scatter as the dog skids to a stop.
non_diegetic_music: An upbeat brass-and-drums march at a fast tempo, pausing briefly at the catch and swelling into a triumphant finish.
```

### input\C3_pull_C9_orbit_C5_move.mp4

- 视频镜头：C3 pull / C9 orbit / C5 move
- 期望标签：`pull` / `orbit` / `move`
- 提示词：

```
integrated_multimodal_description: [Shot 1] 2D-animated, vivid anime style, a close shot begins on two paint-stained hands — a man's and a woman's — pressing the final stroke onto a community mural of a giant orange cat. The camera pulls out with large amplitude at slow speed, gradually revealing the young man in a rust-colored apron, the young woman in a lavender cardigan, the full colorful mural wall, and finally a small shiba inu sitting proudly in front of their finished artwork. The woman (S1) laughs: <d>[English] We actually finished it!</d> [Shot 2] At 00:05.000, the shot cuts to a medium-wide framing of the three of them beside the mural, and the camera performs an arc shot with medium amplitude at slow speed, circling halfway around them from behind the man and woman to a front view as the shiba inu tilts its head at the painted cat. The man (S2) says warmly: <d>[English] The whole neighborhood is going to love this.</d> [Shot 3] At 00:10.000, the camera cuts to a side view and trucks right with medium amplitude at normal speed, following the couple and the dog as they walk away down the street, the mural and its painted cat glowing behind them in the afternoon light while the woman waves back at their artwork.
overall_soundscape: Paintbrushes tap against the wall, paint buckets clink softly, and the shiba inu gives a curious little woof. Footsteps and a light evening breeze accompany the walk down the street.
non_diegetic_music: A gentle piano melody at a moderate tempo, joined by soft plucked strings that brighten as they walk away.
```

### input\C4_pan_C2_push_C6_crane.mp4

- 视频镜头：C4 pan / C2 push / C6 crane
- 期望标签：`pan` / `push` / `crane`
- 提示词：

```
integrated_multimodal_description: [Shot 1] 2D-animated, colorful anime style, a medium shot shows a cheerful open-air flower market in the morning. A young man in a blue jacket holds a small potted sunflower while a young woman in a rainbow-striped sweater and a beagle with floppy ears browse the stalls. The camera pans right with large amplitude at slow speed, sweeping across flower buckets and ribbons before settling on the man as he offers the pot to her. The young woman (S1) gasps happily: <d>[English] Sunflowers! My favorite!</d> [Shot 2] At 00:05.000, the camera cuts to a medium close-up, and the camera pushes in with small amplitude at slow speed as the man (S2) smiles shyly and says: <d>[English] I remembered from last spring.</d> Their two faces grow steadily larger in the frame while the beagle sniffs the sunflower between them. [Shot 3] At 00:10.000, the shot cuts to a low angle at the cobblestones, and the camera pedestals up with medium amplitude at slow speed, rising from the beagle's wagging tail past the couple's smiling faces to reveal the decorated street and the warm morning sky above, the woman hugging the pot to her chest.
overall_soundscape: Market chatter and rattling cart wheels fill the air, paper bags rustle, and the beagle barks once playfully. Wind chimes ring above the stalls while footsteps tap on the cobblestones.
non_diegetic_music: A warm accordion waltz at a moderate tempo, with light tambourine accents and a gentle resolve at the end.
```

### input\C9_orbit_C6_crane_C11_C3_pull.mp4

- 视频镜头：C9 orbit / C6 crane / C3 pull / C11 尾部黑场淡出 + valid_range 收紧
- 期望标签：`orbit` / `crane` / `pull`
- 提示词：

```
integrated_multimodal_description: [Shot 1] 2D-animated, vivid anime style, a medium shot shows a grassy hilltop at golden sunset where a young man in a green-and-white varsity jacket, a young woman in a coral sundress, and a border collie stand beside a freshly built wooden birdhouse. The camera performs an arc shot with medium amplitude at slow speed, circling the three of them as a small blue bird lands on the birdhouse perch. The woman (S1) whispers excitedly: <d>[English] It likes it! It really likes it!</d> The man (S2) chuckles softly: <d>[English] Best housewarming ever.</d> [Shot 2] At 00:05.000, the shot cuts to a close view of the birdhouse, and the camera pedestals up with medium amplitude at slow speed, rising from the tiny blue bird past the friends' glowing faces to reveal the orange-and-pink sunset sky streaked with clouds. [Shot 3] At 00:10.000, the shot cuts to a wide vista of the hilltop, and the camera pulls out with large amplitude at slow speed as the group waves toward the horizon and the bird takes flight; the scene gradually darkens and fades out to complete black over the final second of the video.
overall_soundscape: A warm evening breeze moves through tall grass, mixing with cricket chirps and the flutter of small wings. Wood taps softly, the collie huffs once, and distant birdsong fades as the light dims.
non_diegetic_music: A gentle piano-and-violin theme at a slow tempo that gradually thins out, ending in silence as the frame fades to black.
```
