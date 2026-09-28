




当前希望将分割视频片段让多模态大语言模型逐一进行处理，通过多模态大语言模型进行对视频内容的进行描述生成处供视频生成大模型使用的prompt，让视频模型能够理解视频的内容方便后续重新生成该视频片段

阅读以下文件：
skill\SKILL.md
skill\ref-en.txt
skill\base-en.txt

根据skill中的要求整理一份视频描述的prompt，让大模型根据skill中的要求，将视频内容描述出来，方便后续根据视频内容重新生成新的视频

先通过ffmpeg按时间均匀采样的图像帧,每秒采样3帧,保存为png格式,图片大小最大不能超过1080P分辨率

输入给多模态大模型的包括按时间均匀采样的图像帧(在切割视频目录使用单独的文件夹保存避免文件夹太乱)，还有视频的视频时长，还有视频的字幕内容
输出：视频描述的prompt，将prompt保存在分割文件中，让大模型先输出一段英文的prompt，然后让大模型将prompt翻译成中文，将英文prompt和中文prompt分别保存在分割文件中

后续会将以下内容输入视频生成大模型
- 视频描述的英文prompt内容
- 参考图1 为 `3x3` 宫格故事图，故事顺序从左到右从上到下
- 参考图2 为 视频人物1的参考图
- 参考图3 为 视频人物2的参考图
- 以此类推有多少个主要人物就会有多少个参考图 最多5个
- 参考音频1 为 人物1 的声音参考
- 参考音频2 为 人物2 的声音参考
- 以此类推 最多5个

大语言模型推理框架使用llama.cpp,推理可执行文件位于 llama_cpp\llama_bin 目录下

模型使用：
models\llm\llm_model.gguf
models\llm\mmproj_model.gguf

推理命令参考：
.\llama-server.exe -m llm_model.gguf --mmproj mmproj_model.gguf -ngl 999  -c 50000 -fa on -np 1 --jinja --reasoning off



