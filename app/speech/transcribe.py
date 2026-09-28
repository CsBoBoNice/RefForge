"""Qwen3-ASR 批量转写 + Qwen3-ForcedAligner 词级时间戳。

Qwen3-ASR 原生输出标点，无需 ct-punc；词级时间戳相对于传入音频片段（秒）。
另提供「按标点分句 + 词级时间戳」对齐工具，供更细粒度的说话人区分使用。
"""

import unicodedata

_SENTENCE_END = set("。！？!?…")
_CLOSERS = set("\"'\u201d\u2019)]}】》")


def _split_text_sentences(text):
    """按句末标点切句；英文句点仅在词尾（后接空白/引号/结尾）时算句末。"""
    sentences = []
    buffer = []
    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        buffer.append(ch)
        is_end = ch in _SENTENCE_END or ch == "\n"
        if ch == ".":
            nxt = text[index + 1] if index + 1 < length else ""
            is_end = nxt == "" or nxt.isspace() or nxt in _CLOSERS
        if is_end:
            cursor = index + 1
            while cursor < length and text[cursor] in _CLOSERS:
                buffer.append(text[cursor])
                cursor += 1
            sentences.append("".join(buffer))
            buffer = []
            index = cursor
            continue
        index += 1
    tail = "".join(buffer)
    if tail.strip():
        sentences.append(tail)
    return [part for part in sentences if part.strip()]


def transcribe_clips(asr_model, clips, language=None, sample_rate=16000):
    """批量转写音频片段。

    clips: 单声道 float32 np.ndarray 列表。
    返回与 clips 等长的列表，每项 {
        "text": str, "language": str, "words": [{"text","start","end"}]
    }；失败时该项 text/language 为空。
    """
    if not clips:
        return []
    # qwen_asr 仅接受 str / (np.ndarray, sr)，这里显式传元组。
    audios = [(clip, int(sample_rate)) for clip in clips]
    results = asr_model.transcribe(
        audio=audios, language=language, return_time_stamps=True)

    outputs = []
    for result in results:
        text = str(getattr(result, "text", "") or "").strip()
        lang = str(getattr(result, "language", "") or "").strip()
        words = []
        stamps = getattr(result, "time_stamps", None)
        if stamps is not None:
            for item in getattr(stamps, "items", []):
                words.append({
                    "text": str(item.text),
                    "start": round(float(item.start_time), 3),
                    "end": round(float(item.end_time), 3),
                })
        outputs.append({"text": text, "language": lang, "words": words})
    return outputs


def _is_kept_char(ch):
    if ch == "'":
        return True
    return unicodedata.category(ch)[0] in ("L", "N")


def _is_cjk(ch):
    code = ord(ch)
    return (0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF or 0xF900 <= code <= 0xFAFF
            or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7A3)


def token_count(text):
    """按 Qwen3-ForcedAligner 的英文/中文分词规则估算 token 数。

    英文按空白分词（连续拉丁字母/数字算一个），中日韩字符逐字计数。
    """
    total = 0
    for chunk in str(text).split():
        cleaned = "".join(ch for ch in chunk if _is_kept_char(ch))
        previous_latin = False
        for ch in cleaned:
            if _is_cjk(ch):
                total += 1
                previous_latin = False
            elif not previous_latin:
                total += 1
                previous_latin = True
    return total


def split_sentences_with_words(text, words):
    """把带标点的转写文本按句切分，并映射到词级时间戳。

    返回 [(sentence_text, start_word_index, end_word_index), ...]；
    token 数无法与词表对齐时退化为整段一句。
    """
    clean_words = words or []
    sentences = _split_text_sentences(text or "")
    if not sentences or not clean_words:
        return [(str(text or "").strip(), 0, len(clean_words) - 1)] \
            if clean_words else []
    counts = [token_count(part) for part in sentences]
    if sum(counts) != len(clean_words):
        return [(str(text or "").strip(), 0, len(clean_words) - 1)]
    result = []
    index = 0
    for part, count in zip(sentences, counts):
        if count <= 0:
            continue
        end = min(index + count, len(clean_words)) - 1
        if end < index:
            break
        result.append((part, index, end))
        index = end + 1
    if index < len(clean_words) and result:
        part, start, _ = result[-1]
        result[-1] = (part, start, len(clean_words) - 1)
    return result


def join_text(left, right):
    """拼接两句文本：中日韩之间不加空格，其余加空格。"""
    left = (left or "").rstrip()
    right = (right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    if _is_cjk(left[-1]) or _is_cjk(right[0]):
        return left + right
    return left + " " + right
