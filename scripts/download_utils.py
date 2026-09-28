"""多源下载工具：按实测网速自动选择最快源，失败自动切换下一个源。

仅用标准库（urllib），供 `download_runtime.py` 与各 `download_*_models.py` 复用：

- `probe_speed(url)`：对 URL 做一次 Range 小样本探测，返回字节/秒（失败返回 None）；
- `pick_fastest(urls, group)`：并发探测多个源，返回最快者；同 group 结果在进程内缓存；
- `download_smart(urls, dest, expected, group)`：先选最快源下载，失败自动回退其余源；
- `download(url, dest, expected)`：单源下载（.part 断点续传 + 进度）。
"""

import concurrent.futures
import os
import sys
import time
import urllib.error
import urllib.request

CHUNK = 1 << 20
PROBE_BYTES = 1 << 20
PROBE_TIMEOUT = 15
USER_AGENT = "RefForge-setup"

_FASTEST = {}


def size_ok(path, expected):
    """校验文件存在且大小符合期望。

    expected>=1MB 时要求落在 [0.98, 1.02] 倍区间（文件名固定、来源可变时必须区分
    不同模型）；expected<1MB 时只要求 >1MB。
    """
    if not os.path.isfile(path):
        return False
    size = os.path.getsize(path)
    if expected >= 1_000_000:
        return int(expected * 0.98) <= size <= int(expected * 1.02)
    return size > 1_000_000


def _open(url, extra_headers=None, timeout=120):
    headers = {"User-Agent": USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def probe_speed(url, probe_bytes=PROBE_BYTES, timeout=PROBE_TIMEOUT):
    """对 url 读取前 probe_bytes 字节，返回字节/秒；失败或样本过小返回 None。"""
    started = time.monotonic()
    try:
        with _open(url, {"Range": "bytes=0-%d" % (probe_bytes - 1)}, timeout) as response:
            read = 0
            while read < probe_bytes:
                chunk = response.read(min(65536, probe_bytes - read))
                if not chunk:
                    break
                read += len(chunk)
    except (urllib.error.URLError, OSError):
        return None
    elapsed = max(1e-6, time.monotonic() - started)
    if read < 64 * 1024:
        return None
    return read / elapsed


def pick_fastest(urls, group=None, probe_bytes=PROBE_BYTES, timeout=PROBE_TIMEOUT):
    """并发探测多个源，返回最快的 URL；全部失败时返回第一个。"""
    urls = list(urls)
    if len(urls) <= 1:
        return urls[0] if urls else None
    if group and _FASTEST.get(group) in urls:
        return _FASTEST[group]

    best, best_speed = None, -1.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as pool:
        futures = {pool.submit(probe_speed, url, probe_bytes, timeout): url
                   for url in urls}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                speed = future.result()
            except Exception:
                speed = None
            label = "%.2fMB/s" % (speed / 1e6) if speed else "fail"
            print("[probe] %-9s %s" % (label, url))
            if speed and speed > best_speed:
                best, best_speed = url, speed
    if best is None:
        best = urls[0]
        print("[probe] 全部探测失败，改用：%s" % best)
    else:
        print("[best ] %.2fMB/s %s" % (best_speed / 1e6, best))
    if group:
        _FASTEST[group] = best
    return best


def download(url, dest, expected=0, timeout=120):
    """从单源下载到 dest（.part 断点续传）；expected 为期望字节数（0=不校验）。"""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    downloaded = os.path.getsize(part) if os.path.isfile(part) else 0
    headers = {"Range": "bytes=%d-" % downloaded} if downloaded else {}
    with _open(url, headers, timeout) as response:
        status = getattr(response, "status", 200)
        if status == 200:
            downloaded = 0
        elif status != 206:
            raise RuntimeError("HTTP %s" % status)
        mode = "ab" if downloaded else "wb"
        started = time.time()
        with open(part, mode) as handle:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if expected and downloaded % (20 * CHUNK) < CHUNK:
                    speed = (downloaded / 1e6) / max(1e-6, time.time() - started)
                    sys.stdout.write("    %.1f/%.1fMB  %.2fMB/s\r" % (
                        downloaded / 1e6, expected / 1e6, speed))
                    sys.stdout.flush()
    if expected >= 1_000_000 and downloaded < int(expected * 0.9):
        raise RuntimeError("incomplete: %.1fMB" % (downloaded / 1e6))
    os.replace(part, dest)
    sys.stdout.write("    done %.1fMB\n" % (downloaded / 1e6))
    return True


def download_smart(urls, dest, expected=0, group=None, timeout=120, verify=None):
    """按实测网速选择最快源下载；失败或 verify 校验不通过自动切换其余源。

    verify：可选回调，接收已下载完成的 dest 路径，返回 True 表示内容有效
    （例如归档未被截断）；返回 False 视为该源失败，删除文件并换下一个源。
    """
    urls = list(urls)
    if not urls:
        return False
    best = pick_fastest(urls, group)
    ordered = [best] + [url for url in urls if url != best]
    for url in ordered:
        print("[try ] %s" % url)
        try:
            download(url, dest, expected, timeout)
            if verify is not None and not verify(dest):
                raise RuntimeError("verify failed")
            print("[ok  ] %s" % os.path.basename(dest))
            return True
        except (urllib.error.URLError, RuntimeError, OSError) as exc:
            print("       %s -> %s" % (url, exc))
            if group:
                _FASTEST.pop(group, None)
            for path in (dest, dest + ".part"):
                if os.path.isfile(path):
                    os.remove(path)
    print("[fail] %s" % os.path.basename(dest))
    return False
