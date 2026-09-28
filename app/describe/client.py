"""llama.cpp（llama-server）多模态推理客户端。

负责：启动一次常驻服务（模型只加载一次）-> 逐片段发送「多帧图像 + 文本」请求
-> 结束后停止进程释放显存。HTTP 使用标准库 urllib，不新增依赖。
"""

import base64
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

from io_utils import (ensure_dir, llama_bin_dir, llama_model_dir,
                      llama_server_path)


def _describe_cfg(cfg):
    return (cfg or {}).get("describe") or {}


def model_path(cfg):
    name = str(_describe_cfg(cfg).get("model") or "llm_model.gguf")
    return os.path.join(llama_model_dir(cfg), name)


def mmproj_path(cfg):
    name = str(_describe_cfg(cfg).get("mmproj") or "mmproj_model.gguf")
    return os.path.join(llama_model_dir(cfg), name)


def available(cfg):
    """检查 llama-server 与模型是否就绪，返回 (ok, missing)。"""
    missing = []
    exe = llama_server_path(cfg)
    if not os.path.isfile(exe):
        missing.append("llama-server.exe")
    for label, path in (("model", model_path(cfg)),
                        ("mmproj", mmproj_path(cfg))):
        if not os.path.isfile(path):
            missing.append("describe.%s (%s)" % (label, os.path.basename(path)))
    return (not missing), missing


def _pick_port(host, port):
    for offset in range(0, 50):
        candidate = int(port) + offset
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, candidate))
            sock.close()
            return candidate
        except OSError:
            sock.close()
            continue
    return int(port)


def _data_uri(path):
    with open(path, "rb") as fh:
        payload = base64.b64encode(fh.read()).decode("ascii")
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    return "data:%s;base64,%s" % (mime, payload)


class LlamaServer(object):
    """常驻 llama-server：start() 加载模型一次，chat() 多次推理，close() 释放。"""

    def __init__(self, cfg, logger=None, log_path=None):
        self.cfg = cfg
        self.logger = logger
        self.log_path = log_path
        self.host = str(_describe_cfg(cfg).get("host") or "127.0.0.1")
        self.port = int(_describe_cfg(cfg).get("port") or 8080)
        self.proc = None
        self.base_url = ""
        self._log_fh = None

    def _command(self):
        cfg = _describe_cfg(self.cfg)
        exe = llama_server_path(self.cfg)
        cmd = [
            exe,
            "-m", model_path(self.cfg),
            "--mmproj", mmproj_path(self.cfg),
            "-ngl", str(int(cfg.get("gpu_layers", 999) or 0)),
            "-c", str(int(cfg.get("context", 50000) or 0)),
            "-fa", "on" if cfg.get("flash_attn", True) else "off",
            "-np", "1",
            "--host", self.host,
            "--port", str(self.port),
        ]
        if cfg.get("jinja", True):
            cmd.append("--jinja")
        reasoning = str(cfg.get("reasoning") or "").strip()
        if reasoning:
            cmd += ["--reasoning", reasoning]
        threads = int(cfg.get("threads", 0) or 0)
        if threads > 0:
            cmd += ["-t", str(threads)]
        return cmd

    def start(self):
        ok, missing = available(self.cfg)
        if not ok:
            raise RuntimeError("llama.cpp not ready; missing: %s"
                               % ", ".join(missing))
        self.port = _pick_port(self.host, self.port)
        self.base_url = "http://%s:%d" % (self.host, self.port)
        cmd = self._command()
        if self.logger:
            self.logger.info("  describe server: %s", " ".join(cmd))
        if self.log_path:
            ensure_dir(os.path.dirname(self.log_path))
            self._log_fh = open(self.log_path, "w", encoding="utf-8")
            out = self._log_fh
        else:
            out = subprocess.DEVNULL
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(
            cmd, cwd=llama_bin_dir(self.cfg), stdout=out, stderr=out,
            creationflags=creationflags)
        self._wait_ready(int(_describe_cfg(self.cfg).get(
            "load_timeout_sec", 900) or 900))
        if self.logger:
            self.logger.info("  describe server ready at %s", self.base_url)

    def _wait_ready(self, timeout):
        deadline = time.time() + max(30, int(timeout))
        url = self.base_url + "/health"
        last = ""
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    "llama-server exited early (code=%s)" % self.proc.returncode)
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    body = resp.read().decode("utf-8", "ignore")
                    if resp.status == 200 and '"ok"' in body:
                        return
                    last = body.strip()
            except urllib.error.HTTPError as exc:
                last = "HTTP %s" % exc.code
            except Exception as exc:  # noqa: BLE001 - 服务未起时的连接错误
                last = str(exc)
            time.sleep(1.0)
        raise TimeoutError("llama-server not ready within %ss (%s)"
                           % (timeout, last))

    def _post(self, path, payload, timeout):
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "ignore"))

    def chat(self, system, user_text, image_paths=None, max_tokens=None,
             temperature=None, top_p=None, timeout=None):
        """发送一次多模态对话，返回 (文本, usage)。"""
        cfg = _describe_cfg(self.cfg)
        content = [{"type": "text", "text": user_text}]
        for path in (image_paths or []):
            content.append({
                "type": "image_url",
                "image_url": {"url": _data_uri(path)},
            })
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        payload = {
            "messages": messages,
            "temperature": float(temperature if temperature is not None
                                 else cfg.get("temperature", 0.2)),
            "top_p": float(top_p if top_p is not None
                           else cfg.get("top_p", 0.9)),
            "max_tokens": int(max_tokens if max_tokens is not None
                              else cfg.get("max_tokens", 4096)),
            "stream": False,
        }
        repeat_penalty = cfg.get("repeat_penalty")
        if repeat_penalty is not None:
            payload["repeat_penalty"] = float(repeat_penalty)
        if cfg.get("enable_thinking", False) is False:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        usage = {}
        try:
            data = self._post("/v1/chat/completions", payload,
                              int(timeout or cfg.get("request_timeout_sec",
                                                     1800) or 1800))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            raise RuntimeError("llama-server HTTP %s: %s" % (exc.code, detail))
        usage = data.get("usage") or {}
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("llama-server returned no choices: %s"
                               % json.dumps(data)[:500])
        message = choices[0].get("message") or {}
        text = message.get("content")
        if isinstance(text, list):
            text = "".join(str(part.get("text", "")) for part in text
                           if isinstance(part, dict))
        finish = choices[0].get("finish_reason")
        if finish == "length" and self.logger:
            self.logger.warning("  describe output truncated (max_tokens)")
        return (text or ""), usage

    def close(self):
        proc = self.proc
        self.proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_fh = None
