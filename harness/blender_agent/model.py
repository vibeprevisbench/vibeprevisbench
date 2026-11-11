"""Chat protocol adapter with bounded retries and raw per-attempt recording.

Reuses the old harness's usage and tool-argument normalization, but not its
unbounded rate-limit retry policy. No OpenAI SDK is needed for this adapter.
"""
from __future__ import annotations

import json
import os
import base64
import hashlib
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
import sys

from .llm_util import norm_usage, _fix_args
from .common import write_json
from .tools import INITIAL_IMAGE_CAP


def portable_request(value, recorder):
    """Replace wire data URLs with content-addressed files; reconstructable, no repeated base64."""
    if isinstance(value, dict):
        if value.get("type") == "base64" and value.get("media_type") and isinstance(value.get("data"), str):
            encoded = portable_request("data:" + value["media_type"] + ";base64," + value["data"], recorder)
            encoded["wire_encoding"] = "base64"
            return {**value, "data": encoded}
        return {k: portable_request(v, recorder) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_request(v, recorder) for v in value]
    if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
        prefix, encoded = value.split(";base64,", 1)
        data = base64.b64decode(encoded)
        sha = hashlib.sha256(data).hexdigest()
        path = recorder.root / "trajectory/media" / sha
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"media_ref": str(path.relative_to(recorder.root)), "mime_type": prefix[5:], "sha256": sha,
                "wire_encoding": "data_url_base64"}
    return value


class ModelProtocolError(ValueError):
    """The API finished a response without a usable agent action."""


class ChatModel:
    def __init__(self, base_url, model, api_key="none", max_tokens=32768, reasoning_effort=None, protocol="auto"):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base URL must be HTTP(S) without embedded credentials, query or fragment")
        if protocol not in ("auto", "chat_completions", "responses"):
            raise ValueError("unsupported model protocol")
        self.protocol = ("responses" if model.startswith("gpt-6") else "chat_completions") if protocol == "auto" else protocol
        self.url = base_url.rstrip("/") + ("/responses" if self.protocol == "responses" else "/chat/completions")
        self.model, self.api_key = model, api_key or "none"
        self.max_tokens, self.reasoning_effort = max_tokens, reasoning_effort
        self.stream = self.protocol == "responses" and os.environ.get("BA_STREAM", "0") == "1"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @staticmethod
    def responses_input(messages):
        items = []
        for msg in messages:
            if msg.get("_responses_output") is not None:
                items.extend(msg["_responses_output"])
                continue
            role = msg.get("role", "user")
            if role == "tool":
                items.append({"type": "function_call_output", "call_id": msg["tool_call_id"],
                              "output": msg.get("content") or ""})
                continue
            content = msg.get("content")
            if content:
                if isinstance(content, str):
                    blocks = [{"type": "output_text" if role == "assistant" else "input_text", "text": content}]
                else:
                    blocks = []
                    for block in content:
                        if block.get("type") == "text":
                            blocks.append({"type": "input_text", "text": block["text"]})
                        elif block.get("type") == "image_url":
                            image = block["image_url"]
                            b = {"type": "input_image", "image_url": image["url"]}
                            if image.get("detail"):
                                b["detail"] = image["detail"]
                            blocks.append(b)
                        else:
                            raise ValueError("unsupported input media block")
                items.append({"type": "message", "role": role, "content": blocks})
            for call in msg.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": call["id"],
                              "name": call["function"]["name"],
                              "arguments": call["function"].get("arguments") or "{}"})
        return items

    @staticmethod
    def limit_images(items, limit=50):
        """Keep initial reference images and newest observations under gateway limits."""
        import copy
        result = copy.deepcopy(items)
        positions = [(i,j) for i,item in enumerate(result) if item.get("type") == "message"
                     for j,b in enumerate(item.get("content") or []) if b.get("type") == "input_image"]
        if len(positions) <= limit:
            return result, {"original": len(positions), "retained": len(positions), "omitted": 0}
        user = next((i for i,item in enumerate(result) if item.get("type") == "message" and item.get("role") == "user"), None)
        initial = [p for p in positions if p[0] == user][:min(INITIAL_IMAGE_CAP,limit)]
        newer = [p for p in positions if p not in initial]
        kept = set(initial + newer[max(0, len(newer)-(limit-len(initial))):])
        for i,j in positions:
            if (i,j) not in kept:
                result[i]["content"][j] = {"type": "input_text",
                    "text": "[Earlier observed image omitted from current request to satisfy the 50-image gateway limit. "
                            "Its timestamp and tool text remain above; the original frame/video and full observation remain in the workspace and archived trajectory. Re-observe if needed.]"}
        return result, {"original": len(positions), "retained": len(kept), "omitted": len(positions)-len(kept)}

    def payload(self, messages, tools):
        if self.protocol == "responses":
            items = getattr(self, "resume_input", []) + self.responses_input(messages)
            items, self.image_window = self.limit_images(items, 50)
            payload = {"model": self.model, "input": items,
                       "max_output_tokens": self.max_tokens, "stream": self.stream, "store": False}
            if self.model.startswith("gpt-"):
                payload["include"] = ["reasoning.encrypted_content"]
            if self.reasoning_effort:
                payload["reasoning"] = {"effort": self.reasoning_effort}
            if tools:
                payload["tools"] = [{"type": "function", **t["function"], "strict": False} for t in tools]
                payload["tool_choice"] = "auto"  # planning is optional; the first response may call tools
            return payload
        payload = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "stream": False}
        if tools:
            payload.update(tools=tools, tool_choice="auto")
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def normalized_usage(self, usage):
        if self.protocol != "responses" or usage is None:
            return norm_usage(usage)
        u = dict(usage)
        u["prompt_tokens_details"] = u.get("input_tokens_details")
        u["completion_tokens_details"] = u.get("output_tokens_details")
        return norm_usage(u)

    def reply(self, raw):
        if self.protocol != "responses":
            choice = raw["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("model output truncated; no complete submission")
            msg = choice["message"]
            wire = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "reasoning_content")}
            if not str(msg.get("content") or "").strip() and not msg.get("tool_calls"):
                wire["_protocol_anomaly"] = "completed_without_answer_or_tool_call"
            return wire
        if raw.get("status") == "incomplete":
            raise ValueError("model output incomplete: " + str(raw.get("incomplete_details")))
        if raw.get("status") not in (None, "completed"):
            raise ValueError("model response failed: " + str(raw.get("error") or raw.get("status")))
        output = raw.get("output") or []
        texts, calls = [], []
        for item in output:
            if item.get("type") == "message":
                for block in item.get("content") or []:
                    if block.get("type") == "output_text":
                        texts.append(block.get("text", ""))
            elif item.get("type") == "function_call":
                calls.append({"id": item["call_id"], "type": "function",
                              "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"}})
        wire = {"role": "assistant", "content": "\n".join(texts), "_responses_output": output}
        if calls:
            wire["tool_calls"] = calls
        if not any(text.strip() for text in texts) and not calls:
            wire["_protocol_anomaly"] = "completed_without_answer_or_tool_call"
        return wire

    @staticmethod
    def read_stream(response, folder, deadline=None):
        """Record SSE verbatim and consume the complete terminal response."""
        data_lines = []
        with (folder / "stream_events.jsonl").open("a") as log:
            for raw_line in response:
                if deadline is not None:
                    deadline.remaining()
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif not line and data_lines:
                    data = "\n".join(data_lines)
                    data_lines = []
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    log.flush()
                    if event.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                        return event["response"]
                    if event.get("type") == "error":
                        raise RuntimeError("model stream error: " + str(event.get("message") or event))
        raise ConnectionError("model stream ended before a terminal response")

    def call(self, messages, tools, deadline, recorder, group):
        number = len(recorder.calls) + 1
        recorder.begin_model_call(number, group)
        folder = recorder.root / f"trajectory/calls/{number:03d}"
        payload = self.payload(messages, tools)
        if getattr(self, "image_window", {}).get("omitted"):
            recorder.event("context_images_window", **self.image_window)
        write_json(folder / "request.json", portable_request(payload, recorder))
        recorder.dialogue["request"] = json.loads((folder / "request.json").read_text())
        recorder.save_pending_dialogue()
        began, usage, attempt = time.monotonic(), None, 0
        wire = None
        try:
            for attempt in range(1, 4):
                remaining = deadline.remaining()
                headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key}
                if self.protocol == "anthropic_messages":
                    headers = {"Content-Type": "application/json", "x-api-key": self.api_key,
                               "anthropic-version": "2023-06-01"}
                request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers=headers)
                try:
                    with self.opener.open(request, timeout=remaining) as response:
                        raw = self.read_stream(response, folder, deadline) if self.stream else json.loads(response.read())
                    write_json(folder / f"attempt_{attempt:02d}.json", raw)
                    usage = self.normalized_usage(raw.get("usage"))
                    write_json(folder / "response.json", raw)
                    wire = self.reply(raw)
                    if wire.get("tool_calls"):
                        _fix_args(wire["tool_calls"])
                    wire.setdefault("role", "assistant")
                    return wire
                except urllib.error.HTTPError as exc:
                    detail = exc.read().decode(errors="replace").replace(self.api_key, "[REDACTED]") if self.api_key != "none" else exc.read().decode(errors="replace")
                    exc.close()
                    write_json(folder / f"attempt_{attempt:02d}.json", {"status": exc.code, "error": detail, "usage": None})
                    if exc.code not in (408, 429) and exc.code < 500:
                        raise RuntimeError(f"model HTTP {exc.code}: {detail}") from exc
                    if attempt == 3:
                        raise RuntimeError(f"model HTTP {exc.code}: {detail}") from exc
                except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    write_json(folder / f"attempt_{attempt:02d}.json", {"error": str(exc), "usage": None})
                    if attempt == 3:
                        raise
                time.sleep(min(attempt, deadline.remaining()))
        finally:
            stat = {"call": number, "group": group, "attempts": attempt, "elapsed_s": time.monotonic() - began,
                    "usage": usage, "request": str((folder / "request.json").relative_to(recorder.root))}
            recorder.calls.append(stat)
            recorder.record_model_reply(folder, stat, portable_request(wire, recorder))
            recorder.event("model_call", **stat)
