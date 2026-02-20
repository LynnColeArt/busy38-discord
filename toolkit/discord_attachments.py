"""
Attachment utilities for busy-38-discord.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import mimetypes
import os
import re
import shlex
from pathlib import Path
import hashlib
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple

from core.attachments.intake import (
    ATTACHMENT_DECISION_ACCEPT,
    ATTACHMENT_DECISION_BLOCK,
    ATTACHMENT_DECISION_QUARANTINE,
    extract_attachment_text_preview,
    _assess_attachment_intake,
    attachment_summary_line,
    sanitize_attachment_for_transcript,
)

try:
    import discord
except Exception:  # pragma: no cover
    discord = None


_ATTACH_TAG_RE = re.compile(r"\[\s*attach\s+([^\]]*?)\/\s*\]", re.IGNORECASE)
_ATTACH_INLINE_RE = re.compile(r"\[\s*attach\s*:\s*([^\]]+)\]", re.IGNORECASE)


def _truthy_env(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def _max_attachment_bytes() -> int:
    return max(1, int(os.getenv("DISCORD_ATTACHMENT_MAX_BYTES", "8000000")))


def _max_attachment_files() -> int:
    return max(1, int(os.getenv("DISCORD_ATTACHMENT_MAX_FILES", "10")))


def _infer_filename_from_url(url: str, fallback: str) -> str:
    try:
        parsed = urlparse(url)
        name = Path(parsed.path).name
        if name:
            return name
    except Exception:
        pass
    return fallback


def _infer_filename_from_mime(mime_type: Optional[str], fallback: str) -> str:
    mt = (mime_type or "").strip().lower()
    if not mt:
        return fallback
    ext = mimetypes.guess_extension(mt, strict=False)
    if not ext:
        return fallback
    if "." in Path(fallback).name:
        return fallback
    return f"{fallback}{ext}"


def _decode_data_uri(uri: str) -> Tuple[Optional[str], Optional[bytes], Optional[str]]:
    """
    Decode base64 data URI strings:
    data:<mime>[;...];base64,<payload>
    """
    raw = str(uri or "").strip()
    if not raw.lower().startswith("data:"):
        return None, None, "not a data URI"
    if "," not in raw:
        return None, None, "invalid data URI (missing comma)"
    header, payload = raw.split(",", 1)
    header_l = header.lower()
    if ";base64" not in header_l:
        return None, None, "data URI must include ;base64"
    mime_type = ""
    if ":" in header:
        mime_type = header.split(":", 1)[1].split(";", 1)[0].strip()
    compact = "".join(payload.split())
    try:
        data = base64.b64decode(compact, validate=True)
    except binascii.Error:
        return mime_type or None, None, "invalid base64 payload in data URI"
    return mime_type or None, data, None


def _decode_base64_payload(raw_value: str) -> Tuple[Optional[bytes], Optional[str]]:
    compact = "".join(str(raw_value or "").split())
    if not compact:
        return None, "empty base64 payload"
    try:
        data = base64.b64decode(compact, validate=True)
    except binascii.Error:
        return None, "invalid base64 payload"
    return data, None


def _should_preview_with_ocr() -> bool:
    return _truthy_env("DISCORD_ATTACHMENT_OCR_PREVIEW", "1")


def normalize_attachment_specs(attachments: Any) -> List[Dict[str, Any]]:
    """
    Normalize attachment input into a list of path/url/base64 specs.

    Supported input forms:
    - list of strings and/or objects
    - object with path/url keys
    - JSON string representing list/object
    - comma-separated string of paths/urls
    """
    if attachments is None:
        return []

    payload = attachments
    if isinstance(attachments, str):
        raw = attachments.strip()
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except Exception:
            payload = [part.strip() for part in raw.split(",") if part.strip()]

    if isinstance(payload, dict):
        payload = [payload]

    out: List[Dict[str, Any]] = []
    if not isinstance(payload, list):
        return out

    for item in payload:
        if isinstance(item, str):
            s = item.strip()
            if not s:
                continue
            if s.lower().startswith("data:"):
                out.append({"data_uri": s})
                continue
            if s.lower().startswith(("http://", "https://")):
                out.append({"url": s})
            else:
                out.append({"path": s})
            continue
        if not isinstance(item, dict):
            continue
        entry: Dict[str, Any] = {}
        if item.get("path"):
            entry["path"] = str(item["path"])
        elif item.get("url"):
            entry["url"] = str(item["url"])
        elif item.get("data_uri"):
            entry["data_uri"] = str(item["data_uri"])
        elif item.get("base64") is not None:
            entry["data_base64"] = str(item["base64"])
        elif item.get("data_base64") is not None:
            entry["data_base64"] = str(item["data_base64"])
        elif item.get("b64") is not None:
            entry["data_base64"] = str(item["b64"])
        else:
            continue
        if item.get("filename"):
            entry["filename"] = str(item["filename"])
        if item.get("description"):
            entry["description"] = str(item["description"])
        if item.get("mime_type"):
            entry["mime_type"] = str(item["mime_type"])
        elif item.get("content_type"):
            entry["mime_type"] = str(item["content_type"])
        if "spoiler" in item:
            entry["spoiler"] = bool(item.get("spoiler"))
        out.append(entry)

    return out


def parse_attach_directives(text: Optional[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse inline attachment directives from LLM output and strip them.

    Supported forms:
    - [attach path="/tmp/report.txt" /]
    - [attach url="https://..." filename="report.txt" /]
    - [attach base64="SGVsbG8=" filename="hello.txt" /]
    - [attach data_uri="data:text/plain;base64,SGVsbG8=" filename="hello.txt" /]
    - [attach:/tmp/report.txt]
    """
    source = str(text or "")
    specs: List[Dict[str, Any]] = []

    def _from_attr_tag(match: re.Match[str]) -> str:
        raw = (match.group(1) or "").strip()
        if not raw:
            return ""
        parsed: Dict[str, Any] = {}
        try:
            tokens = shlex.split(raw)
        except Exception:
            tokens = raw.split()
        unnamed: List[str] = []
        for token in tokens:
            if "=" in token:
                key, value = token.split("=", 1)
                parsed[key.strip().lower()] = value.strip()
            else:
                unnamed.append(token.strip())
        if "path" in parsed:
            spec: Dict[str, Any] = {"path": parsed["path"]}
            if "filename" in parsed:
                spec["filename"] = parsed["filename"]
            if "description" in parsed:
                spec["description"] = parsed["description"]
            if "spoiler" in parsed:
                spec["spoiler"] = str(parsed["spoiler"]).lower() in ("1", "true", "yes", "on")
            specs.append(spec)
        elif "url" in parsed:
            spec = {"url": parsed["url"]}
            if "filename" in parsed:
                spec["filename"] = parsed["filename"]
            if "description" in parsed:
                spec["description"] = parsed["description"]
            if "spoiler" in parsed:
                spec["spoiler"] = str(parsed["spoiler"]).lower() in ("1", "true", "yes", "on")
            specs.append(spec)
        elif "base64" in parsed or "data_base64" in parsed or "b64" in parsed:
            spec = {"data_base64": parsed.get("base64") or parsed.get("data_base64") or parsed.get("b64")}
            if "filename" in parsed:
                spec["filename"] = parsed["filename"]
            if "description" in parsed:
                spec["description"] = parsed["description"]
            if "mime_type" in parsed:
                spec["mime_type"] = parsed["mime_type"]
            if "content_type" in parsed:
                spec["mime_type"] = parsed["content_type"]
            if "spoiler" in parsed:
                spec["spoiler"] = str(parsed["spoiler"]).lower() in ("1", "true", "yes", "on")
            specs.append(spec)
        elif "data_uri" in parsed:
            spec = {"data_uri": parsed["data_uri"]}
            if "filename" in parsed:
                spec["filename"] = parsed["filename"]
            if "description" in parsed:
                spec["description"] = parsed["description"]
            if "mime_type" in parsed:
                spec["mime_type"] = parsed["mime_type"]
            if "content_type" in parsed:
                spec["mime_type"] = parsed["content_type"]
            if "spoiler" in parsed:
                spec["spoiler"] = str(parsed["spoiler"]).lower() in ("1", "true", "yes", "on")
            specs.append(spec)
        elif unnamed:
            first = unnamed[0]
            if first.lower().startswith(("http://", "https://")):
                specs.append({"url": first})
            elif first.lower().startswith("data:"):
                specs.append({"data_uri": first})
            else:
                specs.append({"path": first})
        return ""

    stripped = _ATTACH_TAG_RE.sub(_from_attr_tag, source)

    def _from_inline(match: re.Match[str]) -> str:
        raw = (match.group(1) or "").strip()
        if not raw:
            return ""
        if raw.lower().startswith(("http://", "https://")):
            specs.append({"url": raw})
        else:
            specs.append({"path": raw})
        return ""

    stripped = _ATTACH_INLINE_RE.sub(_from_inline, stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, specs


async def build_discord_files(
    attachments: Any,
    *,
    max_bytes: Optional[int] = None,
    max_files: Optional[int] = None,
) -> Tuple[List[Any], List[str]]:
    """
    Resolve normalized attachment specs into discord.File objects.
    """
    specs = normalize_attachment_specs(attachments)
    limit_bytes = int(max_bytes or _max_attachment_bytes())
    limit_files = int(max_files or _max_attachment_files())
    files: List[Any] = []
    errors: List[str] = []

    if not specs:
        return files, errors
    if discord is None:
        return files, ["discord.py not installed; cannot upload attachments"]

    for idx, spec in enumerate(specs):
        if len(files) >= limit_files:
            errors.append(f"skipped attachments beyond limit ({limit_files})")
            break
        try:
            if spec.get("path"):
                p = Path(str(spec["path"])).expanduser()
                if not p.exists() or not p.is_file():
                    errors.append(f"path not found: {p}")
                    continue
                size = p.stat().st_size
                if size > limit_bytes:
                    errors.append(f"path too large ({size} bytes): {p}")
                    continue
                f = discord.File(
                    str(p),
                    filename=str(spec.get("filename") or p.name),
                    description=spec.get("description"),
                    spoiler=bool(spec.get("spoiler", False)),
                )
                files.append(f)
                continue

            data_uri = str(spec.get("data_uri") or "").strip()
            if data_uri:
                mime_type_from_uri, data, decode_err = _decode_data_uri(data_uri)
                if decode_err:
                    errors.append(decode_err)
                    continue
                if data is None:
                    errors.append("invalid data URI attachment")
                    continue
                if len(data) > limit_bytes:
                    errors.append(f"base64 attachment too large ({len(data)} bytes)")
                    continue
                mime_type = str(spec.get("mime_type") or mime_type_from_uri or "").strip().lower() or None
                default_name = _infer_filename_from_mime(mime_type, f"attachment-{idx+1}")
                filename = str(spec.get("filename") or default_name)
                files.append(
                    discord.File(
                        io.BytesIO(data),
                        filename=filename,
                        description=spec.get("description"),
                        spoiler=bool(spec.get("spoiler", False)),
                    )
                )
                continue

            raw_b64 = spec.get("data_base64")
            if raw_b64 is not None:
                data, decode_err = _decode_base64_payload(str(raw_b64))
                if decode_err:
                    errors.append(decode_err)
                    continue
                if data is None:
                    errors.append("invalid base64 attachment")
                    continue
                if len(data) > limit_bytes:
                    errors.append(f"base64 attachment too large ({len(data)} bytes)")
                    continue
                mime_type = str(spec.get("mime_type") or "").strip().lower() or None
                default_name = _infer_filename_from_mime(mime_type, f"attachment-{idx+1}")
                filename = str(spec.get("filename") or default_name)
                files.append(
                    discord.File(
                        io.BytesIO(data),
                        filename=filename,
                        description=spec.get("description"),
                        spoiler=bool(spec.get("spoiler", False)),
                    )
                )
                continue

            url = str(spec.get("url") or "").strip()
            if not url:
                errors.append("attachment spec missing path/url/base64")
                continue
            if not url.lower().startswith(("http://", "https://")):
                errors.append(f"unsupported URL scheme: {url}")
                continue

            try:
                import aiohttp
            except Exception:
                errors.append("aiohttp unavailable; cannot fetch URL attachments")
                continue

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status >= 400:
                        errors.append(f"url fetch failed ({resp.status}): {url}")
                        continue
                    chunks: List[bytes] = []
                    total = 0
                    async for chunk in resp.content.iter_chunked(65536):
                        total += len(chunk)
                        if total > limit_bytes:
                            chunks = []
                            errors.append(f"url attachment too large (> {limit_bytes} bytes): {url}")
                            break
                        chunks.append(chunk)
                    if not chunks:
                        continue
                    data = b"".join(chunks)
            default_name = f"attachment-{idx+1}.bin"
            filename = str(spec.get("filename") or _infer_filename_from_url(url, default_name))
            file_obj = discord.File(
                io.BytesIO(data),
                filename=filename,
                description=spec.get("description"),
                spoiler=bool(spec.get("spoiler", False)),
            )
            files.append(file_obj)
        except Exception as exc:
            errors.append(f"attachment failed ({idx+1}): {exc}")

    return files, errors


def close_discord_files(files: List[Any]) -> None:
    for f in files or []:
        try:
            f.close()
        except Exception:
            pass


async def extract_message_attachments(
    message: Any,
    *,
    include_text_preview: Optional[bool] = None,
    preview_max_bytes: Optional[int] = None,
    preview_max_chars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Extract attachment metadata from a Discord message with optional text previews.
    """
    include_preview = _truthy_env("DISCORD_ATTACHMENT_TEXT_PREVIEW_ENABLE", "1")
    if include_text_preview is not None:
        include_preview = bool(include_text_preview)
    max_preview_bytes = max(1, int(preview_max_bytes or int(os.getenv("DISCORD_ATTACHMENT_TEXT_PREVIEW_MAX_BYTES", "65536"))))
    max_preview_chars = max(1, int(preview_max_chars or int(os.getenv("DISCORD_ATTACHMENT_TEXT_PREVIEW_MAX_CHARS", "1200"))))

    out: List[Dict[str, Any]] = []
    atts = getattr(message, "attachments", None) or []
    for att in atts:
        filename = str(getattr(att, "filename", "") or "")
        content_type = getattr(att, "content_type", None)
        size = int(getattr(att, "size", 0) or 0)
        entry: Dict[str, Any] = {
            "id": int(getattr(att, "id", 0) or 0),
            "filename": filename or "attachment",
            "content_type": content_type,
            "size": size,
            "url": getattr(att, "url", None),
            "proxy_url": getattr(att, "proxy_url", None),
            "is_image": bool(getattr(att, "height", None) and getattr(att, "width", None)),
            "height": getattr(att, "height", None),
            "width": getattr(att, "width", None),
        }

        if include_preview and size > 0 and size <= max_preview_bytes:
            try:
                raw = await att.read()
                if isinstance(raw, bytes) and raw:
                    entry["text_hash"] = hashlib.sha256(raw).hexdigest()
                    text = extract_attachment_text_preview(
                        data=raw,
                        filename=filename,
                        content_type=content_type,
                        max_chars=max_preview_chars,
                        enable_ocr=_should_preview_with_ocr(),
                    )
                    if text:
                        entry["text_preview"] = text
            except Exception:
                pass

        _assess_attachment_intake(entry)
        out.append(entry)
    return out
