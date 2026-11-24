"""Agent/LLM through an online service (the user's own API key). Every service below speaks the same chat format,
so one code path is enough; only the address and the default model differ. The key is stored encrypted for this Windows user (DPAPI)."""
import base64
import ctypes
import sys
from ctypes import wintypes

# id: (title, base address, default model, where to get a key, hint)
SERVICES = {
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "google/gemini-2.0-flash-exp:free", "https://openrouter.ai/keys"),
    "other": ("Другой сервис (AnyModel и др.)", "", "", ""),
}
LOCAL_ONLY = ("cache_prompt", "chat_template_kwargs", "thinking_budget_tokens")   # llama.cpp fields: online services reject them


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def protect(text: str) -> str:
    raw = text.encode("utf-8")
    if sys.platform == "win32":
        try:
            buf = ctypes.create_string_buffer(raw, len(raw))
            src, out = _Blob(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), _Blob()
            if ctypes.windll.crypt32.CryptProtectData(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(out)):
                data = ctypes.string_at(out.pbData, out.cbData)
                ctypes.windll.kernel32.LocalFree(out.pbData)
                return "dpapi:" + base64.b64encode(data).decode()
        except Exception:
            pass
    return "b64:" + base64.b64encode(raw).decode()


def unprotect(s: str) -> str:
    try:
        if s.startswith("dpapi:"):
            raw = base64.b64decode(s[6:])
            buf = ctypes.create_string_buffer(raw, len(raw))
            src, out = _Blob(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), _Blob()
            if ctypes.windll.crypt32.CryptUnprotectData(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(out)):
                data = ctypes.string_at(out.pbData, out.cbData)
                ctypes.windll.kernel32.LocalFree(out.pbData)
                return data.decode("utf-8")
            return ""
        if s.startswith("b64:"):
            return base64.b64decode(s[4:]).decode("utf-8")
    except Exception:
        pass
    return ""


def settings(cfg: dict | None):
    """Ready connection data (url, model, key) or None when the connection is off or incomplete."""
    cfg = cfg or {}
    if not cfg.get("on"):
        return None
    svc = SERVICES.get(cfg.get("svc"), SERVICES["other"])
    url = (cfg.get("url") if cfg.get("svc") == "other" else svc[1]) or ""
    model = cfg.get("model") or svc[2]
    key = unprotect(cfg.get("key") or "")
    if not (url and model and key):
        return None
    return {"url": url.rstrip("/"), "model": model, "key": key}


def prepare(payload: dict, conn: dict):
    """(url, headers, payload) for one chat request to the service."""
    body = {k: v for k, v in payload.items() if k not in LOCAL_ONLY}
    body["model"] = conn["model"]
    return conn["url"] + "/chat/completions", {"Authorization": "Bearer " + conn["key"], "Content-Type": "application/json"}, body


def friendly_error(status: int, text: str) -> str:
    if status in (401, 403):
        return "Ключ не подошёл. Проверьте, что вставили его целиком, без пробелов, и что он от выбранного сервиса."
    if status == 404:
        return "Сервис не нашёл такую модель или адрес. Проверьте название модели."
    if status == 429:
        return "Сервис просит подождать: закончился лимит запросов или деньги на счёте."
    return f"Сервис ответил ошибкой {status}: {text[:200]}"
