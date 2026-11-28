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


# ---- profiles: several saved connections, one of them is active. The active one is also kept in config["llm_api"] (what the requests use).
FIELDS = ("svc", "model", "url", "key")
MAX_PROFILES = 20


def profiles(cfg: dict) -> list:
    """The saved profiles; the single connection of the older versions becomes the first one."""
    got = [dict(p) for p in cfg.get("llm_profiles") or [] if isinstance(p, dict) and p.get("id")]
    old = cfg.get("llm_api") or {}
    if not got and old.get("key"):
        p = {k: old.get(k) or "" for k in FIELDS}
        p.update(id=old.get("id") or "p1", name=title_of(p))
        got = [p]
    return got


def title_of(p: dict) -> str:
    return (p.get("model") or SERVICES.get(p.get("svc"), SERVICES["other"])[0])[:60]


def active_id(cfg: dict, got: list) -> str:
    c = cfg.get("llm_api") or {}
    i = c.get("id") or ("p1" if c.get("key") else "")
    return i if any(p["id"] == i for p in got) else ""


def draft(old: dict | None, body: dict) -> dict:
    """A profile from the fields of the page. An empty key field keeps the saved key, but only for the same service and address:
    a saved key is never sent somewhere else."""
    old = old or {}
    p = dict(old)
    p["svc"] = body.get("svc") if body.get("svc") in SERVICES else old.get("svc") or "openrouter"
    for k in ("model", "url"):
        if k in body:
            p[k] = str(body[k]).strip()[:300]
    key = str(body.get("key") or "").strip()
    if key:
        p["key"] = protect(key)
    elif old and (p["svc"] != (old.get("svc") or "openrouter") or (p["svc"] == "other" and (p.get("url") or "").rstrip("/") != (old.get("url") or "").rstrip("/"))):
        p["key"] = ""
    name = str(body.get("name") or "").strip()[:60]
    p["name"] = name or old.get("name") or title_of(p)
    return p


def change(cfg: dict, body: dict, new_id: str):
    """One action of the page -> (llm_api, llm_profiles). Actions: select, delete, on=False, clear, otherwise save the shown profile and turn it on.
    Raises ValueError with a message for the user."""
    got, cur = profiles(cfg), dict(cfg.get("llm_api") or {})
    cur_id = active_id(cfg, got)
    if cur_id:
        cur["id"] = cur_id
    by = {p["id"]: p for p in got}

    def use(p, on):
        return {**{k: p.get(k) or "" for k in FIELDS}, "id": p["id"], "on": on}
    if body.get("clear"):
        return {}, []
    if body.get("select"):
        p = by.get(body["select"])
        if not p:
            raise ValueError("Такого профиля нет.")
        return use(p, bool(cur.get("on"))), got
    if body.get("delete"):
        got = [p for p in got if p["id"] != body["delete"]]
        return ({} if body["delete"] == cur_id else cur), got
    if body.get("on") is False and "svc" not in body:
        cur["on"] = False
        return cur, got
    old = by.get(body.get("id"))
    p = draft(old, body)
    if not p.get("key"):
        raise ValueError("Вставьте ключ этого сервиса.")
    if not old:
        if len(got) >= MAX_PROFILES:
            raise ValueError("Профилей уже слишком много: удалите ненужный.")
        p["id"] = new_id
        got.append(p)
    else:
        got = [p if x["id"] == p["id"] else x for x in got]
    return use(p, True), got


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
