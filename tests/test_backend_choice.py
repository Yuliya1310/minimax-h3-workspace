"""Проверки выбора сборки движка и страховок при скачивании. Сеть и GPU не нужны.

Эти тесты закрепляют баг: агент всегда качал llama.cpp только с Vulkan, а на машине,
где у загрузчика Vulkan нет зарегистрированного драйвера, движок молча считал на процессоре
(~6 ток/с вместо ~105 на той же RTX 4090). Теперь сборка выбирается по видеокарте, проверяется
через `llama-server --list-devices`, а скачивание отказывается стартовать без места на диске.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import agent  # noqa: E402
import engine  # noqa: E402


class FakeInstaller:
    """Заменяет engine.Installer: ничего не качает, события прогресса никуда не уходят."""

    def __init__(self):
        self.steps = []

    async def step(self, sid, state, **kw):
        self.steps.append((sid, state, kw.get("label", "")))

    async def download(self, *a, **k):
        raise AssertionError("the test must not go to the network")


class Result:
    def __init__(self, text):
        self.stdout, self.stderr, self.returncode = text, "", 0


def test_backend_plan_prefers_cuda_on_nvidia():
    assert agent.backend_plan({"nvidia": True}) == ["cuda", "vulkan", "cpu"]
    assert agent.backend_plan({"nvidia": False}) == ["vulkan", "cpu"]
    assert agent.backend_plan({}) == ["vulkan", "cpu"]
    assert agent.backend_plan({"nvidia": True})[-1] == "cpu"


def test_list_devices_reads_the_engine_output(monkeypatch):
    monkeypatch.setattr(agent.subprocess, "run", lambda *a, **k: Result("Available devices:\n  (none)\n"))
    assert agent.list_devices("llama-server.exe") == []
    monkeypatch.setattr(agent.subprocess, "run", lambda *a, **k: Result(
        "ggml_vulkan: found 0 Vulkan devices\nAvailable devices:\n"
        "  CUDA0: NVIDIA GeForce RTX 4090 (24563 MiB, 23036 MiB free)\n"))
    assert agent.list_devices("llama-server.exe") == [
        "CUDA0: NVIDIA GeForce RTX 4090 (24563 MiB, 23036 MiB free)"]


def test_list_devices_without_exe_is_empty():
    assert agent.list_devices(None) == []


def test_placement_is_read_from_the_engine_log(tmp_path):
    log = tmp_path / "llama.log"
    srv = agent.LlamaServer(tmp_path, log)
    assert srv.placement() is None  # лога ещё нет
    log.write_text("load_tensors: offloading 0/33 layers to GPU\n", encoding="utf-8")
    assert srv.placement() is None  # «offloading» — это план, а не результат
    log.write_text("load_tensors: offloaded 0/33 layers to CPU\n", encoding="utf-8")
    assert srv.placement() == {"layers": 0, "total": 33, "where": "CPU"}
    log.write_text("load_tensors: offloaded 33/33 layers to GPU\n", encoding="utf-8")
    assert srv.placement() == {"layers": 33, "total": 33, "where": "GPU"}


def test_engine_falls_back_until_a_build_sees_a_device(tmp_path, monkeypatch):
    seen = []

    async def fake_fetch(installer, root, llama_dir, backend):
        seen.append(backend)
        llama_dir.mkdir(parents=True, exist_ok=True)
        (llama_dir / "llama-server.exe").write_text("stub", encoding="utf-8")
        return {"title": backend}

    monkeypatch.setattr(agent, "_fetch_engine", fake_fetch)
    monkeypatch.setattr(agent, "list_devices", lambda exe: [])
    asyncio.run(agent._install_engine(FakeInstaller(), tmp_path, tmp_path / "llama.cpp", {"nvidia": True}))
    assert seen == ["cuda", "vulkan", "cpu"]  # сборку без карт выбрасываем, CPU-сборка — крайний случай
    info = json.loads((tmp_path / agent.BACKEND_FILE).read_text(encoding="utf-8"))
    assert info["backend"] == "cpu" and info["devices"] == []


def test_engine_keeps_the_first_build_that_works(tmp_path, monkeypatch):
    seen = []

    async def fake_fetch(installer, root, llama_dir, backend):
        seen.append(backend)
        llama_dir.mkdir(parents=True, exist_ok=True)
        (llama_dir / "llama-server.exe").write_text("stub", encoding="utf-8")
        return {"title": backend}

    monkeypatch.setattr(agent, "_fetch_engine", fake_fetch)
    monkeypatch.setattr(agent, "list_devices", lambda exe: ["CUDA0: NVIDIA GeForce RTX 4090"])
    asyncio.run(agent._install_engine(FakeInstaller(), tmp_path, tmp_path / "llama.cpp", {"nvidia": True}))
    assert seen == ["cuda"]
    info = json.loads((tmp_path / agent.BACKEND_FILE).read_text(encoding="utf-8"))
    assert info == {"backend": "cuda", "title": "cuda", "build": agent.LLAMA_BUILD,
                    "devices": ["CUDA0: NVIDIA GeForce RTX 4090"]}


def test_download_refuses_when_the_disk_is_too_small(tmp_path):
    inst = engine.Installer(tmp_path, session=None, emit=None)
    with pytest.raises(RuntimeError) as err:
        asyncio.run(inst.download("http://127.0.0.1:9/none.bin", tmp_path / "huge.bin",
                                  "models", "огромная модель", 10 ** 15))
    assert "Мало места" in str(err.value)
    assert not (tmp_path / "huge.bin").exists()
