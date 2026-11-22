"""Настоящие обработчики Workspace, временные хранилища и поддельный ComfyUI. Без GPU и сети."""
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import server

PID = '452e648b-04ee-4200-8f13-e49a934535ab'


def call(handler, body=None):
    req = SimpleNamespace(can_read_body=body is not None, json=AsyncMock(return_value=body or {}))
    r = asyncio.run(handler(req))
    return r.status, json.loads(r.text)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for key, value in [('DATA_DIR', tmp_path), ('QUEUE_FILE', tmp_path / 'queue.json'),
                       ('LIBRARY_FILE', tmp_path / 'library.json'), ('LONG_DIR', tmp_path / 'long'),
                       ('REFMODES_FILE', tmp_path / 'refs.json'), ('REFMODES_DIR', tmp_path / 'refs')]:
        monkeypatch.setattr(server, key, value)
    server.QUEUE_FILE.write_text(json.dumps([{'pid': PID, 'saved': time.time(), 'graph': {'1': {'class_type': 'test'}},
        'meta': {'mode': 'custom', 'prompt': 'scene', 'seed': 7}, 'stages': {'1': 'sample'}}]), encoding='utf-8')
    remote = SimpleNamespace(queue={'queue_running': [], 'queue_pending': []}, history={}, posts=[], online=True)
    def open_app():
        app = server.App()
        app.config.data.update(engine='external', llm_variant='qwen3vl-8b')
        async def comfy(method, path, **kwargs):
            if not remote.online:
                raise OSError('test: offline')
            if (method, path) == ('GET', '/queue'):
                return 200, remote.queue
            if method == 'GET' and path.startswith('/history/'):
                pid = path.split('/')[-1]
                return 200, {pid: remote.history[pid]} if pid in remote.history else {}
            if (method, path) == ('POST', '/free'):
                return 200, {}
            if (method, path) == ('POST', '/prompt'):
                remote.posts.append(kwargs['json']); return 200, {'prompt_id': PID, 'number': 1}
            raise AssertionError((method, path))
        app.comfy_json = comfy
        return app
    return open_app, remote


def test_offline_recovery_keeps_saved_entry_even_after_reopening(env):
    open_app, remote = env
    remote.online = False
    status, _ = call(open_app().h_queue_resume)
    assert status == 503
    assert call(open_app().h_queue_saved)[1]['count'] == 1
    assert remote.posts == []


@pytest.mark.parametrize(('bucket', 'state'), [('queue_running', 'running'), ('queue_pending', 'queued')])
def test_reattaches_without_resubmitting_and_tracks_completion(env, bucket, state):
    open_app, remote = env
    remote.queue[bucket].append([1, PID])
    app = open_app()
    status, result = call(app.h_queue_resume)
    assert status == 200
    assert [(j['prompt_id'], j['state']) for j in result['jobs']] == [(PID, state)]
    # Повторный запуск Workspace должен всё ещё помнить исходное задание.
    assert call(open_app().h_queue_saved)[1]['count'] == 1
    remote.history[PID] = finished()
    asyncio.run(app.on_comfy_event({'type': 'execution_success', 'data': {'prompt_id': PID}}))
    assert [x['prompt_id'] for x in call(app.h_library)[1]] == [PID]
    assert call(open_app().h_queue_saved)[1]['count'] == 0
    assert remote.posts == []


def finished():
    return {'status': {'completed': True, 'status_str': 'success'}, 'outputs': {
        'save': {'videos': [{'type': 'output', 'subfolder': '', 'filename': 'scene.mp4'}]}}}


def test_collects_completed_scene_from_history_without_rerendering(env):
    open_app, remote = env
    remote.history[PID] = finished()
    app = open_app()
    status, result = call(app.h_queue_resume)
    assert status == 200 and result['finished'][0]['items'][0]['prompt_id'] == PID
    assert call(open_app().h_queue_saved)[1]['count'] == 0
    call(app.h_queue_resume)
    assert [x['prompt_id'] for x in call(open_app().h_library)[1]] == [PID]
    assert remote.posts == []


def test_only_explicit_retry_of_a_still_missing_id_can_submit(env):
    open_app, remote = env
    app = open_app()
    status, result = call(app.h_queue_resume)
    assert remote.posts == [] and status == 200
    assert [j['prompt_id'] for j in result['missing']] == [PID]
    assert call(open_app().h_queue_saved)[1]['count'] == 1
    status, result = call(app.h_queue_resume, {'restart_id': PID})
    assert status == 200 and len(result['jobs']) == 1 and len(remote.posts) == 1
    call(app.h_queue_resume, {'restart_id': PID})  # повторный клик / второе окно
    assert len(remote.posts) == 1


def test_rechecks_history_after_the_user_saw_the_restart_button(env):
    open_app, remote = env
    app = open_app()
    call(app.h_queue_resume)
    remote.history[PID] = finished()
    status, result = call(app.h_queue_resume, {'restart_id': PID})
    assert remote.posts == [] and status == 200 and result['finished']


def test_error_entry_with_partial_output_is_not_a_finished_clip(env):
    open_app, remote = env
    app = open_app()
    remote.history[PID] = finished()
    remote.history[PID]['status'] = {'status_str': 'error', 'completed': False, 'messages': [
        ['execution_error', {'exception_message': 'test: not enough memory'}]]}
    status, result = call(app.h_queue_resume, {'restart_id': PID})
    assert status == 200 and 'not enough memory' in result['failed'][0]['message']
    assert call(app.h_library)[1] == [] and remote.posts == []


def test_failed_history_lookup_preserves_that_job_and_recovers_other_jobs(env):
    open_app, remote = env
    app = open_app()
    other = 'ec790830-fbff-426f-9ba6-aa2ec7217fb8'
    app.saved_queue.append(dict(app.saved_queue[0], pid=other)); app.save_queue()
    remote.history[other] = finished()
    api = app.comfy_json
    async def sometimes_unavailable(method, path, **kw):
        if path == '/history/' + PID:
            return 502, {}
        return await api(method, path, **kw)
    app.comfy_json = sometimes_unavailable
    _, result = call(app.h_queue_resume, {'restart_id': PID})
    assert [x['prompt_id'] for x in result['deferred']] == [PID]
    assert [x['prompt_id'] for x in result['finished']] == [other]
    assert [x['prompt_id'] for x in call(open_app().h_queue_saved)[1]['items']] == [PID]
    assert not remote.posts


def test_forget_only_the_displayed_entries(env):
    open_app, _ = env
    app = open_app()
    other = 'ec790830-fbff-426f-9ba6-aa2ec7217fb8'
    app.saved_queue.append(dict(app.saved_queue[0], pid=other)); app.save_queue()
    assert call(app.h_queue_discard, {'ids': [PID]})[0] == 200
    assert [x['prompt_id'] for x in call(open_app().h_queue_saved)[1]['items']] == [other]


def test_unknown_history_is_not_permission_to_restart(env):
    open_app, _ = env
    app = open_app()
    api = app.comfy_json
    async def malformed(method, path, **kw):
        if path.startswith('/history'):
            return 200, {'error': 'test: history unavailable'}
        return await api(method, path, **kw)
    app.comfy_json = malformed
    _, result = call(app.h_queue_resume, {'restart_id': PID})
    assert result['deferred'] and not result['jobs']
    assert call(open_app().h_queue_saved)[1]['count'] == 1


def test_parallel_restart_requests_submit_once(env):
    open_app, remote = env
    app = open_app()
    async def run():
        def request():
            return SimpleNamespace(can_read_body=True, json=AsyncMock(return_value={'restart_id': PID}))
        await asyncio.gather(app.h_queue_resume(request()), app.h_queue_resume(request()))
    asyncio.run(run())
    assert len(remote.posts) == 1 and remote.posts[0]['prompt_id'] == PID


def test_cannot_restart_on_another_comfy_server(env):
    open_app, remote = env
    app = open_app()
    remote.queue['queue_running'].append([1, PID])
    call(app.h_queue_resume)
    other_app = open_app()
    other_app.comfy_override = 'http://another.example.test:8188'
    remote.queue['queue_running'].clear()
    _, result = call(other_app.h_queue_resume, {'restart_id': PID})
    assert result['deferred'] and not remote.posts
    assert call(open_app().h_queue_saved)[1]['count'] == 1
