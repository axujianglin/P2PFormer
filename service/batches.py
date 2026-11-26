"""In-process batch coordinator; model calls share the legacy GPU gate."""
import asyncio
import hashlib
import json
import logging
import time
from typing import List

import numpy as np
from fastapi import Depends, File, Form, UploadFile
from pydantic import BaseModel, StrictInt, constr
from starlette.concurrency import run_in_threadpool

from .fusion import reconcile

LOGGER = logging.getLogger("p2pformer.service.batches")


class Tile(BaseModel):
    tile_id: constr(min_length=1, max_length=128)
    x: StrictInt
    y: StrictInt
    width: StrictInt
    height: StrictInt


class Manifest(BaseModel):
    session_id: constr(min_length=1, max_length=128)
    width: StrictInt
    height: StrictInt
    tile_count: StrictInt
    tiles: List[Tile]
    score_threshold: float = .5
    timeout_seconds: StrictInt = 300


def install_batches(app, state, settings, authorize, decode, build, Error):
    sessions = {}
    tasks = set()
    app.state.batch_sessions = sessions

    def get(sid):
        if sid not in sessions:
            raise Error(404, 'not_found', 'Unknown batch session')
        return sessions[sid]

    def view(s):
        return dict(session_id=s['m'].session_id, status=s['status'],
                    received=len(s['hashes']), completed=len(s['results']),
                    tile_count=s['m'].tile_count, failures=s['failures'],
                    final=s['snapshot'] is not None)

    async def predict(s, image):
        while not state.try_start_job():
            if s['snapshot'] is not None:
                raise RuntimeError('batch_closed')
            await asyncio.sleep(.02)
        try:
            if not state.model_ready:
                raise RuntimeError('model_unavailable')
            raw = await run_in_threadpool(state.predictor.infer, image, s['m'].score_threshold)
            return build(raw, image.shape[1], image.shape[0], settings.model_name)
        finally:
            state.finish_job()

    async def finish(s, status):
        if s['snapshot'] is not None or s['finishing']:
            return
        s['finishing'] = True
        s['status'] = 'finalizing'
        m = s['m']
        # Reconciliation runs in a worker; route all reinference through the same GPU gate.
        loop = asyncio.get_running_loop()
        async def crop(roi):
            x0,y0,x1,y1 = roi
            job = asyncio.create_task(predict(s, s['original'][y0:y1, x0:x1].copy()))
            tasks.add(job)
            job.add_done_callback(tasks.discard)
            return await asyncio.wait_for(asyncio.shield(job), 60)
        def reinfer(roi):
            return asyncio.run_coroutine_threadsafe(crop(roi), loop).result()
        try:
            polygons, diagnostics = await run_in_threadpool(
                reconcile, m.session_id, m.width, m.height,
                {t.tile_id:t for t in m.tiles}, dict(s['results']), reinfer)
        except Exception:
            polygons, diagnostics = [], [{'reason':'finalization_failed'}]
        s['status'] = status if status != 'succeeded' else (
            'partial_failed' if s['failures'] or diagnostics else 'succeeded')
        s['snapshot'] = dict(**view(s), coordinate_space='original_image_pixel',
                             polygons=polygons, diagnostics=diagnostics,
                             collection_mode='replace_auto', fusion_version='2',
                             pending_review_count=sum(p['review_status'] == 'pending_review' for p in polygons))
        s['snapshot']['final'] = True
        LOGGER.info(
            'Batch finalized session=%r fusion_version=2 status=%s tiles=%d/%d '
            'final_polygons=%d pending_review=%d diagnostics=%d collection_mode=replace_auto',
            m.session_id, s['status'], len(s['results']), m.tile_count, len(polygons),
            s['snapshot']['pending_review_count'], len(diagnostics))
        s['images'].clear()
        s['original'] = None

    async def worker(s):
        while s['snapshot'] is None and not s['finishing']:
            if s['stop'] or time.monotonic() >= s['deadline']:
                status = s['stop'] or 'timed_out'
                for t in s['m'].tiles:
                    if t.tile_id not in s['results'] and t.tile_id not in s['failures']:
                        s['failures'][t.tile_id] = 'cancelled' if status == 'cancelled' else 'missing_or_unfinished_tile'
                await finish(s, status)
                return
            pending = sorted(set(s['images']) - set(s['results']) - set(s['failures']))
            if pending:
                tid = pending[0]
                try:
                    # Timeout stops waiting, but the GPU gate stays held until the actual call returns.
                    job = asyncio.create_task(predict(s, s['images'][tid]))
                    tasks.add(job)
                    job.add_done_callback(tasks.discard)
                    while not job.done() and not s['stop'] and time.monotonic() < s['deadline']:
                        await asyncio.sleep(.02)
                    if not job.done():
                        raise TimeoutError()
                    s['results'][tid] = job.result()
                except Exception:
                    s['failures'][tid] = 'inference_failed_or_timeout'
            elif len(s['results']) + len(s['failures']) == s['m'].tile_count:
                await finish(s, 'succeeded')
                return
            else:
                await asyncio.sleep(.05)

    async def read(upload):
        data = await upload.read(settings.max_upload_bytes + 1)
        if not data or len(data) > settings.max_upload_bytes:
            raise Error(413, 'image_size', 'Empty or oversized upload')
        return data, decode(data, upload.filename)

    @app.post('/v1/auto/batches', dependencies=[Depends(authorize)])
    async def create(manifest: str = Form(...), original: UploadFile = File(...)):
        try:
            m = Manifest.parse_raw(manifest)
            ids = [t.tile_id for t in m.tiles]
            if not (0 < m.width * m.height <= 64000000 and m.width > 0 and m.height > 0):
                raise ValueError('dimensions')
            if not (0 < m.tile_count == len(ids) <= 4096 and len(set(ids)) == len(ids)):
                raise ValueError('tile IDs')
            if not (0 <= m.score_threshold <= 1 and 1 <= m.timeout_seconds <= 3600):
                raise ValueError('threshold or timeout')
            if not (all(t.x >= 0 and t.y >= 0 and t.width > 0 and t.height > 0 and
                       t.x+t.width <= m.width and t.y+t.height <= m.height for t in m.tiles)):
                raise ValueError('tile ranges')
        except Exception:
            raise Error(400, 'invalid_manifest', 'Invalid dimensions, IDs, threshold or tile ranges')
        data, pixels = await read(original)
        if pixels.shape[:2] != (m.height, m.width):
            raise Error(400, 'dimension_mismatch', 'Original dimensions differ from manifest')
        fingerprint = hashlib.sha256((json.dumps(json.loads(m.json()), sort_keys=True)).encode() + data).hexdigest()
        if m.session_id in sessions:
            s = get(m.session_id)
            if s['fingerprint'] != fingerprint:
                raise Error(409, 'session_conflict', 'Session ID already belongs to another batch')
            return view(s)
        if len(sessions) >= 32:
            raise Error(503, 'session_capacity', 'Session capacity reached; restart after exporting results')
        s = dict(m=m, fingerprint=fingerprint, original=pixels, images={}, hashes={}, results={}, failures={},
                 status='running', snapshot=None, finishing=False, stop=None,
                 deadline=time.monotonic()+m.timeout_seconds)
        sessions[m.session_id] = s
        task = asyncio.create_task(worker(s))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return view(s)

    @app.put('/v1/auto/batches/{sid}/tiles/{tid}', dependencies=[Depends(authorize)])
    async def upload(sid: str, tid: str, image: UploadFile = File(...)):
        s = get(sid)
        t = next((t for t in s['m'].tiles if t.tile_id == tid), None)
        if t is None:
            raise Error(400, 'unknown_tile', 'Tile is not in this batch')
        data, pixels = await read(image)
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        if tid in s['hashes']:
            if s['hashes'][tid] != digest:
                raise Error(409, 'tile_conflict', 'Duplicate tile has different pixels')
            return view(s)
        if s['snapshot'] is not None or s['finishing'] or s['stop']:
            raise Error(409, 'batch_closed', 'Batch no longer accepts tiles')
        if pixels.shape[:2] != (t.height,t.width) or not np.array_equal(
                pixels, s['original'][t.y:t.y+t.height,t.x:t.x+t.width]):
            raise Error(400, 'tile_pixel_mismatch', 'Tile must exactly match the declared original region')
        s['images'][tid] = pixels
        s['hashes'][tid] = digest
        return view(s)

    @app.get('/v1/auto/batches/{sid}', dependencies=[Depends(authorize)])
    async def progress(sid: str):
        return view(get(sid))

    @app.get('/v1/auto/batches/{sid}/result', dependencies=[Depends(authorize)])
    async def result(sid: str):
        s = get(sid)
        if s['snapshot'] is None:
            raise Error(409, 'batch_not_terminal', 'Final snapshot is not ready')
        return s['snapshot']

    @app.post('/v1/auto/batches/{sid}/cancel', dependencies=[Depends(authorize)])
    async def cancel(sid: str):
        s = get(sid)
        if not s['finishing'] and s['snapshot'] is None:
            s['stop'] = 'cancelled'
        return view(s)
