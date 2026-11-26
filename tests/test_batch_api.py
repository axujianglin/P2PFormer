import time
from fastapi.testclient import TestClient
from test_service_api import settings, MockPredictor, image_bytes
from service.app import create_app
import json


def wait_result(client, sid):
    for _ in range(300):
        response=client.get('/v1/auto/batches/'+sid+'/result',headers={'X-API-Key':'test-secret'})
        if response.status_code==200:
            return response.json()
        time.sleep(.01)
    raise AssertionError('no terminal snapshot')


def create(client, sid='s', timeout=5, tiles=None):
    manifest=dict(session_id=sid,width=32,height=20,tile_count=len(tiles or [1]),
                  tiles=tiles or [dict(tile_id='a',x=0,y=0,width=32,height=20)],timeout_seconds=timeout)
    return client.post('/v1/auto/batches',headers={'X-API-Key':'test-secret'},
                       data={'manifest':json.dumps(manifest)},files={'original':('original.png',image_bytes('PNG'))})


def test_batch_idempotent_upload_and_frozen_snapshot(settings):
    with TestClient(create_app(settings,MockPredictor())) as c:
        assert create(c).status_code==200
        assert create(c).status_code==200
        for _ in range(2):
            r=c.put('/v1/auto/batches/s/tiles/a',headers={'X-API-Key':'test-secret'},
                    files={'image':('a.png',image_bytes('PNG'))})
            assert r.status_code==200
        result=wait_result(c,'s')
        assert result['status']=='succeeded' and result['polygons']==[]
        assert result==wait_result(c,'s') and result['received']==1


def test_missing_tiles_timeout(settings):
    with TestClient(create_app(settings,MockPredictor())) as c:
        assert create(c,timeout=1).status_code==200
        result=wait_result(c,'s')
        assert result['status']=='timed_out' and 'a' in result['failures']


def test_cancel(settings):
    with TestClient(create_app(settings,MockPredictor())) as c:
        create(c)
        c.post('/v1/auto/batches/s/cancel',headers={'X-API-Key':'test-secret'})
        assert wait_result(c,'s')['status']=='cancelled'


def test_duplicate_id_and_foreign_tile(settings):
    with TestClient(create_app(settings,MockPredictor())) as c:
        t=dict(tile_id='a',x=0,y=0,width=32,height=20)
        assert create(c,tiles=[t,t]).status_code==400
        create(c)
        assert c.put('/v1/auto/batches/s/tiles/foreign',headers={'X-API-Key':'test-secret'},
                     files={'image':('a.png',image_bytes('PNG'))}).status_code==400

def test_dimension_and_session_conflicts(settings):
    with TestClient(create_app(settings,MockPredictor())) as c:
        assert create(c).status_code==200
        assert create(c,timeout=6).status_code==409
        r=c.put('/v1/auto/batches/s/tiles/a',headers={'X-API-Key':'test-secret'},
                files={'image':('bad.png',image_bytes('PNG',(16,20)))})
        assert r.status_code==400
        c.post('/v1/auto/batches/s/cancel',headers={'X-API-Key':'test-secret'})
        wait_result(c,'s')


def test_model_failure_produces_partial_snapshot(settings):
    with TestClient(create_app(settings,MockPredictor(error=RuntimeError('failure')))) as c:
        create(c)
        c.put('/v1/auto/batches/s/tiles/a',headers={'X-API-Key':'test-secret'},
              files={'image':('a.png',image_bytes('PNG'))})
        result=wait_result(c,'s')
        assert result['status']=='partial_failed' and 'a' in result['failures']
