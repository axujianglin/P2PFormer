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

def test_batch_roi_replaces_fragment_in_final_response(settings):
    from service.predictor import PredictedPolygon

    class Predictor:
        calls = 0

        def infer(self, pixels, threshold):
            self.calls += 1
            if self.calls == 1:
                points = [[50,20],[80,20],[80,50],[50,50]]
            else:
                # First ROI is (34,4,96,66); recovered roof is (45,20)-(90,50).
                assert pixels.shape[:2] == (62,62)
                points = [[11,16],[56,16],[56,46],[11,46]]
            return [PredictedPolygon(points=points,score=.9,score_cls=.9)]

    predictor=Predictor()
    with TestClient(create_app(settings,predictor)) as c:
        manifest=dict(session_id='roi',width=120,height=100,tile_count=1,
                      tiles=[dict(tile_id='a',x=0,y=0,width=80,height=100)])
        r=c.post('/v1/auto/batches',headers={'X-API-Key':'test-secret'},
                 data={'manifest':json.dumps(manifest)},
                 files={'original':('original.png',image_bytes('PNG',(120,100)))})
        assert r.status_code==200
        c.put('/v1/auto/batches/roi/tiles/a',headers={'X-API-Key':'test-secret'},
              files={'image':('a.png',image_bytes('PNG',(80,100)))})
        result=wait_result(c,'roi')
        assert result['status']=='succeeded'
        assert result['collection_mode']=='replace_auto' and result['fusion_version']=='2'
        assert result['pending_review_count']==0 and len(result['polygons'])==1
        p=result['polygons'][0]
        assert p['fusion_method']=='original_roi_reinference'
        assert p['points']==[[45,20],[90,20],[90,50],[45,50]]
        assert p['original_polygon_ids']==['a/building_0001']
        assert predictor.calls==2
