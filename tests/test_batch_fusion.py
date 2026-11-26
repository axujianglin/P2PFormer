from types import SimpleNamespace as NS
import pytest
from service.fusion import reconcile


def rect(x0,y0,x1,y1, ident='p', score=.9):
    return NS(points=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]], id=ident, score=score, score_cls=score)


def tile(x=0, w=80):
    return NS(x=x,y=0,width=w,height=100)


def fuse(results, tiles=None, re=None):
    return reconcile('s',120,100,tiles or {'a':tile(),'b':tile(40)},results,re or (lambda roi: []))


def test_complete_duplicate_quality_order_and_stable_ids():
    a,b=rect(50,20,70,50,score=.8),rect(10,20,30,50)
    first,diag=fuse({'a':[a],'b':[b]})
    second,_=fuse({'b':[b],'a':[a]})
    assert first==second and not diag and len(first)==1
    assert first[0]['score']==.9
    assert first[0]['tile_ids']==['a','b']
    assert first[0]['points'][0]==[50,20]


def test_complete_replaces_truncated():
    out,d=fuse({'a':[rect(50,20,80,50)],'b':[rect(10,20,50,50)]})
    assert len(out)==1 and not d
    assert out[0]['fusion_method']=='quality_selection'


def test_both_truncated_reinfer_replaces_all():
    tiles={'a':tile(0,70),'b':tile(50,70)}
    def re(roi):
        return [rect(40-roi[0],20-roi[1],90-roi[0],50-roi[1])]
    out,d=fuse({'a':[rect(40,20,70,50)],'b':[rect(0,20,40,50)]},tiles,re)
    assert len(out)==1 and not d
    assert out[0]['fusion_method']=='original_roi_reinference'
    assert out[0]['original_polygon_ids']==['a/p','b/p']


def test_zero_reinference_preserves_pending():
    out,d=fuse({'a':[rect(40,20,80,50)],'b':[rect(0,20,50,50)]})
    assert len(out)==2 and d
    assert all(p['review_status']=='pending_review' for p in out)


def test_neighbors_stay_separate():
    out,d=fuse({'a':[rect(45,20,55,40)],'b':[rect(20,20,30,40)]})
    assert len(out)==2 and not d


def test_three_tiles_transitive_and_order():
    tiles={t:tile(0,120) for t in ['a','b','c']}
    results={t:[rect(20,20,50,50)] for t in ['c','a','b']}
    out,d=fuse(results,tiles)
    assert len(out)==1 and out[0]['tile_ids']==['a','b','c'] and not d

def test_reinference_exception_keeps_originals():
    def fail(roi):
        raise RuntimeError('GPU failure')
    out,d=fuse({'a':[rect(40,20,80,50)],'b':[rect(0,20,50,50)]},re=fail)
    assert len(out)==2 and d
    assert all(p['fusion_method']=='unresolved' for p in out)


def test_bbox_overlap_without_polygon_overlap_is_not_duplicate():
    a=NS(points=[[45,20],[55,20],[45,30]],id='p',score=.9,score_cls=.9)
    b=NS(points=[[15,30],[15,22],[7,30]],id='p',score=.9,score_cls=.9)
    out,d=fuse({'a':[a],'b':[b]})
    assert len(out)==2 and not d
