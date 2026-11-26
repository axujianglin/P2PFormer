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

def test_near_seam_inset_fragment_is_reinferred():
    # Model stops 1.5px short of the seam, rather than exactly at x=80.
    def re(roi):
        return [rect(50-roi[0],20-roi[1],100-roi[0],50-roi[1])]
    out,d=fuse({'a':[rect(50,20,78.5,50)]},re=re)
    assert len(out)==1 and not d
    assert out[0]['points'][1][0]==100
    assert out[0]['fusion_method']=='original_roi_reinference'


def test_fragment_can_recover_more_than_twice_its_area():
    def re(roi):
        # Bounds are inside the expanded ROI and source is fully covered.
        return [rect(45-roi[0],15-roi[1],95-roi[0],60-roi[1])]
    out,d=fuse({'a':[rect(65,20,80,50)]},re=re)
    assert len(out)==1 and not d
    assert out[0]['points'][0]==[45,15]


def test_zero_first_crop_retries_with_more_context():
    calls=[]
    def re(roi):
        calls.append(roi)
        if len(calls)==1:
            return []
        return [rect(45-roi[0],20-roi[1],100-roi[0],50-roi[1])]
    out,d=fuse({'a':[rect(50,20,80,50)]},re=re)
    assert len(calls)==2 and calls[1][2]>calls[0][2]
    assert len(out)==1 and not d


def test_contained_fragment_without_edge_flag_requires_roi_validation():
    # Neither detection is near a tile edge; score alone must not select the fragment.
    tiles={'a':tile(0,120),'b':tile(0,120)}
    def re(roi):
        return [rect(30-roi[0],20-roi[1],70-roi[0],60-roi[1])]
    out,d=fuse({'a':[rect(30,20,70,60,score=.8)],
                'b':[rect(40,30,60,50,score=.99)]},tiles,re)
    assert len(out)==1 and not d
    assert out[0]['original_polygon_ids']==['a/p','b/p']
    assert out[0]['fusion_method']=='original_roi_reinference'


def test_failed_retries_include_traceable_reasons():
    out,d=fuse({'a':[rect(50,20,80,50)]})
    assert out[0]['review_status']=='pending_review'
    assert d[0]['attempts']
    assert all(a['reason']=='zero_predictions' for a in d[0]['attempts'])
