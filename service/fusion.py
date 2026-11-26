"""Deterministic, conservative server-side polygon reconciliation."""
import hashlib
import json
import math
from shapely.geometry import Polygon
from shapely.ops import unary_union


def stable_id(session, sources):
    return 'building_' + hashlib.sha256(json.dumps([session, sorted(sources)], separators=(',', ':')).encode()).hexdigest()[:24]


def reconcile(session, width, height, tiles, results, reinfer):
    records = []
    diagnostics = []
    for tid in sorted(results):
        t = tiles[tid]
        for p in results[tid]:
            points = [[x + t.x, y + t.y] for x, y in p.points]
            g = Polygon(points)
            source = tid + '/' + p.id
            if not g.is_valid or g.area <= 0:
                diagnostics.append({'reason': 'invalid_geometry', 'sources': [source]})
                continue
            # Only internal tile edges indicate a potentially truncated building.
            margin = max(2, min(16, min(t.width, t.height) * .02))
            clipped = any((t.x > 0 and x <= margin) or (t.y > 0 and y <= margin) or
                          (t.x + t.width < width and x >= t.width - margin) or
                          (t.y + t.height < height and y >= t.height - margin)
                          for x, y in p.points)
            records.append(dict(g=g, points=points, score=p.score, score_cls=p.score_cls,
                                sources=[source], tile_ids=[tid], clipped=clipped))
    # Geometry overlap AND distinct tile provenance; bbox contact alone is never evidence.
    groups = []
    unseen = set(range(len(records)))
    while unseen:
        group = {min(unseen)}
        unseen -= group
        while True:
            linked = {j for j in unseen if any(
                records[i]['tile_ids'] != records[j]['tile_ids'] and
                (records[i]['g'].intersection(records[j]['g']).area /
                 min(records[i]['g'].area, records[j]['g'].area) >= .5 or
                 (records[i]['clipped'] and records[j]['clipped'] and
                  records[i]['g'].distance(records[j]['g']) <= 1))
                for i in group)}
            if not linked:
                break
            group |= linked
            unseen -= linked
        groups.append([records[i] for i in sorted(group)])
    output = []
    for group in groups:
        sources = sorted(s for r in group for s in r['sources'])
        tids = sorted({t for r in group for t in r['tile_ids']})
        best = sorted(group, key=lambda r: (r['clipped'], -r['score'], -r['g'].area, r['sources']))[0]
        complete = not best['clipped']
        # Require every member to agree with the selected complete roof.
        confirmed = complete and all(
            (best['g'].intersection(r['g']).area / r['g'].area >= .95 if r['clipped'] else
             best['g'].intersection(r['g']).area / best['g'].union(r['g']).area >= .8)
            for r in group)
        method = 'single' if len(group) == 1 else 'quality_selection'
        if not confirmed:
            union = unary_union([r['g'] for r in group])
            x0, y0, x1, y1 = union.bounds
            attempts = []
            seen_rois = set()
            # A fragment's bounds do not establish the full building extent.
            # Retry with increasing original-image context instead of treating
            # one crop/zero detection as a definitive answer.
            for scale in (.5, 1.0, 2.0):
                pad = max(16, math.ceil(max(x1-x0, y1-y0) * scale))
                roi = (max(0, math.floor(x0)-pad), max(0, math.floor(y0)-pad),
                       min(width, math.ceil(x1)+pad), min(height, math.ceil(y1)+pad))
                if roi in seen_rois:
                    continue
                seen_rois.add(roi)
                try:
                    predictions = list(reinfer(roi))
                except Exception:
                    attempts.append({'roi': list(roi), 'reason': 'reinference_exception'})
                    continue
                candidates = []
                rejected = []
                for p in predictions:
                    pts = [[x+roi[0], y+roi[1]] for x,y in p.points]
                    g = Polygon(pts)
                    reason = None
                    if not g.is_valid or g.area <= 0:
                        reason = 'invalid_geometry'
                    elif g.intersection(union).area / union.area < .95:
                        reason = 'insufficient_source_coverage'
                    elif any((roi[0]>0 and x<=roi[0]+2) or (roi[1]>0 and y<=roi[1]+2) or
                             (roi[2]<width and x>=roi[2]-2) or (roi[3]<height and y>=roi[3]-2) for x,y in pts):
                        reason = 'roi_boundary_truncation'
                    elif any(g.intersection(r['g']).area / r['g'].area > .1
                             for r in records if r not in group):
                        reason = 'overlaps_external_detection'
                    # Large recovery is allowed for edge fragments, but still
                    # requires source coverage and no neighboring detection.
                    elif g.area > union.area * (8 if any(r['clipped'] for r in group) else 2):
                        reason = 'unsupported_area_growth'
                    if reason:
                        rejected.append(reason)
                        continue
                    candidates.append(dict(g=g, points=pts, score=p.score, score_cls=p.score_cls))
                if len(candidates) == 1:
                    best = candidates[0]
                    confirmed = True
                    method = 'original_roi_reinference'
                    break
                attempts.append({'roi': list(roi), 'reason':
                                 'zero_predictions' if not predictions else
                                 'multiple_valid_candidates' if len(candidates)>1 else 'validation_rejected',
                                 'rejected': sorted(set(rejected))})
            if not confirmed:
                diagnostics.append({'reason': 'reinference_failed_or_ambiguous',
                                    'sources': sources, 'roi': list(roi), 'attempts': attempts})
        selected = [best] if confirmed else group
        for r in selected:
            provenance = sources if confirmed else r['sources']
            output.append(dict(id=stable_id(session, provenance), points=r['points'],
                               score=r['score'], score_cls=r['score_cls'],
                               tile_ids=tids if confirmed else r['tile_ids'], original_polygon_ids=provenance,
                               fusion_method=method if confirmed else 'unresolved',
                               review_status='confirmed' if confirmed else 'pending_review'))
    return sorted(output, key=lambda p: p['id']), diagnostics
