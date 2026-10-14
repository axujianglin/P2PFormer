# P2PFormer SHORTER HTTP service

This module exposes the repository's existing single-image P2PFormer path as
the synchronous SHORTER SAMPoly protocol. The model is loaded once during
FastAPI startup. Requests are rejected with HTTP 503 while the single GPU slot
is occupied; there is no in-process queue and no asynchronous jobs API.

## Install and run

Install P2PFormer as described in the repository README, then install the
small service dependency set:

```bash
pip install -e .
pip install -r requirements/service.txt
```

Example Linux configuration:

```bash
export P2PFORMER_API_KEY='replace-with-a-secret'
export P2PFORMER_CONFIG='/opt/p2pformer/configs/p2pformer.py'
export P2PFORMER_CHECKPOINT='/opt/p2pformer/checkpoints/model.pth'
export P2PFORMER_MODEL_VERSION='whu-mix-r50-v1'
export P2PFORMER_DEVICE='cuda:0'
export P2PFORMER_HOST='127.0.0.1'
export P2PFORMER_PORT='8000'
export P2PFORMER_MAX_UPLOAD_MB='64'
python -m service.app
```

An empty API key is rejected by default. It can only be enabled explicitly
with `P2PFORMER_ALLOW_EMPTY_API_KEY=true`, which is intended for trusted local
development environments.

Run the isolated service tests (they use a mock predictor and do not load a
checkpoint):

```bash
pip install -r requirements/service-tests.txt
pytest -q -o addopts='' tests/test_service_api.py
```

## Requests

```bash
curl -sS -H 'X-Request-ID: health-001' \
  http://127.0.0.1:8000/health
```

```bash
curl -sS -H 'X-API-Key: replace-with-a-secret' \
  -H 'X-Request-ID: capabilities-001' \
  http://127.0.0.1:8000/v1/capabilities
```

```bash
curl -sS -X POST \
  -H 'X-API-Key: replace-with-a-secret' \
  -H 'X-Request-ID: infer-001' \
  -F 'image=@/data/tile.tif' \
  -F 'score_threshold=0.5' \
  http://127.0.0.1:8000/v1/auto/infer
```

The response always uses original-image pixel coordinates. TIFF georeference
metadata is intentionally ignored by this protocol.

