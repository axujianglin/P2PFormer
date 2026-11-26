# 服务端批次最终集合

现有 `/v1/auto/infer` 不变。批次入口位于 batches.py，融合位于 fusion.py。
本仓库没有远端 P2P 调度/节点回调实现；当前协调器使用同一 predictor 与 GPU 门锁逐片执行，
没有伪造远端节点结果。部署为单进程、单 worker；会话仅在内存中，重启丢失。
生产跨进程调度、持久化和按用户隔离需要另外接入，不能使用多个 uvicorn workers。

## 协议

所有接口均使用原有 `X-API-Key`。session_id 是客户端生成的批次唯一标识，
重用相同 ID 与相同 manifest/原图是幂等的，内容不同返回 409。

1. `POST /v1/auto/batches` multipart: `manifest` JSON 字符串，`original` 原图文件。
2. `PUT /v1/auto/batches/{session_id}/tiles/{tile_id}` multipart: `image` 切片。
3. `GET /v1/auto/batches/{session_id}` 查询进度。
4. `GET /v1/auto/batches/{session_id}/result` 获取冻结快照，未就绪返回 409。
5. `POST /v1/auto/batches/{session_id}/cancel` 请求取消。

manifest 示例：
```json
{"session_id":"image-001-run-1","width":120,"height":100,"tile_count":2,"score_threshold":0.5,"timeout_seconds":300,"tiles":[{"tile_id":"left","x":0,"y":0,"width":80,"height":100},{"tile_id":"right","x":40,"y":0,"width":80,"height":100}]}
```

切片的 width/height 是实际有效图像尺寸，不接收 padding、缩放或旋转。
服务端必须取得原图；不接受仅切片的拼接推断或任意服务器文件路径。
所有切片按 BGR 解码后逐像素验证为原图的声明区域；上传无损裁剪 PNG/TIFF。
如果原图是 JPEG，请从解码后的原图裁剪并保存 PNG，不要再次 JPEG 压缩。
不同批次的切片像素不匹配会拒绝；内容完全相同的区域在语义上不可区分。
重复上传同一 tile_id 不重新推理；像素变化返回 409。不存在公开的任意结果回调入口，
结果始终绑定创建该任务的会话与 tile_id，不能通过客户端注入其他批次结果。

最终结果形状示例（ID 为示意）：
```json
{"session_id":"image-001-run-1","status":"succeeded","received":2,"completed":2,"tile_count":2,"failures":{},"final":true,"coordinate_space":"original_image_pixel","polygons":[{"id":"building_<sha256-prefix>","points":[[50,20],[70,20],[70,50],[50,50]],"score":0.9,"score_cls":0.9,"tile_ids":["left","right"],"original_polygon_ids":["left/building_0001","right/building_0001"],"fusion_method":"quality_selection","review_status":"confirmed"}],"diagnostics":[]}
```

状态：running → finalizing → succeeded / partial_failed / cancelled / timed_out。
成功不表示模型与真实建筑百分之百一致，只表示几何验证通过、任务没有已知失败。
partial_failed 包含推理失败、无效几何或待人工确认候选。
取消/超时会产生现有结果的冻结快照，并列出缺失/未完成切片；迟到结果不再写入快照。
原生 GPU 调用无法安全中断，超时后门锁仍保持到实际推理结束。
finalizing 期间取消不改变已经开始生成的快照。

## 融合与人工处理

按 tile_id 排序映射局部点到原图 x/y，再按跨片真实几何重叠建立连通候选组。
内部切片边缘风险带内的点视为截断迹象，风险带为短边2%，至少2像素、最多16像素。双截断几何距离 ≤1 像素也可成为候选，
但不直接合并。仅 bbox 相交从来不足以合并。
完整轮廓之间 IoU ≥0.8，或完整轮廓包含截断轮廓面积 ≥95%，且组内每个成员都满足证据时，
按完整性、置信度、面积、来源字典序确定性选优。
其他候选组在原图上依次加 50%、100%、200%（至少16像素）上下文重新调用同一模型；通过验证即停止，重复裁区跳过。
重新推理必须唯一覆盖候选组 ≥95%，面积不超过原组2倍（含边缘残片时放宽为8倍），
不能触及 ROI 内边界，也不能吞并组外检测。通过后一次替换全部来源。
这是一套保守的几何启发式，并非建筑语义身份的证明。

重推零轮廓、异常、多解或验证失败时，保留原始候选并标记 pending_review，
diagnostics 包含原因、来源与 ROI。不要把这些候选显示为已完成融合。
相邻独立建筑通常不进入同组；有歧义时保留并交人工确认。
无效几何不进入可渲染集合，诊断记录其来源。
稳定 ID 由 session_id 与排序后的来源 ID 哈希生成，不依赖返回顺序；跨批次不保证同 ID。

客户端接入：先创建批次，再上传全部切片，轮询进度；final=true 后获取 result，
原子替换当前 Auto 集合为 polygons，展示 pending_review 和 failures/diagnostics。
不要附加任何原始切片轮廓，也不要再执行跨片融合。用户编辑集合应由客户端另行保存。

## 配置与 Windows 验证

沿用 service/README.md 的模型、密钥、设备和 P2PFORMER_MAX_UPLOAD_MB。
新增请求项 timeout_seconds 默认300，范围1–3600秒。当前固定限制：32个会话（含终态），
每批最多4096片、原图最多6400万像素，每次ROI重推最多等待60秒，每组最多尝试3个裁区。
达到会话容量前应导出结果并重启；当前不提供持久化和自动淘汰。内存占用随原图及切片增加。
批次截止时间覆盖切片收集/初始推理；终态融合另有每次重推60秒上限。

PowerShell（已按仓库 README 准备模型运行环境）：
```powershell
python -m pip install -e .
python -m pip install -r requirements/service-tests.txt
python -m compileall -q service
python -m pytest -q -o addopts='' tests/test_service_api.py tests/test_batch_api.py tests/test_batch_fusion.py
$env:P2PFORMER_API_KEY='replace-me'
$env:P2PFORMER_CONFIG='D:\models\model.py'
$env:P2PFORMER_CHECKPOINT='D:\models\weights.pth'
$env:P2PFORMER_DEVICE='cuda:0'
python -m service
```
不需要重训或更换模型；CPU协议测试不加载 checkpoint，不能代替真实模型现场验证。


## 排查画面上仍有残片或叠加

fusion_version=2 的结果增加 collection_mode="replace_auto" 和 pending_review_count。
终端 Batch finalized 日志包含最终轮廓数和待处理数。它只在批次终态生成时输出。
仅调用 POST /v1/auto/infer 不会进入批次融合，服务端也无法从单片内容推断原图位置。
客户端必须使用批次协议上传原图、清单和切片，并在 final=true 后整体替换 Auto 集合。
如果仍有多个 pending_review，检查 diagnostics.attempts：zero_predictions、
roi_boundary_truncation、overlaps_external_detection 等会指出失败证据。
服务端不会为了减少可见轮廓数量而隐藏未解决的候选。
本仓库没有客户端代码，新增批次接口不等于既有客户端已经改用该接口。
