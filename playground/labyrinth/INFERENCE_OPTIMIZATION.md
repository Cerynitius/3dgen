# 推理优化备忘录（2026-08-22）

> 对象：Endless Labyrinth 的 WorldMirror 逐 tile 前向。
> 现状基线（实测，RTX PRO 6000 单卡）：~32 视角/476² → 2.5-4 s/tile，
> 其中视图渲染 ~1.0s、模型前向 ~2.0s、后处理 ~0.02s。实时消耗线 0.38 t/s。

## 已证伪（别再做）

| 方向 | 实测 |
|------|------|
| 多 worker 并行 | w1 0.24 t/s → w8 0.13 t/s，单调劣化。单 build 已吃满 GPU，并行只争锁 |
| batch=2 前向 [2,N,3,H,W] | 模型条件路径不认 batch 维（pose 广播维度冲突），结构性拒绝 |
| 降分辨率 476→420 | 3.39s vs 3.33s，噪声内。渲染像素不是瓶颈 |
| 砍视角数 | 视角=质量的生命线（bench 实测随视角数涨），砍质量换速度是本末倒置 |

## 候选路线（按期望收益排序）

### 1. torch.compile（预期 1.3-1.8x，首选）
WorldMirror 是大号 ViT（embed 1024, depth 24），compile 收益主要看 attention/MLP 融合。

- **坑：动态 shape**。视角数随 tile 布局变化（角落相机数量不等），
  每次变化触发重编译（每次几十秒）。对策：视角数 padding 到固定桶
  （如固定 40，不够的拿重复视角/零权重补齐，输出端丢弃），或
  `torch.compile(dynamic=False)` + 锁定 rig 固定视角数
- 模式建议 `mode="max-autotune"`，模型常驻后摊销编译成本
- 验证入口：`recon.infer` 单点包一层即可 A/B

### 2. 闲时预烘（已上线）
推理速度不变，但玩家永远踩现货。_bake_loop 已在跑：队列空时从玩家
最后位置向外烘两圈。这是"感知速度"的最大杠杆，零风险。

### 3. 水平扩展到第二节点（夜里 4090 双卡）
卡内并行证伪了，卡间/机间并行才是真开大 worker：
两个独立 server 实例分片 tile 空间（奇偶 tile 或方位分片），客户端按
tile 坐标哈希路由。4090 双卡 23:00-07:00 窗口可用。吞吐线性翻倍。

### 4. FP8 / 量化（远期）
PRO 6000（Blackwell）支持 FP8，但 WorldMirror 没有现成量化路径，
校准工作量大。等 compile 收益吃干抹净再考虑。

### 5. 注意力实现审查
dense_head 的 `_extract_fused_features` 是显存热点（OOM 记录在那）。
确认是否走 flash-attention/SDPA；若是手写 einsum 注意力，
换 SDPA 通常白捡 10-30% 显存+速度。

## 测量规矩

- 一切改动过 bench_seams + 分环颜色方差指标 + 实机视觉三道验收
- 速度数字以 `/health` 的 mean_build_s 为准（渲染+前向+后处理全计）
- 显存峰值用 nvidia-smi 长时间观察，PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 常驻
