# 推理优化备忘录（2026-08-22）

> 对象：Endless Labyrinth 的 WorldMirror 逐 tile 前向。
> 旧基线（RTX PRO 6000 单卡，~33 视角/476²）：前向 2.0-2.2s，整 tile 2.6s。
> **现状（fastinfer.py 落地后，生产=默认层）：前向 1.5s，tile 字节与基线逐字节相同，
> 整 tile ~1.9-2.1s（33-35 视角实测）。compile（1.4s）/cudagraph 为实验 flag，不进生产。**

## 已证伪（别再做）

| 方向 | 实测 |
|------|------|
| 多 worker 并行 | w1 0.24 t/s → w8 0.13 t/s，单调劣化。单 build 已吃满 GPU，并行只争锁 |
| batch=2 前向 [2,N,3,H,W] | 模型条件路径不认 batch 维（pose 广播维度冲突），结构性拒绝 |
| 降分辨率 476→420 | 3.39s vs 3.33s，噪声内。渲染像素不是瓶颈 |
| 砍视角数 | 视角=质量的生命线（bench 实测随视角数涨），砍质量换速度是本末倒置 |
| 视角数 padding 到固定桶 | 不必要也不正确：dummy 视角参与全局注意力会改变每个真视角的输出；dynamic shape 编译实测不重编译，padding 无需存在 |
| CUDA graph 求提速 | 落地可用、逐字节等价，但**零提速**——rope 同步修掉后 launch 开销已不在关键路径，大核把 GPU 喂满了。留作 flag |

## 已落地（fastinfer.py，逐层可关，全部实测）

| 层（env flag，默认值） | 做什么 | 前向 | tile 字节 |
|---|---|---|---|
| `LAB_SKIP_HEADS=1` | 卸载 depth/pts/norm 三个 DPT 头——tile 管线只消费 splats/gs_depth(+conf)/camera_poses，这三个头占前向 25%，算完即扔 | 2.0→1.75s | 不变 |
| `LAB_ROPE_CACHE=1` | RoPE sin/cos 按 token 网格形状缓存。原实现每 block 两次 `positions.max().item()`，一次前向 ~96 次 CPU 同步 | →1.5s | **逐字节相同** |
| `LAB_SYNC_FREE=1` | 条件路径去同步：normalize_poses 去 NaN 分支（rig 上游已消毒）、相机逆矩阵闭式解、rotmat_to_quat 去 host 常量/布尔索引。graph capture 的前提 | （并入上行） | **逐字节相同** |
| `LAB_COMPILE=1` | trunk 48 block + DINOv2 patch embed + gs_head 动态形状编译（视角数变不重编译；rope 用 `_dynamo.disable` 摘出，否则按形状特化撞 recompile limit） | →1.40s（1.42×） | splat 数漂 ~0.2%，floor_err 不变，bench_seams 与基线一致 |
| `LAB_CUDAGRAPH=1` | 整前向按视角数 keyed 懒捕获，共享 memory pool，无 padding | 无增益 | 逐字节 = 所捕获模式 |

- 验收：bench_seams 5 tile，compile 开 vs 关——x 缝 99.1/99.3%、z 缝 100/100%、
  interior 92.3/92.2%（<5cm），统计一致。
- world version 已把 recon.py + fastinfer.py 源码和 flag 组合折进哈希，改 flag 即换版本，
  浏览器年缓存不可能吐旧 tile。
- **定案：生产保持默认（head-skip + RoPE cache + sync-free），字节恒等优先。**
  compile/cudagraph 留作实验 flag，不进生产。compile 的代价供参考：首次 3 个 build 各
  ~30-43s（动态形状泛化前的编译），之后新视角数不再重编译；开关 flag 即换 world version。
- cudagraph 显存：每视角数一份静态输入/输出 + 共享 pool，峰值 22.5→25.8GB，LRU 上限 12 份。

## 天花板在哪（这次量出来的）

全局注意力就是墙：24 个 global block，每个对 35×1162≈40k token 做 N² attention，
单核 26.7ms ≈ **254 TFLOPS，已贴 bf16 硬件峰**。一次前向仅这一项 ~0.64s。
SDPA 走的是 FLASH 后端；CUDNN 后端同形状 25.5ms（+5%），EFFICIENT 慢一倍。
剩余可挤的：CUDNN 后端 ~30ms/前向、gs_head DPT 降 bf16（~0.1s，需 gs_depth 质量门）、
cam_head 忽略不计。再往下只有 N² 本身，而砍视角已证伪。

## 候选路线（更新后）

### 1. 闲时预烘（已上线）
_bake_loop 在跑：队列空时从玩家最后位置向外烘两圈。"感知速度"的最大杠杆，零风险。

### 2. 水平扩展到第二节点（夜里 4090 双卡）
卡内并行证伪了，卡间/机间并行才是真开大 worker：两个独立 server 实例分片 tile 空间
（奇偶 tile 或方位分片），客户端按 tile 坐标哈希路由。吞吐线性翻倍。

### 3. 视图渲染（rig ~0.4s/tile）
dda 的 Python 步进循环每步 ~25 个小核 + `.any()` 同步。视角间无依赖，可合批一次
raycast；或固定步数后按 (W,H) capture。前向摸到天花板后这是下一块肥肉。

### 4. gs_head 降 bf16 / FP8（远期）
gs_head 的 DPT 主体在 fp32（~0.2s）。降 bf16 需过 gs_depth 质量门（度量尺度逐视角
对 raycast 拟合，能吸收部分噪声，未必掉质量）。FP8 等 compile 收益吃干抹净再说。

## 测量规矩

- 一切改动过 bench_seams + 分环颜色方差指标 + 实机视觉三道验收
- 逐层 A/B 用 `bench_infer.py --save/--compare`：字节级 diff + 前向中位数，
  真基线 = `LAB_SKIP_HEADS=0 LAB_ROPE_CACHE=0 LAB_SYNC_FREE=0`
- 速度数字以 `/health` 的 mean_build_s 为准（渲染+前向+后处理全计）
- 显存峰值用 nvidia-smi 长时间观察，PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  常驻（注意：与 graph capture 不兼容，开 LAB_CUDAGRAPH 时不设）
