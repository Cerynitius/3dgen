# Endless Labyrinth — 当前状态

## 架构一句话

逐 tile 重建：迷宫几何 raycast 出带位姿视图 → WorldMirror 前向 → splat →
度量对齐/裁剪/剪枝 → HTTP 流给客户端。服务器
`experiments/labyrinth/endless/stream_server.py`（8092），客户端
`playground/labyrinth/`（8090 静态服务）。

## 当前配方

**视角 rig（tiles.py）**
- 官方光学：fov 120°、方形 476px（ViT patch14 对齐）、aspect 1.0
- 每机位：平视环 ×4（离心 0.35 m）+ 俯视 ×1（俯角 -0.45）+ 凹角相机（两墙夹角各一，俯角 -0.5）
- context 相机：tile 外一圈 4 机位 × 3 方位角，按方位角均匀分布、可见性优先选取
- 视角数随迷宫拓扑变化（凹角相机取决于墙位），约 32–48

**视图渲染（render.py）**
- 洞穴开凿纹理：低频设计（米级石板/凿痕/沙堆），观测管线载不动 10cm 级砖纹
- 低位火盆（0.55 m）向上打光，墙面光池贴地线；地面亮漫反射；天花板刻意暗
- 渲染出口 NaN 过滤（nan_to_num）

**前向（recon.py + fastinfer.py）**
- 输入再消毒一次 nan_to_num；rig 位姿 + 内参作为条件输入
- splat 位置由 gs_depth 用精确 rig 位姿反投影（`unproject="gt"`），
  逐视角尺度对 raycast 真深度拟合（`scale_from="perview"`）
- 快路径 fastinfer.py 默认开（卸载三个无消费者的 DPT 头、RoPE 缓存、去同步）：
  前向 2.0→1.5s，tile 字节逐字节不变；`LAB_COMPILE=1` 再到 1.4s（长跑服务器建议开），
  细节与实测见 `INFERENCE_OPTIMIZATION.md`

**后处理（tiles.py）**
- 裁剪到 tile 足迹 → 距真实表面 >10 cm 的 splat 剔除 → 0.98 尺度分位过滤 →
  体素剪枝到 400k（透明度按 1-(1-a)^k 补偿，半径 √k 增长）

**服务器（stream_server.py）**
- 环请求按距玩家距离排优先级，玩家脚下的 tile 永远第一
- 闲时预烘：队列空时从玩家最后位置向外烘两圈（低优先级，真请求插队）
- 实心 tile（>92% 占据）直接缓存空 tile
- 单 worker（GPU 已被单 build 饱和）；建议启动时设
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`（代码不设，属运行环境）

**客户端（endless.js / splats.js）**
- splat 场 8M：20 槽 × 400k，cost 驱逐；槽位复用时立即滤掉旧索引
- 槽尾惰性填充（alpha=0, qw=1, scale=1e-4）
- 限速器：前方 tile 未驻留 → 25% 速度
- 加载雾幕（生成时浓、落地缓释）、沙尘微粒、暖黄配色

## 已知上限

- 单 tile ~1.9–2.1 s（33–35 视角实测；快路径默认开）；预烘 + 限速器保证行走覆盖
- 显存在当前视角数下稳定（配合 expandable_segments 启动项）
- 纹理"油画感"是前向 splat 重建的本性，模型不换即此水准

## 下一步

路线与已证伪方向见 `INFERENCE_OPTIMIZATION.md`（前向已摸到全局注意力的硬件峰）：

1. 视图渲染合批/捕获（rig ~0.4s/tile，是前向之后的下一块）
2. 第二生成节点（水平扩展）
