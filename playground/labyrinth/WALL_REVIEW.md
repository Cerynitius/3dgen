# Labyrinth 墙体破碎/断裂排查报告

> 2026-08-19 · 审查者：Kimi Code（K3, thinking max）· 整理：Cherry
> 代码版本：pro6000 `~/HY-World-2.0/playground/labyrinth/`（2026-08-14 快照）
> 方法：Node 探针实测（80 个 seed 的生成器对称性/连通性验证、墙体可行走性测试）+ 全量代码走查
> 注：`endless_selftest.mjs` 未能运行——依赖的 `stream_server.py`（`~/HY-World-2.0/experiments/labyrinth/endless/`）不在本地，selftest 需要 127.0.0.1:8092 的服务端。

## 结论摘要

**生成器（maze.js）是清白的**——80 个 seed 下墙位对称性 0 错误、连通性 0 失败。墙体破碎的根因在渲染与碰撞层：

| 级别 | 问题 | 位置 |
|------|------|------|
| P0-1 | 墙体是"空心纸壳"：只有内表面、没有端帽，可走进墙内并被永久卡住 | geometry.js:105-148, live.js:54-58, live.js:233-238 |
| P0-2 | splat 槽位截断导致墙面补丁状破洞（endless.js 已修，live.js 未修） | live.js:33/78, splats.js:271-272 |
| P1-3 | 排序 worker 与槽位复用竞态：幽灵墙片闪烁 | splats.js:248-254, 278-289 |
| P1-4 | 四元数线格式两处注释矛盾，可能导致墙面 splat 朝向随机（待服务器验证） | splats.js:9 vs 96-101 |
| P1-5 | live.js 用 splat 占用做碰撞：重建上的洞 = 可穿过的墙 | live.js:237, splats.js:336-345 |

## 对三个预设怀疑方向的裁决

1. **wallsNear 不是根因**（且在 maze.js:409 不在 geometry.js）。span=1 数学上足够（cell 3~4m 宽、身体半径 0.34m）。凸角 T×T slab 缺口存在（maze.js:421-424），但身体圆会桥接，blocked 仍成立。另外它只被 controls.js:134 的 makeCollider 引用，live.js/endless.js 都不用。
2. **cellAt 错位是真 bug 但无害**：在 endless.js:59-62（`floor((gx-1.5)/2)`），25% 的位置（plaza 西半）偏到隔壁 cell。消费者只有 HUD 读数和 ensureMaze 窗口居中（被 HALF=28/MARGIN=10 吸收）。装饰性 bug，不破墙。maze.js:397 的 `floor(x/cell)` 是对的。
3. **occupancy.js 与服务器光栅化一致性不是问题**。真正的"表示不一致"在 live.js 的碰撞切换（见 P0-1）。

## P0-1 墙体空心纸壳（主根因）

- `geometry.js:105-148`（buildLevelMesh）：每个带墙 cell 只发一张朝向室内的零厚度面片。相邻 cell 两面之间是 T 宽空腔，墙段末端（T 形交接、墙尽头）从不发端帽面。
- `live.js:54-58`：PY_WALL=2.0 墙厚 + makeMeshCollider(buildLevelMesh(maze))。空腔 2m 宽，0.34m 身体畅行无阻。探针实证：seed=1000 迷宫 24 个墙段末端全部可走入墙体内部；沿墙走到中段两种碰撞体结论相反（24/24）。
- `live.js:233-238` 升级后果：chunk 未到用空心网格碰撞，chunk 到达切成 splat 占用碰撞（splats.js:355-363 AABB）。若已站在墙内，墙面体素将玩家**永久卡进墙体**，只能沿墙腔滑到下一个交接口。玩家视角即"从墙角穿进去，然后墙体破碎/卡死"。
- 附带：`collider3d.js:1-8` 注释声称"碰撞的就是绘制的三角形"，但 live.js 绘制的是 splat，MeshCollider 碰的网格从未被画出——前提不成立。

**修法**：buildLevelMesh 为墙段末端和 T 形交接处补端帽面；或 live.js 改用与 endless.js 一致的碰撞源。

## P0-2 splat 截断打出补丁洞

- `live.js:33`（max=333,312）+ `live.js:78`（24 槽）→ 槽大小恰好 333,312；而 endless.js:96-100 注释记录 voxel_prune 返回目标的 1.1 倍（最多 ~366,643）。
- `splats.js:271-272`：`room = Math.min(n, this.slotSize)` 静默丢弃尾部 ~33k splats。注释自承：截断 voxel-pruned 列表会成片丢失覆盖而非均匀稀疏——即墙上补丁状破洞。
- endless.js 用 18 槽（444,416/槽）已修；live.js 未同步。

**修法**：live.js 改 `slots=21`（380,928/槽）或把 max 降到 300,000。

## P1-3 排序 worker 竞态：幽灵墙片

`splats.js:248-254` 的 onmessage 把 worker 返回的 order 无条件写入 indexBuf，无版本校验。排序往返期间若发生驱逐+槽位复用（splats.js:278-289），旧 order 指向纹理中未覆盖的旧 tile 残段（splats.js:320-326 只上传新行数，槽尾保留旧数据）——被驱逐 tile 的"幽灵墙"以错误深度序画数帧，表现为墙片闪烁/碎裂。

**修法**：每次排序带 slots 版本号，过期响应丢弃。

## P1-4 四元数线格式注释矛盾（待验证）

- `splats.js:9` 文件头注释：wire 格式 `[... | qx qy qz qw | ...]`
- `splats.js:96-101` 着色器注释：声称 "wire 是 w x y z" 并按 `vec4(b.w, c.x, c.y, b.z)` 解包

两者必有一错。若服务器实发 xyzw（文件头为准），则每个 splat 四元数被循环移位，各向异性墙面 splat 旋到随机朝向——**墙看起来就是碎的**。着色器注释声称已按官方导出器验证（rot_0 是标量），大概率文件头注释过期，但需 dump 一个 `/tile` 响应的前 14 个 float 对服务器打包代码确认。

## P1-5 splat 占用做碰撞 = 重建的洞可穿墙

`live.js:237` 对已建成 chunk 用 field.blocked（splats.js:336-345，只统计 y∈[0.35,1.85] 波段内 0.35m 体素有 splat 的格子）。重建在墙上打洞，碰撞墙上就有洞。endless.js:66-69 注释明确说这正是 endless 不用 splat 做碰撞的原因——live.js 反其道行之，且碰撞规则随 chunk 到达瞬变。

## P2（属实但轻微）

- `endless.js:59-62` cellAt 半格偏移（见上文，装饰性）。
- `occupancy.js:22-27`：walls Map 只进不出，长时间漫游线性增长（慢速内存泄漏）。
- `endless_selftest.mjs:54`：`ring > 0` 假条件——服务器与客户端在外环恰好一致时误报 FAIL；`:81-85` 用 cell (c,r) 的 E 墙位做"门可通过"判定，而碰撞代码 occupancy.js:48 实际读 cell (c+1,r) 的 W 墙位——只在"墙位永远对称"假设下等价，测不到真正的不对称。
- `splats.js` 构造函数：8M 容量 → RGBA32F 纹理 2048×15,624，超过部分 GPU 的 MAX_TEXTURE_SIZE（8192）限制，无检查，那类机器上直接黑屏。

## 修复优先级建议

1. **live.js 槽数 24→21**（一行参数，立即消除补丁洞）
2. **buildLevelMesh 补端帽**（几何活，根除"穿墙+卡死"）
3. live.js 碰撞源对齐 endless.js（废弃 splat 占用碰撞）
4. 排序 worker 加版本号
5. dump `/tile` 响应验证四元数序


---

# 第二轮审查 · 猫娘锐评版（2026-08-19 深夜）

> 审查者：Kimi Code（K3, thinking max）· 润色：Cherry
> 标注规则：【实证】=探针跑过，【静态】=代码走查

看完了只想说一句：上一轮修墙修得热火朝天，结果这轮一翻网络层和输入层——好家伙，地基下面还埋着雷呢喵。

## 严重

### 1. fetch 全裸奔：一个超时都没有，挂起即永久死亡

`live.js:115-138`、`endless.js:84-91/119-136/144`，全代码库没有一个 `AbortController`。一个都没有喵。

服务器卡住不响应的时候，fetch 永不 settle，`finally` 永不执行。然后连锁反应来了：live.js 的两个 inflight 槽被永久占满，流式加载整体停摆，HUD 永远显示 "generating" 还不报错——玩家对着一堵不会来的墙干瞪眼。endless.js 更绝，挂起的 tile key 永远赖在 inflight 集合里，入口处 `inflight.has(key)` 直接把它挡在重试门外——**它自己注释里写的 "hole that never fills" 不变量，被它自己亲手打破了**。写注释立 flag 然后自己踩爆，这也算是行为艺术了喵。

修法：每个 fetch 包 `AbortSignal.timeout(...)`，超时清状态允许重试。一天的工作量，治的是"永久卡死"这种病，血赚。

### 2. main.js:831 的 `const fps` 遮蔽：一行代码团灭第一人称导航

`main.js:743` 全局定义了 `fps`（相机状态），`main.js:831` 在 `main()` 里又定义了一遍——这次是 DOM 元素。于是 frame() 和 mousemove 里所有的 `fps.yaw`、`fps.pitch` 全写到 DOM 元素上去了，buildViewMatrix 读的还是那个没人更新的全局对象。**键盘鼠标导航整体失效**。

更妙的是首次按 W：`v3.mad(fps.p, ...)` 里的 `fps.p` 是 DOM 元素的 undefined 属性，`undefined[0]` 直接 TypeError，而且异常发生在 requestAnimationFrame 之前——**RAF 循环当场死亡，画面冻结**。一个变量名，团灭。

不过说句公道话：main.js 是个孤儿文件，没有任何 HTML 引用它。所以这雷埋着呢，谁哪天手贱打开它谁踩。

### 3. addChunk 拒绝后的无限重下载：拿带宽和 GPU 放烟花

live.js:125-126：槽位满了拒绝新区块，requested 一删，0.25 秒后 pump() 又把同一区域选为候选，重新下载几十 MB，再被拒，再下……SCAN=8 的时候候选区 ~96 个对着 24 个槽位，**人站那儿不动，这循环就在那儿持续空转烧带宽烧服务器 GPU**，连个负缓存都不记。endless.js:127-128 同病相怜，0.35 秒一轮。

就……哪怕记一个"刚才被拒过"的小本本呢？修法：给被拒的 key 记个带时间戳的拒绝表，成本不如最差驻留 tile 就跳过。

## 中等

### 4. controls.js 的加速度是帧率相关的【实证】

`vel += (tx - vel) * Math.min(1, k*dt)` 不是时间正确的指数趋近。探针实测：静止按 W 走 0.5 秒，60Hz 位移 -1.5622m，144Hz 位移 -1.5019m，差 3.86%。高刷屏上起步手感更"贼"——电竞显示器玩家莫名其妙占了便宜。修法：`1 - Math.exp(-k * dt)`，一行。

### 5. 没有脱困机制：卡进墙里就是无期徒刑【实证】

controls.js:107-111 每个轴被挡就清零速度，从不 push-out。探针把身体出生在墙板里，100 步后位移 0.0000m，速度全程为零。上一轮 live.js"chunk 到达后被封进墙内"的机械学另一半就在这——**任何路径让身体进了实体，都没有任何恢复路径**。collider3d.js:121 明明已经有 `penetration()` 了，倒是调用它啊喵。

### 6. live.js 把响应里的 meta.cell 和请求键混用

槽位键用请求的 (c,r)，chunkTransform 用响应头里的 meta.cell，chunkCost 又按请求坐标算驱逐成本。服务器要是对区域做了归一化，splats 就按 A 的位置放、按 B 的身份登记驱逐——账房先生和仓库管理员用的不是同一本账。【静态，取决于服务器契约】

### 7. main.js 的 PLY 解析见 CRLF 就死【实证】

`"end_header\n"` 和正则都不匹配 `\r\n`。CRLF 头下正则返回 null，`.exec(header)[1]` 抛 TypeError，worker 直接去世。Windows 工具链导出的 PLY 必现。跨平台测试的时候又是一条好汉。

### 8. 快速失败 + 零退避 = 重试风暴

chunk 500 了？立刻删记录，0.25 秒后原样重试，每个候选每秒最高 8 次，无限持续。endless.js 的 202 轮询每 tile 3Hz。服务器宕机的时候两个客户端像抢修现场一样合力捶它，它都宕机了你们倒是让人家歇会儿啊喵。

## 轻微

- `main.js:816`：拿 Content-Length 开 Uint8Array——服务器走 chunked 编码就直接 RangeError。本地 serve.py 发 Content-Length 所以没事，换个 CDN 就炸。
- `main.js:981-986`：只有 2 个相机时按 "9"，camera=undefined，然后 resize() 在 camera.fx 上再崩一次，崩出连环追尾。拖入 .json 相机文件后投影矩阵还用旧的，焦距错误直到按一次数字键。
- URL 参数无校验：`?radius=abc` → NaN 发给服务器永远 400，`?gain=abc` → NaN uniform 直接黑屏，全都静默吞掉。
- `serve.py`：实测 HTTP/1.0、线程数无上限（slow-loris 可耗尽）、默认 0.0.0.0 + 目录列表全暴露。开发服务器定位下算及格，目录穿越倒是正确拒绝了，20 并发全 200。
- `controls.js:150-158`：指针没锁定 WASD 照样走，hint 却提示点击开始——UX 自相矛盾。Space 被 preventDefault 但没绑定任何功能，占着茅坑。

## 点名区域的阴性结果（查过没病的）

- **mat4.js 全部函数**【实证】：multiply 误差 4e-7、rotationY、视图矩阵刚性、perspective 边界、排序 worker 的深度提取，全对。数学是干净的。
- **dt 钳制防穿隧**【实证】：dt=0.1 加冲刺对着 0.28m 薄墙冲 3000 步，身体最近停在墙外 0.74m，从未穿越。钳制余量够的。
- **maxDepth==minDepth 退化**【实证】：NaN|0=0 全落 bucket 0，不腐坏，虚惊一场。
- **occupancy.js 剩余函数**【静态】：grid/wallAt/ingest 无语义问题。
- **彩蛋**：maze.js 的 growMaze 会 carve 每个非房间 cell，所以 solid rock cell 根本不存在——wallsNear 的岩石分支和 buildLevelMesh 的 !carved 跳过全是死路径。不是 bug，是化石。

## 本轮一句话

生成器干净、数学干净、物理步进大体健康，但**网络层没有任何超时兜底**、**输入层有个遮蔽团灭**、**下载层在空转烧钱**——墙修完了，该修水管和电线了喵。

---

# 第三轮审查 · 修复后的新代码复审（2026-08-20）

> 审查者：Kimi Code（K3, thinking max）
> 范围：仅限本轮新写/重写的部分——netpolicy.js 全文、splats.js 的 gen 版本号与 chunk 增量传输、
> geometry.js 的 buildSolidLevelMesh、controls.js 的 stuck/push-out、occupancy.js 驱逐视界、跨 tile 深度排序
> 方法：Node 探针 `current/round3_probe.mjs`（22 项，21 过 1 故意失败=确认病灶）+ 静态走查。
> 探针跑的是 shipped code：worker 源码从 splats.js 里抽出来 eval，排序矩阵用真值。

## 严重

### 3-1. live.js 调了一个不存在的函数：第一个 chunk 成功即 ReferenceError，全场黑屏

`live.js:142` 调用 `chunkTransform(got.meta)`，但**当前目录里没有任何地方定义它**——
全库 grep 只有上一级旧快照 `../live.js:94` 里有定义。今天重写 live.js 时函数体丢了。

后果链：第一个 chunk 下载成功 → ReferenceError → 被 `live.js:162` 的 catch 吞成
"chunk failed: chunkTransform is not defined" → `defer.hold` 退避重试 → 再下载几十 MB →
再 ReferenceError。**live.js 一个 chunk 都渲染不出来，且永远在重试烧钱**。
netpolicy 的退避让这件事烧得很文明，但治不好。endless.js 不受影响（它内联 `m4.translation`）。

探针实证（round3_probe.mjs §0）：源码里 `chunkTransform(` 有调用、零声明。
修法：把 `chunkTransform` 从旧快照搬回来，或者像 endless.js 一样内联 `m4.translation`。
另外建议 bugfix_test.mjs 加一条静态检查：live.js 里调用的自由变量必须在本文件有声明。

## 中等

### 3-2. splats.js 驱逐复用后，GPU 继续按旧 order 画被驱逐 tile 的残尾——这就是"重影斑块"的一个实锤来源

gen 版本号（splats.js:299）只拦**迟到的排序回包**，没人管**已经在 GPU 里的旧 order**：

- `splats.js:380-385`：槽位复用时 `texSubImage2D` 只上传新 tile 实际需要的行数
  （100 splat 只写 1 行 = 512 splat 位），槽位尾部仍是被驱逐 tile 的旧字节；
- `splats.js:412-413`：`addChunk`  bump 了 generation 却**不动 `this.sortedCount`**，
  indexBuf 里的旧 order 原样继续参与绘制，直到下一次排序回包落地。

探针实证（§2）：1000-splat 的 tile A 被 100-splat 的 tile C 驱逐复用后，`sortedCount`
仍是 2000，旧 order 里有 **488 个索引指向槽 0 中未被覆盖的旧 texel**——被驱逐 tile 的
墙面以残片形式继续画 1 帧以上（排序繁忙时更久）。每次驱逐复用且新 tile 较小时必现，
走路时驱逐频繁，表现为墙面闪烁的碎块/重影斑块。同 key 重传（替换）路径同样中招。

修法（任选）：addChunk 复用槽位时把旧 count 超出新 count 的行清零（只对
`[ceil(room*4/TEX_W), ceil(oldCount*4/TEX_W))` 这段补一次零上传）；或 addChunk 时
`sortedCount = 0`（代价是每驱逐一次全场景闪一空帧）。前者无视觉副作用。

### 3-3. endless.js 驱逐成本混用两种基准：新 tile 按角、旧 tile 按中心，最远差 8.5 m

- `endless.js:191-193`：incoming tile 的 cost = `tileCost(at)`，`at = meta.origin_m` 是
  tile 的**角点**（splats 靠它平移就位）；
- `endless.js:278-283`：驻留 tile 的 `slotCost` 每 0.35 s 用**中心**公式重算
  （`(2*(tc*span)+span-1+1.5)*cell_m`，该公式本身已验证正确）。

两本账在 `splats.js:342` 的 `worstCost <= cost` 里直接比较。span=3、cell_m=2 时同一个
tile 按角算是 17.0 m、按中心算是 7.1 m（探针 §2 实测）——一个其实就在身边的 tile 会
被当成最远的拒绝/驱逐，表现为人走近了某块墙却迟迟不来或突然消失。
修法：统一基准，`fetchTile` 里用中心（`at + tileHalf`）或让 slotCost 循环也用 origin。

## 轻微

### 3-4. withDeadline 把超时后迟到的真实错误吞成"timed out"（netpolicy.js:28）

`catch` 里只判 `ctl.signal.aborted`：fn 若在 deadline 之后抛出真错误（比如 JSON 解析
失败的 SyntaxError），被改报为 "xxx timed out after N ms"，原始错误连 `cause` 都没留。
探针 §1 实证：`TypeError: real bug` 被报成 "probe timed out after 50 ms"。
排查时会把真 bug 引到网络方向。修法：`throw new Error(msg, { cause: e })`。

### 3-5. Defer 的 key 只进不出：没被再次问到的 key 永不回收（netpolicy.js:71-73/90）

条目只有两条出路：`has()` 被问到（过期/走近才删）和 `clear()`（成功才调）。
一个失败/被拒的 tile 一旦不在当前 ring/候选集里，就再没人 `has(key)`——
endless.js:227-229 只对 ring 内 tile 调用 fetchTile。探针 §1：10 万个 hold 后
`size` 仍是 100,000。长会话慢速泄漏，和 occupancy 那个是同一形态，量级更小。
修法：`hold()` 里顺手扫一遍过期项，或给 map 设上限。

### 3-6. Defer 的 tries 把三种等待混在一个计数器里（netpolicy.js:84-85）

拒绝、202、网络错误共用 `tries` 递增。探针 §1：拒绝 3 次后第一次真实网络失败的退避
是 backoff(4)=4000 ms 而非 500 ms；反过来失败次数也会抬高 202 的轮询间隔。
不是正确性问题（方向都是"更保守"），但与注释宣称的"nth consecutive attempt"语义不符。

### 3-7. controls.js push-out：过冲 + 弹出方向任意（controls.js:136-151）

探针 §4 三点实证：

1. **过冲**：0.01 m 的擦碰（身体半径刚蹭进墙面 1 cm）触发完整搜索步，
   横向瞬移 0.17 m——visible pop。最小弹出距离是 `radius*0.5`，与侵入深度无关。
2. **弹出方向由扫描顺序决定，不是来路**：身体在 2 m 厚墙正中央时，角度 0（+x）
   先扫到自由空间，人被**穿过整堵墙弹到另一侧**（x=2.0 → 3.36，单帧 1.36 m 瞬移，
   路径上的墙体不做检查）。正常流程进不了墙（轴分离碰撞挡着），所以这只是异常
   恢复路径的语义问题，但"从哪侧出去"是抛硬币。
3. **搜索半径封顶 2.04 m**（12 环 × 0.17）：比 4.1 m 厚的实心区域更深的陷入
   永远 stuck。maze.js 的岩层分支是死代码、endless 实心 cell 2 m 厚，当前无触发路径，
   但值得注释一句。

未复现"推出墙外"：maze 边界墙的弹出会优先扫到朝内的第一象限方向（角度序 0→2π），
构造不出把玩家送出地图的情形；endless 的 unknown=solid 语义也挡住了弹向未加载区。

## 阴性结果（查过没病的）

- **跨 tile 深度排序的前提已过时**【实证】：当前 worker（splats.js:44-76）把**所有存活
  槽位拼进同一个计数排序**——把 shipped worker 源码抽出来跑，两个槽深度交错时输出
  order 是 `[1,100,0]` 交错排列，且正确回传 gen。**交界交叠区域的顺序是全局管的，没人
  缺位**。玩家看到的持续重影斑块更可能是：相邻 tile/region 各自重建了同一堵共享墙、
  两份都画（客户端无任何跨槽去重，live.js:40-41 注释自承 region 间 "neighbours
  disagree"），叠加 3-2 的驱逐残尾。前者要服务器侧合并边界墙才能根治。
- **buildSolidLevelMesh 的 seen-key 在当前参数下精确**【实证】：cell=4.0/wallT=2.0 和
  cell=3.0/wallT=0.28 下发射盒数 == 独立计算的唯一 slab 数（249==249），零 NaN、
  winding 全对。但 key 是**浮点转字符串**（geometry.js:177）：cell=0.3 时 `(r-1)*S+S !==
  r*S`，同一个共享 slab 算出两个 key——40×40 迷宫实测 1986 盒 vs 应发 1583，25% 重复
  三角形。当前参数碰不到，算 latent。要硬一点就 key 用 `Math.round(v*1e6)`。
- **实体网格与 wallsNear 一致**【实证】：71,007 个板外采样点零分歧（板内是
  包含-vs-表面的既定语义差异，第二轮已裁决，洪水填充仍是 0 穿墙）。
- **occupancy.js 驱逐视界安全**【实证】：对角线走 300 窗口（最坏情形），加载窗内
  0 cell 被误删，峰值 27,949 封顶。`box` 字段（occupancy.js:26）只写不读，死字段。
- **withDeadline 无 timer 泄漏**【实证】：500 次已决 deadline 前后 Timeout 句柄 0→0；
  AbortController 随作用域回收，无 AbortSignal 泄漏路径。
- **push-out 无抖动**【实证】：脱困后 120 帧位移 0.000000 m，stuck 不误报。

## 本轮一句话

前两轮的修复方向都对、测试也都真，但重写时把 `chunkTransform` 弄丢了——live.js 现在是
整体黑屏状态（3-1）；玩家实测的重影斑块有了实锤机制（3-2 驱逐残尾）加一个结构性嫌疑
（邻 tile 重复重建共享墙）；网络层三个边角（3-4/3-5/3-6）不致命但会在长会话里积灰。
