"""City assets generated while walking: one front-elevation photo per building (shop front + sign included).

Pure functions of (world seed, building box), shared by the walk server (which decides what is needed) and
assetgen.py (which runs Qwen-Image-2.1-Turbo and writes the photos). A building is identified by its box, so a
photo never goes stale: revisiting a street shows the same shops.
"""
import math
import os
from pathlib import Path

import numpy as np

import world as Wd

HERE = Path(__file__).resolve().parent
VERSION = "qi21t_v1"


def asset_dir(seed):
    d = HERE / "assets_cache" / f"{VERSION}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


STYLES = [
    "红砖外墙的老式居民楼，木框窗户，部分窗户装着防盗网",
    "米黄色瓷砖外墙的九十年代住宅楼，铝合金窗，窗下挂着空调外机",
    "白色涂料外墙的住宅楼，墙面有一些旧的雨水痕迹，窗台上放着花盆",
    "灰色水刷石外墙的老楼，窗户整齐排列",
    "浅蓝色玻璃窗配白色线条的现代商住楼",
    "骑楼式的老街铺面楼，楼上是拱形窗",
    "暖黄色外墙、深色窗框的楼房",
    "青砖墙面、木格窗的仿古商铺楼",
    "浅灰色面砖外墙的住宅楼，每层有小阳台，阳台上晾着衣服",
    "淡粉色外墙的住宅楼，白色窗框",
]
SHOPS = [  # (sign suffix, what the shop front shows)
    ("大药房", "玻璃门里是一排排药柜，门边有绿色十字灯箱"),
    ("面馆", "玻璃门上贴着红色菜单"),
    ("便利店", "明亮的玻璃门，货架上摆满零食饮料"),
    ("水果店", "门口摆满一箱箱水果"),
    ("理发店", "门口有红蓝白旋转灯柱"),
    ("银行", "深色石材门脸，门边有自动取款机"),
    ("包子铺", "门口有冒着热气的蒸笼"),
    ("五金店", "门口挂着工具和水管"),
    ("眼镜店", "明亮的橱窗里陈列着眼镜"),
    ("花店", "门口摆满鲜花和绿植"),
    ("茶庄", "木质门脸，橱窗里摆着茶叶罐"),
    ("书店", "橱窗里摆着书"),
    ("早餐店", "门口有卖油条豆浆的摊位"),
    ("社区卫生服务站", "白色门脸，门口有蓝色标识牌"),
    ("洗衣店", "橱窗里挂着熨好的衣服"),
    ("照相馆", "橱窗里摆着全家福照片"),
    ("超市", "宽大的玻璃门，门口停着购物车"),
    ("饺子馆", "门上挂着红灯笼"),
    ("烟酒店", "柜台里摆着酒瓶"),
    ("邮政所", "绿色门脸"),
]
PREFIX = ["老王", "福满", "康乐", "惠民", "和平", "新华", "金鑫", "阳光", "幸福", "同仁",
          "百姓", "建设", "长青", "东方", "红星", "丰收", "安康", "平安", "利民", "兴隆"]

# generation sizes (w, h), ~0.5 MP, multiples of 16, indexed by facade aspect w/h
BUCKETS = [(512, 1024), (576, 896), (640, 800), (704, 704), (832, 608), (896, 576)]
B_ASPECT = np.array([w / h for w, h in BUCKETS])


def _front_axis(x0, z0, x1, z1):
    """0: the front faces +-x (street runs along z); 2: front faces +-z. A face is a street front when its plane
    lies HW from a street centre line (multiples of P)."""
    def on_street(a, b):
        return min(abs(c - Wd.P * round(c / Wd.P)) for c in (a, b)) < Wd.HW + 0.05
    fx, fz = on_street(x0, x1), on_street(z0, z1)
    if fx and fz:
        return 2 if (x1 - x0) >= (z1 - z0) else 0
    return 0 if fx else 2


def facade(box_row, prm_row, seed):
    """Building box + params -> dict(key, prompt, w, h, gseed, axis)."""
    x0, _, z0, x1, h, z1 = (float(v) for v in box_row[:6])
    rx = [int(round(v * 100)) for v in (x0, z0, x1, z1)]
    k1, k2 = Wd.h32(seed, *rx), Wd.h32(*rx, seed, 77)
    key = f"{k1:08x}{k2:08x}"
    axis = _front_axis(x0, z0, x1, z1)
    width = (x1 - x0) if axis == 2 else (z1 - z0)
    floors = int((h - Wd.GF) / Wd.FH) + 1
    rng = Wd.Rng(k1, k2)
    style = STYLES[rng.i(len(STYLES))]
    if rng.u() < 0.15:
        ground = "一层是单元门和几扇带防盗网的窗户，门边停着自行车"
    else:
        suf, desc = SHOPS[rng.i(len(SHOPS))]
        name = PREFIX[rng.i(len(PREFIX))] + suf
        ground = f"一层是一家店铺，门头招牌上写着“{name}”，{desc}"
    prompt = (f"一栋{floors}层临街楼房的正立面照片。相机正对墙面平视拍摄，没有透视变形，楼的左右两边和楼顶边缘正好贴着画面边缘，"
              f"画面里只有这一面墙，没有天空、没有马路、没有相邻的建筑。{style}。{ground}。"
              f"白天，阴天柔和光线，写实摄影，细节清晰。")
    w, hh = BUCKETS[int(np.abs(np.log(B_ASPECT) - np.log(width / h)).argmin())]
    return dict(key=key, prompt=prompt, w=w, h=hh, gseed=int(k1 ^ k2) & 0x7FFFFFFF, axis=axis)


def facades_in(xmin, xmax, zmin, zmax, seed, margin=0.0):
    """Every building whose box overlaps the rectangle (grown by margin)."""
    box, boxp = Wd.window(xmin, xmax, zmin, zmax, seed)[:2]
    fac = box[:, 6] == Wd.M_FACADE
    ov = ((box[:, 0] < xmax + margin) & (box[:, 3] > xmin - margin) &
          (box[:, 2] < zmax + margin) & (box[:, 5] > zmin - margin))
    out = []
    for i in np.nonzero(fac & ov)[0]:
        f = facade(box[i], boxp[i], seed)
        f["cx"], f["cz"] = float(box[i, 0] + box[i, 3]) / 2, float(box[i, 2] + box[i, 5]) / 2
        out.append(f)
    return out


def path(seed, key):
    """Cut-outs (key starts with x; facade keys are hex) keep their alpha as PNG; everything else is a JPEG."""
    return asset_dir(seed) / (f"{key}.png" if key.startswith("x") else f"{key}.jpg")


# ------------------------------------------------------------------ props: trees, statues, benches, kiosks ...
SPECIES = ["法国梧桐", "国槐", "银杏", "香樟", "白蜡树", "栾树", "垂柳", "女贞", "榆树", "白玉兰", "水杉", "合欢树"]
STATUES = ["一位老人坐在石头上读书的铜像", "一匹扬蹄奔马的铜像", "一位母亲抱着孩子的铜像", "一位扛着锄头的农民铜像",
           "两只展翅的白鹤不锈钢雕塑", "一个吹笛子的少年铜像", "一头卧牛的铜像", "一位拉二胡的老人铜像"]
KIOSKS = ["报刊亭", "书报亭", "便民服务亭", "彩票站", "早点亭", "修鞋配钥匙亭"]
POSTERS = ["社区健康讲座", "重阳节登高活动", "公园菊花展", "垃圾分类 从我做起", "老年大学春季招生", "广场舞比赛",
           "社区义诊", "防范电信诈骗", "全民健身日", "夏日纳凉晚会", "书法绘画展", "社区读书会"]
TREE_H = {Wd.P_STREET_TREE: (6.5, 8.5), Wd.P_PARK_TREE: (7.0, 10.0), Wd.P_BIG_TREE: (10.0, 12.5)}
_CUT = "透明背景，只有{}本身，没有地面、没有阴影、没有其他物体，侧面平视，完整地出现在画面中，写实摄影，自然光。"


def _k(*xs):
    k1, k2 = Wd.h32(*xs), Wd.h32(*xs, 991)
    return k1, k2, f"{k1:08x}{k2:08x}"


def prop_asset(pr, seed):
    """Props array row -> asset dict (key, prompt, w, h, gseed, kind) or None. Trees share one cut-out per street
    segment (one species along a street), the rest are one asset per object."""
    kind = int(pr[0])
    ga, gb, gc = int(pr[8]), int(pr[9]), int(pr[10])
    if kind in TREE_H:
        k1, k2, key = _k(seed, 31, kind, ga, gb, gc)
        sp = SPECIES[Wd.Rng(k1, k2).i(len(SPECIES))]
        old = "高大的老" if kind == Wd.P_BIG_TREE else ""
        prompt = f"一棵完整的{old}{sp}，从地面上的树干根部到树冠顶部都在画面内，枝叶繁茂。" + _CUT.format("这棵树")
        return dict(key="x" + key, prompt=prompt, w=576, h=896, gseed=k1 & 0x7FFFFFFF, kind="cutout")
    rx, rz = int(round(float(pr[1]) * 100)), int(round(float(pr[3]) * 100))
    k1, k2, key = _k(seed, 37, kind, rx, rz)
    rng = Wd.Rng(k1, k2)
    if kind == Wd.P_STATUE:
        prompt = f"{STATUES[rng.i(len(STATUES))]}，城市广场上的雕塑，没有底座。" + _CUT.format("雕塑")
        return dict(key="x" + key, prompt=prompt, w=576, h=896, gseed=k1 & 0x7FFFFFFF, kind="cutout")
    if kind == Wd.P_BENCH:
        prompt = "一张街边的公共长椅的正面照片，木条椅面和靠背，铸铁扶手和椅腿。" + _CUT.format("长椅").replace("侧面", "正面")
        return dict(key="x" + key, prompt=prompt, w=896, h=576, gseed=k1 & 0x7FFFFFFF, kind="cutout")
    if kind == Wd.P_KIOSK:
        name = KIOSKS[rng.i(len(KIOSKS))]
        prompt = (f"一个城市街头{name}的正面照片，亭子顶上的招牌写着“{name}”，窗口和柜台摆满商品。"
                  + _CUT.format("亭子").replace("侧面", "正面"))
        return dict(key="x" + key, prompt=prompt, w=704, h=704, gseed=k1 & 0x7FFFFFFF, kind="cutout")
    if kind == Wd.P_ADCOL:
        title = POSTERS[rng.i(len(POSTERS))]
        prompt = f"一张竖版的社区宣传海报，标题大字写着“{title}”，配有插图和几行小字说明，海报铺满整个画面，平面设计。"
        return dict(key=key, prompt=prompt, w=512, h=1024, gseed=k1 & 0x7FFFFFFF, kind="plain")
    if kind == Wd.P_CLOCK:
        prompt = "一个方形街钟的钟面正面特写，白色表盘，黑色阿拉伯数字和指针，深绿色金属边框，钟面铺满整个画面，写实摄影。"
        return dict(key=key, prompt=prompt, w=704, h=704, gseed=k1 & 0x7FFFFFFF, kind="plain")
    return None


def prop_shape(pr, seed):
    """Billboard size of a cut-out prop: (y0, height, half width ratio to height is from the photo) and angle."""
    kind = int(pr[0])
    rng = Wd.Rng(seed, 41, int(round(float(pr[1]) * 100)), int(round(float(pr[3]) * 100)))
    if kind in TREE_H:
        lo, hi = TREE_H[kind]
        return 0.0, rng.r(lo, hi), rng.r(0, math.pi / 2)
    if kind == Wd.P_STATUE:
        return float(pr[2]), float(pr[4]), rng.r(0, math.pi / 2)
    if kind == Wd.P_BENCH:
        return 0.0, None, float(pr[6])          # height follows the photo, width = the bench box
    return float(pr[2]), float(pr[4]), float(pr[6])


# ------------------------------------------------------------------ ground: asphalt per street segment, paving per block
ROADS = ["新铺的黑色细粒沥青", "颜色发灰、有细小裂缝的旧沥青", "带几块深色修补痕迹的沥青", "粗颗粒的深灰色沥青"]
PAVES = ["灰色30厘米方形水泥地砖", "红色和灰色相间的长方形透水砖", "人字形铺设的红砖", "浅灰色花岗岩石板",
         "六边形灰色水泥砖", "米黄色和灰色相间的方形地砖", "错缝铺设的深灰色长条石材"]
ROAD_PERIOD, PAVE_PERIOD = 3.0, 2.0      # metres covered by one texture repeat


def road_asset(i, j, ax, seed):
    k1, k2, key = _k(seed, 43, i, j, ax)
    v = ROADS[Wd.Rng(k1, k2).i(len(ROADS))]
    prompt = (f"从正上方垂直向下拍摄的城市马路路面照片，{v}，画面约3米见方，纹理均匀铺满整个画面，"
              f"没有车道线，没有井盖，没有阴影，没有任何物体，均匀的阴天光线，写实摄影。")
    return dict(key="g" + key, prompt=prompt, w=704, h=704, gseed=k1 & 0x7FFFFFFF, kind="seamless")


def walk_asset(i, j, seed):
    k1, k2, key = _k(seed, 47, i, j)
    v = PAVES[Wd.Rng(k1, k2).i(len(PAVES))]
    prompt = (f"从正上方垂直向下拍摄的人行道铺装照片，{v}，画面约2米见方，砖块整齐铺满整个画面，"
              f"没有阴影，没有任何物体，均匀的阴天光线，写实摄影。")
    return dict(key="g" + key, prompt=prompt, w=704, h=704, gseed=k1 & 0x7FFFFFFF, kind="seamless")


def assets_in(xmin, xmax, zmin, zmax, seed, margin=0.0):
    """Everything generated for a rectangle: facades, props and ground, each with cx, cz for ordering."""
    out = facades_in(xmin, xmax, zmin, zmax, seed, margin)
    for f in out:
        f["kind"] = "facade"
    x0, x1, z0, z1 = xmin - margin, xmax + margin, zmin - margin, zmax + margin
    props = Wd.window(x0, x1, z0, z1, seed)[6]
    seen = set()
    for pr in props:
        if not (x0 <= pr[1] <= x1 and z0 <= pr[3] <= z1):
            continue
        a = prop_asset(pr, seed)
        if a and a["key"] not in seen:
            seen.add(a["key"])
            a["cx"], a["cz"] = float(pr[1]), float(pr[3])
            out.append(a)
    for i in range(math.floor(x0 / Wd.P), math.floor(x1 / Wd.P) + 1):
        for j in range(math.floor(z0 / Wd.P), math.floor(z1 / Wd.P) + 1):
            for a, cx, cz in ((road_asset(i, j, 0, seed), (i + 0.5) * Wd.P, j * Wd.P),
                              (road_asset(i, j, 1, seed), i * Wd.P, (j + 0.5) * Wd.P),
                              (walk_asset(i, j, seed), (i + 0.5) * Wd.P, (j + 0.5) * Wd.P)):
                a["cx"], a["cz"] = cx, cz
                out.append(a)
    return out


def have(seed, key):
    return os.path.exists(path(seed, key))
