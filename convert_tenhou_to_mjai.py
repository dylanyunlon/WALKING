#!/usr/bin/env python3
"""
将 Tenhou XML (mjlog) 文件批量转换为 MJAI JSON 格式
用于 WALKING 项目的训练数据准备

修复版本:
    - 完整实现鳴牌 (chi/pon/kan) 的 m 值解析
    - 基于 tenhou-python-bot 项目的解码逻辑

用法:
    python convert_tenhou_to_mjai_fixed.py --input xml/2023/ --output workdir/dataset/2023/
"""

import os
import sys
import gzip
import json
import argparse
import sqlite3
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Tuple
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


# 牌的编码转换
TILE_MAP = {
    # 萬子 (0-35)
    **{i: f"{(i // 4) + 1}m" for i in range(36)},
    # 筒子 (36-71)
    **{i: f"{((i - 36) // 4) + 1}p" for i in range(36, 72)},
    # 索子 (72-107)
    **{i: f"{((i - 72) // 4) + 1}s" for i in range(72, 108)},
    # 字牌 (108-135)
    **{i: ["E", "S", "W", "N", "P", "F", "C"][(i - 108) // 4] for i in range(108, 136)},
}

# 已知的特殊标签（不是摸牌/打牌）
KNOWN_TAGS = {
    "SHUFFLE", "GO", "UN", "BYE", "TAIKYOKU", "INIT", "DORA", 
    "REACH", "N", "AGARI", "RYUKYOKU", "PROF"
}

# 摸牌/打牌标签的正则表达式
DRAW_PATTERN = re.compile(r'^[TUVW](\d+)$')
DISCARD_PATTERN = re.compile(r'^[DEFGdefg](\d+)$')


def tile_to_mjai(tile_id: int) -> str:
    """将 Tenhou 牌 ID 转换为 MJAI 格式
    
    Tenhou 牌编码:
    - 0-35: 萬子 (每种4张, 0-3是1m, 4-7是2m, ...)
    - 36-71: 筒子
    - 72-107: 索子
    - 108-135: 字牌 (东南西北白发中)
    
    特殊红宝牌:
    - 16: 红五萬 (5mr)
    - 52: 红五筒 (5pr)
    - 88: 红五索 (5sr)
    """
    if tile_id == 16:
        return "5mr"
    elif tile_id == 52:
        return "5pr"
    elif tile_id == 88:
        return "5sr"
    return TILE_MAP.get(tile_id, str(tile_id))


@dataclass
class Meld:
    """鳴牌数据结构"""
    type: str = ""
    tiles: List[int] = field(default_factory=list)
    called_tile: int = 0
    who: int = 0
    from_who: int = 0


def parse_meld(who: int, m: int) -> Meld:
    """解析鳴牌的 m 值
    
    基于 Tenhou 的位编码格式:
    - bit 0-1: from_who (相对位置偏移)
    - bit 2: chi 标志
    - bit 3-4: pon/kan 标志
    - bit 5: nuki (北抜き) 标志
    
    参考: https://github.com/MahjongRepository/tenhou-python-bot
    """
    meld = Meld()
    meld.who = who
    meld.from_who = (who + (m & 0x3)) % 4
    
    if m & 0x4:
        # 吃 (Chi)
        parse_chi(m, meld)
    elif m & 0x18:
        # 碰 (Pon) 或 加杠 (Chakan/Shouminkan)
        parse_pon(m, meld)
    elif m & 0x20:
        # 北抜き (三人麻将特有)
        parse_nuki(m, meld)
    else:
        # 杠 (Kan) - 暗杠或大明杠
        parse_kan(m, meld)
    
    return meld


def parse_chi(data: int, meld: Meld):
    """解析吃 (Chi)
    
    吃的编码格式:
    - bit 0-1: who offset
    - bit 2: chi flag (1)
    - bit 3-4: t0 (第一张牌的偏移)
    - bit 5-6: t1 (第二张牌的偏移)
    - bit 7-8: t2 (第三张牌的偏移)
    - bit 10+: base_and_called
    """
    meld.type = "chi"
    
    # 提取三张牌的偏移量
    t0 = (data >> 3) & 0x3
    t1 = (data >> 5) & 0x3
    t2 = (data >> 7) & 0x3
    
    # 基础牌和被叫的牌
    base_and_called = data >> 10
    base = base_and_called // 3
    called = base_and_called % 3
    
    # 转换 base: 从7连续编码转为9连续编码 (跳过8,9等边界)
    base = (base // 7) * 9 + base % 7
    
    # 计算三张牌的实际 ID
    meld.tiles = [
        t0 + 4 * (base + 0),
        t1 + 4 * (base + 1),
        t2 + 4 * (base + 2)
    ]
    meld.called_tile = meld.tiles[called]


def parse_pon(data: int, meld: Meld):
    """解析碰 (Pon) 或加杠 (Chakan/Shouminkan)
    
    碰/加杠的编码格式:
    - bit 0-1: who offset
    - bit 3: pon flag
    - bit 4: chakan flag (加杠)
    - bit 5-6: t4 (未使用的第四张牌)
    - bit 9+: base_and_called
    """
    # t4 表示哪张牌不在碰的组合中
    t4 = (data >> 5) & 0x3
    
    # 根据 t4 确定使用的三张牌
    pon_tiles = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))[t4]
    t0, t1, t2 = pon_tiles
    
    base_and_called = data >> 9
    base = base_and_called // 3
    called = base_and_called % 3
    
    # 检查是否是加杠
    if data & 0x10:
        # 加杠 (Chakan/Shouminkan)
        meld.type = "kakan"
        meld.tiles = [
            t0 + 4 * base,
            t1 + 4 * base,
            t2 + 4 * base,
            t4 + 4 * base  # 加杠的第四张牌
        ]
        meld.called_tile = t4 + 4 * base
    else:
        # 普通碰
        meld.type = "pon"
        meld.tiles = [
            t0 + 4 * base,
            t1 + 4 * base,
            t2 + 4 * base
        ]
        meld.called_tile = meld.tiles[called]


def parse_kan(data: int, meld: Meld):
    """解析杠 (Kan) - 暗杠或大明杠
    
    杠的编码格式:
    - bit 0-1: who offset (0 表示暗杠)
    - bit 8+: base tile
    """
    base_and_called = data >> 8
    base = base_and_called // 4
    called = base_and_called % 4
    
    # 四张牌
    meld.tiles = [
        0 + 4 * base,
        1 + 4 * base,
        2 + 4 * base,
        3 + 4 * base
    ]
    meld.called_tile = meld.tiles[called]
    
    # 判断是暗杠还是大明杠
    from_who_offset = data & 0x3
    if from_who_offset == 0:
        meld.type = "ankan"
    else:
        meld.type = "daiminkan"


def parse_nuki(data: int, meld: Meld):
    """解析北抜き (三人麻将特有)"""
    meld.type = "nuki"
    unused = (data >> 8) & 0x3
    meld.tiles = [unused + 4 * 30]  # 北 (30 * 4 = 120-123)
    meld.called_tile = meld.tiles[0]


def parse_init(attrib: dict) -> list:
    """解析 INIT 标签 (局开始)"""
    events = []
    
    # 开始新局
    seed = attrib.get("seed", "0,0,0,0,0,0").split(",")
    kyoku = int(seed[0])
    honba = int(seed[1])
    kyotaku = int(seed[2])
    
    # 庄家
    oya = int(attrib.get("oya", "0"))
    
    # 点数
    scores = [int(x) * 100 for x in attrib.get("ten", "25000,25000,25000,25000").split(",")]
    
    # 获取每个玩家的手牌
    tehais = []
    for i in range(4):
        hai_str = attrib.get(f"hai{i}", "")
        if hai_str:
            tiles = [tile_to_mjai(int(t)) for t in hai_str.split(",") if t]
        else:
            tiles = []
        tehais.append(tiles)
    
    events.append({
        "type": "start_kyoku",
        "bakaze": ["E", "S", "W", "N"][kyoku // 4],
        "dora_marker": tile_to_mjai(int(seed[5])),
        "kyoku": (kyoku % 4) + 1,
        "honba": honba,
        "kyotaku": kyotaku,
        "oya": oya,
        "scores": scores,
        "tehais": tehais
    })
    
    return events


def parse_draw(tag: str) -> Optional[dict]:
    """解析摸牌标签 (T/U/V/W + 数字)"""
    match = DRAW_PATTERN.match(tag)
    if not match:
        return None
    
    player_map = {"T": 0, "U": 1, "V": 2, "W": 3}
    player = player_map[tag[0]]
    tile_id = int(match.group(1))
    
    return {
        "type": "tsumo",
        "actor": player,
        "pai": tile_to_mjai(tile_id)
    }


def parse_discard(tag: str) -> Optional[dict]:
    """解析打牌标签 (D/E/F/G = 手出, d/e/f/g = 摸切)"""
    match = DISCARD_PATTERN.match(tag)
    if not match:
        return None
    
    player_map = {"D": 0, "E": 1, "F": 2, "G": 3, "d": 0, "e": 1, "f": 2, "g": 3}
    player = player_map[tag[0]]
    tile_id = int(match.group(1))
    tsumogiri = tag[0].islower()
    
    return {
        "type": "dahai",
        "actor": player,
        "pai": tile_to_mjai(tile_id),
        "tsumogiri": tsumogiri
    }


def parse_call(attrib: dict) -> list:
    """解析鳴牌 (N 标签)"""
    events = []
    who = int(attrib.get("who", "0"))
    m = int(attrib.get("m", "0"))
    
    meld = parse_meld(who, m)
    
    if meld.type == "chi":
        # 吃
        # consumed 是手中的两张牌 (不含被叫的牌)
        consumed = [tile_to_mjai(t) for t in meld.tiles if t != meld.called_tile]
        events.append({
            "type": "chi",
            "actor": meld.who,
            "target": meld.from_who,
            "pai": tile_to_mjai(meld.called_tile),
            "consumed": consumed
        })
    elif meld.type == "pon":
        # 碰
        consumed = [tile_to_mjai(t) for t in meld.tiles if t != meld.called_tile]
        events.append({
            "type": "pon",
            "actor": meld.who,
            "target": meld.from_who,
            "pai": tile_to_mjai(meld.called_tile),
            "consumed": consumed
        })
    elif meld.type == "kakan":
        # 加杠
        events.append({
            "type": "kakan",
            "actor": meld.who,
            "pai": tile_to_mjai(meld.called_tile),
            "consumed": [tile_to_mjai(t) for t in meld.tiles[:3]]
        })
    elif meld.type == "daiminkan":
        # 大明杠
        consumed = [tile_to_mjai(t) for t in meld.tiles if t != meld.called_tile]
        events.append({
            "type": "daiminkan",
            "actor": meld.who,
            "target": meld.from_who,
            "pai": tile_to_mjai(meld.called_tile),
            "consumed": consumed
        })
    elif meld.type == "ankan":
        # 暗杠
        events.append({
            "type": "ankan",
            "actor": meld.who,
            "consumed": [tile_to_mjai(t) for t in meld.tiles]
        })
    elif meld.type == "nuki":
        # 北抜き (三人麻将)
        events.append({
            "type": "nukidora",
            "actor": meld.who,
            "pai": tile_to_mjai(meld.called_tile)
        })
    
    return events


def parse_reach(attrib: dict) -> list:
    """解析立直 (REACH 标签)"""
    who = int(attrib.get("who", "0"))
    step = int(attrib.get("step", "1"))
    
    if step == 1:
        return [{"type": "reach", "actor": who}]
    else:
        return [{"type": "reach_accepted", "actor": who}]


def parse_agari(attrib: dict) -> list:
    """解析和牌 (AGARI 标签)"""
    who = int(attrib.get("who", "0"))
    from_who = int(attrib.get("fromWho", str(who)))
    
    # 解析点数变化
    sc = attrib.get("sc", "").split(",")
    deltas = [0, 0, 0, 0]
    if len(sc) >= 8:
        deltas = [int(sc[i * 2 + 1]) * 100 for i in range(4)]
    
    # 解析和牌信息
    hai = attrib.get("hai", "")
    machi = attrib.get("machi", "")
    
    event = {
        "type": "hora",
        "actor": who,
        "target": from_who,
        "deltas": deltas
    }
    
    # 添加和牌的牌
    if machi:
        event["pai"] = tile_to_mjai(int(machi))
    
    return [event]


def parse_ryukyoku(attrib: dict) -> list:
    """解析流局 (RYUKYOKU 标签)"""
    reason = attrib.get("type", "")
    
    # 解析点数变化
    sc = attrib.get("sc", "").split(",")
    deltas = [0, 0, 0, 0]
    if len(sc) >= 8:
        deltas = [int(sc[i * 2 + 1]) * 100 for i in range(4)]
    
    event = {"type": "ryukyoku"}
    if deltas != [0, 0, 0, 0]:
        event["deltas"] = deltas
    if reason:
        event["reason"] = reason
    
    return [event]


def convert_xml_to_mjai(xml_content: str) -> list:
    """将 Tenhou XML 转换为 MJAI 事件列表"""
    events = [{"type": "start_game", "names": ["", "", "", ""]}]
    
    try:
        # 解析 XML
        root = ET.fromstring(xml_content)
        
        for elem in root:
            tag = elem.tag
            attrib = elem.attrib
            
            if tag == "INIT":
                events.extend(parse_init(attrib))
            elif tag == "DORA":
                events.append({
                    "type": "dora",
                    "dora_marker": tile_to_mjai(int(attrib.get("hai", "0")))
                })
            elif tag == "REACH":
                events.extend(parse_reach(attrib))
            elif tag == "N":
                events.extend(parse_call(attrib))
            elif tag == "AGARI":
                events.extend(parse_agari(attrib))
                events.append({"type": "end_kyoku"})
            elif tag == "RYUKYOKU":
                events.extend(parse_ryukyoku(attrib))
                events.append({"type": "end_kyoku"})
            elif tag in KNOWN_TAGS:
                # 跳过其他已知标签 (GO, UN, BYE, TAIKYOKU, SHUFFLE, PROF)
                continue
            else:
                # 尝试解析摸牌/打牌标签
                draw = parse_draw(tag)
                if draw:
                    events.append(draw)
                    continue
                
                discard = parse_discard(tag)
                if discard:
                    events.append(discard)
                    continue
    
    except ET.ParseError as e:
        print(f"XML 解析错误: {e}")
        return []
    
    events.append({"type": "end_game"})
    return events


def process_xml_file(xml_path: Path, output_dir: Path) -> bool:
    """处理单个 XML 文件"""
    try:
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_content = f.read()
        
        events = convert_xml_to_mjai(xml_content)
        
        if not events:
            return False
        
        # 输出为 gzip JSON
        output_path = output_dir / f"{xml_path.stem}.json.gz"
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        
        return True
    except Exception as e:
        print(f"处理 {xml_path} 时出错: {e}")
        return False


def process_db_file(db_path: Path, output_dir: Path, limit: int = None) -> int:
    """从 SQLite 数据库处理日志"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = "SELECT log_id, content FROM logs WHERE content IS NOT NULL"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    
    count = 0
    for log_id, content in cursor:
        if content:
            events = convert_xml_to_mjai(content)
            if events:
                output_path = output_dir / f"{log_id}.json.gz"
                with gzip.open(output_path, "wt", encoding="utf-8") as f:
                    for event in events:
                        f.write(json.dumps(event, ensure_ascii=False) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"已处理 {count} 个文件...")
    
    conn.close()
    return count


def test_meld_parsing():
    """测试鳴牌解析"""
    print("测试鳴牌解析...")
    
    # 测试用例来自 tenhou-python-bot 的测试
    # Pon: m=34314, who=3 -> tiles=[89, 90, 91]
    meld = parse_meld(3, 34314)
    assert meld.type == "pon", f"Expected pon, got {meld.type}"
    assert meld.tiles == [89, 90, 91], f"Expected [89, 90, 91], got {meld.tiles}"
    print(f"  Pon 测试通过: tiles={meld.tiles}, called={meld.called_tile}")
    
    # Kan: m=13825, who=3 -> tiles=[52, 53, 54, 55]
    meld = parse_meld(3, 13825)
    assert meld.type in ["ankan", "daiminkan"], f"Expected kan type, got {meld.type}"
    assert meld.tiles == [52, 53, 54, 55], f"Expected [52, 53, 54, 55], got {meld.tiles}"
    print(f"  Kan 测试通过: type={meld.type}, tiles={meld.tiles}")
    
    # Chakan: m=18547, who=3 -> tiles=[48, 49, 50, 51]
    meld = parse_meld(3, 18547)
    assert meld.type == "kakan", f"Expected kakan, got {meld.type}"
    assert meld.tiles == [48, 49, 50, 51], f"Expected [48, 49, 50, 51], got {meld.tiles}"
    print(f"  Kakan 测试通过: tiles={meld.tiles}")
    
    # Chi: m=27031, who=3 -> tiles=[42, 44, 51]
    meld = parse_meld(3, 27031)
    assert meld.type == "chi", f"Expected chi, got {meld.type}"
    assert meld.tiles == [42, 44, 51], f"Expected [42, 44, 51], got {meld.tiles}"
    print(f"  Chi 测试通过: tiles={meld.tiles}, called={meld.called_tile}")
    
    print("所有测试通过!")


def main():
    parser = argparse.ArgumentParser(description="将 Tenhou XML 转换为 MJAI JSON 格式 (修复版)")
    parser.add_argument("--input", "-i", help="输入目录或数据库文件")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--limit", "-l", type=int, help="限制处理数量")
    parser.add_argument("--workers", "-w", type=int, default=4, help="并行处理数")
    parser.add_argument("--test", action="store_true", help="运行测试")
    args = parser.parse_args()
    
    if args.test:
        test_meld_parsing()
        return
    
    if not args.input or not args.output:
        parser.error("--input 和 --output 参数是必需的 (除非使用 --test)")
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.suffix == ".db":
        # 从数据库处理
        count = process_db_file(input_path, output_dir, args.limit)
        print(f"总共处理 {count} 个日志")
    elif input_path.is_dir():
        # 从目录批量处理 XML 文件
        xml_files = list(input_path.glob("*.xml"))
        if args.limit:
            xml_files = xml_files[:args.limit]
        
        success = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_xml_file, f, output_dir): f for f in xml_files}
            for future in as_completed(futures):
                if future.result():
                    success += 1
                if success % 100 == 0 and success > 0:
                    print(f"已处理 {success}/{len(xml_files)} 个文件...")
        
        print(f"成功处理 {success}/{len(xml_files)} 个文件")
    else:
        print(f"输入路径不存在或格式不支持: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()