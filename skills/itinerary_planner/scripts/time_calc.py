#!/usr/bin/env python3
"""
行程时间计算工具 — 给 itinerary_planner Worker 使用。

根据景点之间的距离和游览时长，自动计算每个时间段的行程安排。
当 SKILL.md 提到"需要估算时间"时，Agent 可以读取本脚本获取计算公式。

用法: python scripts/time_calc.py <distance_km> <visit_hours> [speed_kmh]
输出: JSON 格式的各时段预估时间
"""
import sys
import json


def calc_schedule(distance_km: float, visit_hours: float, speed_kmh: float = 4.0) -> dict:
    """根据距离和游览时间，生成典型一日游时间表。

    Args:
        distance_km: 景点间总距离（公里）
        visit_hours: 核心游览所需小时数
        speed_kmh: 步行速度，默认 4 km/h（景区步行节奏）
    """
    travel_min = round(distance_km / speed_kmh * 60)
    lunch_min = 60
    rest_min = 30

    schedule = {
        "08:00": "出发前往景区",
        f"08:{travel_min:02d}": f"抵达景点（步行约 {travel_min} 分钟 / {distance_km}km）",
        f"09:00": f"开始游览（预计 {visit_hours} 小时）",
        "12:00": "午餐时间（60 分钟）",
        "13:00": "下午继续游览或周边探索",
        "15:30": "休息补给（30 分钟）",
        "16:00": "自由活动或返回",
        "18:00": "结束行程",
    }

    # 如果有午后续游，插入时段
    if visit_hours > 4:
        schedule["14:00"] = "深度游览 / 第二景点"

    return {
        "distance_km": distance_km,
        "visit_hours": visit_hours,
        "walking_speed_kmh": speed_kmh,
        "travel_time_min": travel_min,
        "schedule": schedule,
        "total_duration": f"{travel_min + lunch_min + rest_min + int(visit_hours * 60)} 分钟",
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        # 无参时输出使用说明，方便 agent 理解如何调用
        print(json.dumps({
            "tool": "行程时间计算器",
            "usage": "python scripts/time_calc.py <distance_km> <visit_hours> [speed_kmh]",
            "example": "python scripts/time_calc.py 3.5 4.0 4.5",
            "params": {
                "distance_km": "景点间步行距离（公里）",
                "visit_hours": "核心游览时长（小时）",
                "speed_kmh": "步行速度 km/h，默认 4.0（景区节奏）"
            }
        }, ensure_ascii=False, indent=2))
    else:
        dist = float(sys.argv[1])
        hours = float(sys.argv[2])
        speed = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
        result = calc_schedule(dist, hours, speed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
