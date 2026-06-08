---
name: itinerary_planner
description: >
  制定旅行行程规划时使用。不要 spawn 一个 Worker 包揽全部——
  Coordinator 按本 SOP 逐步 spawn 基础工具 Worker 收集数据，
  最后由 Coordinator 综合所有结果直接输出行程给用户。
allowed_tools:
  - search_knowledge_base
  - weather_api
  - travel_advice_api
  - weather_forecast_api
  - distance_api
  - around_search_api
  - walking_plan_api
  - get_current_time
---

# 行程规划编排手册（Coordinator 专用）

按本流程逐步派发 Worker 收集数据，最后综合输出。

## 第一步：并行收集基础数据

以下 Worker 互不依赖，**同时 spawn**：

| Worker (agent_name) | 查什么 | directive 示例 |
|---------------------|--------|----------------|
| `get_current_time` | 当前时间，确定行程日期 | "获取当前时间" |
| `search_knowledge_base` | 景点介绍、开放时间、门票、美食推荐 | "查询{目的地}的景点介绍、开放时间、门票价格和特色美食" |
| `weather_api` | 目标日期天气 | "查询{目的地}{日期}的天气" |
| `weather_forecast_api` | 未来几天预报 | "查询{目的地}未来3天天气预报" |
| `travel_advice_api` | 穿衣/紫外线/运动建议 | "查询{目的地}的生活指数建议" |

**并发原则**：以上 5 个 Worker 在同一轮全部发出，不要串行。

## 第二步：距离与路线（依赖第一步知识库结果）

知识库返回景点列表后，对主要景点 spawn：

| Worker (agent_name) | 查什么 |
|---------------------|--------|
| `distance_api` | 景点间距离（起点→终点） |
| `walking_plan_api` | 步行路线详情 |
| `around_search_api` | 景点周边餐厅、酒店 POI |

多个景点距离查询可以并行发出。

## 第三步：综合输出

所有 Worker 结果到齐后，Coordinator 直接为用户输出行程。不要 spawn 额外 Worker。

### 时间计算参考
| 参数 | 值 |
|------|-----|
| 步行速度 | 4 km/h |
| 午餐 | 60 min |
| 休息 | 30 min |
| 单景点游览 | 2-3 h |

> 详细计算逻辑见 `scripts/time_calc.py`。

### 输出格式
```
## 行程方案
**日期**：YYYY年MM月DD日（周X）  **天气**：XX°C~XX°C，天气状况

### 上午
- **HH:MM - HH:MM** 景点 — 说明（步行耗时）

### 中午
- **HH:MM - HH:MM** 午餐 — 推荐理由

### 下午
- **HH:MM - HH:MM** 景点 — 说明

### 晚上
- **HH:MM - HH:MM** 晚餐/活动 — 说明

## 费用参考
| 项目 | 费用 |
|------|------|

## 注意事项
- 穿衣建议
- 交通提示
```
