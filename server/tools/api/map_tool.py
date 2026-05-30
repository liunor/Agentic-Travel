import httpx
import urllib.parse
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from configs.settings import settings
from utils.logger import get_logger

logger = get_logger("shiliu.tools.map")

AMAP_API_KEY = settings.AMAP_API_KEY
BASE_AMAP_URL = settings.amap_base_url


# ==========================================
# 输出/输入 Spec 定义
# ==========================================
class ErrorResponse(BaseModel):
    error: str


class WalkingPlanInput(BaseModel):
    origin: str = Field(..., description="起点地址")
    destination: str = Field(..., description="终点地址")


class WalkingPlanResponse(BaseModel):
    origin: str
    destination: str
    distance_meters: str
    duration_seconds: str
    steps: List[str]


class DistanceInput(BaseModel):
    origin: str = Field(..., description="起点地址")
    destination: str = Field(..., description="终点地址")
    type: str = Field(default="straight", description="距离类型: straight, driving, walking")


class DistanceResponse(BaseModel):
    origin: str
    destination: str
    distance_type: str
    distance_meters: str
    duration_seconds: Optional[str] = None


class SearchAroundInput(BaseModel):
    address: str = Field(..., description="中心地址")
    keyword: str = Field(..., description="搜索关键词")
    radius: int = Field(default=1000, description="搜索半径(米)")
    page_size: int = Field(default=5, description="返回结果数量")


class POIItem(BaseModel):
    name: str
    address: str
    tel: str


class SearchAroundResponse(BaseModel):
    center_address: str
    keyword: str
    total_count: str
    pois: List[POIItem]


class StaticMapInput(BaseModel):
    address: str = Field(..., description="地图中心地址")
    zoom: int = Field(default=15, description="地图缩放级别(1-19)")


class StaticMapResponse(BaseModel):
    address: str
    map_url: str
    zoom: int


# ==========================================
# 内部辅助函数
# ==========================================
async def _get_lng_lat(address: str) -> tuple[str, str] | None:
    """ 获取地址的经纬度坐标

    Args:
        address: 地址字符串，支持直接传入"经度,纬度"格式的坐标

    Returns:
          (经度, 纬度)元组，或None表示解析失败
    """
    if "," in address and len(address.split(",")) == 2:
        try:
            float(address.split(",")[0])
            return address.split(",")[0], address.split(",")[1]
        except ValueError:
            pass

    params = {"address": address, "key": AMAP_API_KEY, "output": "json"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{BASE_AMAP_URL}/geocode/geo", params=params)
            data = resp.json()
            if data.get("status") == "1" and data.get("geocodes"):
                lng_lat = data["geocodes"][0]["location"].split(",")
                return lng_lat[0], lng_lat[1]
            return None
        except Exception as e:
            logger.exception("高德编码异常", address=address, error=str(e))
            return None


# ==========================================
# 结构化工具函数 (@tool)
# ==========================================
@tool("walking_plan_api", args_schema=WalkingPlanInput)
async def get_walking_plan(origin: str, destination: str) -> str:
    """查询起点到终点的步行规划路线（含距离、耗时、详细步骤）"""
    logger.info("步行规划", origin=origin, dest=destination)
    o_lnglat = await _get_lng_lat(origin)
    d_lnglat = await _get_lng_lat(destination)
    if not o_lnglat or not d_lnglat:
        return ErrorResponse(error="地址解析失败").model_dump_json()

    params = {
        "origin": f"{o_lnglat[0]},{o_lnglat[1]}",
        "destination": f"{d_lnglat[0]},{d_lnglat[1]}",
        "key": AMAP_API_KEY, "output": "json"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{BASE_AMAP_URL}/direction/walking", params=params)
            data = resp.json()
            if data.get("status") == "1" and data.get("route"):
                route = data["route"]["paths"][0]
                steps = [f"{s['instruction']} ({s['distance']}米)" for s in route["steps"]]
                # 精简过长的步骤省 token
                if len(steps) > 5:
                    steps = steps[:3] + ["...(中间省略)..."] + steps[-2:]

                res = WalkingPlanResponse(
                    origin=origin, destination=destination,
                    distance_meters=route['distance'],
                    duration_seconds=route['duration'],
                    steps=steps
                )
                return res.model_dump_json()
            return ErrorResponse(error=data.get("info", "规划失败")).model_dump_json()
        except Exception as e:
            return ErrorResponse(error=str(e)).model_dump_json()


@tool("distance_api", args_schema=DistanceInput)
async def get_distance(origin: str, destination: str, type: str = "straight") -> str:
    """测量两点间的距离（支持直线/驾车/步行）"""
    o_lnglat = await _get_lng_lat(origin)
    d_lnglat = await _get_lng_lat(destination)
    if not o_lnglat or not d_lnglat:
        return ErrorResponse(error="地址解析失败").model_dump_json()

    type_map = {"straight": "1", "driving": "0", "walking": "2"}
    amap_type = type_map.get(type, "1")

    params = {
        "origins": f"{o_lnglat[0]},{o_lnglat[1]}",
        "destination": f"{d_lnglat[0]},{d_lnglat[1]}",
        "type": amap_type, "key": AMAP_API_KEY, "output": "json"
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_AMAP_URL}/distance", params=params)
            data = resp.json()
            if data.get("status") == "1" and data.get("results"):
                result = data["results"][0]
                res = DistanceResponse(
                    origin=origin, destination=destination, distance_type=type,
                    distance_meters=result["distance"],
                    duration_seconds=result.get("duration")
                )
                return res.model_dump_json()
            return ErrorResponse(error="测量失败").model_dump_json()
        except Exception as e:
            return ErrorResponse(error=str(e)).model_dump_json()


@tool("around_search_api", args_schema=SearchAroundInput)
async def search_around(address: str, keyword: str, radius: int = 1000, page_size: int = 5) -> str:
    """搜索指定地址周边的POI（如餐厅、酒店等）。最多返回5条结果。"""
    page_size = min(page_size, 5)  # 硬限制，防止 token 爆炸
    center_lng_lat = await _get_lng_lat(address)
    if not center_lng_lat:
        return ErrorResponse(error="中心地址解析失败").model_dump_json()

    params = {
        "location": f"{center_lng_lat[0]},{center_lng_lat[1]}",
        "keywords": keyword, "radius": radius, "page_size": page_size,
        "output": "json", "key": AMAP_API_KEY
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_AMAP_URL}/place/around", params=params)
            data = resp.json()
            if data.get("status") == "1" and data.get("pois"):
                pois = [POIItem(name=p['name'], address=p['address'], tel=p.get('tel', '无')) for p in
                        data["pois"][:page_size]]
                res = SearchAroundResponse(center_address=address, keyword=keyword, total_count=data['count'],
                                           pois=pois)
                return res.model_dump_json()
            return ErrorResponse(error="周边搜索失败").model_dump_json()
        except Exception as e:
            return ErrorResponse(error=str(e)).model_dump_json()


@tool("static_map_api", args_schema=StaticMapInput)
async def get_static_map(address: str, zoom: int = 15) -> str:
    """生成指定地址的静态地图图片URL"""
    center_lng_lat = await _get_lng_lat(address)
    if not center_lng_lat:
        return ErrorResponse(error="地址解析失败").model_dump_json()

    location = f"{center_lng_lat[0]},{center_lng_lat[1]}"
    params = {
        "location": location, "zoom": zoom, "size": "600*400",
        "markers": f"mid,0xFF0000,A:{location}", "key": AMAP_API_KEY
    }

    query_string = urllib.parse.urlencode(params)
    map_url = f"{BASE_AMAP_URL}/staticmap?{query_string}"

    res = StaticMapResponse(address=address, map_url=map_url, zoom=zoom)
    return res.model_dump_json()