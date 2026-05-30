import urllib.parse
import httpx
import asyncio
import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from configs.settings import settings
from utils.logger import get_logger

logger = get_logger("shiliu.tools.weather")

QWEATHER_API_KEY = settings.QWEATHER_API_KEY
BASE_HOST = settings.qweather_base_url
if not QWEATHER_API_KEY:
    logger.error("未找到 QWEATHER_API_KEY")


# ==========================================
# 输出/输入 Spec 定义
# ==========================================
class CityInputSpec(BaseModel):
    city: str = Field(..., description="城市名称（如：北京、峨眉山、乐山）")


class AstronomyInputSpec(BaseModel):
    city: str = Field(..., description="城市名称")
    date: Optional[str] = Field(default=None, description="查询日期，格式YYYYMMDD，默认今日")


# 统一的错误回包格式
class ErrorResponse(BaseModel):
    error: str


class WeatherResponse(BaseModel):
    city: str
    temp: str
    feels_like: str
    condition: str
    wind_dir: str
    wind_scale: str
    humidity: str
    update_time: str


class AdviceItem(BaseModel):
    name: str
    category: str
    text: str


class TravelAdviceResponse(BaseModel):
    city: str
    advice_list: List[AdviceItem]


class AstronomyResponse(BaseModel):
    city: str
    date: str
    sunrise: str
    sunset: str
    moonrise: str
    moonset: str
    moon_phase: str


class DailyForecast(BaseModel):
    date: str
    text_day: str
    temp_min: str
    temp_max: str


class ForecastResponse(BaseModel):
    city: str
    forecasts: List[DailyForecast]


# ==========================================
# 内部辅助函数
# ==========================================
async def _get_location_id(location_name: str) -> str | None:
    url = f"{BASE_HOST}/geo/v2/city/lookup"
    params = {"location": location_name, "key": QWEATHER_API_KEY, "lang": "zh", "number": 1}

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "200" and data.get("location"):
                return data["location"][0]["id"]
            return None
        except Exception as e:
            logger.exception("GeoAPI 查询异常", city=location_name, error=str(e))
            return None


# ==========================================
# 结构化工具函数 (@tool)
# ==========================================
@tool("weather_api", args_schema=CityInputSpec)
async def get_current_weather(city: str) -> str:
    """查询指定城市的实时天气预报和气温。"""
    logger.info("调用实时天气", city=city)
    location_id = await _get_location_id(city)
    if not location_id:
        return ErrorResponse(error=f"未能识别城市名「{city}」").model_dump_json()

    params = {"location": location_id, "key": QWEATHER_API_KEY}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{BASE_HOST}/v7/weather/now", params=params)
            data = resp.json()
            if data.get("code") == "200":
                now = data["now"]
                res = WeatherResponse(
                    city=city, temp=now['temp'], feels_like=now['feelsLike'],
                    condition=now['text'], wind_dir=now['windDir'],
                    wind_scale=now['windScale'], humidity=now['humidity'],
                    update_time=data['updateTime']
                )
                return res.model_dump_json()
            return ErrorResponse(error=f"API错误: {data.get('code')}").model_dump_json()
        except Exception as e:
            return ErrorResponse(error=f"请求异常: {str(e)}").model_dump_json()


@tool("travel_advice_api", args_schema=CityInputSpec)
async def get_travel_advice(city: str) -> str:
    """获取针对该城市的生活指数建议（穿衣、紫外线、运动、旅游建议等）。"""
    logger.info("调用生活指数", city=city)
    location_id = await _get_location_id(city)
    if not location_id:
        return ErrorResponse(error=f"未能识别城市名「{city}」").model_dump_json()

    params = {"location": location_id, "key": QWEATHER_API_KEY, "type": "1,3,5", "lang": "zh"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{BASE_HOST}/v7/indices/1d", params=params)
            data = resp.json()
            if data.get("code") == "200" and data.get("daily"):
                items = [AdviceItem(name=i['name'], category=i['category'], text=i['text']) for i in data["daily"]]
                return TravelAdviceResponse(city=city, advice_list=items).model_dump_json()
            return ErrorResponse(error="暂无可用的生活指数数据").model_dump_json()
        except Exception as e:
            return ErrorResponse(error=f"请求异常: {str(e)}").model_dump_json()


@tool("astronomy_api", args_schema=AstronomyInputSpec)
async def get_astronomy_info(city: str, date: str = None) -> str:
    """查询指定城市的日出日落、月升月落及当前月相，适用于摄影、夜游规划。"""
    if not date:
        date = datetime.datetime.now().strftime("%Y%m%d")

    logger.info("调用天文信息", city=city, date=date)
    location_id = await _get_location_id(city)
    if not location_id:
        return ErrorResponse(error=f"未能识别城市名「{city}」").model_dump_json()

    params = {"location": location_id, "key": QWEATHER_API_KEY, "date": date}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            sun_resp, moon_resp = await asyncio.gather(
                client.get(f"{BASE_HOST}/v7/astronomy/sun", params=params),
                client.get(f"{BASE_HOST}/v7/astronomy/moon", params=params)
            )
            sun_data, moon_data = sun_resp.json(), moon_resp.json()

            if sun_data.get("code") == "200" and moon_data.get("code") == "200":
                sun_daily = sun_data.get("daily", [{}])[0]
                moon_daily = moon_data.get("daily", [{}])[0]
                res = AstronomyResponse(
                    city=city, date=date,
                    sunrise=sun_daily.get("sunrise", "未知"),
                    sunset=sun_daily.get("sunset", "未知"),
                    moonrise=moon_daily.get("moonrise", "未知"),
                    moonset=moon_daily.get("moonset", "未知"),
                    moon_phase=moon_daily.get("moonPhaseName", "未知")
                )
                return res.model_dump_json()
            return ErrorResponse(error="天文数据获取失败").model_dump_json()
        except Exception as e:
            return ErrorResponse(error=f"请求异常: {str(e)}").model_dump_json()


@tool("weather_forecast_api", args_schema=CityInputSpec)
async def get_weather_forecast(city: str) -> str:
    """查询指定城市未来3天（含今天、明天、后天）的天气预报。"""
    logger.info("调用天气预报", city=city)
    location_id = await _get_location_id(city)
    if not location_id:
        return ErrorResponse(error=f"未能识别城市名「{city}」").model_dump_json()

    params = {"location": location_id, "key": QWEATHER_API_KEY}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{BASE_HOST}/v7/weather/3d", params=params)
            data = resp.json()
            if data.get("code") == "200" and data.get("daily"):
                forecasts = [
                    DailyForecast(date=d['fxDate'], text_day=d['textDay'], temp_min=d['tempMin'], temp_max=d['tempMax'])
                    for d in data["daily"]
                ]
                return ForecastResponse(city=city, forecasts=forecasts).model_dump_json()
            return ErrorResponse(error="预报获取失败").model_dump_json()
        except Exception as e:
            return ErrorResponse(error=f"请求异常: {str(e)}").model_dump_json()