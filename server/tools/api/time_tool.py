import datetime
from pydantic import BaseModel
from langchain_core.tools import tool

from utils.logger import get_logger

logger = get_logger("shiliu.tools.time")


class TimeResponse(BaseModel):
    datetime: str
    date: str
    time: str
    weekday: str
    timezone: str = "Asia/Shanghai"


@tool("get_current_time")
async def get_current_time() -> str:
    """获取当前真实系统时间，返回 ISO 格式的日期、时间、星期。"""
    now = datetime.datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    response = TimeResponse(
        datetime=now.strftime("%Y-%m-%d %H:%M:%S"),
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M:%S"),
        weekday=weekdays[now.weekday()],
    )
    logger.info("返回当前时间", datetime=response.datetime)
    return response.model_dump_json()
