import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from configs.settings import settings
from utils.logger import get_logger

logger = get_logger("shiliu.tools.image")

SEEDREAM_API_KEY = settings.volcengine_API_KEY
SEEDREAM_CONF = settings.image_llm
SEEDREAM_MODEL_ID = SEEDREAM_CONF["model_id"]
SEEDREAM_BASE_URL = SEEDREAM_CONF["base_url"]


class ImageGenerationInput(BaseModel):
    prompt: str = Field(..., description="图片生成提示词，描述期望的画面内容与风格")
    size: str = Field(default="1024x1024", description="图片尺寸，如 1024x1024、1792x1024")


class ImageGenerationResponse(BaseModel):
    prompt: str
    image_url: str
    revised_prompt: str = ""
    size: str = "1024x1024"


class ImageGenerationError(BaseModel):
    error: str


@tool("generate_image_tool", args_schema=ImageGenerationInput)
async def generate_image_tool(prompt: str, size: str = "1024x1024") -> str:
    """文生图画画，根据文本描述生成图片，返回图片 URL。"""
    logger.info("发起图片生成", prompt=prompt[:80])
    if not SEEDREAM_API_KEY:
        return ImageGenerationError(error="volcengine_API_KEY 未配置").model_dump_json()

    payload = {
        "model": SEEDREAM_MODEL_ID,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }

    headers = {
        "Authorization": f"Bearer {SEEDREAM_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{SEEDREAM_BASE_URL}/images/generations",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            image_url = ""
            if "data" in data and len(data["data"]) > 0:
                image_url = data["data"][0].get("url", "")

            response = ImageGenerationResponse(
                prompt=prompt,
                image_url=image_url,
                revised_prompt=data.get("data", [{}])[0].get("revised_prompt", ""),
                size=size,
            )
            logger.info("图片生成完成", prompt=prompt[:80])
            return response.model_dump_json()

        except Exception as e:
            logger.exception("图片生成异常", error=str(e))
            return ImageGenerationError(error=f"图片生成异常: {str(e)}").model_dump_json()
