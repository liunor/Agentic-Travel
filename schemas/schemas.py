import json
from pydantic import BaseModel, Field
from typing import Optional, Union, Dict, Any


class SendMessageInputSpec(BaseModel):
    to: str = Field(..., description="接收者：队友名称或 Agent ID")
    # 完美保留源码的 summary 字段
    summary: Optional[str] = Field(default=None, description="在 UI 中作为预览显示的 5-10 词简短摘要")
    # 支持纯文本消息，或者结构化的字典信令 (如 shutdown_request)
    message: Union[str, Dict[str, Any]] = Field(..., description="纯文本消息内容或结构化消息（StructuredMessage）")

class SendMessageOutputSpec(BaseModel):
    success: bool
    message: str

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)