from typing import Optional, Literal
from pydantic import BaseModel, Field, ConfigDict

class RagContextSpec(BaseModel):
    """RAG 检索到的有效片段"""
    model_config = ConfigDict(strict=True)
    content: str = Field(..., description="知识点的具体文字内容")
    source: str = Field(..., description="来源文档或知识库")
    score: float = Field(default=0.0, description="向量检索相关性得分")

class WorkerUsageSpec(BaseModel):
    total_tokens: int = 0
    tool_uses: int = 0
    duration_ms: int = 0

class WorkerNotificationSpec(BaseModel):
    task_id: str = Field(..., description="特工的唯一标识 ID")
    status: Literal["started", "completed", "failed", "killed"] = Field(..., description="任务执行状态")
    summary: str = Field(..., description="简短的状态摘要")
    result: Optional[str] = Field(default=None, description="Worker 的最终文本回复（仅在完成时存在）")
    usage: Optional[WorkerUsageSpec] = Field(default=None, description="遥测消耗统计")