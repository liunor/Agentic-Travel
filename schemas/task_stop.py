from typing import Optional
from pydantic import BaseModel, Field, model_validator

class TaskStopInputSpec(BaseModel):
    task_id: Optional[str] = Field(default=None, description="要停止的后台任务的 ID")
    # 100% 保留源码的向下兼容设计
    shell_id: Optional[str] = Field(default=None, description="已弃用：请使用 task_id")

    @model_validator(mode='after')
    def validate_ids(self) -> 'TaskStopInputSpec':
        if not self.task_id and not self.shell_id:
            raise ValueError("缺少必要参数：task_id")
        return self

class TaskStopOutputSpec(BaseModel):
    success: bool = Field(..., description="操作是否成功")
    message: str = Field(..., description="操作的状态描述或错误详情")
    task_id: Optional[str] = Field(default=None, description="已停止任务的 ID")
    task_type: Optional[str] = Field(default=None, description="已停止任务的类型")
    command: Optional[str] = Field(default=None, description="已停止任务的命令或描述")
    error_code: Optional[int] = Field(default=None, description="错误码（失败时提供）")