import re
import secrets
from typing import Any, Optional

UUID_REGEX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE
)


def validate_uuid(maybe_uuid: Any) -> Optional[str]:
    """ 验证输入是否为合法的 UUID 字符串。

    Args:
        maybe_uuid: 待检查的值，任意类型。仅当其为字符串且匹配 UUID 正则时被认为合法。

    Returns:
        Optional[str]: 如果输入是合法的 UUID 字符串，返回该字符串；否则返回 None。
    """
    if not isinstance(maybe_uuid, str):
        return None

    return maybe_uuid if UUID_REGEX.match(maybe_uuid) else None


def create_agent_id(label: Optional[str] = None) -> str:
    """ 生成带前缀的 agent id，格式用于与任务 ID 保持一致。

    Args:
        label: 可选标签字符串，若提供则作为前缀的一部分并以连字符分隔。

    Returns:
        str: 生成的 agent id，格式为 "{label}-随机16字符十六进制字符串"。

    """
    suffix = secrets.token_hex(8)

    if label:
        return f"{label}-{suffix}"
    return f"agent-{suffix}"