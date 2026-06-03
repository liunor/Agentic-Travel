import json
import re
from typing import List, Dict, Any, Union

# CJK 字符范围：中文、日文汉字、韩文汉字 & 假名/谚文
_CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ가-힯]')

def _count_cjk_chars(s: str) -> int:
    return len(_CJK_RE.findall(s))

def get_token_count_from_usage(usage: Dict[str, Any]) -> int:
    """从 API 返回的 usage_metadata 中提取精确 token 总数（优先取 total_tokens，其次累加各分项）。"""
    if "total_tokens" in usage:
        return usage["total_tokens"]

    input_t = usage.get("input_tokens", 0)
    output_t = usage.get("output_tokens", 0)
    cache_creation_t = usage.get("cache_creation_input_tokens", 0)
    cache_read_t = usage.get("cache_read_input_tokens", 0)

    return input_t + output_t + cache_creation_t + cache_read_t

def rough_token_count_estimation(content: Union[str, Dict, List, None]) -> int:
    """
    粗略估算 token 数，针对中英文混合文本分别处理：
    - CJK 字符：BPE 分词下每个字 ≈ 1.5 token（DeepSeek/Claude 均适用）
    - 非 CJK（英文/数字/标点）：≈ 4 chars/token
    - JSON 结构体：≈ 2 chars/token（括号和引号密度高）
    """
    if not content:
        return 0

    if isinstance(content, (dict, list)):
        content_str = json.dumps(content, ensure_ascii=False)
        return len(content_str) // 2
    else:
        content_str = str(content)
        cjk_count = _count_cjk_chars(content_str)
        non_cjk_count = len(content_str) - cjk_count
        return int(cjk_count * 0.67 + non_cjk_count * 0.25)

def rough_estimation_for_messages(messages: List[Any]) -> int:
    """对消息列表进行粗略 token 估算。遍历每条消息的 content、toolCalls/tool_calls、toolResult 字段并累加。"""
    total = 0
    for msg in messages:
        if hasattr(msg, "content"):
            total += rough_token_count_estimation(msg.content)
            
        if hasattr(msg, "toolCalls") and msg.toolCalls:
            total += rough_token_count_estimation(msg.toolCalls)
            
        if hasattr(msg, "toolResult") and msg.toolResult:
            total += rough_token_count_estimation(msg.toolResult)
            
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            total += rough_token_count_estimation(msg.tool_calls)
            
    return total

def token_count_with_estimation(messages: List[Any]) -> int:
    """混合计费策略：从后向前找到最后一条带 usage/usage_metadata 的消息作为精确锚点，锚点之后的消息用粗略估算补齐。若无锚点则全量估算。"""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        usage = getattr(msg, "usage", None) or getattr(msg, "usage_metadata", None)
        
        if usage and isinstance(usage, dict):
            exact_tokens = get_token_count_from_usage(usage)
            subsequent_msgs = messages[i+1:]
            estimated_tokens = rough_estimation_for_messages(subsequent_msgs)
            return exact_tokens + estimated_tokens

    return rough_estimation_for_messages(messages)
