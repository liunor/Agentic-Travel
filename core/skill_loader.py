"""
skill_loader 模块

该模块实现了基于 Markdown 文件与 YAML 配置的技能加载器，供 Planner/Agent 使用。
主要功能包括：
- 从目录 `skills`（可通过 SkillLoader(skills_dir=...) 指定）读取 `base_tools.yaml`，作为通用基础工具映射。
- 扫描并解析目录下以 `---` YAML 头部开头的 `.md` 技能文件，生成 `MarkdownSkillSpec`（基于 pydantic）。
- 提供查询、格式化输出等便捷方法（例如 `get_planner_listing`、`is_skill`、`is_base_tool`）。

Classes:
    MarkdownSkillSpec
        Pydantic 模型，包含字段：
        - name: 技能名
        - when_to_use: 给 Planner 的技能摘要
        - allowed_tools: 允许使用的原子工具列表
        - prompt_sop: Markdown 正文（作为 Prompt 指令）

    SkillLoader
        负责加载和管理技能与基础工具映射。主要方法：
        - _load_base_tools_config(): 读取 `base_tools.yaml` 并填充实例属性 `base_tools`（异常通过 logger 记录）。
        - _load_all_md_skills(): 扫描 `skills` 目录下的 `.md` 文件并调用 `_parse_md_file`。
        - _parse_md_file(filepath): 解析单个 Markdown 文件的 YAML 头部与正文，返回 `MarkdownSkillSpec` 或 `None`。
        - get_planner_listing(): 返回格式化的技能与工具列表字符串，供 Planner 使用。
        - is_skill(agent_name) / is_base_tool(agent_name): 判断名称是否为已加载的技能或基础工具。


YAML / Markdown 约定:
    - `base_tools.yaml`：应为工具名到描述的字典映射，例如 { tool_name: "描述" }。
    - Markdown 技能文件：必须以三横线 `---` 开头并包含 YAML 头部，必须包含至少 `name` 与 `when_to_use` 字段，正文部分作为 `prompt_sop`。

Side effects:
    - 如果 `skills` 目录不存在，初始化时会自动创建该目录。
    - `SkillLoader` 初始化会自动加载 `base_tools.yaml` 与目录下的 `.md` 文件；加载失败通过 `logger` 记录，不会抛出给外部调用者。
"""
import os
import yaml
from typing import Dict, List
from pydantic import BaseModel, Field
from utils.logger import get_logger

logger = get_logger("shiliu.core.skill_loader")

class MarkdownSkillSpec(BaseModel):
    name: str = Field(..., description="技能名称（如 trip_planner）")
    when_to_use: str = Field(..., description="给 Planner 看的技能摘要")
    allowed_tools: List[str] = Field(default_factory=list, description="允许使用的原子工具名称")
    prompt_sop: str = Field(..., description="markdown中的纯正文（Prompt 指令）")


class SkillLoader:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = skills_dir
        self.skills: Dict[str, MarkdownSkillSpec] = {}
        self.base_tools: Dict[str, str] = {}
        self._load_base_tools_config()
        self._load_all_md_skills()

    def _load_base_tools_config(self):
        """加载基础工具配置，提供给 Planner 使用

        读取 skills/base_tools.yaml 文件，成功或失败均不返回数据，函数通过修改实例属性来生效。

        Returns:
            None:修改实例属性生效
        """
        config_path = os.path.join(self.skills_dir, "base_tools.yaml")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    self.base_tools = yaml.safe_load(f) or {}
                logger.info("基础工具配置加载成功", count=len(self.base_tools))
            except Exception as e:
                logger.exception("加载 base_tools.yaml 失败")
        else:
            logger.warning("未找到 base_tools.yaml，基础工具列表为空")

    def _load_all_md_skills(self):
        """扫描加载所有技能。

        支持两种格式（按优先级）：
        1. 目录格式: skills/<skill_name>/SKILL.md （推荐）
        2. 单文件格式: skills/<skill_name>.md （兼容旧版）

        技能放入 self.skills 字典中，键为技能名称。

        Returns:
            None:修改实例属性生效
        """
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir, exist_ok=True)
            logger.info("技能目录不存在，已自动创建", path=self.skills_dir)
            return

        loaded = 0

        # 优先扫描子目录中的 SKILL.md（Agent Skill 目录格式）
        for entry in os.listdir(self.skills_dir):
            entry_path = os.path.join(self.skills_dir, entry)
            if os.path.isdir(entry_path):
                skill_file = os.path.join(entry_path, "SKILL.md")
                if os.path.isfile(skill_file):
                    skill = self._parse_md_file(skill_file)
                    if skill:
                        self.skills[skill.name] = skill
                        loaded += 1
                        logger.debug("成功加载目录格式技能", skill_name=skill.name, dir=entry)

        # 兼容旧版：扫描 skills/ 下的单文件 .md
        for filename in os.listdir(self.skills_dir):
            filepath = os.path.join(self.skills_dir, filename)
            if os.path.isfile(filepath) and filename.endswith(".md"):
                # 如果已经通过 SKILL.md 加载了同名技能，跳过
                skill_name_candidate = filename[:-3]
                if skill_name_candidate in self.skills:
                    continue
                skill = self._parse_md_file(filepath)
                if skill:
                    self.skills[skill.name] = skill
                    loaded += 1
                    logger.debug("成功加载单文件技能", skill_name=skill.name, file=filename)

        logger.info("技能加载完成", total=loaded)

    def _parse_md_file(self, filepath: str) -> MarkdownSkillSpec | None:
        """ 解析单个 Markdown 文件，提取技能信息

        Args:
            filepath: md 技能文件路径

        Returns:
            MarkdownSkillSpec: 解析成功返回技能对象
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    yaml_text = parts[1]
                    md_body = parts[2].strip()
                    try:
                        meta = yaml.safe_load(yaml_text)
                        # 兼容 Agent Skill 格式（description）和旧版格式（when_to_use）
                        when_to_use = meta.get("when_to_use") or meta.get("description")
                        return MarkdownSkillSpec(
                            name=meta.get("name"),
                            when_to_use=when_to_use,
                            allowed_tools=meta.get("allowed_tools", []),
                            prompt_sop=md_body
                        )
                    except Exception as e:
                        logger.exception("解析 MD 文件中的 YAML 头部失败", filepath=filepath)
                else:
                    logger.warning("MD文件格式不规范：未找到成对的分隔符(---)", filepath=filepath)
            else:
                logger.warning("忽略无效的MD文件：文件未以(---)开头", filepath=filepath)

        except Exception as e:
            logger.exception("读取 MD 文件发生系统异常", filepath=filepath)

        return None

    def get_planner_listing(self) -> str:
        """生成给 Planner 使用的技能和工具列表字符串。

        工具按优先级分三组：本地知识库（必须优先）、专用外部工具、兜底搜索。
        """
        lines = ["【可用 Markdown 高级专家技能包】（优先分配）："]
        for name, skill in self.skills.items():
            lines.append(f"- {name}: {skill.when_to_use}")

        # ── 知识库单独列出，强调优先 ──
        kb_desc = self.base_tools.pop("search_knowledge_base", None)
        web_desc = self.base_tools.pop("web_search", None)
        image_desc = self.base_tools.pop("generate_image_tool", None)

        if kb_desc:
            lines.append("\n【本地知识库 — 必须优先使用】：")
            lines.append(f"- search_knowledge_base: {kb_desc}")

        lines.append("\n【专用外部工具 — 实时数据 & 地图】：")
        for name, desc in self.base_tools.items():
            lines.append(f"- {name}: {desc}")

        if web_desc:
            lines.append("\n【兜底 — 仅在以上全部无结果时使用】：")
            lines.append(f"- web_search: {web_desc}")

        if image_desc:
            lines.append(f"- generate_image_tool: {image_desc}")

        # 恢复原字典
        if kb_desc:
            self.base_tools["search_knowledge_base"] = kb_desc
        if web_desc:
            self.base_tools["web_search"] = web_desc
        if image_desc:
            self.base_tools["generate_image_tool"] = image_desc

        return "\n".join(lines)

    def is_skill(self, agent_name: str) -> bool:
        return agent_name in self.skills

    def is_base_tool(self, agent_name: str) -> bool:
        return agent_name in self.base_tools

skill_loader = SkillLoader()