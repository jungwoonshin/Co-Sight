# Copyright 2025 ZTE Corporation.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

from app.agent_dispatcher.domain.plan.action.skill.mcp.const import LOCAL_MCP
from app.agent_dispatcher.domain.plan.action.skill.mcp.engine import MCPEngine
import asyncio
from contextlib import contextmanager


@contextmanager
def async_event_loop():
    """Thread-safe event loop context manager"""
    try:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
    finally:
        # Clean up event loop
        loop.close()
        asyncio.set_event_loop(None)

def convert_skill_to_tool(skill, lang='en') -> dict:
    """Convert skill to tool format for llm.create_with_tools

    Args:
        skill: The skill dictionary returned by skill functions
        lang: Language code ('en' or 'zh')

    Returns:
        dict: Tool format for llm.create_with_tools
    """
    tools=[]
    if skill['skill_type'] not in [LOCAL_MCP]:
        parameters = skill['function'].get("parameters").copy()

        if 'properties' in parameters:
            for prop_name, prop_value in parameters['properties'].items():
                if lang in prop_value:
                    prop_value['description'] = prop_value[lang]
                    for key in ['zh', 'en']:
                        if key in prop_value:
                            del prop_value[key]

        result = {
            "type": "function",
            "function": {
                "name": skill['skill_name'],
                "description": skill[f'description_{lang}'],
                "parameters": parameters
            }
        }
        tools.append(result)
    return tools


def get_mcp_tools(skills):
    tools = []
    for skill in skills:
        if skill.skill_type in [LOCAL_MCP]:
            try:
                with async_event_loop() as loop:
                    mcp_tools = loop.run_until_complete(
                        MCPEngine.get_mcp_tools(
                            skill.skill_name,
                            skill.mcp_server_config
                        )
                    )
                print(f"mcp_tools:{mcp_tools}")
                result = {
                    "mcp_name": skill.skill_name,
                    "mcp_config": skill.mcp_server_config,
                    "mcp_tools": mcp_tools
                }
                tools.append(result)
            except Exception as e:
                logger.error(f"Failed to get MCP tools for {skill.skill_name}: {str(e)}", exc_info=True)
    return tools

def convert_mcp_tools(mcp_configs):
    """Improved MCP tool conversion"""
    tools = []

    for mcp_config in mcp_configs:
        for mcp_tool in mcp_config.get('mcp_tools', []):
            try:
                # Get input parameter schema
                input_schema = getattr(mcp_tool, 'inputSchema', {})

                # Build parameter structure
                parameters = {
                    "type": "object",
                    "properties": input_schema.get('properties', {}),
                    "required": input_schema.get('required', [])
                }

                # Clean empty fields
                parameters = {k: v for k, v in parameters.items() if v}

                # Build tool format
                tool_config = {
                    "type": "function",
                    "function": {
                        "name": getattr(mcp_tool, 'name', 'unnamed_tool'),
                        "description": getattr(mcp_tool, 'description', 'No description'),
                        "parameters": parameters
                    }
                }

                # Add necessary validation
                if not tool_config["function"]["name"]:
                    raise ValueError("MCP tool name is required")

                tools.append(tool_config)

            except Exception as e:
                logger.error(f"Failed to convert MCP tool: {str(e)}", exc_info=True)
                continue

    return tools