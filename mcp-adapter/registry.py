"""Shared tool registry — imported by server.py and all tools/* modules."""

# Global tool registry: name -> {description, input_schema, handler}
TOOLS: dict[str, dict] = {}


def register_tool(name: str, description: str, input_schema: dict):
    """Decorator: register a function as an MCP tool."""
    def wrapper(fn):
        TOOLS[name] = {
            "description": description,
            "input_schema": input_schema,
            "handler": fn,
        }
        return fn
    return wrapper