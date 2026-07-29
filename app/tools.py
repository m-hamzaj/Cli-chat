"""The two tools the model can call. Both return a string; both catch their
own errors so a bad expression or a dead URL never crashes the chat loop."""

import ast
import operator
import urllib.request
import urllib.error
import re

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression using +, -, *, /, ** (power), "
                            "% (modulo only, NOT percent), and parentheses. Express percentages as "
                            "decimals, e.g. '15% of 240' is '0.15 * 240'. "
                            "Use this for any math instead of computing it yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '12.5 * (3 + 7) / 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL over HTTP(S) and return its text content (truncated). "
                            "Use this when the user asks about a specific web page or link.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL including scheme, e.g. https://example.com",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculator(expression):
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


def fetch_url(url):
    try:
        if not re.match(r"^https?://", url):
            url = "https://" + url
        req = urllib.request.Request(url, headers={"User-Agent": "cli-chat-tool/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read(20000).decode(errors="replace")
        text = re.sub(r"<[^>]+>", " ", raw)          # strip tags, rough but dependency-free
        text = re.sub(r"\s+", " ", text).strip()
        return text[:2000] if text else "(empty response body)"
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} fetching {url}"
    except urllib.error.URLError as e:
        return f"Error: could not reach {url} ({e.reason})"
    except Exception as e:
        return f"Error fetching {url}: {e}"


def run_tool(name, arguments):
    """Dispatch by name. Never raises -- returns an error string instead."""
    try:
        if name == "calculator":
            return calculator(arguments.get("expression", ""))
        if name == "fetch_url":
            return fetch_url(arguments.get("url", ""))
        return f"Error: unknown tool '{name}'"
    except Exception as e:
        return f"Error: tool '{name}' failed unexpectedly ({e})"
