"""LLM 输出解析与参数验证."""
import inspect
import warnings
from typing import Any, Callable, Dict, List, Union, get_args, get_origin

from langchain_core.utils.function_calling import convert_to_openai_tool


def validate_tool_args(args: Dict[str, Any], signature: inspect.Signature) -> Dict[str, Any]:
    """验证并过滤 LLM 生成的参数，确保符合函数签名."""
    valid_params = signature.parameters
    validated = {}
    
    for param_name, param in valid_params.items():
        if param_name in args:
            value = args[param_name]
            # 类型转换（简单处理）
            if param.annotation is int and isinstance(value, (int, float, str)):
                validated[param_name] = int(value)
            elif param.annotation is float and isinstance(value, (int, float, str)):
                validated[param_name] = float(value)
            elif param.annotation is bool and isinstance(value, str):
                validated[param_name] = value.lower() in ("true", "1", "yes")
            else:
                validated[param_name] = value
        elif param.default is inspect.Parameter.empty:
            # 必填参数缺失
            raise ValueError(f"Missing required parameter: {param_name}")
        # 可选参数缺失：不添加，使用函数默认值
    
    return validated


def build_openai_tool_schema(fn: Callable[..., Any], tool_name: str) -> Dict[str, Any]:
    tool_schema = convert_to_openai_tool(fn)

    if not isinstance(tool_schema, dict):
        raise ValueError(f"Invalid tool schema generated for {tool_name}")
    if tool_schema.get("type") != "function":
        raise ValueError(f"Unsupported tool schema type for {tool_name}: {tool_schema.get('type')}")

    function_def = tool_schema.get("function")
    if not isinstance(function_def, dict):
        raise ValueError(f"Missing function definition in schema for {tool_name}")

    function_def["name"] = tool_name
    parameters = function_def.get("parameters")
    if isinstance(parameters, dict):
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.setdefault("additionalProperties", False)
    else:
        function_def["parameters"] = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    return tool_schema


def _annotation_to_json_schema(annotation: Any) -> Dict[str, Any]:
    if annotation in (inspect.Parameter.empty, Any):
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (list, List):
        item_schema = _annotation_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_schema}

    if origin in (dict, Dict):
        value_schema = _annotation_to_json_schema(args[1]) if len(args) > 1 else {}
        schema: Dict[str, Any] = {"type": "object"}
        if value_schema:
            schema["additionalProperties"] = value_schema
        return schema

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_json_schema(non_none[0])
        return {"anyOf": [_annotation_to_json_schema(a) for a in non_none]} if non_none else {}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    return {}


def build_parameters_json_schema(signature: inspect.Signature) -> Dict[str, Any]:
    warnings.warn(
        "build_parameters_json_schema is deprecated; use build_openai_tool_schema with convert_to_openai_tool instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        schema = _annotation_to_json_schema(param.annotation)
        if not schema:
            schema = {"type": "string"}
        properties[name] = schema

        if param.default is inspect.Parameter.empty:
            required.append(name)

    result: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def _format_annotation(annotation: Any) -> str:
    if annotation == inspect.Parameter.empty:
        return "Any"
    return annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)


def format_tool_signature(signature: inspect.Signature) -> str:
    """将函数签名格式化为 LLM 可读的字符串."""
    parts = []
    for name, param in signature.parameters.items():
        annotation = param.annotation
        default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
        parts.append(f"{name}: {_format_annotation(annotation)}{default}")
    
    return_annotation = signature.return_annotation
    return_str = f" -> {_format_annotation(return_annotation)}" if return_annotation != inspect.Parameter.empty else ""
    
    return f"({', '.join(parts)}){return_str}"
