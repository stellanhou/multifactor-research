"""Strict Pydantic validation of the JSON-schema subset used by research roles.

Existing role schemas remain the single source for prompts and validation.
This is deliberately not a general JSON Schema interpreter.
"""
import json
from typing import Annotated, Any, Union

from langchain_core.output_parsers import BaseOutputParser

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model


class OutputFormatError(ValueError):
    """An invalid model reply, eligible for the caller's bounded correction loop."""


class ReplyModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)


def validation_message(exc: ValidationError) -> str:
    return json.dumps(exc.errors(include_url=False, include_input=False, include_context=False),
                      ensure_ascii=False)


def _type(schema, name):
    if not schema:
        return Any
    supported = {"type", "properties", "required", "additionalProperties", "items", "anyOf",
                 "enum", "const", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                 "minLength", "maxLength", "minItems", "maxItems", "description", "title"}
    if set(schema) - supported:
        raise TypeError(f"unsupported internal schema keys: {set(schema) - supported}")
    if "anyOf" in schema:
        return Union[tuple(_type(s, name + str(i)) for i, s in enumerate(schema["anyOf"]))]
    kind = schema["type"]
    if kind == "object":
        properties = schema["properties"]
        # All current research role fields are required, including nullable fields.
        if set(schema["required"]) != set(properties):
            raise TypeError("research schemas must explicitly require every field")
        return create_model(name, __base__=ReplyModel, **{
            k: (_type(s, name + "_" + k), Field(description=s.get("description")))
            for k, s in properties.items()})
    annotation = list[_type(schema["items"], name + "_item")] if kind == "array" else {
        "string": str, "integer": int, "number": float, "boolean": bool, "null": type(None)}[kind]
    constraints = {target: schema[source] for source, target in {
        "minimum": "ge", "maximum": "le", "exclusiveMinimum": "gt", "exclusiveMaximum": "lt",
        "minLength": "min_length", "maxLength": "max_length", "minItems": "min_length",
        "maxItems": "max_length"}.items() if source in schema}
    if kind == "number":
        constraints["allow_inf_nan"] = False
    annotation = Annotated[annotation, Field(strict=True, **constraints)] if kind != "null" else annotation
    if "enum" in schema or "const" in schema:
        choices = schema["enum"] if "enum" in schema else [schema["const"]]
        def allowed(value):
            if value not in choices:
                raise ValueError(f"expected one of {choices}")
            return value
        annotation = Annotated[annotation, AfterValidator(allowed)]
    return annotation


def _extras(value, schema, extras, pointer):
    if "anyOf" in schema:
        if value is None:
            return
        branches = [branch for branch in schema["anyOf"] if branch.get("type") != "null"]
        if len(branches) == 1:
            schema = branches[0]
        else:
            # The controlled experiment union is discriminated by its metric.
            matches = []
            for index, branch in enumerate(branches):
                try:
                    TypeAdapter(_type(branch, f"AnyOfBranch{index}")).validate_python(value, strict=True)
                except ValidationError:
                    continue
                matches.append(branch)
            if not matches and isinstance(value, dict):
                matches = [branch for branch in branches
                           if value.get("metric") in branch.get("properties", {}).get("metric", {}).get("enum", [])]
                if not matches and {"horizon_hours", "displacement_hours"} & value.keys():
                    matches = [branch for branch in branches
                               if "horizon_hours" in branch.get("properties", {})]
            if not matches:
                return
            schema = matches[0]
    if schema.get("type") == "object" and isinstance(value, dict):
        for key, item in value.items():
            path = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in schema["properties"]:
                extras.append(path)
            else:
                _extras(item, schema["properties"][key], extras, path)
    elif schema.get("type") == "array" and isinstance(value, list):
        for i, item in enumerate(value):
            _extras(item, schema["items"], extras, f"{pointer}/{i}")


def validate_schema(value, schema, extras, pointer=""):
    adapter = TypeAdapter(_type(schema, "RoleReply"))
    _extras(value, schema, extras, pointer)
    try:
        parsed = adapter.validate_python(value, strict=True)
    except ValidationError as exc:
        raise OutputFormatError(validation_message(exc)) from exc
    return adapter.dump_python(parsed, mode="json")


class StrictReplyParser(BaseOutputParser[dict]):
    """LangChain parser retaining the project's strict JSON/Pydantic contract."""
    reply_model: type[BaseModel]

    def parse(self, text: str) -> dict:
        try:
            return self.reply_model.model_validate(json.loads(text), strict=True).model_dump(mode="json")
        except (json.JSONDecodeError, ValidationError) as exc:
            raise OutputFormatError(validation_message(exc) if isinstance(exc, ValidationError) else str(exc)) from exc


def parse_reply(content, model):
    """No partial JSON repair, Markdown stripping, or numeric/boolean coercion."""
    if not isinstance(content, str):
        raise OutputFormatError("reply must be JSON text")
    return StrictReplyParser(reply_model=model).invoke(content)
