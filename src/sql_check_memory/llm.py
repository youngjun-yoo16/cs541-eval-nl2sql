"""Small HTTP client for the OpenAI Responses API."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class OpenAIClient:
    """Send prompts to OpenAI and extract the model's text responses."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 180,
    ) -> None:
        """Set the endpoint, model, and timeout using options or environment values."""
        environment_base_url = os.getenv("OPENAI_BASE_URL")
        self.base_url = base_url or environment_base_url or "https://api.openai.com/v1"
        self.base_url = self.base_url.rstrip("/")

        environment_model = os.getenv("OPENAI_MODEL")
        self.model = model or environment_model or "gpt-6.2-sol"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.timeout = timeout

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        """Send one prompt to the Responses API and return its generated text."""
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add your API key to the environment, then retry."
            )

        request_body = {
            "model": self.model,
            "instructions": system,
            "input": user,
            "store": False,
        }
        if json_mode:
            request_body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "sql_check_proposal",
                    "strict": True,
                    "schema": _proposal_schema(),
                }
            }

        request = _make_api_request(
            self.base_url,
            self.api_key,
            request_body,
        )
        response_data = _send_api_request(request, self.timeout, self.base_url)
        return _extract_response_text(response_data)

    def propose_check(self, example: dict[str, Any]) -> dict[str, Any]:
        """Ask the model to describe a reusable check for one SQL mistake pair."""
        instructions = """
Analyze the incorrect and corrected SQL pair. Propose one reusable check.
Return the fields required by the JSON schema. Choose only an allowed rule ID.
Never return executable code or arbitrary SQL. Use rule_id=unknown if none of
the available checks explains the mistake. Generalize only what the pair supports.
""".strip()
        example_text = json.dumps(example, ensure_ascii=False)
        response_text = self.chat(instructions, example_text, json_mode=True)
        return _extract_json(response_text)

    def generate_sql(self, question: str, schema: str) -> str:
        """Ask the model to answer a natural-language question with one SQL query."""
        instructions = """
Generate one DuckDB-compatible, read-only SELECT query.
Return SQL only. Do not use DDL, DML, PRAGMA, COPY, ATTACH, or multiple statements.
Use only the tables and columns in the supplied schema.
""".strip()
        prompt = f"Schema:\n{schema}\n\nQuestion: {question}"
        return self.chat(instructions, prompt)

    def suggest_fix(
        self,
        question: str,
        sql: str,
        findings: list[dict[str, Any]],
        schema: str,
    ) -> str:
        """Suggest a corrected query using the question, schema, and check findings."""
        instructions = """
Return one corrected DuckDB SELECT statement only.
Preserve the question's requested meaning and use only the supplied schema.
If the findings are not enough to safely correct the query, return the original SQL.
""".strip()
        prompt_data = {
            "schema": schema,
            "question": question,
            "candidate_sql": sql,
            "diagnostics": findings,
        }
        prompt = json.dumps(prompt_data, ensure_ascii=False)
        return self.chat(instructions, prompt)


def _make_api_request(
    base_url: str,
    api_key: str,
    body: dict[str, Any],
) -> urllib.request.Request:
    """Build an authenticated POST request for the Responses API endpoint."""
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    return urllib.request.Request(
        f"{base_url}/responses",
        data=payload,
        headers=headers,
        method="POST",
    )


def _send_api_request(
    request: urllib.request.Request,
    timeout: int,
    base_url: str,
) -> dict[str, Any]:
    """Send the request and turn common HTTP and connection failures into clear errors."""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"OpenAI API returned HTTP {exc.code}: {error_text}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Could not reach the OpenAI API at {base_url}: {exc}"
        ) from exc

    try:
        response_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The OpenAI API returned a response that was not valid JSON.") from exc

    if not isinstance(response_data, dict):
        raise RuntimeError("The OpenAI API returned an unexpected response format.")
    return response_data


def _extract_response_text(response_data: dict[str, Any]) -> str:
    """Collect generated text from the message items in a Responses API result."""
    text_parts = []
    output_items = response_data.get("output", [])

    for output_item in output_items:
        if output_item.get("type") != "message":
            continue

        content_items = output_item.get("content", [])
        for content_item in content_items:
            if content_item.get("type") != "output_text":
                continue

            text = content_item.get("text")
            if isinstance(text, str):
                text_parts.append(text)

    if not text_parts:
        raise RuntimeError(
            f"OpenAI response did not contain output text: {response_data}"
        )

    return "\n".join(text_parts).strip()


def _proposal_schema() -> dict[str, Any]:
    """Return the JSON Schema that limits model check proposals to known fields."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "rule_id",
            "applicability_terms",
            "tables",
            "columns",
            "explanation",
            "suggested_correction",
            "confidence",
        ],
        "properties": {
            "rule_id": {
                "type": "string",
                "enum": [
                    "order_direction_intent",
                    "count_distinct_intent",
                    "count_nullable_column",
                    "cartesian_join",
                    "unknown",
                ],
            },
            "applicability_terms": {
                "type": "array",
                "items": {"type": "string"},
            },
            "tables": {"type": "array", "items": {"type": "string"}},
            "columns": {"type": "array", "items": {"type": "string"}},
            "explanation": {"type": "string"},
            "suggested_correction": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, including fenced JSON text."""
    cleaned_text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )

    try:
        result = json.loads(cleaned_text)
    except json.JSONDecodeError:
        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Model did not return a JSON object: {text[:300]}")
        result = json.loads(cleaned_text[start : end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object from model output.")
    return result
