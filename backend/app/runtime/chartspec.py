"""ChartSpec — charts as verified structured output, not code.

Analysts write `./artifacts/chart_*.json` in the sandbox; each is validated
(schema + dataset-column check) exactly the way numeric claims are verified.
An invalid chart is rejected with a reason, never rendered. The frontend
mirrors this schema in Zod (frontend/src/lib/chartSpec.ts).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

MAX_DATA_ROWS = 500


class ChartSpec(BaseModel):
    kind: Literal["line", "bar", "scatter", "pie", "histogram"]
    title: str = ""
    x: str
    y: list[str] = Field(min_length=1)
    data: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_DATA_ROWS)
    x_label: str = ""
    y_label: str = ""
    source_columns: list[str] = Field(min_length=1)  # dataset columns this derives from


def extract_chart_specs(
    artifacts: list[dict], columns: list[str]
) -> tuple[list[ChartSpec], list[str]]:
    """Validate chart_*.json sandbox artifacts. Returns (valid specs, rejection reasons)."""
    specs: list[ChartSpec] = []
    rejections: list[str] = []
    for artifact in artifacts:
        name = artifact.get("name", "")
        if not name.startswith("chart_"):
            continue
        if artifact.get("kind") != "json":
            rejections.append(f"{name}: not valid JSON")
            continue
        try:
            spec = ChartSpec.model_validate(artifact.get("data"))
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first["loc"]) or "spec"
            rejections.append(f"{name}: invalid spec — {loc}: {first['msg']}")
            continue
        unknown = [c for c in spec.source_columns if c not in columns]
        if unknown:
            rejections.append(f"{name}: references nonexistent column(s): {', '.join(unknown)}")
            continue
        fields = [spec.x, *spec.y]
        missing = [f for f in fields if f not in spec.data[0]]
        if missing:
            rejections.append(f"{name}: data rows missing field(s): {', '.join(missing)}")
            continue
        specs.append(spec)
    return specs, rejections
