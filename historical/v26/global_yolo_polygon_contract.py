from __future__ import annotations

from typing import Any, Dict


def _is_point(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def has_usable_polygon(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    if len(value) >= 3 and all(_is_point(point) for point in value):
        return True
    return any(
        isinstance(polygon, (list, tuple))
        and len(polygon) >= 3
        and all(_is_point(point) for point in polygon)
        for polygon in value
    )


def person_has_usable_polygon(person: Any) -> bool:
    if not isinstance(person, dict):
        return False
    value = person.get("mask_polygons") or person.get("mask_polygon") or person.get("polygons")
    return has_usable_polygon(value)


def summarize_polygon_contract(
    *,
    persons_total: int,
    persons_with_polygon: int,
    polygon_output_enabled: bool,
    dropped_batches: int = 0,
) -> Dict[str, Any]:
    total = max(0, int(persons_total))
    with_polygon = max(0, min(total, int(persons_with_polygon)))
    missing = total - with_polygon
    dropped = max(0, int(dropped_batches))
    if not polygon_output_enabled:
        state = "disabled"
        ok = False
    elif dropped > 0:
        state = "invalid_postprocess_drop"
        ok = False
    elif missing > 0:
        state = "invalid_person_without_polygon"
        ok = False
    elif total == 0:
        state = "valid_empty"
        ok = True
    else:
        state = "valid_complete"
        ok = True
    return {
        "polygon_contract_ok": bool(ok),
        "polygon_contract_state": state,
        "polygon_output_enabled": bool(polygon_output_enabled),
        "published_persons_total": total,
        "published_persons_with_polygon": with_polygon,
        "published_persons_without_polygon": missing,
        "published_persons_with_polygon_ratio": 1.0 if total == 0 else with_polygon / float(total),
        "postprocess_dropped_batches": dropped,
    }
