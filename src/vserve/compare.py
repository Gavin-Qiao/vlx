"""Compare models against a workload."""


def _best_slots(limits_entry: dict) -> tuple[int, str]:
    best_slots = 0
    best_dtype = "auto"
    for dtype, slots in limits_entry.items():
        if slots is not None and slots > best_slots:
            best_slots = slots
            best_dtype = dtype
    return best_slots, best_dtype


def _find_context_level(limits: dict, context: int) -> str | None:
    levels = sorted(int(k) for k in limits.get("limits", {}).keys())
    for level in levels:
        if level >= context:
            return str(level)
    return None


def filter_models_for_workload(
    *, all_limits: list[dict], users: int, context: int,
) -> list[dict]:
    result = []
    for lim in all_limits:
        ctx_key = _find_context_level(lim, context)
        if ctx_key is None:
            continue
        entry = lim.get("limits", {}).get(ctx_key, {})
        best_slots, best_dtype = _best_slots(entry)
        if best_slots >= users:
            result.append({
                **lim,
                "_matched_context": int(ctx_key),
                "_matched_slots": best_slots,
                "_matched_dtype": best_dtype,
            })
    return result


def rank_models(fits: list[dict]) -> list[dict]:
    return sorted(fits, key=lambda m: m["_matched_slots"], reverse=True)
