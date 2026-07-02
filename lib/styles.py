"""Streamlit CSS injection."""

CUSTOM_CSS = """
<style>
.main-card {
    border: 1px solid rgba(120,120,120,0.25);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
    background: rgba(50, 80, 140, 0.06);
}
.result-dr {
    border-left: 6px solid #d9534f;
}
.result-nodr {
    border-left: 6px solid #2e8b57;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 8px;
}
.badge-high { background: rgba(46, 139, 87, 0.18); color: #2e8b57; }
.badge-medium { background: rgba(255, 165, 0, 0.18); color: #c97d00; }
.badge-low { background: rgba(217, 83, 79, 0.18); color: #d9534f; }
.small-muted { opacity: 0.8; font-size: 0.92rem; }
</style>
"""


def confidence_level(confidence):
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.65:
        return "medium"
    return "low"


def format_confidence_badge(confidence):
    level = confidence_level(confidence)
    label = level.capitalize()
    return f'<span class="badge badge-{level}">{label} confidence</span>'
