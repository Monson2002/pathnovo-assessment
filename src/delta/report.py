import json
import os
from typing import Optional, Tuple
from src.delta.engine import ChangeType, DeltaResult


def generate_markdown_report(delta_result: DeltaResult) -> str:
    """Generate a human-readable Markdown report from a DeltaResult."""
    lines = [
        f"# Document Delta Report: {delta_result.pid_a} vs {delta_result.pid_b}",
        "",
        "## 1. Executive Summary",
        "",
        "| Change Type | Count |",
        "| --- | --- |",
        f"| **Added** | {delta_result.summary.get('added', 0)} |",
        f"| **Removed** | {delta_result.summary.get('removed', 0)} |",
        f"| **Modified** | {delta_result.summary.get('modified', 0)} |",
        f"| **Total Changes** | {delta_result.summary.get('total_changes', 0)} |",
        "",
        "---",
        "",
        "## 2. Detailed Changes",
        "",
    ]

    if not delta_result.items:
        lines.append("No changes detected between the document revisions.")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "| Item # | Change Type | Item Type | Page | Description | Old Value | New Value | Confidence |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    for idx, item in enumerate(delta_result.items, start=1):
        old_val = (item.old_value or "-").replace("\n", " ").replace("|", "\\|")
        new_val = (item.new_value or "-").replace("\n", " ").replace("|", "\\|")
        desc = item.description.replace("\n", " ").replace("|", "\\|")

        lines.append(
            f"| #{idx} | `{item.change_type.value}` | `{item.item_type}` | Page {item.page} | "
            f"{desc} | `{old_val}` | `{new_val}` | {item.confidence:.2f} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Itemized Breakdown")
    lines.append("")

    # Group by change type
    for change_type in [ChangeType.MODIFIED, ChangeType.ADDED, ChangeType.REMOVED]:
        type_items = [
            (idx + 1, item)
            for idx, item in enumerate(delta_result.items)
            if item.change_type == change_type
        ]
        if not type_items:
            continue

        lines.append(
            f"### {change_type.value.capitalize()} Elements ({len(type_items)})"
        )
        lines.append("")

        for item_num, item in type_items:
            lines.append(
                f"- **Item #{item_num}** `[Delta Report, Item #{item_num}]` (Page {item.page}, {item.item_type}):"
            )
            lines.append(f"  - **Description**: {item.description}")
            if item.old_value:
                lines.append(f"  - **Old Value**: `{item.old_value}`")
            if item.new_value:
                lines.append(f"  - **New Value**: `{item.new_value}`")
            loc = item.location
            lines.append(
                f"  - **Bounding Box**: `(x0: {loc.x0:.3f}, y0: {loc.y0:.3f}, x1: {loc.x1:.3f}, y1: {loc.y1:.3f})`"
            )
            lines.append(f"  - **Confidence**: {item.confidence:.2f}")
            lines.append("")

    return "\n".join(lines)


def generate_json_report(delta_result: DeltaResult) -> str:
    """Generate a machine-parseable JSON report from a DeltaResult."""
    return json.dumps(delta_result.model_dump(), indent=2)


def generate_delta_report(
    delta_result: DeltaResult, output_dir: Optional[str] = None
) -> Tuple[str, str]:
    """Generate Markdown and JSON delta reports, optionally saving them to output_dir."""
    md_content = generate_markdown_report(delta_result)
    json_content = generate_json_report(delta_result)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, "delta_report.md")
        json_path = os.path.join(output_dir, "delta_report.json")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_content)

    return md_content, json_content
