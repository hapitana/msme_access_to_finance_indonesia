from pathlib import Path
import svy


project_dir = Path(__file__).resolve().parent.parent

output_dir = (
    project_dir
    / "data"
    / "processed"
    / "script59_svy_variance_inspection"
)

output_dir.mkdir(parents=True, exist_ok=True)

package_dir = Path(svy.__file__).resolve().parent

search_terms = [
    "wr",
    "with_replacement",
    "without_replacement",
    "pop_size",
    "fpc",
    "singleton",
    "variance",
    "variance_estimator",
    "prob",
    "taylor",
    "linearized",
    "stratum",
    "psu",
]

max_hits_per_file = 35
context_before = 5
context_after = 10

print("=" * 75)
print("SVY VARIANCE IMPLEMENTATION INSPECTION")
print("=" * 75)

print(f"Package directory: {package_dir}")
print(f"Output directory: {output_dir}")

python_files = sorted(
    package_dir.rglob("*.py")
)

print(f"Python source files found: {len(python_files)}")

report = []

report.append("SVY VARIANCE IMPLEMENTATION INSPECTION")
report.append("=" * 75)
report.append(f"Package directory: {package_dir}")
report.append(f"Python source files: {len(python_files)}")

total_hits = 0

for file_path in python_files:
    try:
        lines = file_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except Exception as error:
        report.append(
            f"READ ERROR: {file_path}: {error}"
        )
        continue

    matching_lines = []

    for line_number, line in enumerate(lines):
        line_lower = line.lower()

        matched_terms = [
            term
            for term in search_terms
            if term in line_lower
        ]

        if matched_terms:
            matching_lines.append(
                (line_number, matched_terms)
            )

    if not matching_lines:
        continue

    selected_hits = matching_lines[:max_hits_per_file]

    file_report = []

    file_report.append("")
    file_report.append("=" * 75)
    file_report.append(
        f"FILE: {file_path.relative_to(package_dir)}"
    )
    file_report.append(
        f"Total matching lines: {len(matching_lines)}"
    )
    file_report.append(
        f"Displayed matching lines: {len(selected_hits)}"
    )
    file_report.append("=" * 75)

    displayed_ranges = []

    for line_number, matched_terms in selected_hits:
        start = max(
            0,
            line_number - context_before,
        )

        end = min(
            len(lines),
            line_number + context_after + 1,
        )

        if displayed_ranges and start <= displayed_ranges[-1][1]:
            displayed_ranges[-1] = (
                displayed_ranges[-1][0],
                max(displayed_ranges[-1][1], end),
            )
        else:
            displayed_ranges.append((start, end))

    for start, end in displayed_ranges:
        file_report.append("")
        file_report.append(
            f"Lines {start + 1}-{end}"
        )

        for index in range(start, end):
            marker = (
                ">>"
                if any(
                    index == hit_line
                    for hit_line, _ in selected_hits
                )
                else "  "
            )

            file_report.append(
                f"{marker} {index + 1:5d}: {lines[index]}"
            )

    report.extend(file_report)

    total_hits += len(matching_lines)

report.append("")
report.append("=" * 75)
report.append("INSPECTION SUMMARY")
report.append("=" * 75)
report.append(
    f"Python files inspected: {len(python_files)}"
)
report.append(
    f"Total matching lines: {total_hits}"
)
report.append(
    "Search terms: " + ", ".join(search_terms)
)

report_path = (
    output_dir
    / "svy_variance_source_inspection.txt"
)

report_path.write_text(
    "\n".join(report),
    encoding="utf-8",
)

print()
print(f"Total matching lines: {total_hits}")
print(f"Report saved: {report_path}")

print()
print("Inspection completed.")