from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]
output_path = project_dir / "script55_stdout.txt"
result_path = project_dir / "script55_required_sections.txt"

target_sections = {
    15: "15. COVARIANCE MATRIX NUMERICAL VALIDATION",
    21: "21. TARGETED GLM DEGREES OF FREEDOM SOURCE AUDIT",
    22: "22. ADJUSTED WALD F AND P-VALUE VERIFICATION",
}

lines = output_path.read_text(
    encoding="utf-8",
    errors="replace",
).splitlines()

section_starts = {}

for index, line in enumerate(lines):
    normalized_line = line.strip()

    for section_number, title in target_sections.items():
        if normalized_line.endswith(title):
            section_starts[section_number] = index

selected_sections = []

for section_number, title in target_sections.items():
    if section_number not in section_starts:
        print(f"Section not found: {title}")
        continue

    start = section_starts[section_number]
    end = len(lines)

    for index in range(start + 1, len(lines)):
        line = lines[index].strip()

        if any(
            line.endswith(next_title)
            for next_title in [
                "16. COEFFICIENT AND INFERENCE VALIDATION",
                "22. ADJUSTED WALD F AND P-VALUE VERIFICATION",
                "23. FINAL OUTPUT",
            ]
        ):
            end = index
            break

    selected_sections.append(
        "\n".join(lines[start:end])
    )

result_path.write_text(
    "\n\n".join(selected_sections),
    encoding="utf-8",
)

print("EXTRACTION COMPLETED")
print(f"Sections found: {len(selected_sections)}")
print(f"Output file: {result_path}")
print(f"Output size: {result_path.stat().st_size} bytes")