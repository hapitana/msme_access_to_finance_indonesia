from pathlib import Path

import polars as pl


project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

inventory_path = processed_dir / "explanatory_variable_inventory.csv"

inventory = pl.read_csv(inventory_path)

search_groups = {
    "banking": [
        "account",
        "checking",
        "current account",
        "bank",
        "savings",
        "deposit",
        "transaction"
    ],
    "ownership": [
        "owner",
        "owned",
        "ownership",
        "shareholder",
        "share",
        "capital",
        "foreign",
        "domestic",
        "private",
        "equity"
    ],
    "export": [
        "export",
        "sales",
        "exported"
    ]
}

results = []

print("=" * 78)
print("WBES INDONESIA 2023: EXPANDED VARIABLE CONCEPT SEARCH")
print("=" * 78)

for group, terms in search_groups.items():
    print("\n" + "=" * 78)
    print(f"CONCEPT GROUP: {group.upper()}")
    print("=" * 78)

    group_matches = []

    for term in terms:
        matches = inventory.filter(
            pl.col("label")
            .fill_null("")
            .str.to_lowercase()
            .str.contains(term, literal=True)
        )

        if matches.height > 0:
            matches = matches.with_columns(
                pl.lit(group).alias("concept_group"),
                pl.lit(term).alias("matched_term")
            )

            group_matches.append(matches)

    if group_matches:
        combined = (
            pl.concat(group_matches, how="diagonal")
            .unique(subset=["variable", "matched_term"])
            .sort(["variable", "matched_term"])
        )

        results.append(combined)

        for row in combined.iter_rows(named=True):
            print("\n" + "-" * 78)
            print(f"Variable: {row['variable']}")
            print(f"Label: {row['label']}")
            print(f"Matched term: {row['matched_term']}")
            print(f"Valid observations: {row['valid_observations']}")
            print(f"Missing observations: {row['missing_observations']}")
            print(f"Unique values: {row['unique_values']}")
            print(f"Value labels: {row['value_labels']}")
    else:
        print("No matching variables found.")

if results:
    output = (
        pl.concat(results, how="diagonal")
        .unique(subset=["variable", "matched_term"])
        .sort(["concept_group", "variable"])
    )

    output_path = processed_dir / "expanded_variable_concept_search.csv"
    output.write_csv(output_path)

    print("\nOutput file:")
    print(output_path)

print("\nExpanded concept search completed.")