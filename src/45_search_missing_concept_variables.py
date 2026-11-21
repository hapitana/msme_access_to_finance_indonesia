from pathlib import Path

import polars as pl


project_dir = Path(__file__).resolve().parent.parent
processed_dir = project_dir / "data" / "processed"

inventory_path = processed_dir / "explanatory_variable_inventory.csv"

inventory = pl.read_csv(inventory_path)

search_terms = [
    "account",
    "bank",
    "checking",
    "savings",
    "female owner",
    "female ownership",
    "women ownership",
    "foreign ownership",
    "foreign owned",
    "ownership",
    "export",
    "establishment",
    "year began"
]

print("=" * 78)
print("WBES INDONESIA 2023: MISSING CONCEPT VARIABLE SEARCH")
print("=" * 78)

for term in search_terms:
    matches = (
        inventory
        .filter(
            pl.col("label")
            .str.to_lowercase()
            .str.contains(term, literal=True)
        )
        .select(
            "variable",
            "label",
            "valid_observations",
            "missing_observations",
            "unique_values",
            "value_labels"
        )
        .sort("variable")
    )

    print("\n" + "-" * 78)
    print(f"Search term: {term}")
    print(matches)

output_path = processed_dir / "missing_concept_variable_search.csv"

all_matches = (
    pl.concat(
        [
            inventory
            .filter(
                pl.col("label")
                .str.to_lowercase()
                .str.contains(term, literal=True)
            )
            .with_columns(pl.lit(term).alias("search_term"))
            for term in search_terms
        ],
        how="diagonal"
    )
    .unique(subset=["variable", "search_term"])
    .sort(["search_term", "variable"])
)

all_matches.write_csv(output_path)

print("\nOutput file:")
print(output_path)

print("\nConcept variable search completed.")