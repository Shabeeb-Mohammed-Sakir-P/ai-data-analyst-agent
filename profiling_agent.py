import os
import sqlite3
import tempfile

import pandas as pd

from llm_client import call_llm


# Number of rows loaded into memory at one time.
# Keeping this relatively small helps Render stay below its 512 MB limit.
CHUNK_SIZE = 5000

# Maximum number of unique normalized text values we track per column.
# This prevents extremely high-cardinality columns from consuming too much RAM.
MAX_CATEGORY_VALUES = 10000

# Maximum number of numeric values retained per column for IQR calculation.
# This gives us a memory-safe approximation of the outlier calculation
# for very large datasets.
MAX_OUTLIER_SAMPLE = 10000


def detect_dtype_issues(df: pd.DataFrame) -> list:
    """
    Finds columns that LOOK like they should be numeric, but contain
    some non-numeric values mixed in.
    """

    issues = []

    for col in df.select_dtypes(include="object").columns:
        non_null_values = df[col].dropna()

        if len(non_null_values) == 0:
            continue

        numeric_convertible = pd.to_numeric(
            non_null_values,
            errors="coerce"
        )

        percent_numeric = numeric_convertible.notna().mean()

        if 0.5 < percent_numeric < 1.0:
            bad_values = (
                non_null_values[numeric_convertible.isna()]
                .astype(str)
                .drop_duplicates()
                .head(5)
                .tolist()
            )

            issues.append({
                "column": col,
                "percent_numeric": round(percent_numeric * 100, 1),
                "example_bad_values": bad_values
            })

    return issues


def analyze_dataset(filepath: str) -> dict:
    """
    Memory-efficient CSV profiling.

    The CSV is processed in chunks instead of loading the entire
    dataset into RAM at once.

    Returns:
        dict containing:
        - number of rows
        - number of columns
        - column names
        - duplicate rows
        - missing values
        - dtype issues
        - category inconsistencies
        - numeric outliers
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"Profiling CSV: {file_size_mb:.2f} MB")

    # ---------------------------------------------------------
    # First chunk
    # ---------------------------------------------------------

    first_chunk = pd.read_csv(
        filepath,
        nrows=CHUNK_SIZE,
        low_memory=True
    )

    if first_chunk.empty:
        return {
            "num_rows": 0,
            "num_columns": len(first_chunk.columns),
            "columns": list(first_chunk.columns),
            "duplicate_rows": 0,
            "missing_values": {},
            "dtype_issues": [],
            "category_inconsistencies": {},
            "outliers": {},
        }

    columns = list(first_chunk.columns)
    num_columns = len(columns)

    # ---------------------------------------------------------
    # Counters
    # ---------------------------------------------------------

    total_rows = 0

    # Missing values
    missing_counts = {
        col: 0
        for col in columns
    }

    # Dtype issues
    dtype_numeric_counts = {
        col: 0
        for col in columns
    }

    dtype_non_null_counts = {
        col: 0
        for col in columns
    }

    dtype_bad_examples = {
        col: []
        for col in columns
    }

    # ---------------------------------------------------------
    # Category inconsistencies
    # ---------------------------------------------------------

    # Example:
    #
    # "North", "north", " NORTH "
    #
    # all become:
    #
    # "north"
    #
    # and we remember the different original forms.

    category_groups = {
        col: {}
        for col in columns
    }

    category_tracking_disabled = set()

    # ---------------------------------------------------------
    # Numeric values for outlier detection
    # ---------------------------------------------------------

    numeric_samples = {
        col: []
        for col in columns
    }

    numeric_sample_counts = {
        col: 0
        for col in columns
    }

    # ---------------------------------------------------------
    # Duplicate detection using SQLite
    # ---------------------------------------------------------
    #
    # We use a temporary SQLite database rather than keeping
    # millions of row hashes inside a Python set.
    #
    # This keeps duplicate detection much more memory efficient.
    #

    temp_db = tempfile.NamedTemporaryFile(
        suffix=".db",
        delete=False
    )

    temp_db_path = temp_db.name
    temp_db.close()

    conn = sqlite3.connect(temp_db_path)

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE row_hashes (
            row_hash TEXT PRIMARY KEY
        )
        """
    )

    conn.commit()

    duplicate_rows = 0

    # ---------------------------------------------------------
    # Process CSV in chunks
    # ---------------------------------------------------------

    try:

        for chunk in pd.read_csv(
            filepath,
            chunksize=CHUNK_SIZE,
            low_memory=True
        ):

            chunk_rows = len(chunk)
            total_rows += chunk_rows

            # -------------------------------------------------
            # Missing values
            # -------------------------------------------------

            for col in columns:
                missing_counts[col] += int(
                    chunk[col].isna().sum()
                )

            # -------------------------------------------------
            # Duplicate rows
            # -------------------------------------------------

            row_hashes = pd.util.hash_pandas_object(
                chunk,
                index=False
            )

            unique_hashes = row_hashes.drop_duplicates()

            new_hashes = [
                (str(int(value)),)
                for value in unique_hashes
            ]

            before_count = cursor.execute(
                "SELECT COUNT(*) FROM row_hashes"
            ).fetchone()[0]

            cursor.executemany(
                "INSERT OR IGNORE INTO row_hashes(row_hash) VALUES (?)",
                new_hashes
            )

            conn.commit()

            after_count = cursor.execute(
                "SELECT COUNT(*) FROM row_hashes"
            ).fetchone()[0]

            # Number of unique rows already present / duplicates
            duplicates_in_chunk = (
                chunk_rows
                - len(unique_hashes)
                + (len(unique_hashes) - (after_count - before_count))
            )

            duplicate_rows += duplicates_in_chunk

            # -------------------------------------------------
            # Object/text columns
            # -------------------------------------------------

            object_columns = chunk.select_dtypes(
                include="object"
            ).columns

            for col in object_columns:

                non_null_values = chunk[col].dropna()

                if len(non_null_values) == 0:
                    continue

                # ---------------------------------------------
                # Dtype issue detection
                # ---------------------------------------------

                converted = pd.to_numeric(
                    non_null_values,
                    errors="coerce"
                )

                dtype_non_null_counts[col] += len(
                    non_null_values
                )

                dtype_numeric_counts[col] += int(
                    converted.notna().sum()
                )

                bad_values = (
                    non_null_values[converted.isna()]
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                )

                if bad_values:

                    existing = dtype_bad_examples[col]

                    for value in bad_values:

                        if value not in existing:
                            existing.append(value)

                        if len(existing) >= 5:
                            break

                # ---------------------------------------------
                # Category inconsistency detection
                # ---------------------------------------------

                if col not in category_tracking_disabled:

                    unique_values = (
                        non_null_values
                        .astype(str)
                        .drop_duplicates()
                    )

                    groups = category_groups[col]

                    for value in unique_values:

                        normalized = value.lower().strip()

                        if normalized not in groups:
                            groups[normalized] = set()

                        groups[normalized].add(value)

                        # Stop tracking this column if it becomes
                        # too high-cardinality.
                        if len(groups) > MAX_CATEGORY_VALUES:

                            category_tracking_disabled.add(col)
                            category_groups[col] = {}
                            break

            # -------------------------------------------------
            # Numeric columns
            # -------------------------------------------------

            numeric_columns = chunk.select_dtypes(
                include="number"
            ).columns

            for col in numeric_columns:

                values = (
                    chunk[col]
                    .dropna()
                    .astype(float)
                    .tolist()
                )

                if not values:
                    continue

                current_count = numeric_sample_counts[col]

                # Keep only a limited sample for very large datasets.
                remaining = MAX_OUTLIER_SAMPLE - current_count

                if remaining > 0:

                    values_to_store = values[:remaining]

                    numeric_samples[col].extend(
                        values_to_store
                    )

                    numeric_sample_counts[col] += len(
                        values_to_store
                    )

            # Explicitly release the chunk before the next one.
            del chunk

    finally:

        conn.close()

        try:
            os.remove(temp_db_path)
        except OSError:
            pass

    # ---------------------------------------------------------
    # Build dtype issues
    # ---------------------------------------------------------

    dtype_issues = []

    for col in columns:

        non_null_count = dtype_non_null_counts[col]

        if non_null_count == 0:
            continue

        numeric_count = dtype_numeric_counts[col]

        percent_numeric = numeric_count / non_null_count

        if 0.5 < percent_numeric < 1.0:

            dtype_issues.append({
                "column": col,
                "percent_numeric": round(
                    percent_numeric * 100,
                    1
                ),
                "example_bad_values": dtype_bad_examples[col][:5]
            })

    # ---------------------------------------------------------
    # Build missing-value findings
    # ---------------------------------------------------------

    missing_values = {}

    for col in columns:

        missing_count = missing_counts[col]

        if missing_count > 0:

            if total_rows > 0:

                missing_pct = round(
                    (missing_count / total_rows) * 100,
                    1
                )

            else:
                missing_pct = 0.0

            missing_values[col] = (
                f"{missing_count} missing ({missing_pct}%)"
            )

    # ---------------------------------------------------------
    # Build category inconsistencies
    # ---------------------------------------------------------

    category_inconsistencies = {}

    for col, groups in category_groups.items():

        if col in category_tracking_disabled:
            continue

        inconsistent = {}

        for normalized, variants in groups.items():

            if len(variants) > 1:

                inconsistent[normalized] = sorted(
                    list(variants)
                )

        if inconsistent:

            category_inconsistencies[col] = inconsistent

    # ---------------------------------------------------------
    # Build outlier findings
    # ---------------------------------------------------------

    outliers = {}

    for col, values in numeric_samples.items():

        if len(values) < 4:
            continue

        series = pd.Series(values)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = int(
            (
                (series < lower_bound)
                |
                (series > upper_bound)
            ).sum()
        )

        if outlier_count > 0:

            outliers[col] = outlier_count

    # ---------------------------------------------------------
    # Final findings
    # ---------------------------------------------------------

    findings = {
        "num_rows": total_rows,
        "num_columns": num_columns,
        "columns": columns,
        "duplicate_rows": int(duplicate_rows),
        "missing_values": missing_values,
        "dtype_issues": dtype_issues,
        "category_inconsistencies": category_inconsistencies,
        "outliers": outliers,
    }

    print(
        f"Profiling complete: "
        f"{total_rows} rows × {num_columns} columns"
    )

    return findings


def summarize_findings(findings: dict) -> str:
    """
    Takes the raw profiling findings and asks an LLM to explain them
    in plain, human-readable language.
    """

    prompt = f"""You are a data analyst assistant. Below are automated profiling
results for a dataset, in JSON format. Write a short, clear summary (4-6 sentences)
explaining the data quality issues found, in plain English a non-technical person
could understand. Be specific about which columns have problems and roughly how
severe each issue is. Do not just repeat the JSON — actually explain it.

Findings:

{findings}

"""

    return call_llm(prompt)


# -------------------------------------------------------------
# Quick local test
# -------------------------------------------------------------

if __name__ == "__main__":

    results = analyze_dataset(
        "data/sample_messy_customers.csv"
    )

    import json

    print("=== Raw findings ===")

    print(
        json.dumps(
            results,
            indent=2,
            default=str
        )
    )

    print("\n=== Plain-English summary ===")

    summary = summarize_findings(results)

    print(summary)