import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from llm_client import call_llm
from utils import parse_json_response


def analyze_for_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Computes objective facts about the dataset relevant to preparing it
    for model building — no LLM involved yet, just pandas/numpy.
    """
    analysis = {
        "numeric_columns": [],
        "categorical_columns": [],
        "low_variance_columns": [],
        "high_cardinality_columns": [],
        "highly_correlated_pairs": [],
    }

    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(include="object")

    analysis["numeric_columns"] = list(numeric_df.columns)
    analysis["categorical_columns"] = list(categorical_df.columns)

    # Low variance = the column barely changes, so it likely adds no predictive value
    for col in numeric_df.columns:
        if numeric_df[col].std() < 0.01:
            analysis["low_variance_columns"].append(col)

    # High cardinality = too many unique categories (e.g. names, emails, IDs) —
    # one-hot encoding these would create hundreds of useless columns
    for col in categorical_df.columns:
        unique_ratio = categorical_df[col].nunique() / len(categorical_df)
        if unique_ratio > 0.5:
            analysis["high_cardinality_columns"].append(col)

    # Multicollinearity check: numeric columns that are strongly correlated
    # with each other carry redundant information
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr().abs()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.85:
                    analysis["highly_correlated_pairs"].append({
                        "columns": [corr_matrix.columns[i], corr_matrix.columns[j]],
                        "correlation": round(corr_matrix.iloc[i, j], 3)
                    })

    return analysis


def propose_feature_engineering_actions(analysis: dict) -> list:
    """
    Asks the LLM to propose specific feature engineering actions based on
    the objective analysis above. Same human-in-the-loop structure as
    the Cleaning Agent — proposals only, never auto-applied.
    """
    prompt = f"""You are a machine learning engineer preparing a dataset for
model building. Based on the analysis below, propose specific feature
engineering actions. Respond with ONLY a JSON array (no other text, no code
fences) where each item has: "action", "column" (a SINGLE column name, never
a list — one action per column, even if several columns need the same
action), "reason", "severity" ("low", "medium", or "high").

Valid action types: "one_hot_encode" (for low-cardinality categories),
"label_encode" (for ordinal or binary categories), "scale_numeric",
"drop_low_variance", "drop_high_cardinality", "drop_correlated_feature".

High-cardinality columns like IDs, names, or emails should almost always be
dropped or excluded, not encoded. If multiple columns need the same action
(e.g. dropping several high-cardinality columns), create one SEPARATE action
item per column, not one item with multiple columns.

Analysis:
{analysis}

Respond with ONLY the JSON array, starting with [ and ending with ]."""

    raw_response = call_llm(prompt)
    return parse_json_response(raw_response)


def apply_feature_engineering_action(df: pd.DataFrame, action: dict) -> pd.DataFrame:
    """
    Applies ONE approved feature engineering action to the DataFrame.
    Only ever called after human approval, same principle as the Cleaning Agent.
    """
    action_type = action.get("action")
    column = action.get("column")

    if action_type == "one_hot_encode":
        return pd.get_dummies(df, columns=[column], prefix=column)

    elif action_type == "label_encode":
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        return df

    elif action_type == "scale_numeric":
        scaler = StandardScaler()
        df[column] = scaler.fit_transform(df[[column]])
        return df

    elif action_type in ("drop_low_variance", "drop_high_cardinality"):
        return df.drop(columns=[column])

    elif action_type == "drop_correlated_feature":
        # column may be a list of two correlated columns — drop just one of them
        col_to_drop = column[1] if isinstance(column, list) else column
        return df.drop(columns=[col_to_drop])

    else:
        print(f"Warning: unknown action type '{action_type}', skipping.")
        return df


# Quick test — only runs if you execute this file directly
if __name__ == "__main__":
    from cleaning_agent import apply_cleaning_action

    df = pd.read_csv("data/sample_messy_customers.csv")
    df = apply_cleaning_action(df, {"action": "remove_duplicates", "column": "all"})
    df = apply_cleaning_action(df, {"action": "standardize_categories", "column": "region"})
    df = apply_cleaning_action(df, {"action": "standardize_categories", "column": "is_active"})
    df = apply_cleaning_action(df, {"action": "fix_dtype", "column": "age"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "age"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "monthly_spend"})
    df = apply_cleaning_action(df, {"action": "impute_missing", "column": "region"})

    analysis = analyze_for_feature_engineering(df)
    print("=== Feature Engineering Analysis ===")
    print(analysis)

    actions = propose_feature_engineering_actions(analysis)
    print(f"\n=== Proposed {len(actions)} actions ===\n")
    for i, action in enumerate(actions, 1):
        print(f"{i}. [{action.get('severity', '?').upper()}] {action.get('action')} "
              f"on {action.get('column')}")
        print(f"   Reason: {action.get('reason')}\n")

    print("=== Testing apply: one-hot encoding 'region' ===")
    df_encoded = apply_feature_engineering_action(df, {"action": "one_hot_encode", "column": "region"})
    print(f"Columns after encoding: {list(df_encoded.columns)}")