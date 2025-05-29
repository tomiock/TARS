import ast
from warnings import warn

import pandas as pd

from scripts.lang_embeddings import language_tokenizer
from scripts.lang_embeddings import EmbeddingLookup as LookupLang
from scripts.lang_embeddings import load_embedding_data as load_lang

from scripts.industry_embeddings import industry_tokenizer
from scripts.industry_embeddings import EmbeddingLookup as LookupIndustry
from scripts.industry_embeddings import load_embedding_data as load_industry

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def main():
    df_tasks = pd.read_csv("../data/data_sample.csv")
    df_translators = pd.read_pickle("../data/translators_enhanced.pkl")

    df_tasks = drop_columns_tasks(df_tasks)

    positives = merge_dataframes(df_tasks, df_translators)

    positives = dataframe_rename(positives)

    data_embed_lang = load_lang("../scripts/language_embeddings.pkl")
    data_embed_industry = load_industry("../scripts/industry_embeddings.pkl")

    lang_lookup = LookupLang(data_embed_lang)
    industry_lookup = LookupIndustry(data_embed_industry)

    positives["SOURCE_LANG_task"] = (
        positives["SOURCE_LANG_task"]
        .apply(language_tokenizer)
        .apply(lang_lookup.get_vector)
    )
    positives["TARGET_LANG_task"] = (
        positives["TARGET_LANG_task"]
        .apply(language_tokenizer)
        .apply(lang_lookup.get_vector)
    )
    positives["INDUSTRY_task"] = (
        positives["INDUSTRY_task"]
        .apply(industry_tokenizer)
        .apply(industry_lookup.get_vector)
    )

    columns_to_drop = [
        "TASK_ID",
        "QUALITY_EVALUATION",
        "SOURCE_LANG_translator",
        "TARGET_LANG_translator",
        "MANUFACTURER_INDUSTRY_translator",
    ]

    positives.drop(columns=columns_to_drop, inplace=True)

    positives["PM_task"] = positives["PM_task"].astype("category")
    positives["PM_translator"] = positives["PM_translator"].astype("category")
    positives["TRANSLATOR_NAME"] = positives["TRANSLATOR_NAME"].astype("category")
    positives["TASK_TYPE_task"] = positives["TASK_TYPE_task"].astype("category")
    positives["TASK_TYPE_translator"] = positives["TASK_TYPE_translator"].astype(
        "category"
    )

    task_categorical = [
        "PM_task",
        "TASK_TYPE_task",
    ]

    translator_categorical = [
        "PM_translator",
        "TASK_TYPE_translator",
    ]

    task_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    translator_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    task_encoder.fit(positives[task_categorical].copy())
    translator_encoder.fit(positives[translator_categorical].copy())

    task_encoded_array = task_encoder.transform(positives[task_categorical].copy())
    translator_encoded_array = translator_encoder.transform(
        positives[translator_categorical].copy()
    )

    task_categorical_vector_name = "task_categorical_vector"
    positives[task_categorical_vector_name] = pd.Series(
        list(task_encoded_array), index=positives.index
    ).apply(np.array)

    translator_categorical_vector_name = "translator_categorical_vector"
    positives[translator_categorical_vector_name] = pd.Series(
        list(translator_encoded_array), index=positives.index
    ).apply(np.array)

    columns_to_drop = (
        task_categorical + translator_categorical + ["Unnamed: 0", "TRANSLATOR_NAME"]
    )
    positives.drop(columns=columns_to_drop, inplace=True, errors="ignore")

    scalar_columns = [
        "FORECAST_task",
        "FORECAST_mean",
        "HOURLY_RATE_task",
        "HOURLY_RATE_translator",
        "HOURLY_RATE_mean",
        "QUALITY_EVALUATION_mean",
    ]

    scalar = StandardScaler()

    scalar_data = positives[scalar_columns].values
    scalar.fit(scalar_data)
    scalar_data = scalar.transform(scalar_data)

    positives = convert_strings2arrays(positives)

    positives.to_pickle("../data/positives.pkl")


def convert_strings2arrays(dataframe) -> pd.DataFrame:
    vector_string_cols = [
        "SOURCE_LANG_task",
        "TARGET_LANG_task",
        "INDUSTRY_task",
        "SOURCE_LANG_EMBED_translator",
        "TARGET_LANG_EMBED_translator",
        "INDUSTRY_EMBED_translator",
        "task_categorical_vector",
        "translator_categorical_vector",
    ]

    for col in vector_string_cols:
        if col in dataframe.columns:  # Check if the column exists in the positives
            print(f"Attempting to convert column '{col}' from string to list/array...")
            # Check if the column is of object dtype, which is typical for strings or mixed types
            if dataframe[col].dtype == "object":
                try:
                    # Use a lambda function with ast.literal_eval to convert each string cell
                    # Add checks within the lambda:
                    # 1. Check if the value is a string and not NaN (using pd.notna)
                    # 2. Use ast.literal_eval to parse the string
                    # 3. Convert the result to a NumPy array with float32 dtype
                    # Handle cases where the value might already be a list/array (if conversion was partially done)
                    dataframe[col] = dataframe[col].apply(
                        lambda x: np.array(ast.literal_eval(x), dtype=np.float32)
                        if isinstance(x, str) and pd.notna(x)
                        else (
                            np.array(x, dtype=np.float32)
                            if isinstance(x, (list, np.ndarray))
                            else x
                        )  # Keep existing arrays/lists, pass others through
                    )
                    print(f"Conversion successful for column '{col}'.")
                except Exception as e:
                    # Catch potential errors during parsing (e.g., malformed string)
                    warn(
                        f"Warning: Could not convert column '{col}' from string to list/array. "
                        f"There might be problematic values. Error: {e}"
                    )
                    # You might want to inspect the problematic rows or column content here
            else:
                print(
                    f"Column '{col}' is not of object dtype ({dataframe[col].dtype}). Skipping string conversion."
                )

        else:
            warn(f"Warning: Column '{col}' not found in the positives. Skipping.")

    return dataframe


def drop_columns_tasks(dataframe):
    columns_drop_tasks = [
        "PROJECT_ID",
        "START",
        "END",
        "ASSIGNED",
        "READY",
        "WORKING",
        "DELIVERED",
        "RECEIVED",
        "CLOSE",
        "COST",
        "MANUFACTURER",
        "MANUFACTURER_SECTOR",
        "MANUFACTURER_INDUSTRY_GROUP",
        "MANUFACTURER_SUBINDUSTRY",
        "_work_ready",
        "_time_taken",
        "_time_reception",
        "_time_to_close",
    ]

    return dataframe.drop(columns=columns_drop_tasks, inplace=True)


def merge_dataframes(df_tasks, df_translators):
    return pd.merge(
        df_tasks,
        df_translators,
        on="TRANSLATORS",
        how="left",
        suffixes=("_task", "_translators"),
    )


def dataframe_rename(dataframe) -> pd.DataFrame:
    columns = {
        "TRANSLATOR": "TRANSLATOR_NAME",
        "FORECAST": "FORECAST_task",
        "MANUFACTURER_INDUSTRY_task": "INDUSTRY_task",
        "SOURCE_LANG_EMBED": "SOURCE_LANG_EMBED_translator",
        "TARGET_LANG_EMBED": "TARGET_LANG_EMBED_translator",
        "INDUSTRY_EMBED": "INDUSTRY_EMBED_translator",
    }

    return dataframe.rename(columns, inplace=True)
