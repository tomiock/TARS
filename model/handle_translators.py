from warnings import warn

from scripts.lang_embeddings import language_tokenizer

import pandas as pd

from scripts.lang_embeddings import EmbeddingLookup as EmbeddingLanguage
from scripts.lang_embeddings import load_embedding_data as load_languages

from scripts.industry_embeddings import EmbeddingLookup as EmbeddingIndustry
from scripts.industry_embeddings import load_embedding_data as load_industries
from scripts.industry_embeddings import industry_tokenizer


def main():
    df_task = pd.read_csv("../data/data_enhanced.csv")
    df_translators = pd.read_csv("../data/translators.csv")
    df_translators = handle_translator_dataframe(df_translators)

    category_columns = [
        "MANUFACTURER_INDUSTRY",
        "TASK_TYPE",
        "PM",
    ]

    df_translators = add_most_frequent_category(
        df=df_task,
        target_df=df_translators,
        group_col="TRANSLATOR",
        category_cols=category_columns,
    )

    numerical_columns = [
        "FORECAST",
        "HOURLY_RATE",
        "QUALITY_EVALUATION",
    ]

    df_translators = add_mean_numerical_value(
        df=df_task,
        target_df=df_translators,
        group_col="TRANSLATOR",
        numerical_cols=numerical_columns,
    )

    df_translators = df_translators[~df_translators["MANUFACTURER_INDUSTRY"].is_null()]

    to_rename = {
        "HOURLY_RATE_x": "HOURLY_RATE",
        "HOURLY_RATE_y": "HOURLY_RATE_AVG_TASK",
    }

    df_translators.rename(columns=to_rename, inplace=True)

    # -- Codify the languages --
    data_languages = load_languages("../scripts/language_embeddings.pkl")
    lookup = EmbeddingLanguage(loaded_data=data_languages)

    source_lang_embed = df_translators["SOURCE_LANG"].apply(lookup.get_vector)
    target_lang_embed = df_translators["TARGET_LANG"].apply(lookup.get_vector)

    df_translators["SOURCE_LANG_EMBED"] = source_lang_embed
    df_translators["TARGET_LANG_EMBED"] = target_lang_embed

    # -- Codify the languages --
    data_industries = load_industries("../scripts/industry_embeddings.pkl")
    lookup = EmbeddingIndustry(loaded_data=data_industries)

    industry_translators = (
        df_translators["MANUFACTURER_INDUSTRY"]
        .apply(industry_tokenizer)
        .apply(lookup.get_vector)
    )
    df_translators["MANUFACTURER_INDUSTRY"] = industry_translators

    df_translators.to_pickle("../data/translators_enhanced.csv")


def handle_translator_dataframe(dataframe: pd.DataFrame):
    return (
        dataframe.groupby("TRANSLATOR")
        .agg(
            SOURCE_LANG=("SOURCE_LANG", lambda x: language_tokenizer(x.mode()[0])),
            TARGET_LANG=("TARGET_LANG", lambda x: language_tokenizer(x.mode()[0])),
            HOURLY_RATE=("HOURLY_RATE", "mean"),
        )
        .reset_index()
    )


def add_most_frequent_category(
    df: pd.DataFrame, target_df: pd.DataFrame, group_col: str, category_cols: list
) -> pd.DataFrame:
    merged_df = target_df.copy()

    for col in category_cols:
        if col not in df.columns:
            warn(f"Warning: Column '{col}' not found in the input DataFrame. Skipping.")
            continue

        count_series = df.groupby([group_col, col]).size().reset_index(name="count")
        idx_max_count = count_series.groupby(group_col)["count"].idxmax()

        most_frequent_categories = count_series.loc[idx_max_count]
        most_frequent_categories = most_frequent_categories[[group_col, col]]

        merged_df = pd.merge(
            merged_df,
            most_frequent_categories,
            on=group_col,
            how="left",  # Use left merge to keep all rows from target_df
        )
        print(f"Merged most frequent '{col}' into target DataFrame.")

    return merged_df


def add_mean_numerical_value(
    df: pd.DataFrame, target_df: pd.DataFrame, group_col: str, numerical_cols: list
) -> pd.DataFrame:
    merged_df = target_df.copy()

    for col in numerical_cols:
        if col not in df.columns:
            warn(f"Warning: Column '{col}' not found in the input DataFrame. Skipping.")
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            warn(
                f"Warning: Column '{col}' is not of numeric dtype. Skipping mean calculation."
            )
            continue

        mean_values = df.groupby(group_col)[col].mean().reset_index()

        new_col_name = f"{col}_mean"
        mean_values.rename(columns={col: new_col_name}, inplace=True)

        merged_df = pd.merge(
            merged_df,
            mean_values,
            on=group_col,
            how="left",
        )
        print(f"Merged mean of '{col}' into target DataFrame as '{new_col_name}'.")

    return merged_df
