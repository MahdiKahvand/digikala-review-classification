import hashlib
import html
import re
import unicodedata

import emoji
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

split_seed=2026

def normalize_persian_text(text):

    if not isinstance(text, str):
        return ""

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Normalize Unicode
    text = unicodedata.normalize("NFC", text)

    # 4. Normalize Arabic characters
    text = text.translate(str.maketrans({
        "ي": "ی",
        "ى": "ی",
        "ك": "ک"
    }))

    # 5. Remove Arabic diacritics and Tatweel
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("ـ", "")

    # 6. Remove zero-width and directional characters
    text = re.sub(
        r"[\u200B\u200E\u200F\u202A-\u202E\u2060\uFEFF]",
        "",
        text
    )

    # 7. Normalize Persian and Arabic digits
    digit_translation = str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
        "01234567890123456789"
    )
    text = text.translate(digit_translation)

    # 8. Replace URLs
    text = re.sub(
        r"(https?://\S+|www\.\S+)",
        " URL ",
        text,
        flags=re.IGNORECASE
    )

    # 9. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text



def handle_emojis(text):

    if not isinstance(text, str):
        return ""

    # Convert emojis to English textual descriptions
    text = emoji.demojize(text, language="en")

    # Remove colons
    text = text.replace(":", " ")

    # Convert underscores to spaces
    text = text.replace("_", " ")

    # Normalize extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def create_rating_band(df):

    rating = pd.to_numeric(df["rate"], errors="coerce")

    df["rating_band"] = pd.cut(
        rating,
        bins=[0, 2, 3, 5],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    return df



def create_processing_df(df):

    required_columns = [
        "id",
        "product_id",
        "body",
        "recommendation_status",
        "rating_band"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df_process = df[required_columns].copy()

    return df_process



def separate_labeled_data(df):

    labeled_df = df[
        df["recommendation_status"].notna()
    ].copy()

    unlabeled_df = df[
        df["recommendation_status"].isna()
    ].copy()

    return labeled_df, unlabeled_df




def split_by_product(
    df,
    group_col="product_id",
    target_col="recommendation_status",
    rating_col="rating_band",
    split_seed = 2026
):

    # 1. Validate required columns
    required_cols = [group_col, target_col, rating_col]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # 2. Validate product IDs
    if df[group_col].isna().any():
        raise ValueError(
            "Missing product IDs must be resolved before splitting."
        )

    if df[group_col].nunique() < 7:
        raise ValueError("At least 7 product groups are required.")

    # 3. Create combined stratification labels
    strata = (
        df[target_col].astype("string").fillna("MISSING")
        + "__"
        + df[rating_col].astype("string").fillna("MISSING")
    )

    # 4. Encode product groups
    groups = pd.factorize(df[group_col])[0]

    # 5. First split: Train + Validation / Test
    first_split = StratifiedGroupKFold(
        n_splits=7,
        shuffle=True,
        random_state=split_seed
    )

    train_val_idx, test_idx = next(
        first_split.split(
            np.zeros(len(df)),
            strata,
            groups
        )
    )

    # 6. Second split: Train / Validation
    second_split = StratifiedGroupKFold(
        n_splits=6,
        shuffle=True,
        random_state=split_seed
    )

    train_local_idx, val_local_idx = next(
        second_split.split(
            np.zeros(len(train_val_idx)),
            strata.iloc[train_val_idx],
            groups[train_val_idx]
        )
    )

    train_idx = train_val_idx[train_local_idx]
    val_idx = train_val_idx[val_local_idx]

    # 7. Create datasets
    train = df.iloc[train_idx].copy()
    validation = df.iloc[val_idx].copy()
    test = df.iloc[test_idx].copy()

    # 8. Verify product separation
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    return train, validation, test



MASK_SEED = "digikala-semi-supervised-v1"
MASK_RATE = 0.20
def stable_hash(row_id: str) -> int:

    key = f"{MASK_SEED}|{row_id}".encode("utf-8")  # noqa: UP012

    return int(
        hashlib.sha256(key).hexdigest()[:16],
        16
    )



def apply_public_masking(
    train_df,
    id_col="id",
    target_col="recommendation_status",
    mask_rate=MASK_RATE
):

    # 1. Validate required columns
    required_cols = [id_col, target_col]

    missing_cols = [
        col for col in required_cols
        if col not in train_df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if not 0 <= mask_rate <= 1:
        raise ValueError("mask_rate must be between 0 and 1.")

    # 2. Preserve original training data
    train_original = train_df.copy()

    # 3. Create public training data
    train_public = train_df.copy()

    # 4. Select labeled rows
    labeled_mask = train_df[target_col].notna()

    labeled_df = train_df.loc[labeled_mask].copy()

    # Stable hash requires unique, non-missing IDs
    if labeled_df[id_col].isna().any():
        raise ValueError("Labeled rows contain missing IDs.")

    if labeled_df[id_col].duplicated().any():
        raise ValueError(
            "Duplicate IDs exist among labeled training rows."
        )

    # 5. Create hash column
    labeled_df["_stable_hash"] = labeled_df[id_col].map(
        lambda x: stable_hash(str(x))
    )

    # 6. Initialize manifest
    manifest = pd.DataFrame({
        id_col: train_df[id_col].to_numpy(),
        "original_label": train_df[target_col].to_numpy(),
        "stable_hash": pd.NA,
        "is_masked": False
    }, index=train_df.index)

    manifest["stable_hash"] = manifest["stable_hash"].astype("object")

    # 7. Mask within each recommendation class
    mask_counts = []

    for class_name, class_df in labeled_df.groupby(
        target_col,
        sort=True
    ):

        class_count = len(class_df)

        mask_count = int(np.floor(mask_rate * class_count))

        sorted_class = class_df.sort_values(
            "_stable_hash",
            kind="mergesort"
        )

        masked_idx = sorted_class.index[:mask_count]

        # Hide labels in public training data
        train_public.loc[
            masked_idx,
            target_col
        ] = pd.NA

        # Update manifest
        manifest.loc[
            class_df.index,
            "stable_hash"
        ] = class_df["_stable_hash"]

        manifest.loc[
            masked_idx,
            "is_masked"
        ] = True

        mask_counts.append({
            "class": class_name,
            "total_count": class_count,
            "masked_count": mask_count,
            "remaining_count": class_count - mask_count,
            "mask_rate_actual": mask_count / class_count,
            "rounding": "floor"
        })

    # 8. Create counts report
    counts_df = pd.DataFrame(mask_counts)

    return (
        train_public,
        train_original,
        counts_df,
        manifest
    )