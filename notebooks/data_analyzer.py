import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

DATA_NAME = "digikala-comments.csv"
DATA_PATH = Path(os.path.join("..","data","raw" , DATA_NAME))

def find_data(MORE_detail=False):
    print("The current file location : ",DATA_PATH.cwd())
    print("File exists:", DATA_PATH.exists())

    if DATA_PATH.exists():
        print("File size:", round(os.path.getsize(DATA_PATH) / 1024**3, 2), "GB")


    if MORE_detail:
        print("\n============== DATA STATUS ==============\n")
        status_checker(DATA_PATH)



def df_maker():
    return pd.read_csv(DATA_PATH)


def status_checker(df_sample , SHOW_HEAD=True):
    
    
    raw , column = df_sample.shape
    features = df_sample.columns.tolist()
    unique_products = df_sample["product_id"].dropna().unique()
    unique_sellers = df_sample["seller_code"].dropna().unique()
    review_per_products = int(df_sample["product_id"].dropna().value_counts().mean())



    print(f"number of sampels : {raw}")
    print(f"number of features : {column}")
    print(f"number of products : {len(unique_products)}")
    print(f"number of sellers : {len(unique_sellers)}")
    print(f"number of mean review : {review_per_products}")
    print(f"features : {features}")
    print(f"\nData types:\n{df_sample.dtypes}")

    if SHOW_HEAD:
        print(f"some records:\n{display(df_sample.head())} ")






def missing_summary(df):
    
    missed_data = df.isna().sum()

    missing_percentage = (missed_data / len(df)) * 100

    missing_stats = pd.DataFrame({
        "missing_count": missed_data,
        "missing_percentage": missing_percentage
    })

    return missing_stats




def analyze_buyer_missing(df):
   
    result = (
        df.groupby("is_buyer", dropna=False)
        .agg(
            total_count=("recommendation_status", "size"),
            missing_count=("recommendation_status", "count")
        )
    )

    result["missing_count"] = (
        result["total_count"] - result["missing_count"]
    )

    result["missing_percentage"] = (
        result["missing_count"] / result["total_count"] * 100
    )

    return result



def analyze_rate_missing(df):
    
    result = (
        df.groupby(
            df["recommendation_status"].isna(),
            dropna=False
        )["rate"]
        .agg(["count", "mean", "median", "std"])
    )

    result.index = [
        "Available",
        "Missing"
    ]

    return result


def analyze_product_frequency(df):
    product_counts = df["product_id"].value_counts()

    product_frequency = df["product_id"].map(product_counts)

    temp = pd.DataFrame({
        "recommendation_missing": df["recommendation_status"].isna(),
        "product_frequency": product_frequency
    })

    result = (
        temp.groupby("recommendation_missing")["product_frequency"]
        .agg(["count", "mean", "median", "std"])
    )

    result.index = ["Available", "Missing"]

    return result




def analyze_other_metadata(df):
    target = "recommendation_status"

    excluded_columns = [
        target,
        "is_buyer",
        "rate",
        "product_id"
    ]

    metadata_columns = [
        col for col in df.columns
        if col not in excluded_columns
        and df[col].nunique(dropna=True) <= 20
    ]

    results = {}

    for col in metadata_columns:

        result = (
            df.groupby(col, dropna=False)[target]
            .agg(
                total_count="size",
                available_count="count"
            )
        )

        result["missing_count"] = (
            result["total_count"] - result["available_count"]
        )

        result["missing_percentage"] = (
            result["missing_count"] /
            result["total_count"] * 100
        )

        results[col] = result

    return results




def report_missing(df):

    # A. Missing summary
    missing_stats = missing_summary(df)

    # B. Buyer relationship
    buyer_analysis = analyze_buyer_missing(df)

    # C. Rate relationship
    rate_analysis = analyze_rate_missing(df)

    # D. Product frequency relationship
    product_analysis = analyze_product_frequency(df)

    # E. Other metadata
    metadata_analysis = analyze_other_metadata(df)

    # Display results

    display(Markdown("## A. Missing Summary"))

    missing_stats = missing_stats.sort_values(
        "missing_percentage", ascending=False
    )

    display(
        missing_stats.style
        .format({
            "missing_count": "{:,.0f}",
            "missing_percentage": "{:.2f}%"
        })
        .background_gradient(subset=["missing_percentage"], cmap="Reds")
    )

    display(Markdown("## B. Missing by Buyer"))

    display(
        buyer_analysis.style.format({
            "total_count": "{:,.0f}",
            "missing_count": "{:,.0f}",
            "missing_percentage": "{:.2f}%"
        })
    )

    display(Markdown("## C. Missing by Rate"))

    display(
        rate_analysis.style.format({
            "count": "{:,.0f}",
            "mean": "{:.2f}",
            "median": "{:.2f}",
            "std": "{:.2f}"
        })
    )

    display(Markdown("## D. Missing by Product Frequency"))

    display(
        product_analysis.style.format({
            "count": "{:,.0f}",
            "mean": "{:.2f}",
            "median": "{:.2f}",
            "std": "{:.2f}"
        })
    )

    display(Markdown("## E. Missing by Other Metadata"))

    for col, result in metadata_analysis.items():

        display(Markdown(f"### {col}"))

        display(
            result.style.format({
                "total_count": "{:,.0f}",
                "available_count": "{:,.0f}",
                "missing_count": "{:,.0f}",
                "missing_percentage": "{:.2f}%"
            })
        )



def analyze_rate_recommendation(df):

    # Prepare data
    valid_df = df[df["rate"].between(0, 5)].copy()

    # 1. Recommendation distribution
    recommendation_counts = (
        df["recommendation_status"]
        .value_counts(dropna=False)
    )

    # 2. Recommendation percentage by rate
    rate_percentage = pd.crosstab(
        valid_df["rate"],
        valid_df["recommendation_status"],
        normalize="index"
    ) * 100

    # 3. Recommendation counts by rate
    rate_counts = pd.crosstab(
        valid_df["rate"],
        valid_df["recommendation_status"]
    )

    # Plot
    _ , axes = plt.subplots(3, 1, figsize=(16, 20))

    # Chart 1
    recommendation_counts.plot(
        kind="bar",
        ax=axes[0]
    )

    axes[0].set_title("Recommendation Status Distribution")
    axes[0].set_xlabel("Recommendation Status")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    # Chart 2
    rate_percentage.plot(
        kind="bar",
        stacked=True,
        ax=axes[1],
        width=0.85
    )

    axes[1].set_title("Recommendation Percentage by Rate")
    axes[1].set_xlabel("Rate")
    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=90)

    # Chart 3
    rate_counts.plot(
        kind="bar",
        stacked=True,
        ax=axes[2],
        width=0.85
    )

    axes[2].set_title("Recommendation Count by Rate")
    axes[2].set_xlabel("Rate")
    axes[2].set_ylabel("Count")
    axes[2].tick_params(axis="x", rotation=90)

    plt.tight_layout()
    plt.show()

    return {
        "recommendation_counts": recommendation_counts,
        "rate_percentage": rate_percentage,
        "rate_counts": rate_counts
    }





def analyze_body_length(df):

    # Prepare data
    data = df.copy()

    data["body"] = data["body"].astype("string")

    data = data.dropna(subset=["body"]).copy()

    # A. Character and word length
    data["char_length"] = data["body"].str.len()
    data["word_length"] = data["body"].str.split().str.len()

    # B. Descriptive statistics
    length_stats = data[
        ["char_length", "word_length"]
    ].describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).T

    length_stats = length_stats[
        ["count", "mean", "std", "min",
         "25%", "50%", "75%", "90%", "95%", "99%", "max"]
    ]

    # C. Very short and very long reviews
    short_threshold = 5
    long_threshold = data["word_length"].quantile(0.95)

    short_reviews = data["word_length"] <= short_threshold
    long_reviews = data["word_length"] >= long_threshold

    review_length_summary = pd.DataFrame({
        "count": [
            len(data),
            short_reviews.sum(),
            long_reviews.sum()
        ],
        "percentage": [
            100,
            short_reviews.mean() * 100,
            long_reviews.mean() * 100
        ]
    }, index=[
        "All Reviews",
        f"Very Short (<= {short_threshold} words)",
        f"Very Long (>= {long_threshold:.0f} words)"
    ])

    # D. Compare length across classes
    class_col = (
        "rating_band"
        if "rating_band" in data.columns
        else "recommendation_status"
    )

    class_summary = (
        data.groupby(class_col, dropna=False)
        .agg(
            review_count=("body", "count"),
            mean_words=("word_length", "mean"),
            median_words=("word_length", "median"),
            mean_chars=("char_length", "mean"),
            median_chars=("char_length", "median")
        )
    )

    # E. Display results
  

    display(Markdown("## A. Review Length Statistics"))

    display(
        length_stats.style.format("{:,.2f}")
    )

    display(Markdown("## B. Short and Long Reviews"))

    display(
        review_length_summary.style.format({
            "count": "{:,.0f}",
            "percentage": "{:.2f}%"
        })
    )

    display(Markdown(f"## C. Review Length by {class_col}"))

    display(
        class_summary.style.format({
            "review_count": "{:,.0f}",
            "mean_words": "{:.2f}",
            "median_words": "{:.2f}",
            "mean_chars": "{:.2f}",
            "median_chars": "{:.2f}"
        })
    )

    # F. Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Character length distribution
    axes[0, 0].hist(
        data["char_length"],
        bins=50
    )
    axes[0, 0].set_title("Character Length Distribution")
    axes[0, 0].set_xlabel("Characters")
    axes[0, 0].set_ylabel("Number of Reviews")

    # Word length distribution
    axes[0, 1].hist(
        data["word_length"],
        bins=50
    )
    axes[0, 1].set_title("Word Length Distribution")
    axes[0, 1].set_xlabel("Words")
    axes[0, 1].set_ylabel("Number of Reviews")

    # Compare word length across classes
    data.boxplot(
        column="word_length",
        by=class_col,
        ax=axes[1, 0],
        showfliers=False
    )
    axes[1, 0].set_title("Word Length by Class")
    axes[1, 0].set_xlabel(class_col)
    axes[1, 0].set_ylabel("Number of Words")

    # Compare character length across classes
    data.boxplot(
        column="char_length",
        by=class_col,
        ax=axes[1, 1],
        showfliers=False
    )
    axes[1, 1].set_title("Character Length by Class")
    axes[1, 1].set_xlabel(class_col)
    axes[1, 1].set_ylabel("Number of Characters")

    fig.suptitle("Review Length Analysis")
    plt.tight_layout()
    plt.show()




def analyze_duplicates_and_products(df):

    total_rows = len(df)

    # A. Duplicate IDs
    duplicate_ids = df["id"].duplicated(keep="first")

    id_duplicate_count = duplicate_ids.sum()
    id_duplicate_groups = df.loc[
        df["id"].notna()
    ].groupby("id").size()

    id_duplicate_groups = (id_duplicate_groups > 1).sum()

    id_summary = pd.DataFrame({
        "count": [
            id_duplicate_count,
            id_duplicate_groups
        ],
        "percentage": [
            id_duplicate_count / total_rows * 100,
            id_duplicate_groups / df["id"].nunique() * 100
            if df["id"].nunique() else 0
        ]
    }, index=[
        "Duplicate ID Rows",
        "Duplicate ID Groups"
    ])


    # B. Duplicate Review Bodies
    body = (
        df["body"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    valid_body = body.notna() & body.ne("")

    body_counts = body[valid_body].value_counts()

    duplicate_body_rows = (
        body_counts[body_counts > 1] - 1
    ).sum()

    duplicate_body_groups = (
        body_counts > 1
    ).sum()

    valid_body_count = valid_body.sum()

    body_summary = pd.DataFrame({
        "count": [
            duplicate_body_rows,
            duplicate_body_groups
        ],
        "percentage": [
            duplicate_body_rows / valid_body_count * 100
            if valid_body_count else 0,
            duplicate_body_groups / body_counts.size * 100
            if body_counts.size else 0
        ]
    }, index=[
        "Duplicate Body Rows",
        "Duplicate Body Groups"
    ])


    # C. Identical Texts with Conflicting Labels

    label_col = (
        "rating_band"
        if "rating_band" in df.columns
        else "recommendation_status"
    )

    label_data = pd.DataFrame({
        "body": body,
        "label": df[label_col]
    })

    label_data = label_data[
        valid_body & label_data["label"].notna()
    ]

    text_label_groups = (
        label_data.groupby("body")["label"]
        .agg(
            label_count="nunique",
            review_count="size",
            labels=lambda x: list(x.unique())
        )
    )

    conflicting_labels = text_label_groups[
        text_label_groups["label_count"] > 1
    ].sort_values("review_count", ascending=False)

    conflict_summary = pd.DataFrame({
        "count": [
            len(conflicting_labels),
            conflicting_labels["review_count"].sum()
        ],
        "percentage": [
            len(conflicting_labels) / len(text_label_groups) * 100
            if len(text_label_groups) else 0,
            conflicting_labels["review_count"].sum()
            / len(label_data) * 100
            if len(label_data) else 0
        ]
    }, index=[
        "Conflicting Text Groups",
        "Reviews in Conflicting Groups"
    ])


    # D. Product Concentration

    product_counts = (
        df["product_id"]
        .dropna()
        .value_counts()
    )

    product_summary = product_counts.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).to_frame("reviews_per_product")

    top_products = product_counts.head(10).to_frame(
        "review_count"
    )

    top_products["percentage"] = (
        top_products["review_count"] / total_rows * 100
    )

    concentration = pd.DataFrame({
        "product_group": [
            "Top 1% Products",
            "Top 5% Products",
            "Top 10% Products"
        ],
        "review_percentage": [
            product_counts.head(
                max(1, int(len(product_counts) * 0.01))
            ).sum() / product_counts.sum() * 100,

            product_counts.head(
                max(1, int(len(product_counts) * 0.05))
            ).sum() / product_counts.sum() * 100,

            product_counts.head(
                max(1, int(len(product_counts) * 0.10))
            ).sum() / product_counts.sum() * 100
        ]
    })


    # Display results

    display(Markdown("## A. Duplicate IDs"))

    display(id_summary.style.format({
        "count": "{:,.0f}",
        "percentage": "{:.2f}%"
    }))

    display(Markdown("## B. Duplicate Review Bodies"))

    display(body_summary.style.format({
        "count": "{:,.0f}",
        "percentage": "{:.2f}%"
    }))

    display(Markdown(
        f"## C. Identical Texts with Conflicting Labels ({label_col})"
    ))

    display(conflict_summary.style.format({
        "count": "{:,.0f}",
        "percentage": "{:.2f}%"
    }))

    display(Markdown("### Most Frequent Conflicting Texts"))

    display(conflicting_labels.head(10))

    display(Markdown("## D. Product Review Distribution"))

    display(product_summary.style.format("{:,.2f}"))

    display(Markdown("### Top 10 Products"))

    display(top_products.style.format({
        "review_count": "{:,.0f}",
        "percentage": "{:.2f}%"
    }))

    display(Markdown("### Product Concentration"))

    display(concentration.style.format({
        "review_percentage": "{:.2f}%"
    }))


    # Visualizations

    _ , axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Duplicate IDs and Bodies

    duplicate_chart = pd.Series({
        "Duplicate IDs": id_duplicate_count,
        "Duplicate Bodies": duplicate_body_rows
    })

    duplicate_chart.plot(
        kind="bar",
        ax=axes[0, 0]
    )

    axes[0, 0].set_title("Duplicate Records")
    axes[0, 0].set_ylabel("Number of Records")
    axes[0, 0].tick_params(axis="x", rotation=0)

    # Chart 2: Conflicting Labels

    axes[0, 1].bar(
        ["Conflicting Text Groups"],
        [len(conflicting_labels)]
    )

    axes[0, 1].set_title("Identical Texts with Conflicting Labels")
    axes[0, 1].set_ylabel("Number of Groups")

    # Chart 3: Top 20 Products

    product_counts.head(20).sort_values().plot(
        kind="barh",
        ax=axes[1, 0]
    )

    axes[1, 0].set_title("Top 20 Products by Review Count")
    axes[1, 0].set_xlabel("Number of Reviews")
    axes[1, 0].set_ylabel("Product ID")

    # Chart 4: Reviews per Product

    axes[1, 1].hist(
        product_counts,
        bins=50
    )

    axes[1, 1].set_title("Distribution of Reviews per Product")
    axes[1, 1].set_xlabel("Number of Reviews")
    axes[1, 1].set_ylabel("Number of Products")

    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.show()


    # return {
    #     "id_summary": id_summary,
    #     "body_summary": body_summary,
    #     "conflict_summary": conflict_summary,
    #     "conflicting_labels": conflicting_labels,
    #     "product_summary": product_summary,
    #     "top_products": top_products,
    #     "product_concentration": concentration
    # }



def analyze_exact_duplicates(df):

    total_rows = len(df)

    duplicate_mask = df.duplicated(keep="first")

    duplicate_count = duplicate_mask.sum()

    unique_count = total_rows - duplicate_count

    duplicate_percentage = (
        duplicate_count / total_rows * 100
        if total_rows else 0
    )

    summary = pd.DataFrame({
        "count": [
            total_rows,
            unique_count,
            duplicate_count
        ],
        "percentage": [
            100 if total_rows else 0,
            unique_count / total_rows * 100 if total_rows else 0,
            duplicate_percentage
        ]
    }, index=[
        "Total Records",
        "Unique Records",
        "Duplicate Records"
    ])

    display(summary.style.format({
        "count": "{:,.0f}",
        "percentage": "{:.2f}%"
    }))

    display(Markdown("### Sample Duplicate Records"))

    display(
        df[duplicate_mask].head(10)
    )



def compare_label_distribution(train, validation, test, target_col="recommendation_status"):

    datasets = {
        "Train": train,
        "Validation": validation,
        "Test": test
    }

    result = pd.concat(
        {
            name: data[target_col].value_counts(normalize=True) * 100
            for name, data in datasets.items()
        },
        axis=1
    ).fillna(0)

    return result.round(2)



def check_split_strata(df):

    strata = (
        df["recommendation_status"]
        .astype("string")
        .fillna("MISSING")
        + "__"
        + df["rating_band"]
        .astype("string")
        .fillna("MISSING")
    )

    return strata.value_counts().sort_values()
