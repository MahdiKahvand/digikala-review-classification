# Digikala Review Classification

Persian customer review classification (recommendation status and rating band) using TF-IDF, a fine-tuned multi-task BERT, and semi-supervised self-training.

## Overview

The goal is to predict two labels from the text of a Persian customer review:

| Task | Classes |
|---|---|
| `recommendation_status` | `recommended`, `not_recommended`, `no_idea` |
| `rating_band` | `Low` (rating ≤ 2), `Medium` (2 < rating ≤ 3), `High` (3 < rating ≤ 5) |

Three approaches are compared:

1. **TF-IDF + Logistic Regression** (baseline)
2. **Supervised multi-task BERT** (ParsBERT, two classification heads)
3. **Semi-supervised BERT** (self-training with pseudo-labels, using only 30% of the labels)

## Data

- **Source:** Digikala customer reviews (6,156,289 raw reviews).
- **After cleaning:** 6,153,060 reviews (5,258,990 labeled, 894,070 unlabeled).
- **Split:** by product (`product_id`) using `StratifiedGroupKFold`, so no product appears in more than one split.

| Split | Rows |
|---|---|
| Train | 3,756,421 |
| Validation | 751,284 |+
| Test | 751,285 |

- **Text cleaning:** HTML/URL removal, Unicode and Arabic-to-Persian character normalization, digit normalization, whitespace cleanup (`Cleaning_Splitting.py`).
- **Semi-supervised setup:** 20% of the `recommendation_status` labels are masked per class in the public training set (`apply_public_masking`), and the BERT self-training notebook keeps labels for 30% of the labeled training rows.

> **Note:** the data contains real user reviews. Check the original data terms and privacy requirements before redistributing it.

## Methods

| Method | Details |
|---|---|
| TF-IDF + Logistic Regression | 20,000 features, 1–2 grams, `class_weight="balanced"`, `MultiOutputClassifier` |
| Supervised BERT | `HooshvareLab/bert-fa-base-uncased`, shared encoder with two heads, max length 96, 1 epoch |
| Semi-supervised BERT | Supervised stage on 30% labels, then 2 rounds of self-training with confidence threshold 0.90 |

## Results

Macro-averaged precision/recall/F1. TF-IDF was evaluated on validation only.

**Validation**

| Model | Rec. Acc | Rec. F1 | Rating Acc | Rating F1 | Mean F1 |
|---|---|---|---|---|---|
| TF-IDF + LR | 0.7934 | 0.6658 | 0.6352 | 0.5568 | 0.6113 |
| Supervised BERT (100% labels) | 0.8790 | 0.7139 | 0.7459 | 0.5930 | 0.6534 |
| Semi-supervised BERT (30% labels) | 0.8756 | 0.7068 | 0.7429 | 0.5908 | 0.6488 |

**Test**

| Model | Rec. Acc | Rec. F1 | Rating Acc | Rating F1 |
|---|---|---|---|---|
| Supervised BERT | 0.8787 | 0.7138 | 0.7457 | 0.5924 |
| Semi-supervised BERT | 0.8752 | 0.7066 | 0.7427 | 0.5904 |

**Key findings**

- BERT clearly outperforms TF-IDF (about +8.5 points accuracy on recommendation and +11 on rating).
- About 79% of training reviews are `recommended`, so recommendation accuracy should be read together with F1.
- Semi-supervised BERT with only 30% of the labels comes within about 0.3–0.4 points accuracy of the fully supervised model.
- Self-training itself added a small gain (combined F1 0.6434 → 0.6488), since only about 14k pseudo-labels were added per round.

## Project Structure

```
digikala-review-classification/
│   README.md
│
├───reports
│   └───analyzing data
│           reprot.docx
│
├───notebooks
│       descriptive_analysis.ipynb
│       data_analyzer.py
│       Cleaning_Splitting.py
│       superwised.ipynb
│       making_clean_data.ipynb
│       semisupervised.ipynb
│
├───data
│       README.txt
│
└───artifacts
        split_manifest.json

```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the `torch` build that matches your CUDA version from [pytorch.org](https://pytorch.org).

## Usage

Run the notebooks in this order:

1. `01_descriptive_analysis.ipynb`
2. `main.ipynb` (cleaning, splitting, TF-IDF baseline)
3. `project3superwised.ipynb` (supervised BERT)
4. `project3_colab_semisupervised.ipynb` (semi-supervised BERT)

The BERT notebooks were run on Google Colab (NVIDIA L4 GPU) and expect the data and checkpoints under `MyDrive/project33/`. Adjust the paths at the top of each notebook if you run them elsewhere.

## Data and Model Downloads

Large files are not stored in this repository.

- **Dataset (train / validation / test CSV files):** `<add Hugging Face link>`
- **Trained checkpoints (`.pth`):** `<add Hugging Face link>`
- **Pretrained model:** [`HooshvareLab/bert-fa-base-uncased`](https://huggingface.co/HooshvareLab/bert-fa-base-uncased) (downloaded automatically by `transformers`)

## Limitations

- BERT models were trained for only 1 epoch with a single random seed, so small differences (below ~0.5 points) may be noise.
- Hyperparameters differ between models (for example batch size 20 vs 10), so comparisons are not perfectly controlled.
- TF-IDF results are reported on validation only.
- The `rating_band` task remains difficult (F1 ≈ 0.59), mainly due to the ambiguous middle class.

## Acknowledgements

- Pretrained model: [ParsBERT / HooshvareLab](https://huggingface.co/HooshvareLab/bert-fa-base-uncased)
- Data: Digikala customer reviews

## License

`<choose a license, e.g. MIT>`