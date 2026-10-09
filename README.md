# Explainable Arabic Sentiment Analysis

**Comparing Classical NLP with Arabic Transformer-Based Models**

This project investigates Arabic sentiment classification by comparing a classical Natural Language Processing (NLP) approach with a pretrained Arabic Transformer model. The experiment evaluates **TF-IDF + Logistic Regression** against **MARBERT** using a cleaned Arabic social-media dataset.

The study combines Arabic-specific text preprocessing, data-quality auditing, leakage-aware evaluation, model comparison, error analysis, and statistical uncertainty estimation.

## Project Overview

Arabic sentiment analysis presents several challenges, including dialectal variation, spelling differences, informal expressions, emojis, and context-dependent sentiment.

This project addresses these challenges by comparing two approaches:

- **TF-IDF + Logistic Regression:** A classical, interpretable NLP baseline based on word unigrams and bigrams.
- **MARBERT:** A Transformer-based Arabic language model fine-tuned for binary sentiment classification.

Both approaches use the same reconstructed training, validation, and test partitions. Their performance is evaluated using multiple classification metrics, confusion matrices, ROC and precision-recall curves, and additional error analysis.

## Research Objective

The main research question is:

> How does contextual Arabic language representation affect sentiment classification compared with a traditional TF-IDF-based approach?

The experiment also investigates the importance of data quality, the effect of Arabic text normalization, differences in model errors, and uncertainty in the observed performance difference.

## Dataset

**Arabic Sentiment Twitter Corpus**

- **Dataset source:** [Hugging Face — Arabic Sentiment Twitter Corpus](https://huggingface.co/datasets/asas-ai/Arabic_Sentiment_Twitter_Corpus)
- **Original source:** [Kaggle — Arabic Sentiment Twitter Corpus](https://www.kaggle.com/datasets/mksaad/arabic-sentiment-twitter-corpus)
- **Language:** Arabic
- **Task:** Binary sentiment classification
- **Text column:** `tweet`
- **Label column:** `label`
- **Sentiment classes:** Negative (`neg`) and Positive (`pos`)

The dataset contains Arabic social-media text with positive and negative sentiment labels.

### Data Quality and Cleaning

The initial dataset contained 56,795 observations across its original training and test splits. The preliminary audit identified substantial duplication and overlap:

| Data Quality Check | Result |
|---|---:|
| Original training observations | 45,275 |
| Original test observations | 11,520 |
| Duplicate text occurrences in training | 15,826 |
| Duplicate text occurrences in test | 2,703 |
| Unique texts shared across original splits | 2,512 |
| Unique raw texts with conflicting labels | 122 |

All occurrences of conflicting raw texts were removed, followed by global deduplication and Arabic text normalization. After the final cleaning and deduplication steps, the analytical dataset contained **35,601 unique Arabic texts**.

Because the original splits contained cross-split overlap and inconsistent labels, the experiment reconstructs its train-validation-test partitions. Consequently, the final test set is a newly created hold-out set rather than the dataset provider's original test split.

## Methodology

The project follows this workflow:

```text
Arabic Sentiment Twitter Corpus
             |
             v
     Dataset Quality Audit
             |
             v
 Arabic Text Cleaning and Normalization
             |
             v
  Conflict Removal and Deduplication
             |
             v
    Exploratory Data Analysis
             |
             v
 Stratified Train / Validation / Test Split
             |
             v
       Text Leakage Verification
             |
       +-----+-----+
       |           |
       v           v
     TF-IDF      MARBERT
       |           |
       v           v
 Logistic Reg.  Fine-Tuning
       |           |
       +-----+-----+
             |
             v
      Model Comparison
             |
             v
   Test Evaluation and Error Analysis
             |
             v
   Bootstrap Uncertainty Analysis
             |
             v
     Results and Conclusion
```

## Data Preprocessing

The preprocessing stage includes:

- Inspecting missing values and duplicate text.
- Detecting overlap and inconsistent labels across the original splits.
- Removing texts associated with conflicting sentiment labels.
- Normalizing common Arabic Alef variations and the character `ى`.
- Removing URLs, user mentions, Tatweel, and invisible formatting characters where detected.
- Preserving hashtag content, emojis, punctuation, and sentiment-bearing words.
- Removing repeated normalized texts before creating the final partitions.
- Encoding negative sentiment as `0` and positive sentiment as `1`.

The final dataset contains:

| Sentiment | Observations | Percentage |
|---|---:|---:|
| Negative | 18,277 | 51.34% |
| Positive | 17,324 | 48.66% |
| **Total** | **35,601** | **100.00%** |

The class distribution is relatively balanced, so corrective class weighting was not enabled for the baseline model.

## Train, Validation, and Test Split

A stratified 70/15/15 split was applied to the final cleaned dataset.

| Partition | Observations |
|---|---:|
| Training | 24,920 |
| Validation | 5,340 |
| Test | 5,341 |
| **Total** | **35,601** |

The final overlap checks returned zero shared cleaned texts between training and validation, training and test, and validation and test.

The TF-IDF vocabulary was fitted only on training data, and MARBERT was fine-tuned only on training data. The validation set was used for comparison, while the reconstructed test set was reserved for final evaluation.

## Models

### 1. TF-IDF + Logistic Regression

The classical baseline represents Arabic text numerically using word unigrams and bigrams. Logistic Regression then classifies each text as negative or positive.

The fitted vocabulary contained **40,261 features**. Logistic Regression provides an interpretable baseline because its feature coefficients indicate terms associated with each sentiment class.

### 2. MARBERT

MARBERT is a pretrained Arabic Transformer model designed to handle both Modern Standard Arabic and dialectal Arabic, with pretraining on a large Arabic Twitter corpus. It was fine-tuned for binary sentiment classification using a maximum sequence length of 128 tokens.

- **Model:** [`UBC-NLP/MARBERT`](https://huggingface.co/UBC-NLP/MARBERT)
- **Training epochs:** 2
- **Maximum sequence length:** 128 tokens
- **Training time in the recorded run:** approximately 6.30 minutes

Longer texts may be truncated during tokenization, which is considered in the limitations of the experiment.

## Evaluation Metrics

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Balanced Accuracy

Additional analyses include confusion matrices, ROC curves, precision-recall curves, TF-IDF feature interpretation, validation error comparison, text-length subgroup evaluation, and paired bootstrap confidence intervals.

## Validation Results

| Metric | TF-IDF + Logistic Regression | MARBERT |
|---|---:|---:|
| Accuracy | 0.6740 | **0.9360** |
| Precision | 0.6864 | **0.9403** |
| Recall | 0.6079 | **0.9273** |
| F1-score | 0.6448 | **0.9337** |
| ROC-AUC | 0.7378 | **0.9854** |
| Average Precision | 0.7363 | **0.9846** |
| Balanced Accuracy | 0.6723 | **0.9357** |

MARBERT achieved higher performance across all reported validation metrics and was selected for the final comparison based on validation F1-score.

## Final Test Results

The final evaluation was performed on the reconstructed hold-out test set.

| Metric | TF-IDF + Logistic Regression | MARBERT |
|---|---:|---:|
| Accuracy | 0.6703 | **0.9397** |
| Precision | 0.6848 | **0.9484** |
| Recall | 0.5979 | **0.9265** |
| F1-score | 0.6383 | **0.9373** |
| ROC-AUC | 0.7235 | **0.9870** |
| Average Precision | 0.7244 | **0.9863** |
| Balanced Accuracy | 0.6684 | **0.9394** |

MARBERT achieved the strongest result across all seven reported metrics. Its test F1-score was **0.9373**, compared with **0.6383** for the classical baseline.

The observed F1-score difference was approximately **0.2990 in favor of MARBERT**.

## Error Analysis

The test-set confusion matrices produced the following results:

| Model | True Negatives | False Positives | False Negatives | True Positives |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 2,026 | 716 | 1,045 | 1,554 |
| MARBERT | 2,611 | 131 | 191 | 2,408 |

MARBERT substantially reduced both false-positive and false-negative predictions compared with TF-IDF.

The validation error analysis showed:

| Measure | Count |
|---|---:|
| Validation texts | 5,340 |
| TF-IDF errors | 1,741 |
| MARBERT errors | 342 |
| Both models correct | 3,451 |
| TF-IDF wrong, MARBERT correct | 1,547 |
| MARBERT wrong, TF-IDF correct | 148 |
| Both models wrong | 194 |

These results show that MARBERT corrected many cases misclassified by the classical baseline, although some texts remained difficult for both models.

## Statistical Uncertainty Analysis

A paired bootstrap analysis with 1,000 replications estimated uncertainty in the test F1-score difference.

| Measure | Result |
|---|---:|
| F1 difference (MARBERT − TF-IDF) | 0.29901 |
| 95% confidence interval — lower bound | 0.28378 |
| 95% confidence interval — upper bound | 0.31552 |
| Bootstrap replications | 1,000 |
| Confidence interval includes zero | No |

The 95% confidence interval did not include zero, supporting a positive observed F1 difference for this reconstructed test sample. This result should not be interpreted as proof that the same improvement will occur across all Arabic domains or dialects.

## Performance by Text Length

The test-set analysis also compared short, medium, and long texts.

| Length Group | Model | Observations | Accuracy | F1-score |
|---|---|---:|---:|---:|
| Short (0–10 words) | TF-IDF | 2,877 | 0.6594 | 0.5706 |
| Short (0–10 words) | MARBERT | 2,877 | 0.9639 | 0.9577 |
| Medium (11–30 words) | TF-IDF | 2,445 | 0.8222 | 0.8088 |
| Medium (11–30 words) | MARBERT | 2,445 | 0.9121 | 0.9191 |
| Long (>30 words) | TF-IDF | 19 | 0.7895 | 0.8462 |
| Long (>30 words) | MARBERT | 19 | 0.8421 | 0.8696 |

MARBERT outperformed TF-IDF in all three length groups. However, the long-text group contained only 19 observations, so its results are insufficient for a reliable general conclusion about long Arabic texts.

## Main Findings

The experiment produced the following findings:

1. The original train and test splits contained repeated texts and inconsistent labels, requiring data cleaning and reconstruction of the evaluation partitions.
2. The final dataset contained 35,601 unique cleaned Arabic texts with a relatively balanced sentiment distribution.
3. TF-IDF + Logistic Regression provided an interpretable classical baseline but achieved a test F1-score of 0.6383.
4. MARBERT achieved a test F1-score of 0.9373 and ROC-AUC of 0.9870.
5. MARBERT produced substantially fewer false positives and false negatives than TF-IDF.
6. The paired bootstrap analysis supported a positive F1 difference on the reconstructed test sample.
7. MARBERT also performed better across the evaluated text-length groups, though the long-text group was small.

Overall, the results indicate that contextual Arabic language representations provided a substantial performance advantage over the tested TF-IDF baseline on this dataset and experimental setup.

## Limitations

- The corpus represents Arabic social-media text and may not generalize directly to formal Arabic or other domains.
- The provider's original train and test splits were reconstructed because of overlap and inconsistent labels.
- Conflicting text-label records were removed rather than manually adjudicated.
- MARBERT uses a maximum sequence length of 128 tokens, so some longer texts are truncated.
- The study uses binary sentiment labels and does not model neutral sentiment or fine-grained emotions.
- The Transformer was trained with a fixed configuration and a limited number of epochs.
- The long-text subgroup contained only 19 test observations.
- Broader generalization would require external evaluation on a separately collected Arabic dataset.

## Repository Structure

```text
Explainable-Arabic-Sentiment-Analysis/
│
├── Explainable_Arabic_Sentiment_Analysis_COMPLETE.ipynb
└── README.md
```

The notebook loads the dataset directly from Hugging Face. The dataset file does not need to be uploaded separately to this repository.

## How to Run

1. Clone or download this repository.
2. Open `Explainable_Arabic_Sentiment_Analysis_COMPLETE.ipynb` in Google Colab.
3. Select a GPU runtime for faster MARBERT fine-tuning.
4. Run the cells from top to bottom.
5. Allow the notebook to download the dataset, preprocess the text, create new partitions, train both models, and produce the evaluation outputs.

The environment setup is included in the notebook.

## References

1. Abdul-Mageed, M., Elmadany, A., & Nagoudi, E. M. B. (2021). **ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic.** Proceedings of ACL-IJCNLP 2021.  
   https://aclanthology.org/2021.acl-long.551/

2. ASAS AI. **Arabic Sentiment Twitter Corpus.** Hugging Face Datasets.  
   https://huggingface.co/datasets/asas-ai/Arabic_Sentiment_Twitter_Corpus

3. UBC-NLP. **MARBERT Model Card.** Hugging Face Model Hub.  
   https://huggingface.co/UBC-NLP/MARBERT

4. Pedregosa, F., et al. (2011). **Scikit-learn: Machine Learning in Python.** *Journal of Machine Learning Research*, 12, 2825–2830.

5. Hugging Face. **Transformers Documentation.**  
   https://huggingface.co/docs/transformers

6. Hugging Face. **Datasets Documentation.**  
   https://huggingface.co/docs/datasets

---

**Project Status:** Completed experimental workflow; final results are based on the recorded notebook run.
