**Chapter 2: Performance Evaluation**

*   **Evaluation Metrics (Fundamental)**
    *   **Precision:** Percentage of relevant items retrieved.
    *   **Recall:** Percentage of relevant items retrieved from the total set of relevant items.
    *   **F-Measure (Fβ-score):**  Combines precision and recall, weighted by a parameter β that determines relative importance.
    *   **F1-score:** A special case of F-Measure when  β = 1.
    *   **Fallout:** The ratio of non-relevant items retrieved over the total non-relevant items in the corpus.

*   **Evaluation Approaches**
    *   **Macro-Evaluation:** Average of per-query precision/recall.
    *   **Micro-Evaluation:** Aggregate across all queries, then calculate precision/recall.

*   **Retrieval with Order:**
    *   **Mean Reciprocal Rank (MRR):** Average reciprocal rank of the first relevant document. Focuses on the top result's rank.
    *   **Precision-Recall Curve:** Precision and recall values for top-k results.
    *   **Precision at k (P@k):** Precision at a specific rank k.
    *   **R-precision:** Precision when recall hits a threshold rt.
    *   **Average Precision (AP):** Area under the precision-recall curve.
    *   **Mean Average Precision (MAP):** Average AP across a set of queries.

*  **Retrieval with Graded Relevance**
    *   **Cumulative Gain (CG-k):** Sum of relevance scores up to rank k.
    *   **Normalized Cumulative Gain (nCG-k):** Normalized CG-k by dividing by the ideal CG-k.
    *   **Discounted Cumulative Gain (DCG-k):** Penalizes relevant documents in lower ranks.
    *   **Normalized Discounted Cumulative Gain (nDCG-k):** Normalized DCG-k by dividing by the ideal DCG-k.

*   **Confusion Matrix (Classification tasks)**
    *   **True Positives (TP):** Correct positive predictions.
    *   **False Positives (FP):** Incorrect positive predictions.
    *   **True Negatives (TN):** Correct negative predictions.
    *   **False Negatives (FN):** Incorrect negative predictions.
    *   **Positive Predictive Value (PPV) or Precision:** TP/(TP+FP), how many predicted positives are actual positives.
    *   **False Discovery Rate (FDR):** FP/(TP+FP) = 1 - PPV
    *   **Negative Predicative Value (NPV):** TN/(TN+FN), how many predicted negatives are actual negatives.
    *   **False Omission Rate (FOR):** FN/(TN+FN) = 1 - NPV.
    *   **True Positive Rate (TPR) or Recall or Sensitivity:** TP/(TP+FN), how many actual positives are predicted as positive
    *   **False Negative Rate (FNR):** FN/(TP+FN) = 1-TPR
    *  **True Negative Rate (TNR) or Specificity:** TN/(TN+FP), how many actual negatives are predicted as negative.
    *   **False Positive Rate (FPR) or Fall-Out:** FP/(TN+FP) = 1-TNR
    *   **Accuracy (ACC):** (TP + TN) / Total.
    *   **Error Rate (ERR) or Misclassification Rate:** (FP + FN) / Total = 1 – ACC.
    *   **Prevalence:** P / Total.
    *   **F1-score:** 2 \* precision \* recall / (precision + recall), harmonic mean between precision and recall.

*   **Optimizing Hyperparameters**
   *  The notion of a threshold, usually with respect to a score or rank.
   *   Receiver operating characteristics (ROC) curve
   *   Area under the ROC curve (AUC)
   *   Mean Average Precision (MAP)
