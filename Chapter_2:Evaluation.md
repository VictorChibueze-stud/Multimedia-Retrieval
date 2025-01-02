**Chapter 2: Performance Evaluation**

**2.1 Introduction**

*   **Retrieval Benchmarks:**
    *   **Definition**: Standardized datasets, queries, and relevance assessments used to evaluate information retrieval systems. They provide a controlled environment to compare different methods.
    *   **Use Case**: Essential for objectively comparing the performance of various retrieval algorithms.
    *   **Example**: MS MARCO (large-scale reading comprehension dataset), TREC datasets (diverse set of text retrieval tasks).
    *   **Pros:** Allows for fair and reproducible evaluations, aids in identifying strengths and weaknesses of different retrieval systems.
    *   **Cons:** Can be biased towards the type of data included in the benchmark, can be computationally expensive to build and maintain.
    *   **Dependencies:** Need for expert assessors to determine ground truth (relevance judgments), good query representation.
*   **Dense vs. Sparse Assessments:**
    *   **Definition:**
        *   **Dense Assessment:** Most or all retrieved documents are assessed for relevance to a query. This usually happens in text retrieval competitions.
        *   **Sparse Assessment:** Only a small subset of retrieved documents are assessed for relevance. Used for very large datasets where complete assessments are infeasible.
    *   **Use Case**:
        *   Dense assessments are preferred for accuracy in smaller benchmarks.
        *   Sparse assessments are practical for large datasets to reduce the assessment efforts.
    *   **Example**: TREC is typically using dense assessments whereas MS Marco is a typical example of sparse assessments.
    *   **Pros:** Dense assessments provide a clearer picture of a model's performance, especially in the ranking of search results (relative ranking) but are expensive. Sparse assessments allow for evaluation of large datasets.
    *   **Cons:** Dense assessments can be expensive and time-consuming to create. Sparse assessments may not give an accurate view of all situations and may miss out on relevant documents that were not assessed.
    *   **Dependencies:** The need for efficient evaluation tools, good sampling methods (in case of sparse assessments).
*   **Performance Goals in Web Search vs. Expert Search:**
    *   **Definition:** Different user needs and expectations lead to varying performance goals.
    *   **Use Case:** Web search focuses on precision, aiming for relevant results in the top-10, while expert search focuses on recall to ensure complete results.
    *   **Example**: Web search: quickly find a web page; expert search: find all relevant documents related to a patent or scientific finding
    *   **Pros:** Tailors evaluation metrics and search algorithms to specific use cases.
    *   **Cons:** Requires a clear understanding of user needs and contexts for different tasks.
    *   **Dependencies:** Good evaluation metrics, clear definition of user scenarios, clear definition of use cases.

**2.2 Boolean Retrieval**

*   **Precision:**
    *   **Definition**: The proportion of retrieved documents that are relevant to the query. It answers "how many of the retrieved documents are actually relevant?"
    *   **Formula**: `p = |Retrieved ∩ Relevant| / |Retrieved|`
    *   **Use Case**: Important when you need only the most relevant information and can tolerate missing some relevant documents, such as fact-checking queries, or cases where you have many irrelevant items in the collection.
    *   **Example**: If a search retrieves 10 documents, and 5 are relevant, the precision is 5/10 = 50%.
    *   **Pros:** Measures the accuracy of the retrieved results, shows how clean the results are
    *   **Cons:** Does not account for how many relevant documents were missed, can be low even if many relevant documents are not retrieved.
*   **Recall:**
    *   **Definition**: The proportion of relevant documents that are retrieved by the search system. It answers the question "how many of the relevant documents have been retrieved?"
    *   **Formula**: `r = |Retrieved ∩ Relevant| / |Relevant|`
    *   **Use Case**: Important when you need to retrieve all or most of the relevant information, regardless of the number of retrieved documents. Such as patent searches, systematic reviews of literature etc.
    *   **Example**: If there are 20 relevant documents, and the search system retrieves 10 of them, the recall is 10/20 = 50%.
    *   **Pros:** Shows how complete the retrieved results are, accounts for all relevant documents in the collection.
    *   **Cons:** Does not account for irrelevant documents that are returned. It can be high even when many irrelevant documents are also retrieved.
*   **Fallout:**
    *   **Definition**: The proportion of irrelevant documents that are retrieved by the search system. It answers "how many of the irrelevant items are returned?"
    *   **Formula**: `f = |Retrieved \ Relevant| / |All \ Relevant|`
    *   **Use Case**: This is especially important when the number of relevant results are very small, and the remaining results have high chances of being irrelevant and thus creating overhead for the user.
    *   **Example**: Out of 100 documents in the collection, 20 are relevant and 10 retrieved. 7 of the retrieved documents are relevant, and thus 3 are irrelevant. There are 80 irrelevant documents. Therefore fallout = 3/80 = 3.75%.
    *   **Pros:** Shows how well the search system avoid irrelevant results.
    *   **Cons:** It does not measure how well the search system retrieved relevant documents.
*   **F-Measure / F1-Score:**
    *   **Definition**: The harmonic mean of precision and recall, providing a single metric that balances both. It is suitable when you want to measure both at the same time.
    *   **Formula**: `F = (2 * precision * recall) / (precision + recall)`
    *   **Use Case**: Useful for comparing performance when both precision and recall are important. Also used in optimizing machine learning algorithms and hyperparameter tuning.
    *   **Example**: If precision = 60% and recall = 40%, then F1 = (2 * 0.6 * 0.4)/(0.6 + 0.4) = 0.48 or 48%
    *   **Pros:** Balances precision and recall, giving a single performance metric.
    *   **Cons:** Might not be suitable if precision or recall is more important than the other in a specific use case.
        *   `Fβ = (β² + 1). p . r / (β2.p+r)`
        *   ẞ determines the importance of recall over precision
        *   ẞ=0, only precision is considered;
        *  ẞ=∞, only recall is considered
*   **Macro-evaluation:**
    *   **Definition**: The average of a metric across all queries independently (for example, average of precision or recall over all queries). The "query" in macro and micro evaluation is a specific, evaluated request for information. It is not just a user search, but rather part of a controlled experiment with known relevance assessments. These queries are often created to evaluate all use cases of the information retrieval system in question. The main goal is to understand how well the entire system performs, and not just a subset of the use cases.
    *   **Use Case**: Useful when you want to treat all queries as equally important.
    *   **Example**: Compute precision for each of the 100 queries, and then average these 100 precision values.
    *   **Pros:** Provides a balanced performance view across all queries, avoids overemphasizing results for queries with many relevant documents.
    *   **Cons:** Can be skewed if you have queries with wildly different sizes of relevant documents. Not suitable if some queries are more important than others.
    *   **Dependencies:** Requires query level results.
*   **Micro-evaluation:**
    *   **Definition**: The metric is computed over the sum of all (relevant / retrieved) documents. (For example sum of true positives over the sum of retrieved documents)
    *   **Use Case**: Useful when you want to prioritize performance on queries with more relevant documents.
    *   **Example**: Add all true positives from all queries. Add all retrieved documents from all queries. Divide the sum of true positives with the sum of retrieved documents to compute micro-precision
    *   **Pros:** Prioritizes performance for the queries with more relevant documents.
    *   **Cons:** May overemphasize the performance of methods on large queries while diminishing the performance on smaller queries.
    *   **Dependencies:** Requires query level results.

**2.3 Retrieval with Order**

*   **Mean Reciprocal Rank (MRR):**
    *   **Rank:** The position of the first relevant document that your search system returns for a particular query.
    *   **Reciprocal Rank:** The reciprocal of the rank of the first relevant document. It's the value you get after taking the rank and calculating 1/rank.
So:
If the first relevant document is at rank 1, the reciprocal rank is 1/1 = 1.
If the first relevant document is at rank 2, the reciprocal rank is 1/2 = 0.5.
If the first relevant document is at rank 5, the reciprocal rank is 1/5 = 0.2.
If no relevant document is returned, some systems will consider the reciprocal rank is zero.
    *   **Definition:** The average of all the reciprocal rank scores, across all of the queries in your evaluation set. This average gives you a single number to compare and evaluate your search systems.
    *   **Formula:** `MRR = 1/|Q| * Σ(1/rank_i)` where `rank_i` is the rank of the first relevant document for query `i`
    *   **Use Case:** Important when the first relevant document is the most critical, for example, in question answering, or fact checking. Also relevant in situations with sparse relevance assessments
    *   **Example**: Given queries with ranks 1, 3, 2, MRR = (1/1 + 1/3 + 1/2)/3 = (1 + 0.333 + 0.5)/3 = 0.611.
    *   **Pros:** Emphasizes top-ranked results, useful when other metrics can't be computed due to sparse evaluation data, such as sparse assessments.
    *   **Cons:** Ignores relevance beyond the first relevant result.
*   **Precision-Recall Curve:**
    *   **Definition:** A plot of precision against recall at different ranking thresholds. It visualizes how precision changes as recall increases and helps understand the trade-offs
    *   **Use Case:** Analyzing how precision and recall values trade off at varying ranking positions.
    *   **Example:** Consider a system that retrieves documents based on a ranking score. We can compute the precision and recall at each rank (1, 2, 3, etc) and plot them. When we plot we can observe that precision increases with new relevant results but may decrease if new results are irrelevant. Similarly recall increases only when new relevant documents are discovered.
    *   **Pros:** Provides detailed insights into the performance of the ranked list of results.
    *   **Cons:** Hard to compare curves if they cross each other.
    *   **Dependencies:** Needs ranked list of results
*   **System Efficiency:**
    *   **Definition:** Measures how close a system is to the ideal system with both precision and recall equal to 1. It uses the distance from a system's performance to the perfect performance.
    *   **Formula:**  `E = 1 – d/√2`, where `d` is the distance to the ideal point (1,1).
    *   **Use Case:** Provides a single metric that captures the overall performance, but not used frequently.
    *   **Example:** A system with a score of (0.7, 0.7) has a higher system efficiency than a system at (0.2, 0.9)
    *   **Pros:** Gives a single value to assess the overall performance by considering both recall and precision,
    *   **Cons:** Not as informative as precision/ recall individually or the precision/recall curve.
    *   **Dependencies:** Requires precision and recall metrics.
*   **Precision at k (P@k):**
    *   **Definition:** Precision computed using only the top *k* results of the ranking.
    *   **Formula:** `P@k = |Relevant in Top-k| / k`
    *   **Use Case:** Important when users mostly examine the top results, for example, in web search or fact-checking queries.
    *   **Example:** If the top 5 results contain 3 relevant documents, P@5 = 3/5 = 60%.
    *   **Pros:** Focuses on top results, easy to understand and calculate.
    *   **Cons:** Ignores the overall position of relevant documents.
*   **R-Precision:**
    *   **Definition:** Precision computed once a threshold recall value *r* is achieved. For example, once a system has retrieved 20% of all relevant results.
    *   **Use Case:**  Useful when a threshold recall value is important for specific application, and you need to understand the precision at this threshold.
    *   **Example:** With `r` set to 20%, we determine precision when 20% of the relevant results have been retrieved.
    *   **Pros:** Provides insight at a specific level of recall.
    *   **Cons:** Requires knowing the overall number of relevant documents.
*   **Average Precision (AP):**
    *   **Definition:** The average of the precision values, computed at all positions where a relevant document is retrieved.
    *   **Formula:**  The formula includes the precision *pk* at position *k* and the binary variable, that is one if the document at position *k* is a relevant document and zero if the document at position *k* is not relevant. The sum of precisions at each position *k* is then divided by the number of relevant documents.
    *   **Use Case:** Comprehensive performance measure, accounts for precision at all rank positions that return a relevant document.
    *   **Example:**  AP is high if relevant documents appear at the top.
    *   **Pros:** Provides insights into both precision and recall performance across the entire ranking.
    *   **Cons:** Can be difficult to intuitively understand the score, especially if you do not understand how it was derived (as the average of precision at each rank that retrieved a relevant document).
*   **Mean Average Precision (MAP):**
    *   **Definition:** The average of the AP scores across all queries.
    *   **Formula**: `MAP = 1/|Q| * ΣAP(qi)`, where AP(qi) is the average precision for query *i*
    *   **Use Case:** Standard for ranked results across multiple queries.
    *   **Example:** Compute the AP for 100 queries. Then, the average of these 100 AP values are MAP
    *   **Pros:** Aggregates performance across queries for a comprehensive evaluation.
    *   **Cons:**  Less meaningful if the use case has only one query.

**2.4 Retrieval with Graded Relevance**

*   **Cumulative Gain (CG):**
    *   **Definition:** The sum of graded relevance values in the ranking up to position *k*.
    *  **Formula:** `CGk = Σreli` from *i*=1 to *k*, where reli is the graded relevance at position *i*
    *   **Use Case:** Assessing overall relevance up to a given position, by taking into account the graded relevance.
    *   **Example:** If the top 3 documents have graded relevance of 3, 1, and 2, the CG3 = 3 + 1 + 2 = 6.
    *   **Pros:** Considers the relevance of documents with different levels of relevance (graded relevance), easy to understand and implement.
    *   **Cons:** Does not consider the position of results in the ranking.
*   **Discounted Cumulative Gain (DCG):**
    *   **Definition:** Cumulative gain with a penalty (discount) for lower ranks.
    *   **Formula:** `DCGk = Σ(reli / log2(i+1))` from *i*=1 to *k*
    *   **Use Case:** Important when the ranking position is an important factor, as higher-ranked documents are valued more than lower ranked documents.
    *   **Example:** If the top 3 documents have graded relevance of 3, 1, and 2, the DCG3 = 3/log2(2) + 1/log2(3) + 2/log2(4) = 3 + 0.63 + 1 = 4.63
    *   **Pros:** Takes ranking order into account; higher-ranked documents contribute more,
    *   **Cons:** DCG scores of different queries may have different lengths, making comparisons hard.
*   **Normalized Discounted Cumulative Gain (nDCG):**
    *   **Definition:** DCG normalized by the ideal DCG (IDCG), which is the DCG of the optimal ranking.
    *   **Formula:** `nDCGk = DCGk / IDCGk` where IDCG is the DCG of ideal ranking
    *   **Use Case:** Comparing the performance of ranked results with varying lengths.
    *   **Example:** An nDCG score of 0.98 means that the search result performs 98% as good as the ideal search result.
    *   **Pros:** Normalizes the DCG scores to range between 0 and 1, allowing for meaningful comparisons between results of different lengths and use cases.
    *   **Cons:** IDCG has to be computed for each query, and that can be expensive.

**2.5 The Confusion Matrix**

*   **True Positives (TP):**
    *   **Definition:** The number of cases where the model correctly predicted the positive class.
    *   **Use Case:** Counting the correct predictions for the positive class.
    *   **Example**: In cancer detection: the model correctly identifies that a patient has cancer.
*   **False Positives (FP):**
    *   **Definition:** The number of cases where the model incorrectly predicted the positive class (Type I error).
    *   **Use Case:** Counting the number of incorrect positive predictions.
    *   **Example**:  In spam detection, the model incorrectly marks a non-spam email as spam.
*   **False Negatives (FN):**
    *   **Definition:** The number of cases where the model incorrectly predicted the negative class (Type II error).
    *   **Use Case:** Counting the number of incorrect negative predictions.
    *   **Example**:  In fraud detection, a model fails to identify a fraudulent transaction.
*   **True Negatives (TN):**
    *   **Definition:** The number of cases where the model correctly predicted the negative class.
    *   **Use Case:** Counting the number of correctly predicted cases for the negative class.
    *   **Example**: In disease detection, the model correctly identifies a healthy person as healthy.
*   **Positive Predictive Value (PPV) / Precision:**
    *   **Definition:** The proportion of predicted positives that are actually positive.
    *   **Formula:** `PPV = TP / (TP + FP)`
    *   **Use Case:** Important when we need to minimize false positive, for example, in medical diagnosis.
    *   **Example:** If 100 people are predicted to have cancer and 80 of those actually do, PPV = 80/100 = 80%
    *   **Note**: Same as precision from section 2.2 but in a different context of classification and with different naming.
*   **False Discovery Rate (FDR):**
    *   **Definition:** The proportion of predicted positives that are actually false. The complement of PPV.
    *   **Formula:** `FDR = FP / (TP + FP) = 1 - PPV`
    *   **Use Case:** Used in scientific literature to indicate how many of the findings are wrong in a set of research findings. Important when we need to minimize false positives
    *   **Example:** If 100 people are predicted to have cancer and 20 of those are actually healthy, FDR = 20/100 = 20%.
*   **Negative Predictive Value (NPV):**
    *   **Definition:** The proportion of predicted negatives that are actually negative.
    *   **Formula:** `NPV = TN / (TN + FN)`
    *   **Use Case:** Important when we need to minimize false negatives, for example, in a diagnostic test that identifies people that do not have a disease.
    *   **Example:** If 100 people are predicted not to have cancer, and 90 of those are actually healthy, NPV = 90/100= 90%.
*   **False Omission Rate (FOR):**
    *   **Definition:** The proportion of predicted negatives that are actually positive. The complement of NPV.
    *   **Formula:** `FOR = FN / (TN + FN) = 1 - NPV`
    *   **Use Case:** Useful for understanding how often a test will miss a positive case, important when we need to minimize false negatives.
    *   **Example:** If 100 people are predicted not to have cancer and 10 of those are actually sick, FOR = 10/100=10%.
*   **True Positive Rate (TPR) / Recall / Sensitivity:**
    *   **Definition:** The proportion of actual positives that are correctly predicted as positive.
    *   **Formula:** `TPR = TP / (TP + FN)`
    *   **Use Case:** Important when we need to maximize true positive findings, such as a test that detects people that have a certain disease.
    *  **Example:** If there are 100 people with cancer, and a test correctly identifies 80 as having cancer, the TPR= 80/100 = 80%.
    *   **Note**: The same as recall in the context of information retrieval.
*   **False Negative Rate (FNR):**
    *   **Definition:** The proportion of actual positives that are incorrectly predicted as negative. The complement of TPR.
    *   **Formula:** `FNR = FN / (TP + FN) = 1 - TPR`
    *   **Use Case:** It is useful in contexts where it is important to know how many of the positive cases are misclassified
    *   **Example:** If there are 100 people with cancer, and a test misclassifies 20 as healthy, then FNR = 20/100 = 20%.
*   **True Negative Rate (TNR) / Specificity:**
    *   **Definition:** The proportion of actual negatives that are correctly predicted as negative.
    *   **Formula:** `TNR = TN / (TN + FP)`
    *   **Use Case:** Important when we need to minimize the false positives (incorrect positive findings) for a disease, for example, in diagnostic tests that detects people that are healthy.
    *   **Example:**  If 100 people do not have cancer, and the test correctly identifies 95 as healthy, then TNR = 95/100= 95%.
*   **False Positive Rate (FPR) / Fall-Out:**
    *   **Definition:** The proportion of actual negatives that are incorrectly predicted as positive. The complement of TNR.
    *   **Formula:** `FPR = FP / (TN + FP) = 1 - TNR`
    *   **Use Case:** Useful when it is important to see how often the system raises an incorrect alarm when there is no real condition, such as spam detection system.
    *   **Example:** If 100 people do not have cancer, and the test incorrectly identifies 5 as having cancer, then FPR= 5/100 = 5%
*   **Accuracy (ACC):**
    *   **Definition:** The overall proportion of correct predictions.
    *   **Formula:** `ACC = (TP + TN) / (TP + TN + FP + FN)`
    *   **Use Case:** Provides a general performance view of the classifier.
    *   **Example:** If the model correctly predicts 80% of the time, then accuracy is 80%.
    *   **Cons:** Accuracy may be biased with imbalanced datasets.
*   **Error Rate (ERR) / Misclassification Rate:**
    *   **Definition:** The overall proportion of incorrect predictions
    *   **Formula:** `ERR = (FP + FN) / (TP + TN + FP + FN) = 1 - ACC`
    *   **Use Case:** Understanding how many predictions are wrong in the model.
    *  **Example:** If the model misclassifies 20% of the cases, the error rate is 20%
*   **Prevalence:**
    *  **Definition**: The proportion of the positive cases in the dataset.
    *  **Formula**: `P = |Positive| / |All|`
    *  **Use Case:** Important to understand the distribution of positive and negative cases as the metrics that are derived based on the confusion matrix can vary depending on the prevalence. It helps to indicate if there is a bias in the evaluation.
    * **Example**: In medical testing, a disease with a prevalence of 1% will drastically change the diagnostic interpretation compared to a disease with a prevalence of 50%.
*   **F1-Score (revisited):**
    *   **Definition**: Harmonic mean of precision and recall but in the context of a confusion matrix.
    *   **Formula:** F1 = 2 * PPV * TPR / (PPV + TPR) = 2TP / (2TP+FP+FN)
    *   **Use Case:** Provides a balanced performance metric between PPV and TPR, important in classification with imbalanced datasets.
    *   **Note**: The same F1-score as in the section 2.2 but in the context of a confusion matrix and with different naming of variables.
*  **Multiclass Confusion Matrix:**
    *   **Definition**: A matrix showing the agreement between the predicted and the actual classes when there are more than two classes in the dataset.
    *  **Use Case**: Visualizes the classification performance between multiple classes and helps identify areas where a model may have difficulties.
    *  **Example**: An image classification model that classifies images of dogs, cats, and birds.
        * The diagonals of the matrix represent correctly classified instances,
        * Non-diagonal elements show the misclassifications between each class.
    *   **Pros:** Helps analyze the model's prediction accuracy by category.
    *   **Cons:** Can become very difficult to read and analyze when there are many classes.
* **Collapsing the Multiclass Confusion Matrix:**
    *   **Definition:** Converting a multi-class problem into a series of binary problems by selecting one class as the positive class, and all other classes are grouped into the negative class.
    *  **Use Case:** Analyze the performance of individual classes in a multi-class setting using binary performance metrics.
    * **Example**: In image classification with "dog", "cat" and "bird", if we are interested in "cat", then we can merge "dog" and "bird" into a new class of "non cat", then use metrics from binary classifiers.
    * **Pros**: Allows us to use well known metrics derived from binary confusion matrix, and provides a very good understanding of model performance in multi-class settings
    * **Cons**: Requires multiple evaluations when interested in the results of multiple classes.

**2.6 Optimizing Hyperparameters**

*   **Threshold Tuning for Binary Classifiers:**
    *   **Definition**: Adjusting the decision boundary (threshold) of a binary classifier to optimize a specific evaluation metric. In other words the hyperparameter of the model needs to be adjusted to improve performance in the desired area.
    *   **Use Case:** Crucial in classification tasks where the cost of false positives or false negatives varies.
    *   **Example**: Adjusting the decision threshold in a diabetes detection system to balance sensitivity and specificity.
    *   **Pros:** Optimizes a classifier for a specific use case.
    *   **Cons:** Requires a validation set to determine the appropriate threshold.
*   **The Distributions of Scores for True Positives and True Negatives:**
    *   **Definition:** The distributions of model scores for the items which actually belong to the positive class (true positive items) and negative class (true negative items).
    *   **Use Case:** Helps visualize the separability of classes and understand the effect of thresholding on outcomes.
    *   **Example**: A graph showing two overlapping bell curves – one for diabetic scores and another for healthy scores.
    *   **Pros:** Allows a graphical analysis of the model’s prediction performance, helps to understand why predictions fail.
    *   **Cons:** Requires knowledge of the actual classes and how the model predicts them.
*   **True Negative Rate (TNR):**
    *   **Definition:** As defined in the previous section, it is the proportion of correctly predicted negative cases.
    *   **Formula:** `TNR = TN / (TN + FP)`
    *  **Use Case:** Important when we need to minimize false positives.
    *  **Example:**  For an automated diagnosis, if 100 healthy patients are tested, the model should correctly identify as many of them as healthy.
*   **False Positive Rate (FPR):**
    *   **Definition:** As defined in the previous section, it is the proportion of actual negatives that are incorrectly classified as positive.
    *   **Formula:** `FPR = FP / (TN + FP) = 1 - TNR`
    *   **Use Case:** Useful when it is important to see how often the system raises an incorrect alarm when there is no real condition.
    *   **Example:** Spam email detection: how many of the regular emails are marked as spam
*   **False Negative Rate (FNR):**
     *   **Definition:** As defined in the previous section, it is the proportion of the actual positives that are classified incorrectly as negative.
    *   **Formula:** `FNR = FN / (TP + FN) = 1 - TPR`
    *   **Use Case:** Useful to understand how many of the positive cases are misclassified by the model, for example, in disease detection.
    *   **Example:** A tumor detection system may fail to identify an actual malignant tumor.
*   **True Positive Rate (TPR):**
    *   **Definition:** As defined in the previous section, it is the proportion of the actual positives that are correctly classified as positive.
    *   **Formula:** `TPR = TP / (TP + FN)`
    *   **Use Case:** Important when we need to maximize correct positive findings.
    *   **Example:** Detection of actual fraudulent transactions in online shopping
*   **Receiver Operating Characteristic (ROC) curve:**
    *   **Definition:** A plot that visualizes the performance of a binary classifier by varying the classification threshold, plotting TPR vs. FPR.
    *   **Use Case:** Helps choose the best threshold for different classification tasks, understand the overall performance of the classification model by displaying how the TPR and FPR metrics trade off with each other for varying threshold values.
    *   **Example:** Plotting the TPR against FPR for different thresholds in a classification task.
    *   **Pros:** Provides insights into model performance at all classification thresholds.
    *   **Cons:** Less intuitive for specific use cases than direct metrics.
*  **Area Under the Curve (AUC):**
    *  **Definition**: The area under the ROC curve, representing the overall performance of a binary classifier.
    * **Use Case**: Summarizes the ROC curve into a single metric, providing an estimate of model’s performance across all possible threshold values.
    * **Example**: A perfect classifier that separates positive and negative classes completely has an AUC value of 1
    * **Pros**: Provides single metric to summarize the performance, gives a clear distinction on how the model performs.
    * **Cons**: May be less interpretable than the actual ROC curve for a specific use case with specific thresholds.

**2.7 Literature and Links**

* This section provides the resources and the relevant papers that can deepen your knowledge about the topics covered in the chapter.

This is a comprehensive list of all the concepts, models, and algorithms covered in your lecture notes.

**Next Steps**

Now you have a solid overview of what you're expected to know. As we move forward, we can dive into deeper discussions on:

*   **Relationships:** I'll help you connect the concepts. For example, how does the confusion matrix relate to metrics like precision, recall, and F1-score?
*   **Examples**: We can create more complex scenarios to test your understanding.
*   **Oral Exam Strategies**: We can develop good approaches to structure the answers to the oral exam questions.
*   **Answering Potential Questions**: I can ask you questions to challenge your knowledge.

Let me know when you're ready to start with a deeper analysis!
