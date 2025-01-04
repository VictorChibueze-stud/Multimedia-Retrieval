### Chapter3: Classical Text Retrieval

**Part 1: Core Concepts and Preprocessing**

**1.1.  Core Definitions**

*   **Classical Text Retrieval:** The process of finding relevant text documents within a collection based on a user's textual query, primarily using techniques developed prior to the rise of modern machine learning.
*   **Terms:** Individual words or phrases extracted from documents and used as the basic units for indexing and search.
*  **Document Representation:** How each document is represented internally, usually as a mathematical model.
*   **Query Representation:** How user queries are converted to the same representation as documents.
*   **Relevance Ranking:** The process of ordering retrieved documents based on their perceived relevance to a user's query.
*   **Retrieval Status Value (RSV):** A score representing the relevance of a document to a query. In some models, this score also represents a probability of relevancy.

**1.2. Preprocessing: Transforming Raw Text**

Before indexing and searching, text needs to be cleaned and transformed, or pre-processed. This phase includes steps like:

   *   **Extracting Text (and MetaData):** The initial step where we retrieve the content from various document formats (HTML, PDF, etc.) and identify structure, like title, meta-information, body content, link anchors etc. This involves techniques like HTML parsing. For example, retrieving the article text from an HTML page and metadata from its header section.
      *   **Pros:** Enables access to text, allows enriching content with metadata.
      *   **Cons:** Requires parsing logic, can be complex for different formats
   *  **Splitting:** Dividing a document into smaller, more manageable units (chunks) for indexing. Common methods are:
       *  **Fixed-Size Chunks:** splitting text into chunks of fixed token number (words or chars).
           *  **How it Works:** Chunks are created every *n* tokens.
           *  **Pros:** Simple, uniform chunk sizes for normalization.
           *  **Cons:** May split sentences or paragraphs, not contextually meaningful, can lead to poor retrieval quality.
       *   **NLP-Based Chunks:** Chunks are made of complete sentences until a minimum length is reached.
           *   **How it Works:** Sentences are grouped together until they reach some minimum size.
           *   **Pros:** Chunks are more contextually coherent, more semantically meaningful.
           *   **Cons:**  Chunk sizes can vary, can be more computational expensive due to the NLP part.
       *   **Metadata-Based Chunks:** Use structural markers (paragraphs, chapters) for splitting.
            *  **How it Works:** Documents are split based on metadata (e.g. HTML tags).
            *  **Pros:** Chunks are contextually coherent, often corresponds to logical sections in a document, good compromise between context and size.
           *   **Cons:** Can be difficult to implement on unstructured text formats.
       *  **Semantic Chunks:** Use NLP to group sentences based on topic/concept.
            *  **How it Works:** Machine learning techniques are used to identify similar sentences, which are grouped together.
           *  **Pros:** Chunks are more contextually coherent, better semantic relationship of concepts within the chunks
           *  **Cons:** more computationally expensive, higher risk of error.
   *   **Tokenization:** The process of breaking text into individual units called tokens, usually words.
       *   **How it Works:** Basic tokenization splits by whitespace, but can be complex for other languages.
       *   **Pros:** Simplifies text for indexing
       *  **Cons:**  Can lead to different word forms being treated as separate, unrelated terms
   *   **Stemming:**  Reducing words to their root form by removing affixes. For example, "running," "ran," and "runs" all stem to "run."
       *   **How it Works:** Implements rules for suffix removal.
       *  **Pros:** Reduces vocabulary size, improves matching by merging different forms of words
       *   **Cons:** Results in non-linguistic forms that can sometimes be inaccurate.
           * **Most Important 20%:** *The Porter Stemming algorithm is the best-known. It consists of a series of rules applied in order based on the "measure" of the word, which is the number of alternating consonants and vowels, which are marked by `C` and `V` respectively.*
   *  **Lemmatization:** Reducing words to their dictionary form based on the context of each word. For example, "better" becomes "good".
         *   **How it Works:** Uses a dictionary and part-of-speech analysis to return the root word.
         *   **Pros:** Produces more linguistically correct results than stemming
         *   **Cons:**  More complex and computationally expensive than stemming
   * **Stop Word Removal:** Eliminating common words (like "the," "a," "is") that often have little semantic value and high occurrence frequencies.
        *  **How it Works:** Uses a pre-defined list of stop-words.
        *  **Pros:** Reduces index size and improves search efficiency by focusing on more descriptive words.
        *  **Cons:** Can remove contextually relevant words, might need custom stop lists.
   * **Character and Fragment-Based Tokenization**: Uses a specified number of characters for tokenization, which is particularly useful in modern large language models to avoid OOV (Out of Vocabulary) issues.
       *  **How it Works:** Creates tokens from a sequence of characters, i.e. every 3 letters.
       *   **Pros:** Can deal with OOV issues
       *  **Cons:** Resulting tokens can be less meaningful

**1.3 Term Weightings & Bag-of-Words**

*  **Term Frequency (TF):** The number of times a term appears in a document. TF is used to reflect the importance of a word within a document.
*   **Bag-of-Words (BOW):** A simple text representation where the order of terms is ignored, focusing instead on the presence and frequency of terms.
    *   **How it Works:** A document is represented as a vector, where the i-th entry reflects the TF value of the i-th word in the vocabulary.
    *   **Pros:** Simple and efficient to compute
    *  **Cons:** Ignores word order and context
*   **Set-of-Words Model:** A simplified text representation that only indicates whether a term is present (1) or absent (0) in a document, disregarding term frequencies and word order.
    *   **How it Works:**  A document is represented as a vector, where the i-th entry is 1 if the i-th word is present, and 0 if it is not.
    *   **Pros:** Simple and compact representation.
    *   **Cons:** Ignores term frequencies and word order and context.
*  **Inverse Document Frequency (IDF):** Measures how rare a term is across the document collection. It's used to assign higher weights to less frequent terms since they are considered more important in distinguishing documents from each other.
    *   **How it Works:** Calculated as  `idf(t) = log(N/df(t))`, where N is the total documents and df(t) is the number of documents a term t appears in.
    *   **Pros:** Gives less importance to words that appear often and adds significance to words that appear seldom.
    *   **Cons:** May not work well for small collections
*   **TF-IDF (Term Frequency-Inverse Document Frequency):**  Combines TF and IDF to weight terms based on their frequency within a document and their rarity across the collection. `tf-idf(t,d) = tf(t,d) * idf(t)`.

      *   **How it Works:**  Each document is represented as a vector, where the i-th entry is equal to the TF-IDF value of the i-th word in the vocabulary.
      *   **Pros:** Provides better ranking compared to using only TF, highlights keywords that are more important for particular document compared to a wider document collection.
       *   **Cons:** Does not capture word order and semantic relationships, is biased to larger documents.

**1.4 Discriminative Power of Terms**
*    **How it works:**  Terms are ranked based on their ability to differentiate documents. The discriminative power of a term describes how effective a term is in differentiating relevant documents from non-relevant ones.
*   **Pros:** Can be used to identify terms that are important for retrieval
*   **Cons:** Can be complex to compute.
    * **Most Important 20%:** *Terms appearing in very few or very many documents receive lower scores. The best terms are somewhere in between.*

**1.5 Zipf's Law:**
*   **How it Works:** A statistical law stating that a small number of terms appear most of the time, and that the frequency of a term is inversely proportional to its rank in the frequency list.
*   **Pros:** Can be used to predict word distributions, shows that top terms do not add much value.
*   **Cons:** Not a perfect fit, and can have variations between the different collection, can be used to prune the vocabulary, or weigh terms by importance.
    * **Most Important 20%:** *The most common words contribute little to differentiating document relevance*

**Part 2: Retrieval Models**

**2.1 Boolean Model**

*  **How it Works:** Documents are represented as a set of terms. Queries use Boolean operators (AND, OR, NOT) to combine term predicates. A document matches a query if it satisfies the full Boolean expression.
*  **Pros:** Simple to understand and implement, exact matching, support for explicit query structuring
*  **Cons:** Lack of ranking, no partial matching, can be difficult to use for complex information needs.
    * **Most Important 20%:** *Documents are returned if they completely fulfill the query, no notion of ranking.*

    * **Use Case:** Boolean retrieval is great for applications where precision is paramount and document ranking is not critical, such as filtering or searching for specific data based on metadata, such as searching for files on a local drive or performing data retrieval from database.

**2.2 Extended Boolean Model**

*   **How it Works:** Extends the Boolean model by assigning scores to documents based on how well they satisfy the query, allowing for partial matches. It uses normalized vectors, term occurrences and idf-weighting. Similarity is computed using fuzzy algebraic, fuzzy set, or soft-boolean operators.
*  **Pros:** Supports ranking, allows partial matches, improves usability over the standard Boolean model.
*   **Cons:**  Heuristic similarity scores lack theoretical foundation, still require users to form complex boolean queries.
     * **Most Important 20%:** *It provides a soft matching approach, meaning results are ranked, and it uses bag of words for document representation.*
*   **Operators:**
    *   **Fuzzy Algebraic Operators:** `sim(A AND B) = sim(A) * sim(B)` and `sim(A OR B) = sim(A) + sim(B) - sim(A) * sim(B)`. Only works with two operands.
    *   **Fuzzy Set Operators:** `sim(A AND B) = min(sim(A), sim(B))` and  `sim(A OR B) = max(sim(A), sim(B))`. Works with multiple operands.
    *  **Soft Boolean Operators:** Similar to fuzzy set operators but uses a scaling factor between 0 and 1 to control the influence of min and max function on the resulting score.
    *   **P-Norm Operator:**  A generalization of soft boolean operators using the concept of distance and the p-norm.
*  **Use Case:** Enhanced Boolean retrieval can be used for applications with medium sized collection where partial matching is important, but users still prefer the boolean query model.

**2.3 Vector Space Model**

*   **How it Works:** Documents and queries are represented as vectors in a high-dimensional space. The relevance of a document is determined by the similarity between its vector and the query vector, often using measures like dot product and cosine similarity.
*   **Pros:**  Intuitive and efficient, supports partial matching and ranking, captures term importance (using tf-idf).
*   **Cons:** Heuristic scoring, ignores term dependencies, can be biased by authors, long documents are favored.
   * **Most Important 20%:** *Documents and queries are represented by vectors, and similarity is measured by the inner product or cosine of the vectors.*
*  **Similarity Measures**
    * **Inner Vector Product:** A simple sum of element-wise vector product without normalization.
       * **Pros:** Can be computed quickly.
       * **Cons:** Can favor documents with many terms, not normalized
    * **Cosine Similarity:** The cosine of the angle between document and query vectors, normalized vectors are often used.
       *  **Pros:** Not influenced by the vector's magnitude, normalized between 0 and 1.
       * **Cons:**  Ignores differences in term occurrences

  *   **Use Case:** General-purpose retrieval that works well with a large variety of use cases such as web searches, document indexing.

**2.4 Probabilistic Retrieval (BIR Model)**

*   **How it Works:**   Documents are treated as sets of terms. Documents are ranked based on the probability of their relevance to a given query using the Binary Independence Model. It starts with initial estimates and refines with user feedback.
*   **Pros:** Formal probabilistic foundation, robust performance with user feedback, incorporates a notion of relevancy.
*  **Cons:**  Strong assumptions, lacks term dependency, relies on user feedback to achieve better performance.
  * **Most Important 20%:** *Provides a probabilistic approach, documents are ranked using relevance scores, user feedback is an important part of the model.*
*   **Core Concepts:**
    *   `P(R|D)`: Probability that document `D` is relevant to a query.
    *   `P(NR|D)`: Probability that document `D` is not relevant.
    *   `r_j`: Probability of term `t_j` being in a relevant document.
    *   `n_j`: Probability of term `t_j` being in a non-relevant document.
*   **Assumptions**
    *   Term frequency is not used, document representation is set-of-words.
    *   Terms are treated as statistically independent from each other.
    *   Absent terms in the query do not have an impact on ranking.
*   **Use Case:** Particularly suitable for scenarios where the user interaction is possible and the system can improve on its own, such as a personal document indexing system.

**2.5 BM25 Model**

*  **How it Works:** Builds on the Vector Space model, but incorporates a probabilistic framework, utilizing advanced term frequency saturation and document length normalization. It uses BM25 formula for scoring, enhancing the retrieval accuracy and relevancy of the results.
*  **Pros:**  Very effective at retrieval, addresses many shortcomings of previous models, can be fine-tuned using hyper-parameters, good overall model.
*  **Cons:** Heuristic weighting, lacks term dependency, can require adjustment based on search context.
 * **Most Important 20%:** *The main improvements over the Vector space model are that they introduce term saturation to avoid overemphasizing long and frequently appearing words in documents and introduces document length normalization to avoid penalizing shorter documents. They use TF and IDF values. Hyper-parameters such as k and b can be used to fine-tune retrieval.*
* **Key Elements:**
    *   **Term Frequency Saturation:**  A function used to dampen the impact of repeated terms by using a non-linear saturation, as to prevent keyword spamming. The formula is `tf_k = tf * (k+1) / (tf+k)`, where k is a parameter that control the saturation, usually between `[1,2]`.
    *   **Document Length Normalization:**  A formula that introduces a bias for short documents, by decreasing the term frequency. The BM25 formula is `tf_k(D) = tf_k / (1 + b*(|D| / avgdl))`, where `|D|` is the document's length, `avgdl` is the average document length and b is a length normalization parameter (values in range `[0,1]`).
    *  **IDF:** Uses the following idf function, which prevents negative idf values: `idf = log ((N - df + 0.5)/(df+0.5))`
*   **Use Case:** Widely used in popular search engines due to its overall robustness and solid performance.

**Part 3: Indexing**

**3.1 Inverted Index**

*   **How it Works:** A data structure that maps each term to a list of document IDs that contain that term. Instead of storing documents, the system stores terms and corresponding document IDs.
*  **Pros:** Enables fast document lookup and is widely used in real-world search solutions, minimizes time spent on processing irrelevant documents, efficient for sparse matrices.
*  **Cons:** Requires a vocabulary of terms, which can be a storage overhead, requires the overhead of maintaining the vocabulary.
    * **Most Important 20%:** *The inverted index data-structure stores documents per term, not terms per document.*
*   **Structure:**
   *   **Vocabulary:** The set of all unique terms.
   *  **Postings:** A list containing the document IDs that have a specific word.
 *  **Use Cases:** Foundational indexing technique for classical text retrieval models, used in modern search tools like Apache Lucene, Elasticsearch.

**3.2 Indexing using SQL Databases**

* **Concept:**  Instead of using a specialized search engine,  SQL databases can be employed to implement basic retrieval systems. The data structures are: `document`, `posting`, `vocabulary` and `query` tables. SQL statements are used for constructing inverted indexes and performing query retrieval and calculations.
   *   **Pros:** Utilizes existing SQL databases, can be used to make SQL based back-ends more searchable, can be good as a starting point.
   *  **Cons:** Performance is not on par with optimized systems like Lucene, SQL databases are not optimized for large text datasets, can be more complex to implement than specialized tools.

**3.3 Types of Indexing**
* **Document-at-a-Time (DAAT)**
     *  **How it Works:** Evaluates all queries for a single document by streaming documents from the inverted index. Each document's relevance score is calculated, and documents are added to a priority queue in order.
        *   **Pros:** Simple and fast method for simple retrieval.
        *  **Cons:** Does not work well when long posting lists are present.
* **Term-at-a-Time (TAAT)**
     *  **How it Works:** Evaluates all documents for one query term. For each term, the score for all documents that contain the term is increased. Then the next term is processed.
        *   **Pros:** Less complicated, and can evaluate any number of query terms.
        *   **Cons:** Does not retain the top k results, all documents need to be read and stored in a temporary dictionary.

**Part 4: Advanced Concepts**

**4.1 Fuzzy Logic Operators**
* **How it Works:** Use membership values and rules to provide a way of partial matching
    *   **Fuzzy Algebraic Operators:** `sim(A AND B) = sim(A) * sim(B)` and `sim(A OR B) = sim(A) + sim(B) - sim(A) * sim(B)`. Only works with two operands.
    *  **Fuzzy Set Operators:** `sim(A AND B) = min(sim(A), sim(B))` and  `sim(A OR B) = max(sim(A), sim(B))`. Works with multiple operands.
  * **Pros:** Enables partial matching using membership values.
 * **Cons:** Heuristic scoring,  lacks flexibility and can not be extended.

**4.2 Soft Boolean Operators:**
* **How it Works:**  Extends fuzzy operators with a tuning parameter between `[0,1]`. `sim(A AND B) = (1 − a) · min{sim(A), sim(B)} + a · max{sim(A), sim(B)}`, where  `a` controls the trade-off between `min` and `max`. Similarly, with OR operator,  `sim(A OR B) = (1 − β)· min{sim(A), sim(B)} + β• max{sim(A), sim(B)}`, where `β` controls the trade-off.
    *   **Pros:**  Similar to fuzzy set operators with the addition of a hyper-parameter that can be used to tune the behavior of the operators.
   * **Cons:**  Lacks  theoretical foundations.

**4.3 P-Norm Distance:**
* **How it Works:**  The p-norm provides a measure of how far a document vector is from the ideal vector, the query vector. `(sum((1 - sim(query, doc))^p) / K )^(1/p)`.
  *   **Pros:** Generalizes soft boolean operators, and can model a wide range of different behaviors.
   * **Cons:**  Has to be optimized by selecting different hyper-parameters.

**Part 5: Putting it All Together and the Big Picture**

**5.1 Relationships**

*   **Preprocessing and Models:** Preprocessing steps are crucial to any retrieval model, as they transform raw text to make it suitable for indexing and retrieval.
*   **Boolean Model and Other Models:** Serves as a building block for other models, providing exact matching, although the extended model can be seen as a way to improve on it.
*   **Vector Space and BM25:**  BM25 can be seen as an extension of Vector Space model by adding a probabilistic foundation, and thus a deeper theoretical approach.
*  **BIR model and User Feedback**: User feedback improves the relevance using probabilities.

**5.2 Choosing the Right Tool**
*   **Simple Boolean Retrieval:** Simple searches with clear criteria and no need for ranking.
*   **Extended Boolean Model:** Search systems requiring partial matches, but the boolean query language is still used.
*  **Vector Space Model:** General text search where good ranking and partial matching are needed.
*  **Probabilistic Retrieval (BIR):**  Text retrieval systems where user feedback can improve the relevance of the results.
* **BM25:** Best overall performance for full text search with good ranking and partial matching.
* **Lucene:**  A robust and widely used open-source search library, with support for all the models and algorithms described above.
* **Elasticsearch/OpenSearch/Solr:** Are enterprise-level systems with strong horizontal scaling capabilities that are built upon Lucene.

**5.3 Most Important 20% to understand 80% of the concepts:**
1.  The processing phase turns the raw text into searchable units (terms) using splitting, tokenization and stemming and stop-word removals.
2.  Terms are often weighted by their frequency in the document (TF) and across a whole collection (IDF).
3. Boolean Retrieval is an exact match algorithm, other models are often soft-matching algorithms that provide ranking.
4. Vector Space models use documents and queries as vectors with similarity measured by inner product or cosine.
5. Probabilistic models work by calculating the probability of document relevancy.
6. BM25 model is one of the best models and is based on tf-idf and length normalization and term saturation.
7. All models need an efficient indexing strategy to deliver results in a reasonable time.
8. The Inverted index is crucial for fast look-ups and performance optimization in most retrieval systems.
9.  In practical search solutions, the inverted index stores the terms with their respective documents, while storing all documents is not required.
10. Sharding is used in enterprise level solutions to improve performance and scaling.

**Part 6: Comparisons and Specific Examples**

**6.1 BIR vs. BM25**

*   **Foundation:** BIR is rooted in probability theory, calculating the likelihood of a document being relevant based on term presence in relevant and non-relevant documents using set-of-words (binary) document representation. BM25, on the other hand, builds on vector space principles, adjusting term frequencies and document lengths within a probabilistic framework using bag-of-words document representation.
*   **Term Weighting:** BIR assigns weights (`c_j`) by estimating the probability of a term appearing in relevant (`r_j`) and non-relevant (`n_j`) documents, often relying on user feedback. BM25 uses a combination of TF (with saturation), IDF (with an offset), and document length normalization.
*   **User Feedback:** User feedback is a core component of BIR, allowing it to iteratively refine relevance estimates. BM25 can adapt based on feedback but does not depend on it to function.
*  **Practicality:** BM25 is generally more practical and used more often as it does not require user feedback, while BIR needs the user feedback to improve its performance.
*   **Performance:** BM25 often outperforms BIR in practical applications as it addresses the challenges of overemphasizing long documents and frequent terms in documents from the beginning. BIR can achieve similar performance if enough feedback is provided.
*   **Complexity:** BIR is relatively more complex due to the iterative nature of weight refinement, whereas BM25 is simpler to implement once document statistics are pre-computed.
*  **Use Case:**
    *   **BIR** : Is good for systems where user interaction is present and a high level of customization is needed.
    *   **BM25** : Is a robust model for general purpose search systems, with good overall ranking quality and without the need for user feedback.
    *  **Example:** *In a system for a librarian to categorize books, BM25 would be a good starting point for indexing, but the BIR model might be suitable when the system allows for user feedback to further train the results.*

**6.2 Vector Space Model vs. BM25**

*  **Term Weighting:** While both models use TF and IDF, Vector Space uses them more directly, while BM25 uses term frequency saturation with the formula `tf_k = tf * (k+1) / (tf+k)`.
* **Document Length:** The Vector Space model does not explicitly handle document length normalization, instead rely on the cosine similarity. BM25 uses a complex normalization model based on the length of the document and the average document length to avoid favoring longer documents. `tf_k(D) = tf_k / (1 + b*(|D| / avgdl))`.
*   **Basis:** Vector space model is built on the idea that documents and queries are mathematical vectors and the relevance is measured through similarity. The BM25 model builds on top of vector space by adding a probabilistic approach to better fine-tune ranking.
*   **Complexity:** Vector Space is simpler to implement, but can be tuned through TF and IDF. BM25 includes more hyperparameters that give a finer level of control over the search relevance, but can also make it harder to find a perfect result.
*  **Performance** BM25 is generally superior because it implements techniques like term saturation and document length normalization.
*   **Use Case:**
    *   **Vector Space Model**: Is a good starting point with solid retrieval performance, it might need hyper-parameter tuning to work better.
    *   **BM25:** Is a more robust model, used as a benchmark for other models, is the go-to model for many full text search applications.
    *   **Example:** *A university library for the general public might prefer using BM25, whereas a simplified version might use a Vector Space model as it is easier to implement.*

**Part 7: Splitting, Stemming, and Lemmatization Techniques**

**7.1 Splitting Techniques (Revisited)**

*   **Fixed-Size Chunks:**
    *   **Example:** For a document "The cat sat on the mat. The dog barked loudly.", with a chunk size of 5 tokens, the first chunk would be "The cat sat on the", and the second "mat. The dog barked loudly.".
    *   **Pros:** Simple implementation, good if the system requires small uniform chunks.
    *   **Cons:** Can interrupt the logical flow of text.
*   **NLP-Based Chunks:**
    *   **Example:** Using the same text, the chunks would be "The cat sat on the mat.", "The dog barked loudly."
    *   **Pros:** Preserves sentence boundaries, offers a more semantically coherent unit than the fix-sized chunk approach.
    *   **Cons:** Requires more computational resources for NLP sentence extraction and may have inconsistent chunk size.
*   **Metadata-Based Chunks:**
    *   **Example:** *An HTML document where we use the tag `<p>` to separate paragraphs into chunks.*
    *   **Pros:** Easy to implement for structured data, the context of the paragraph is retained.
    *   **Cons:** Can result in different chunk sizes based on document structure.
*   **Semantic-Based Chunks:**
    *  **Example:**  Sentences like "The cat is on the table" and "the dog is also there" would be combined due to the similar topic they represent.
    *  **Pros:** Grouping sentences based on meaning.
    * **Cons** Is computationally expensive and might require a machine learning model.
*  **Use Cases:**
    *   **Fixed-Size Chunks:** Good if performance is the priority and a minimal overhead is acceptable, can be used in small datasets.
    *   **NLP-Based Chunks:** Best for situations where more context is needed for the text.
    *   **Metadata-Based Chunks:** Best to use if document format is structured, such as in books or PDF documents.
     * **Semantic-Based Chunks:** When context is the most important, and computational costs are not an issue.

**7.2 Stemming**

*   **Porter Stemmer:**
    *   **How it Works:** A rule-based algorithm that systematically removes suffixes based on a set of predefined rules, for example  `SSES -> SS`,  `IES -> I` and so on. The rules are applied in a cascading manner.
    *   **Example:** "computers," "computing," and "computerization" all stem to "comput".
    *   **Pros:** Efficient, widely used, and good results for English, and can be easily implemented.
    *   **Cons:** Can produce non-linguistic stems, which are not a real word. Can lead to over-stemming (e.g., "university" and "universe" to the same stem) or under-stemming (e.g., "good" and "better").
 * **Most Important 20%:** *The algorithm is a series of rules that are applied based on the measure of the word, that can be expressed in terms of consonants and vowels `[C](VC)m[V]`*
*  **Use Cases:** Good in general purpose applications where search speed and simple implementation is preferred, and over-stemming and under-stemming are acceptable.

**7.3 Lemmatization**

*   **How it Works:** Uses a dictionary and morphological analysis to reduce a word to its dictionary form, taking the context into account.
    *   **Example:** "better" becomes "good", "mice" becomes "mouse".
    *   **Pros:** Produces linguistic root words, handles irregular forms better than stemming, more semantically relevant results.
    *   **Cons:** More computationally expensive, needs dictionaries for each language.
*  **Use Cases:**  Is best when linguistic accuracy is important, such as question answering systems or chatbots.
   * **Example:** *In a system where the correct meaning is important, for example, a medical dataset, Lemmatization is preferred.*

**7.4 Stemming vs. Lemmatization**

*   **Stemming:** Faster but less accurate, good for general search.
*   **Lemmatization:** Slower but more accurate, good for applications where precision is critical.
*  **When to Use Which:**
    *   **Stemming:** Ideal for search engines and systems where speed is paramount.
    *   **Lemmatization:**  Use for applications requiring accurate natural language processing, like machine translation or information extraction.

**Part 8: Indexing Methods Revisited**

*   **Inverted Index Details:**
    *  **Postings Structure:** Postings can contain only document ids (for boolean models), or document ids with corresponding term frequencies, or also with term positions.
    *   **Set Operations:** Set intersection and union operations are implemented using the method of streaming sorted posting lists.
    *   **Optimization:** Postings lists are stored in compressed formats.
*   **SQL Based Indexing:**
    * **Advantages:**  SQL databases are readily available, do not require custom implementations, are good for smaller datasets.
    *  **Disadvantages:** Performance is lower than specialized systems, slow joins, SQL databases are not optimized for the operations in large text collections.
    *   **Use Case:** Useful to quickly test ideas, or add limited search capabilities to existing SQL databases.
*   **Document-at-a-time(DAAT) vs Term-at-a-time(TAAT):**
     *   **DAAT:**  Good for single term or phrase based search queries, where the document ranking is fast, because all terms for a document are evaluated at once.
    *   **TAAT:** A good approach when more than a few query terms are used, but can cause performance issues because all documents need to be evaluated for a specific term.
   *  **When to Use:** DAAT is preferred when documents need to be evaluated for their whole term, while TAAT is used when each term should be evaluated separately.

