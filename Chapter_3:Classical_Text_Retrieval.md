**Chapter 3: Classical Text Retrieval**

**3.1 Introduction**

*   **Boolean Retrieval Systems:**
    *   **Definition:** A retrieval system that uses Boolean logic (AND, OR, NOT) to determine document relevance. Documents either fully match the query criteria or do not.
    *   **Use Case:** Used when precise matching is required and the user needs to include or exclude terms. Early systems where only 'retrieval' is important and no ranking/scoring is required.
    *   **Example:** Searching for "apple AND NOT (macintosh OR iphone)". This query retrieves documents with "apple" but *not* "macintosh" or "iphone" .
    *   **Pros:** Simple to understand and implement, good for precise matching, allows control over results via Boolean operators.
    *   **Cons:** No ranking, can lead to either too many or too few results, does not handle partial matches, limited in expressing complex user needs.
    *  **The 20%:** The core concept is that documents are either fully matching or not matching. No notion of ranking of the results is available. This leads to Boolean logic with AND/OR/NOT as operators.
*   **Extended Boolean Model:**
    *   **Definition**: An extension to the standard Boolean model, incorporating term weighting and the notion of partial matches. It uses document representation as vectors (bag-of-words) and computes a score.
    *   **Use Case**: Used when partial matches and ranking are needed, instead of strict Boolean matching.
    *   **Example**: The query "cat AND dog" may return a document with just "cat" and gives it a low score and a higher score for document with "cat" and "dog".
    *   **Pros**: Supports partial matches, provides ranked results, uses term frequencies and inverse document frequency to compute the scores.
    *   **Cons**: Heuristic scores, complex query expressions may be difficult for users, lower quality compared to advanced models
    *  **The 20%:** Key extensions are partial matching and ranking. Uses vector representation and similarity scores in conjunction with Boolean operators.
*  **Vector Space Model:**
    *   **Definition**: Documents and queries are represented as vectors in a high-dimensional space where each dimension corresponds to a term, using TF-IDF to compute the vector components. Similarity is computed using vector algebra.
    *   **Use Case**: Most widely used model for ranking results.
    *   **Example**: A query for "cat" and a document vector with high "cat" weight is deemed more similar than a document with lower "cat" weight. The model is able to also return results that do not have all query terms (partial matching)
    *   **Pros**: Provides ranked results, handles partial matches, uses term weighting, and does not need strict Boolean logic
    *   **Cons**: The TF-IDF model is heuristic, the independence assumption might be flawed in some scenarios (synonyms/homonyms)
    *  **The 20%:** Documents and queries are vectors in a high-dimensional space. Similarity between the two is computed using cosine measure or inner vector product.
*   **Probabilistic Retrieval Models:**
    *   **Definition:** A family of models that use probability theory to estimate the likelihood of a document being relevant to a query.
    *   **Use Case:** Provides a principled approach to ranking results by modeling the probability of relevance.
    *   **Example:** Calculating the probabilities of a document being relevant given a query term.
    *   **Pros:** Strong theoretical foundation, handles partial matches.
    *   **Cons:** The assumptions are often not valid, more complex to implement, requires user feedback for effective learning.
    *  **The 20%:** It tries to formally establish the probability of a document being relevant for a given query, unlike the heuristic approaches from Boolean and vector space models.
*  **BM25 (Okapi Best Match 25):**
    *   **Definition:** A widely used ranking function based on a probabilistic framework and adapted from the vector space model.
    *   **Use Case:** An industry-standard model used in major search applications for ranking.
    *   **Example:** Returns highly relevant results by balancing various factors like term frequency, document length, and term specificity.
    *   **Pros:** High performance in practice, accounts for document length, term saturation, includes an inverse document frequency (IDF) component.
    *   **Cons**: Complex formula and heuristics, with limited theoretical backgrounds.
    *  **The 20%:** BM25 builds on vector space models and incorporates concepts from probabilistic models. Most of the adjustments are based on heuristic approaches to the base vector space model.

**3.2 Fundamentals**

*   **Offline Phase:**
    *   **Definition:** The pre-processing step where documents are analyzed, and features are extracted to create an index. This happens before any query is received.
    *   **Use Case**: Allows for efficient search by pre-computing information.
    *   **Example**: Creating an inverted index from a collection of documents.
    *   **Pros**: Faster search since the work is done before the queries are sent.
    *   **Cons**: Requires time to compute the index.
    *  **The 20%**: Pre processing of documents by feature extraction and storing in an efficient index is critical.
*   **Online Phase:**
    *   **Definition:** The phase where a user query is processed and matched against the pre-built index to retrieve relevant documents.
    *  **Use Case**: Serves user requests by querying the index and returning ranked documents.
    *   **Example**: Responding to user query "red car" by querying the index and retrieving the most relevant documents
    *   **Pros**: Responds to user queries.
    *   **Cons**: Can be slow if indexes are not setup correctly.
    *  **The 20%**: Serves the users and queries the index created in the offline phase to provide results.
*   **Feature Extraction:**
    *   **Definition:** The process of selecting and extracting relevant information from a document that will form the basis of the index.
    *   **Use Case:** To reduce the document to manageable components for storage and retrieval.
    *   **Example:** Breaking down a text document into individual words or n-grams
    *   **Pros**: Reduces the search space to smaller components.
    *   **Cons**: Has to be done correctly as this will determine the search space.
    *  **The 20%**: A process that extracts manageable components from documents to form the basis of the index.
*   **Inverted File Index:**
    *   **Definition:** An index structure that maps each term to the list of documents containing that term. Efficient in accessing document lists for each term.
    *   **Use Case**: To provide a list of document IDs for a given query term.
    *   **Example**: An index entry: "cat -> \[doc1, doc3, doc5]".
    *   **Pros**: Allows faster access to documents that contain query terms, saves space and enhances retrieval speed by only working with the relevant documents, as opposed to all.
    *   **Cons**: Needs to be computed and stored.
    *  **The 20%**: Key innovation in search systems that makes searching extremely fast. Provides document lists for query terms.
*   **Retrieval Status Value (RSV):**
    *   **Definition:** A score calculated by retrieval models to assess the relevance of a document to a query (typically for the extended Boolean model and Vector space model).
    *  **Use Case**: Provides a mechanism to rank the documents based on the matching level with the query.
    *   **Example**: A document score of 0.9 is more relevant to a given query compared to a document with a score of 0.5.
    *   **Pros**: Allows for ranking of documents and selection of the most relevant results.
    *   **Cons**: May not be accurate if the underlying scoring model is inaccurate.
    *  **The 20%:** Defines how relevant a document is. Each retrieval algorithm uses its own methods to compute RSV.

**3.2.1 Step 1: Extract**

*   **Text Extraction (from HTML):**
    *   **Definition:** The process of removing structural information and tags from documents (such as HTML), isolating the text content and the metadata.
    *   **Use Case**: Enables processing of raw text without interference from markup language.
    *   **Example**: Removing the HTML tags to retrieve the text in `<p>This is a paragraph</p>`.
    *   **Pros**: Enables indexing of actual text in source documents.
    *   **Cons**: Can miss out on relevant structural information from markup (for example emphasis, formatting etc).
    *  **The 20%:** The process of separating metadata and the actual text in HTML and other types of documents.

**3.2.2 Step 2: Split**

*   **Document Splitting:**
    *   **Definition:** The process of dividing a large document into smaller, more manageable chunks for better indexing and relevance assessment.
    *   **Use Case**: Enhances retrieval precision, particularly for large documents.
    *   **Example**: Breaking down a book into chapters, paragraphs, or sentences.
    *   **Pros**: Higher precision due to shorter documents.
    *   **Cons**: Can create too many entries.
    *  **The 20%:** Large documents should be split into smaller units to enhance precision (context is very important here).

**3.2.3 Step 3: Tokenize**

*   **Tokenization:**
    *   **Definition:** The process of breaking down text into individual tokens (usually words or characters).
    *   **Use Case**: Provides the basic unit for indexing and searching.
    *   **Example:** Splitting "Hello world!" into "Hello" and "world".
    *   **Pros**: Allows for representation of the document as a list of tokens.
    *   **Cons**: Requires care to handle compound words, word variations, special characters, numbers etc.
    *  **The 20%:** Turns text into a sequence of tokens which are used to create the index.
*   **Stemming:**
    *   **Definition:** The process of reducing inflected words to their base or root form. Reduces tokens to similar forms to enhance matching between queries and documents.
    *   **Use Case**: Improves recall by matching different forms of a word (e.g., "running" and "ran" become "run").
    *  **Example:** Reducing "fishing", "fished", and "fisher" to "fish."
    *  **Pros**: Improves search recall, reduces the vocabulary size, groups similar words together.
    *  **Cons**: Can be too aggressive, grouping non-related terms, also can miss important variations in the text.
    *  **The 20%:** Groups similar words into the same root form to enhance recall.
*   **N-grams:**
    *   **Definition:** The sequence of 'n' tokens extracted from the text.
    *   **Use Case**: For detecting phrases or for search applications where we may want to detect a sequence of tokens.
    *   **Example**: The 2-gram of "hello world" is 'hello world'.
    *   **Pros**: Captures multi-word expressions.
    *   **Cons**: Increases vocabulary size, may be computationally intensive, may contain non-relevant sequences
    *   **The 20%:** Extracts sequences of tokens from the text to detect phrases.
*   **Lemmatization:**
     *   **Definition:** Reducing words to their dictionary form (lemma).
    *   **Use Case**: Enhances recall and precision by matching words to their root form. More accurate version of stemming, but not as widely used.
    *   **Example:** Reducing "better" to "good."
     *   **Pros**: More accurate compared to stemming,
    *   **Cons**: Computationally complex, language dependent, needs dictionaries.
     *  **The 20%**: Reduces words to their dictionary form as a more accurate version of stemming.

**3.2.4 Step 4: Summarize**

*   **Inverse Document Frequency (IDF):**
    *   **Definition:** A metric measuring how unique a term is across the collection of documents. (Higher idf implies the term is very unique).
    *   **Use Case:** Used to reduce the weights of common terms and boost the weights of rare or unique terms.
    *   **Formula:** idf(t) = log(N/df(t)), where N is the total number of documents and df(t) is document frequency of term t
    *   **Example:** The idf of "the" will be lower compared to "jaguar".
    *   **Pros:** Reduces influence of common words, increases influence of rare words
    *   **Cons:** Can be biased with certain datasets.
    *  **The 20%:** Key metric in weighting of terms. Reduces the effect of common words and increases the effect of unique words.
*   **Stop Words:**
    *   **Definition:** Commonly occurring words that are not informative for retrieval (e.g., "the," "a," "is").
    *   **Use Case:** Typically removed from the document to reduce the size of the index, and to reduce noise.
    *   **Example**: Removing words like "the", "a" from the index and queries.
    *  **Pros**: Reduces index size, can potentially improve performance.
    *   **Cons**: May miss out on specific use cases (e.g. "To be or not to be" cannot be searched).
    *  **The 20%**: Common terms are removed to enhance the effectiveness of retrieval.
*  **Zipf's Law:**
    *   **Definition**: A principle that the frequency of an element is inversely proportional to its rank. For example the most common word will occur twice as much compared to the second most common word. This also applies to term occurrences.
    *   **Use Case**:  Used to inform term selection strategy by observing the distribution of terms.
    *  **Example:** In any language, words like "the" are most common and occur much more frequently compared to other words.
    *   **Pros**: Help create automated stopword lists without the need for manual creation.
    *   **Cons**: Distribution can change and requires to be computed for individual use cases.
     *  **The 20%**: Helps automate stopword lists by observing distribution of terms.
*  **Discriminative Power of Terms:**
     *   **Definition**: The ability of a term to distinguish relevant and non-relevant documents. It measures if a term is a strong indicator of document content, or it occurs everywhere.
     *   **Use Case:** Used to select appropriate weights for terms in document representation.
    *   **Example**: The term "the" is present in almost all the documents, therefore its discriminating power is low. A unique technical term will have a high discriminating power.
    *   **Pros**: Helps in selecting relevant terms and removing noise from the text.
    *   **Cons**: The approach can miss specific aspects.
    * **The 20%**: Ability of a term to distinguish relevant documents from non-relevant ones.

**3.3 Text Retrieval Models**

*   **Term Frequency (tf):**
    *   **Definition:** The raw count of how many times a term appears in a document.
    *   **Use Case:** Measures the importance of terms inside the document.
    *   **Example:** The number of times "cat" is in "the cat sat on the cat". tf = 2
    *   **Pros:** Simple metric.
    *   **Cons:** Does not take into account the length of the documents.
    *   **The 20%:** Shows how often a term occurs within a document.
* **Document Length**
    *  **Definition**: Number of tokens in a document.
    *  **Use Case**: To understand if a longer document will be ranked higher, we need to use document length.
    * **Example**: A longer document with the same term frequency as a shorter document should not automatically be ranked higher compared to shorter documents.
    * **Pros**: Helps in normalizing term frequencies.
    * **Cons**: Needs to be computed for each document.
     * **The 20%**: Length of a document influences how term frequency is interpreted.

**3.3.1 Standard Boolean Model**
*(Covered in the introduction, but summarizing for clarity)*

* **Standard Boolean Model**: (Same as in Introduction)
  *   **Definition:** A retrieval system that uses Boolean logic (AND, OR, NOT) to determine document relevance. Documents either fully match the query criteria or do not.
    *   **Use Case:** Used when precise matching is required and the user needs to include or exclude terms. Early systems where only 'retrieval' is important and no ranking/scoring is required.
    *   **Example:** Searching for "apple AND NOT (macintosh OR iphone)". This query retrieves documents with "apple" but *not* "macintosh" or "iphone" .
    *   **Pros:** Simple to understand and implement, good for precise matching, allows control over results via Boolean operators.
    *   **Cons:** No ranking, can lead to either too many or too few results, does not handle partial matches, limited in expressing complex user needs.
      *  **The 20%:** The core concept is that documents are either fully matching or not matching. No notion of ranking of the results is available. This leads to Boolean logic with AND/OR/NOT as operators.

**3.3.2 Extended Boolean Model**
 *(Covered in the introduction, but summarizing for clarity)*
*   **Extended Boolean Model:** (Same as in Introduction)
    *   **Definition**: An extension to the standard Boolean model, incorporating term weighting and the notion of partial matches. It uses document representation as vectors (bag-of-words) and computes a score.
    *   **Use Case**: Used when partial matches and ranking are needed, instead of strict Boolean matching.
    *   **Example**: The query "cat AND dog" may return a document with just "cat" and gives it a low score and a higher score for document with "cat" and "dog".
    *   **Pros**: Supports partial matches, provides ranked results, uses term frequencies and inverse document frequency to compute the scores.
    *   **Cons**: Heuristic scores, complex query expressions may be difficult for users, lower quality compared to advanced models
    *  **The 20%:** Key extensions are partial matching and ranking. Uses vector representation and similarity scores in conjunction with Boolean operators.
*   **Fuzzy Algebraic, Fuzzy Set and Soft Boolean Operators:**
    *  **Definition**: Different ways to combine similarity scores in an Extended Boolean Model to generate the final document score
    *  **Use Cases**: These are different heuristic approaches to computing a score for partial matches in Boolean queries
    *   **Pros**: Allows use of scores in Boolean logic
    *   **Cons**: Heuristic and does not have a strong theoretical background
    *   **The 20%**: The different combination methods for the similarity scores of a document in extended boolean model.
*   **P-Norm Model:**
    *   **Definition:** A scoring approach that uses the p-norm distance for calculating document scores.
    *  **Use Cases**: Another alternative approach to computing a similarity between a document and the query using p-norm which uses p as a hyperparameter to define document similarity.
    *   **Pros**: Allows for flexibility in computation of similarity between documents and queries.
    *   **Cons**: Computationally intensive.
    *  **The 20%**: Another method for computing score in extended boolean model using hyperparameterized distance function.

**3.3.3 Vector Space Retrieval**

*(Covered in the introduction, but summarizing for clarity)*

*   **Vector Space Model:** (Same as in Introduction)
    *   **Definition**: Documents and queries are represented as vectors in a high-dimensional space where each dimension corresponds to a term, using TF-IDF to compute the vector components. Similarity is computed using vector algebra.
    *   **Use Case**: Most widely used model for ranking results.
    *   **Example**: A query for "cat" and a document vector with high "cat" weight is deemed more similar than a document with lower "cat" weight. The model is able to also return results that do not have all query terms (partial matching)
    *   **Pros**: Provides ranked results, handles partial matches, uses term weighting, and does not need strict Boolean logic
    *   **Cons**: The TF-IDF model is heuristic, the independence assumption might be flawed in some scenarios (synonyms/homonyms)
    *  **The 20%:** Documents and queries are vectors in a high-dimensional space. Similarity between the two is computed using cosine measure or inner vector product.
*   **Inner Vector Product:**
    *   **Definition:** A method for calculating document-query similarity by multiplying corresponding components of their vectors and adding them. It gives a raw score without restricting the values to a range.
    *   **Formula:** `sim(Q, Di) = q . d`
    *   **Use Case**: Ranking documents by computing a similarity score to the query (can be used to compute top-k results).
    *   **Pros**: Simple to implement and efficient in computation.
    *   **Cons**: Not confined to any given range (not a probability) and might give incorrect results for documents with large term frequencies.
    *   **The 20%:** A dot product between query and document vectors, computed by simple multiplication and summation.
*   **Cosine Measure:**
    *   **Definition:** A method of computing the similarity between a document and a query based on the angle between their vectors.
    *   **Formula:** `sim(Q, Di) = (q.d)/(||q|| * ||d||)`
    *   **Use Case:** Measures how well a document is aligned with the query by computing the angle between them.
    *   **Pros**: Provides more accurate results, accounts for document lengths, provides scores in the range of \[0,1]
    *   **Cons**: Can be computationally expensive compared to inner product.
    *   **The 20%:** Similarity of a document is calculated using the cosine of the angle between document and query vectors.

**3.3.4 Probabilistic Retrieval**

*(Covered in the introduction, but summarizing for clarity)*

*   **Probabilistic Retrieval Models:** (Same as in Introduction)
    *   **Definition:** A family of models that use probability theory to estimate the likelihood of a document being relevant to a query.
    *   **Use Case:** Provides a principled approach to ranking results by modeling the probability of relevance.
    *   **Example:** Calculating the probabilities of a document being relevant given a query term.
    *   **Pros:** Strong theoretical foundation, handles partial matches.
    *   **Cons:** The assumptions are often not valid, more complex to implement, requires user feedback for effective learning.
    *  **The 20%:** It tries to formally establish the probability of a document being relevant for a given query, unlike the heuristic approaches from Boolean and vector space models.
*   **Binary Independence Model (BIR):**
    *   **Definition:** A simple probabilistic model for information retrieval based on multiple independence assumptions.
    *   **Use Case:** Establishes the foundation for other probabilistic models.
    *   **Example:** Documents are represented as binary vectors (only term presence or absence matters), with query-specific term relevance probabilities calculated from user feedback.
    *   **Pros:** Establishes a simple probabilistic approach to information retrieval, good for understanding underlying concepts.
    *   **Cons:** Assumes complete independence, document length is not considered, term frequencies are ignored, relies on user feedback.
    *   **The 20%**: Simple probabilistic model that relies on several independece assumptions, and uses feedback to update the model. It lays the foundation for more advanced models like BM25.

**3.3.5 Okapi Best Match 25 (BM25)**
 *(Covered in the introduction, but summarizing for clarity)*
*   **BM25 (Okapi Best Match 25):** (Same as in Introduction)
    *   **Definition:** A widely used ranking function based on a probabilistic framework and adapted from the vector space model.
    *   **Use Case:** An industry-standard model used in major search applications for ranking.
    *   **Example:** Returns highly relevant results by balancing various factors like term frequency, document length, and term specificity.
    *   **Pros:** High performance in practice, accounts for document length, term saturation, includes an inverse document frequency (IDF) component.
    *   **Cons**: Complex formula and heuristics, with limited theoretical backgrounds.
    *  **The 20%:** BM25 builds on vector space models and incorporates concepts from probabilistic models. Most of the adjustments are based on heuristic approaches to the base vector space model.
*   **Term Saturation**:
    *  **Definition**: The effect of excessively frequent terms not contributing to the overall score.
    *   **Use Case**: The method is used to reduce the effect of long document lengths and common words.
    *   **Example:** Using tf^k to saturate term frequency based on a hyperparameter k to control how fast tf increases with term frequency.
    *   **Pros**: Improves the ranking by reducing the influence of overly frequent terms.
    *   **Cons**: Needs parameter tuning and the saturation point varies.
    * **The 20%**: Term frequencies are saturated to improve the document scores
*   **Document Length Normalization**:
    *  **Definition**: Taking into account document length and normalizing the term frequencies to balance long and short documents.
    *  **Use Case**: Used to avoid scenarios where long documents are favored by having more term frequencies.
    * **Example**: Term frequencies of documents are scaled based on the document length using the average document length.
    *   **Pros**: Improves scores by normalizing term frequencies with respect to document lengths.
    *   **Cons**: Needs a hyperparameter and document length to perform adjustments
    *   **The 20%**: Documents with different lengths should be treated fairly using document length normalization.

**3.4 Indexing Structures**

*   **Inverted Index:**
    *   **Definition**: (Same as earlier): An index structure that maps each term to the list of documents containing that term. Efficient in accessing document lists for each term.
    *   **Use Case**: To provide a list of document IDs for a given query term.
    *   **Example**: An index entry: "cat -> \[doc1, doc3, doc5]".
    *   **Pros**: Allows faster access to documents that contain query terms, saves space and enhances retrieval speed by only working with the relevant documents, as opposed to all.
    *   **Cons**: Needs to be computed and stored.
    *  **The 20%**: Key innovation in search systems that makes searching extremely fast. Provides document lists for query terms.
*  **Document-at-a-Time (DAAT)**:
    *   **Definition:** A search approach where we process each document to compute the score. This needs sorted lists of document ids for each term
    *   **Use Case**: Implements Boolean OR queries, and allows for fast retrieval of top-k results.
    *   **Pros**: Streamline processing, efficient retrieval of top-k results, fast to compute.
    *   **Cons**: It needs sorted lists of document ids
    *  **The 20%:** Process documents by a stream of document ids.
*   **Term-at-a-Time (TAAT):**
   *   **Definition:** A search approach where we process each term in the query to compute the scores. The scores for each term are added to the document scores.
    *   **Use Case**: Implements Boolean OR queries, provides a list of documents at the end.
    *   **Pros**: Streamline processing.
    *   **Cons**: Might need large space as we need to keep track of all documents.
    *  **The 20%**: Process terms in the query sequentially and updating the scores for each document.
*   **Inverted File with Document Lengths (Vector Space Model/BM25):**
    *   **Definition:** Storing document lengths with the document ids (TF-IDF scores), instead of just the list of documents.
    *  **Use Cases**: Enables computation of Vector Space Models and BM25 scores which utilize document lengths and document frequencies.
    *   **Pros**: Enables efficient computation of all vector space retrieval models.
    *   **Cons**: Increases storage costs.
    *  **The 20%**: Storing document lengths in conjunction with document ids and term frequencies makes vector space and bm25 models practical.
*   **Term Dependency and Term Proximity** (Mentioned but not fully defined in this chapter)
   *   **Definition:** Proximity: Closeness of terms to each other. Term Dependence: Identifying dependent/correlated terms
   * **Use cases**: Provides more accurate results based on phrases, multi-word expressions and identifying similar terms (for example synonym recognition).
    * **Pros:** Improve semantic relevance of results.
    * **Cons:** Can be computationally intensive.
      *  **The 20%**: Detecting multi-word expressions and synonyms will enhance the effectiveness of the search system.

**3.4.4 Inverted Files Implementation with SQL**
    *   **Definition**: The different SQL code implementations to build and query the database for different retrieval techniques
     * **Use Cases:** A step by step example to see how to implement inverted files using SQL.
    *  **Pros**: A practical overview of how to implement inverted files with SQL
    *   **Cons**: Can be overwhelming with the numerous SQL code examples
    * **The 20%:** Provides a practical overview of a traditional implementation of an information retrieval system.

**3.5 Lucene - Open Source Text Search**
 *   **Definition**: A robust open source framework for building search applications
    *   **Use Cases**: Used to create scalable and performant information retrieval systems
    *   **Pros**: A wide range of features and a strong and active community behind it.
    *   **Cons**: Steep learning curve.
    * **The 20%**: Powerful library which is the basis of many industrial solutions.
*   **Lucene Analyzer:**
    *   **Definition:** Classes in Lucene that define how text is tokenized, stemmed, and filtered before indexing. It performs a tokenization and a linguistic transformation of text.
    *   **Use Case:** Prepares document content for indexing and searching.
    *   **Example:** The StandardAnalyzer and EnglishAnalyzer.
    *   **Pros:** Highly customizable.
    *   **Cons:** Needs to be well-configured for each use case.
    *    **The 20%**: The building block of any Lucene based system for preparing text for search.
* **Lucene Field Type**
    * **Definition**: Specifies how the document's contents should be indexed, stored, and retrieved.
   *  **Use Cases**: Defines how different parts of the documents are to be treated.
    *   **Pros**: Allows for powerful index strategies and separation of text and metadata.
    *   **Cons**: Needs to be chosen wisely as there can be many different field types.
     * **The 20%**: Enables precise definitions for each part of the document.
*  **Lucene IndexWriter**:
    *   **Definition**: Class that is responsible for index creation and document insertion in Lucene.
    *   **Use Case:** Used to create an index and add content to it.
    *   **Pros**: Highly efficient and well-structured indexing capabilities.
    *   **Cons**: Can lead to index corruption if not implemented correctly.
    * **The 20%**: The core component of adding/removing and updating documents in an index.
*   **Lucene IndexSearcher:**
    *   **Definition:** Class that is responsible for searching and retrieving document from an index.
    *   **Use Case:** Used to search and query an index built using Lucene.
    *   **Pros:** Simple interfaces and powerful querying language.
    *   **Cons:** Needs well-formed query for complex searches.
    *  **The 20%:** The main method for searching the index, utilizing a query.
*   **Lucene Query Parser**:
    *   **Definition:** Classes used for parsing user queries in to their respective component of Lucene.
    *   **Use Case**: Translates user query into a series of search actions
    *   **Pros**: User friendly and powerful syntax for searching.
    *   **Cons**: Needs to understand the syntax for more advanced search scenarios.
    * **The 20%**: Translates the users query into Lucene's internal operations.

**3.5.1 Apache Solr, Elasticsearch, and OpenSearch**

*   **Sharding:**
    *   **Definition:** Distributing an index across multiple machines for scalability and redundancy.
    *   **Use Case**: Used to overcome index limits and enhance parallel query processing and scalability
    *   **Example**: An index of 100GB may be divided into 5 shards, each residing on a different machine.
    *   **Pros**: Increased performance for queries and enables scalability
    *   **Cons**: Requires extra infrastructure and management
    *   **The 20%:** Distributing the index across multiple machines to enhance the scalability.
*   **Replication:**
    *   **Definition:** Creating redundant copies of shards for better availability and failure tolerance.
    *   **Use Case**: Used to ensure search is working even if one or more server fails.
    *  **Example**: A shard with one replica means that one copy of the shard is available in case of a failure.
    *   **Pros**: Higher availability.
    *   **Cons**: Extra storage costs.
    *   **The 20%**: Copy of data in the index to ensure availability.
*  **Segment-Based Architecture**:
   * **Definition**: Core structure of Lucene that divides the index into segments.
    * **Use Case**: Supports parallel indexing and search by having smaller segments.
    *  **Pros**: Highly optimized and performant.
    * **Cons**: Involves complexities with document deletions and updates.
      * **The 20%**: The foundation for building a search index by dividing them into smaller units.
*  **Geoproximity Policy**:
    *  **Definition:** The concept of routing search requests based on the geographic locations of the users.
    *  **Use Cases:** Improves performance of search system by bringing the search nodes closer to the users geographically.
    *  **Pros**: Reduces network latencies and improves performance.
    *  **Cons**: Requires geographic location data of users.
      * **The 20%**: Routes user requests to the closest geographical region.

**3.6 Literature and Links**

*   This section lists the key resources (text books, articles and libraries) for further studies.

**Relationships:**

*   **Boolean Retrieval Systems vs. Extended Boolean Model:** The Extended Boolean Model improves upon the Boolean Model by adding support for ranking and partial matches, rather than just strict matching based on Boolean rules.

*   **Vector Space Model and BM25:** BM25 can be seen as an extension of the vector space model, with refined term weighting. The vector space model forms the base for BM25.

*   **Tokenization, Stemming, Lemmatization:** Tokenization provides the basic tokens, stemming and lemmatization then normalize those tokens by reducing to root words.

*   **Inverted Index and DAAT/TAAT:** The inverted index allows document access in the DAAT/TAAT models. DAAT and TAAT are different algorithms used to traverse the inverted index.

*   **Lucene Analyzer, FieldType, IndexWriter, IndexSearcher**: These are Lucene classes working together to create index, query, and tokenize text.

*   **Sharding and Replication**: Sharding provides scalable access to the index by dividing it into smaller chunks. Replication provides redundancy to enhance reliability and availability.

**Examples in the Lecture Notes (Rephrased for Intuitive Understanding)**

*   **Stop Words Example**: In searching for "the history of cats," 'the' and 'of' do not offer relevance and may be discarded using a stop word list.

*   **Stemming Example**: When searching for "runs," the stemming can reduce the term to "run," also matching results containing the word "running".

*   **Tokenization Example**: the query "New York" is tokenized to 'New' and 'York' and the n-gram to 'New York'.

*   **Term-Document matrix**: Using the example
