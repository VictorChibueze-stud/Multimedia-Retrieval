**Part 1: Introduction and Chunking Text**

**1.1 Introduction**

*   **Goal of the Chapter**: To enhance classical text retrieval models by incorporating techniques from NLP and machine learning, focusing on advanced methods for tokenization, embeddings, and text classification, while exploring generative AI for improved user experience.
*   **Key Areas Covered**:
    *   Revisiting document splitting techniques.
    *   Exploring alternative tokenization methods.
    *   Fundamental NLP techniques: part-of-speech tagging, chunking.
    *   Dimensionality reduction techniques: Latent Semantic Indexing (LSI).
    *   Modern AI-based Embeddings.
    *   Text classification methods.
    *   Introducing Generative AI for enhanced retrieval.
*   **Significance:** The chapter bridges the gap between traditional text retrieval and modern techniques, setting the stage for more sophisticated information retrieval systems.

**1.2 Chunking Text: Revisited**

*   **Definition:**  The process of dividing documents into smaller, meaningful pieces (chunks) for indexing and retrieval. Unlike earlier simpler methods (e.g. splitting at fixed number of tokens), this chapter explores more context-aware techniques.
*   **Purpose:**
    *   Enables more precise retrieval by connecting the query to specific passages.
    *   Facilitates more effective text summarization.
    *   Provides context for sentiment analysis by detecting sentiment changes within documents.
    *   Prepares data for machine learning by dividing large documents into manageable inputs.
    *   Enables context-aware question answering through Retrieval Augmented Generation (RAG).
    *   Improves translation and natural language understanding.
*   **Common Use Cases and Their Needs:**
    *   **Large Documents**: Need splitting for targeted retrieval.  *Example:* splitting a novel for chapter specific search.
    *   **Text Summarization**: Paragraph-level summaries using smaller chunks. *Example:* summarizing a news article with paragraphs.
    *  **Sentiment Analysis**: Evaluate sentiment changes within documents using smaller segments. *Example:* review where sentiment changes between paragraphs.
    *   **Large Language Models**: Training using smaller, coherent input chunks. *Example:* providing context for training a generative text model.
    *   **Retrieval Augmented Generation (RAG)**: Context is important, and must be combined with the query in the prompt. *Example:* providing context from text collection to a LLM based chatbot.
    *   **Machine Translation:** Need word and sentence context when translating. *Example:* correct verb conjugation by considering the context of the sentence.
    *  **Natural Language Processing**: Chunks used for semantical analysis with word meanings dependant on context.  *Example:* Named Entity Recognition with larger text blocks.
*   **Types of Splitting Techniques:**
    *   **Fixed-Size Chunks:** Documents divided into chunks of constant word or character lengths.
        *   **How it Works:** Split document every *n* tokens or characters.
        *   **Pros:** Simplicity, uniformity, straightforward implementation.
        *   **Cons:** May split sentences/paragraphs abruptly, ignoring context.
        *  **Example:** splitting a text every 1000 characters (with a possible overlap of 200 characters.)
    *   **NLP-Based Chunks (Sentence Boundaries):**  Text is split based on sentence boundaries, often merging sentences until a minimum length is reached.
        *   **How it Works:** Identify sentences, merge them if they are smaller than a defined threshold.
        *   **Pros:** Preserves sentence context, more meaningful units for retrieval.
        *   **Cons:** Still not ideal because paragraphs can span more than one sentence, variations in chunk sizes.
        *  **Example:** sentences are separated using NLTK or spaCy tools and merged into chunks of 1000 characters minimum length.
    *   **Structural Splitting (Metadata):** Chunks are determined by structural elements like chapters, paragraphs, or sections.
        *   **How it Works:** Leverages document structure metadata, e.g. HTML or Markdown tags.
        *   **Pros:** Aligning chunks with structural divisions. Useful for books, PDFs and HTML documents.
        *   **Cons:** Depends on structural information availability, variance of chunk sizes across the documents.
        *   **Example:** splitting based on HTML tags `<p>` or `<div>`.
     *   **Semantic Splitting:** Sentences or paragraphs are grouped based on semantic similarities of their content.
           *  **How it Works:**  Machine learning models group parts of the text based on the meaning they contain.
           * **Pros:** Provides semantically related chunks.
           *  **Cons:** Computationally expensive.
           * **Example:** merging all sentences that belong to a specific discussion point.
   *  **Hierarchical Chunking:** Uses a combination of the above techniques. In particular for RAG, a larger document provides context and smaller documents provide specific retrieval.
         *   **How it Works:** Splits document into larger chunks, which are divided into smaller subchunks.
         *   **Pros:** Provides both a broader context and focused details for retrieval.
         *   **Cons:** More computationally expensive to implement.
        * **Example:** splitting a PDF document into pages, and each page into sentences.
    * **Most Important 20%**: *Choose a splitting method that fits the needs of the application at hand. Use overlapping for the fixed size splitting approaches. You can use structural approaches, when the document structure is known, semantic approaches when context is required.*
*   **Relation to Previous Chapter:** Extends splitting techniques mentioned in previous chapter using NLP and ML. The concept of splitting remains the same, but this chapter explores splitting based on content.

**Part 2: Tokenization Revisited**

**2.1 Basic Tokenization: Challenges**

*   **Review:**  Basic tokenization involves separating words by spaces and non-word characters.
*  **Limitations:**
    *   Fails to recognize possessives like “'s” properly.
    *   Struggles with numbers, percentages, and currencies, treating them as distinct tokens or breaking them apart.
    *   Abbreviatons such as "U.K." or "Dr." are not captured, nor are complex abbreviations.
    *   Omits or incorrectly interprets punctuations, leading to inaccurate sentence analysis.
    *   Does not account for different word forms and therefore needs methods like stemming or lemmatization.
*  **Example:** “I buy my parents’ 10% of U.K. startup for $1.4 billion. Dr. Watson's cat called Mrs. Hersley and it was w.r.o.n.g., more to come ...” is tokenized in a very ineffective way.
    *   **Why:** Punctuation and numbers are separated, possessives are omitted.

**2.2 Improved Tokenization with NLTK and spaCy**

*   **Overview:** Introduces word-based tokenizers that utilize machine learning to resolve issues with basic tokenization.
*   **Enhancements:**
    *   Better handling of possessives.
    *   Improved processing of numbers, percentages, and currencies.
    *   Correct interpretation of common abbreviations (including multi-dot ones).
    *   Preservation of interpunctuation for sentence analysis.
*   **Example:** Using NLTK or SpaCy produces more accurate tokens for the example from the previous section.

**2.3 Tokenization for Retrieval (Clean Up Actions)**

*  **Objective:** To create clean token lists by focusing on tokens that are important for text retrieval.
*   **Cleanup Steps:**
    *   Removing short tokens such as single letters
    *   Excluding non-word tokens, including special characters, most number and hyphens/dot abbreviations.
    *   Converting Unicode to ASCII characters to reduce the size of the vocabulary and improve matching capabilities.
    *   Applying case conversion to improve matching by disregarding capitalization.
    *  **Why**: Provides the essential elements for retrieval and reduces the storage needs for the vocabulary.
    *  **Example**: Removes numbers, special chars, and convert all tokens to lowercase.

**2.4  Addressing Complex Scenarios**

*  **Scriptio Continua:** Text without spacing, requires more complex models to detect token boundaries (e.g., Chinese). *Example:* a Chinese sentence using multiple words together without any whitespace.
*  **Programming Literals:** Variables that do not contain spaces, need to be broken into tokens using a programming language tokenizer. *Example:*  `camelCaseName` into `camel` and `Case` and `Name`.
*  **Spoken Language Transcriptions:** The transcribed words (phonemes) are not segmented, but are continuous. *Example:* a phoneme stream representing a sequence of spoken words.
*   **Methods for Complex Tokenization**
    *   **Dictionary-based + HMM/NN:** A hybrid approach, first looking into a dictionary, and then using a model to resolve ambiguities.
    *   **Sub-word based tokenization:** Creates tokens that are parts of words.
       *    **Pros**: Provides flexibility, resolves issues with OOV (Out-of-Vocabulary) words, creates smaller vocabulary.
       *    **Cons**: Can make the tokens less meaningful, requires models to interpret.
    *   **Examples:**
        *   A phoneme stream is split into overlapping sequences of phonemes.
        *  Words are split into parts (i.e. syllables or affixes)

**2.5 N-grams and Phrases**

*   **Definition:**  Combining multiple words into a single token (bi-grams for two, tri-grams for three).
*  **Objective**: to capture phrases with distinct meanings. *Example:* `"New York City"` has more meaning than individual terms.
*  **Naive Approach:**
    *   Extract all possible n-grams and add to vocabulary.
    *  **Limitations:** Many stop words (e.g. "of the") are combined and many infrequent n-grams.
    *  **Example:** "of the" has high frequency.
*  **Enhanced Approaches:**
    *   Filtering out n-grams that include stop words or single letter words.
    *   Use Pointwise Mutual Information (PMI) to highlight n-grams that occur more frequently together.
    *   Use Likelihood Ratio (LHR) to select the most relevant n-grams based on their relation to each other.
     * **Pros:**  Can highlight named entities that occur together, capture context.
    *  **Cons:** Vocabulary size increases exponentially, a lot of parameters need to be tuned.
    * **Most Important 20%**: *PMI highlights terms that co-occur more than would be expected, LHR looks at the probability of the co-occurrence.*

**2.6 Bridging the Gap to Machine Learning**

*   **Need for Numerical Representations:** Machine learning models require numerical input, not text, to handle the information.
*   **Challenges**: Mapping tokens to numerical values: A simple mapping can lead to meaningless relationship between values.
* **Solution**
   *   **One-Hot Vectors:**  Creating a vector with a length equal to the size of the vocabulary, where each token is represented with a 1 at its ID index.
        *  **Pros:**  Simple to implement.
       * **Cons:** Vectors are sparse and have no information about semantic proximity.
   *   **Embeddings Layers:** Transforms one hot vectors to a dense lower-dimensional space, learning semantic similarities between terms.
      *  **Pros:** Provides a useful transformation with the benefit of neural network structures, can learn semantic proximity.
        * **Cons:** Can add computational complexity.
    *   **Positional Encoding**: Positional encoding methods are used to inject information about the word position into embeddings.
       * **Pros:** Retains the sequential nature of language.
        *  **Cons**:  Can add additional complexity to the structure.

**Part 3: Summary (Part 1)**
* This part established the need for context-aware splitting and the need for a more robust tokenization method for complex use cases and machine learning, showing how to combine simple approaches for effective results.
* We have explored splitting techniques with a focus on their use cases.
* We have discussed improved tokenization with language specific tools.
* We have seen sub-word and character-based approaches and their necessity for large language models.
* We established a bridge between traditional tokenization with modern word embedding techniques using one-hot vectors and how embedding and positional encoding layers are used.

**Part 4: Lemmatization and Linguistic Transformation**

**4.1 Stemming Revisited**

*   **Rule-Based Stemmers:**
    *   **How They Work:** Transform words based on a predefined set of rules to map different word forms to the same stem.
    *   **Pros:** Fast and efficient, easy to implement, good for a range of languages.
    *   **Cons:** Can produce non-linguistic or incorrect stems, limited accuracy.
    *   **Example**: Porter stemmer (English).
        *  **Most Important 20%**: They are not necessarily linguistically correct, but they group words with same root. They also have a set of rules that are implemented in order.
*   **Dictionary-Based Stemmers:**
    *   **How They Work:** Use dictionaries and exception lists to map words to their root form. Often incorporate rules to account for minor regularities.
    *   **Pros:** More linguistically accurate, handles irregular inflections, creates actual words.
    *   **Cons:** Requires a dictionary for each language and has a larger computational complexity.
     *   **Example**: WordNet and spaCy based stemmers for English, German, and French.
*   **Snowball Stemmer Framework:**
    *   **How It Works:**  A framework for creating rule-based stemmers for multiple languages. It provides a language to define the rules and generate code.
    *   **Pros:** Supports a range of languages, efficient, and flexible rule definition.
    *   **Cons:** Still produces pseudo-stems.
*   **Key Idea:** Stemmers and lemmatizers are useful to group different word forms, even with the drawback of producing incorrect forms.

**4.2 Lemmatization**

*   **How it Works:** Reduces a word to its dictionary form, using part-of-speech tagging and morphology to derive the stem.
*   **Pros:** More linguistically accurate, handles irregular forms.
*   **Cons:** Computationally more expensive and needs a dictionary for each language.
*   **Example:** Reducing words like "better" to "good".
* **When to Use Which:**
    * **Stemming:** Good for general purpose search applications.
    *   **Lemmatization:** Best for applications with specific language needs or high accuracy requirements.

**4.3  Handling Compound Words**
*   **Definition:**  Words formed by combining two or more base words (e.g., 'skyscraper' or 'Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz').
* **Challenge:** Traditional methods treat them as one term, while they can be a combination of multiple meaningful terms.
* **Approaches:**
   * **Rule-based Splitting:** Split based on syllabification or hyphenation. *Example:* “must-have” is split into “must” and “have”.
    * **Morphological Splitting:**  Uses linguistic analysis to split compounds into root words. *Example:* "Wolkenkratzer" is split into "Wolken" and "Kratzer".
* **Classification of Compounds:**
   * **Endocentric Compounds:** Meaning is derived from its constituent parts (e.g., “sunglasses”).
   * **Exocentric Compounds:** Meaning is different from its constituent parts (e.g., “skyscraper”).
*  **Importance**: Combining splitting of compound words with regular terms enhances the vocabulary to deal with the complexity of many different languages, improving search performance.

**4.4  Pointwise Mutual Information (PMI) and Likelihood Ratios (LHR) (Revisited)**

*   **PMI:** Measures word associations by comparing observed co-occurrence frequency to expected frequencies.
    *   **How it Works**:  Compares the ratio of probability of words appearing together with the probability of the words appearing independently from each other.
        *   **PMI(t1,t2) = log( p(t1,t2) / (p(t1) * p(t2) ) )**
    *   **Pros**: Highlights unique bigrams, good for filtering less-significant pairs.
    *   **Cons**: Can be biased by infrequent terms, can lead to an uncontrolled vocabulary
    *  **Example**: Selecting meaningful word pairs like “New York” instead of “of the”.
*   **LHR:** Compares the probability of words co-occurring against the probability of words occurring independently by using maximum likelihood estimations.
    *   **How it Works**: Similar to PMI, but provides a more robust approach to sparse data.
    *   **Pros:** Can also be used to obtain meaningful bi-grams, less biased by infrequent terms than PMI.
    *  **Cons**: Requires a training step.
    *  **Example:** Highlights "Sherlock Holmes" over "I am" using Likelihood Ratios.
*   **Relationship:** Both PMI and LHR are used for bi-gram selection, but they use different techniques and have different outcomes.

**Part 5: Part of Speech Tagging (POS) and Named Entity Recognition (NER)**

**5.1 Part-of-Speech Tagging (POS)**
* **Definition:** The method to assign a grammatical class to each word in a text, based on its syntactic role and context.
*  **Purpose:**
     *   Enhances stop word filtering (e.g., retaining "it" as noun and removing as a pronoun).
     *   Improves query processing by identifying parts of a question.
     *   Facilitates understanding of sentence structure for further linguistic transformations.
*   **POS Tagging Methods:**
    *   **Rule-based tagging:** Employs hand-crafted rules to assign tags based on word context.
        *   **How it Works:** if word ends with "-ing" is often a Verb, gerund
        *   **Pros:** Simple, fast processing, minimal training data needed.
       * **Cons:** Limited accuracy, requires a lot of rules, language specific.
    *   **Stochastic POS Tagging (Hidden Markov Model):** Utilizes a model with hidden states that follow a markov chain, and output words, with associated probabilities. Viterbi algorithm is used to find optimal sequence of hidden states.
        *   **How it Works:** Applies probability to find a sequence of hidden states that maximizes the likelihood of producing a given sequence of tokens.
        *   **Pros:** Can be trained on a set of sentences and words, captures dependencies between words, performs well with different use cases.
       * **Cons:** Requires training data and is computationally more expensive than rule-based systems.
      * **Core Components**: Hidden States (POS tags), Observations (words), Transition Probabilities (probability of a tag following the current tag), and Emission Probabilities (probability of a word being generated by a certain tag).
    *   **Transformation-Based Tagging:** Iteratively corrects errors from a rule based initial tagging using other rules.
        *   **How it Works:** start with a simple rule-based tagger and then refine errors iteratively using new rules based on training data.
        *  **Pros**: combines rules with training data to improve performance.
        *   **Cons**: Can overfit training data.
    *   **Deep Learning POS Tagging:** Employs neural networks to learn complex patterns from training data.
         * **How it Works:** Uses neural networks, specialized for sequence data, such as transformer networks, to output part-of-speech tags based on the input sentence.
        * **Pros:** High accuracy, captures dependencies between words, can also adapt to different languages.
        * **Cons:** Requires training data, computational resources, and might not improve performance enough compared to other models.
 *   **Use Cases:** Useful in query processing, sentence structure analysis, and for other NLP tasks.
 *   **Most Important 20%**: *It assigns grammatical classes to words, by looking at their position and their function in a sentence. This is used to identify and resolve ambiguities.*

**5.2 Named Entity Recognition (NER)**

*   **Definition:** The method for classifying named entities in the text.
*   **Purpose:** To identify and classify named entities, including people, locations, organizations, etc., extracting the key aspects in the text.
*   **NER Tags:** Common categories include PERSON, LOCATION, ORGANIZATION, etc.
*   **Implementation**: Similar to POS tagging.
    *   Uses rules, machine learning algorithms, or deep neural networks,
    *   Can use language specific tools such as NLTK and spaCy.
   *   **Importance:** Can be used to direct searches to specific data, like people or locations.
*   **Use Cases:** Enhancing search accuracy for names, dates, and locations, and to extract structured data.

**Part 6: Latent Semantic Analysis (LSI)**

*   **Definition:** A method to understand semantic meanings by reducing document-term matrices to lower-dimensional representation, revealing latent topics.
*   **How It Works:**
     *  Uses Singular Value Decomposition (SVD) on document-term matrix to identify the core dimensions of the text collection.
     *   The SVD decomposes the matrix into three matrices: U, S, and VT. U matrix represents terms, S contains singular values that define the importance of latent topics, and VT represents documents.
   *   Reduces dimensionality by removing lower singular values and corresponding columns.
   *   Projects document and query vectors to the lower dimensional topic space.
   *   Calculates document relevancy by vector similarity in the reduced topic space.
*  **Key Elements**:
    * **Singular Value Decomposition (SVD):**  A way to decompose a matrix and reduce its dimensionality.
    * **Latent Topics:**  Dimensions in the reduced space represent abstract, underlying topics present in the data.
    * **Dimensionality Reduction:** By reducing the number of dimension, the method makes complex data easier to handle.
*   **Pros:** Captures semantic similarities, finds latent relationships, reduces data dimensionality.
*   **Cons:** Computationally expensive for very large collections, and needs recalculations for every new text, limited ability to integrate with inverted indexes.
     *  **Most Important 20%:** *It uses SVD on a document-term matrix, creating hidden topics and allowing semantic retrieval. Also it removes the need for manual term selection.*
*  **Use Cases:** Useful when large collections need to be semantically searched using a vector space model.

**Part 7: Embeddings**

**7.1 Embeddings vs LSI:**

*  **Difference from LSI:** Embeddings directly map words/phrases to a lower dimensional vector space, while LSI maps documents into topics.
*  **Advantages of Embeddings**:
   *  Can capture context by utilizing neural networks.
   *  Can work on subword levels.
   * Can work on entire documents with transformer networks.
   *  Training is much faster than in LSI.

**7.2 Word2vec, GloVe, and fastText**

*   **Overview:** Models that learn vector representations of words by exploring surrounding words.
*   **Word2vec**
   *  **Core Concept**: Uses context windows around words and uses self supervised learning to find vectors for words.
    *   **Skip-Gram Model:** Predicts the surrounding words from the given center word. *Example:* "red" and "apple" are both surrounding words to "apple".
    *  **CBOW Model:** Predicts the center word from the surrounding words. *Example:* "the apple is __" , "tastes" would be the missing center word.
    *   **Pros:** Can find contextually similar words, self supervised learning, very efficient.
     *   **Cons:** Ignores the order of words.
   * **Use Case:** General purpose word embedding model, a good starting point to learn more about text embeddings.
*   **GloVe**
    *   **How it Works:** Builds co-occurrence matrices from corpus statistics, which are used to train word vectors.
    *   **Pros:** Fast training, global overview of the document collection.
    *   **Cons:** Cannot generalize well to words that were not in the corpus.
*    **Use Case:** Useful when global relationship of the terms needs to be learned from a dataset.
*   **fastText**
    *  **How it Works:** Uses word pieces for sub-word tokenization and builds word embeddings using these sub-words.
    *   **Pros:** Handles OOV, and misspellings, and can be trained more quickly than word2vec.
    *   **Cons:** The vocabulary becomes larger, and the vectors are not directly understandable to a human.
    *   **Use Case:** More robust in edge cases or where spelling or OOV (Out Of Vocabulary) issues are expected.
    * **Most Important 20%**: They learn vectors from words by considering the contexts they appear in, and create a semantically aware vector space for words

**7.3 Transformer-Based Embeddings**

*   **Key Idea:** Uses a self attention method, which encodes the position and relationships of words using transformer encoder layers.
*   **Features:**
     *   Uses word pieces or byte pair encodings for tokens.
     *    Learns the semantic of the words using large number of transformer layers.
    *   Adds positional encoding for the model to better understand the order of tokens.
* **Process:**
     *  The input sentence is transformed into embeddings and position encodings.
     *   The output is used in different tasks depending on the model.
        *    For transformers, the output can be used for classification.
*   **Advantages:** Can capture complex and context specific relationships.
*   **Limitations:** Computationally heavy.
*   **Use Case:** Is the basis of modern Natural Language Processing and it's many different applications.

**7.4 Sentence Embeddings and How to Obtain Them**

*   **Token-Based Embeddings + Pooling:**  Compute embeddings for all tokens and aggregate them using a pooling layer (mean or max). *Example:* averaging all the word embeddings of the sentence.
    *   **Pros:** Easy to implement.
    *   **Cons:** Can loose information by not considering sentence level context.
*  **Encoder Output of BERT + Pooling:** Utilizes the contextualized embeddings from the BERT encoder (special [CLS] token or all tokens) and aggregates them using pooling. *Example:* pooling or using the special `[CLS]` token from the output layer of a BERT model to obtain sentence embeddings.
    *  **Pros:** captures richer features with sentence level context.
    *  **Cons:**  Can be sub-optimal because BERT was not trained for sentence embeddings.
*  **Sentence Transformers:** Uses a BERT model specifically trained for producing sentence vectors using a fine tuning process.
      * **How it works**: A  BERT model is finetuned with sentence pairs using a Siamese network and produces a new model which generates good sentence embeddings.
     *  **Pros**: Best method to generate sentence embeddings.
     *  **Cons**: Requires additional training and special model.
* **Cross Encoder:** Does not aim to embed, instead it is used for calculating similarity of sentences, by outputting a probability between 0 and 1.
      * **How it works:** Two sentences are input together and a model based on transformer is trained to rate the relation between them.
       * **Pros:** Produces highly accurate scores.
        * **Cons:** not applicable to tasks that require sentence embeddings.
*  **Most Important 20%**: *Sentence embeddings are essential to calculate sentence similarity and are the building block for many AI-based text retrieval tools.*

**Part 8: Text Classification**

* **Definition:** Assigning pre-defined labels to text documents or passages.
*   **Difference from Topic Modeling:** Classification relies on supervised learning with pre-defined classes while topic modeling is unsupervised and aims to discover hidden structures or topic in the text.
*   **Naïve Bayes for Classification (Revisited):**
     * **Use Case:** Highly accurate classification on limited input data.
    * **Pros:** Simple, fast and efficient, with limited training needs.
    * **Cons:** Relies on the independence assumption which is not true in real world scenarios.
*   **TextCNN:**
    *   **How It Works:** Uses embeddings and 1D convolution to extract patterns and generate classes.
    *   **Pros:** Effective for capturing local patterns, computationally efficient, good for many classification tasks.
    *   **Cons:** May not handle long-range dependencies as well as Transformers.
*   **Transformer-Based Classifiers:**
   *   **How it Works:** Uses pre-trained transformer models and tunes them for classification tasks.
   *   **Pros:** Captures long range dependencies, very effective for many different use cases.
    *  **Cons**: Computationally expensive, require lot of data for tuning.
   * **Most Important 20%**:  *Transformers provides state of the art results but require heavy computational power. TextCNN provides good results for many use cases with limited computational needs.*

**8.1 Classification by Prompt Engineering**

*   **How It Works:** Utilizes a large language model to extract classes from the given document based on prompt instructions.
*   **Pros**: Simple approach for many different use cases.
*   **Cons**: Output can be biased by training data, does not have access to external data sources.
   * **Example**: a prompt that categorizes text into different categories and outputs a JSON file.
   * **Most Important 20%**: *This approach can bypass the need to implement specific deep-learning models for classification.*

**8.2 Text Clustering**

*   **Definition:**  Unsupervised method to group similar documents.
*   **Methods:**
    *   **Dimensionality reduction:** Embeddings are visualized using techniques like PCA.
    *  **Clustering Algorithms:** such as K-means can be used to group data in clusters and then outliers are identified by measuring the distance to existing clusters.
*   **Importance:**  Identifies related documents, reveals structure, detects outliers in the data.

