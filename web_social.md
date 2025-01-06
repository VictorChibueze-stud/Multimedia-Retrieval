**1. Introduction**

Web retrieval is about finding relevant information on the internet in response to a user's query. It's more complex than just searching for keywords; it involves ranking pages based on their relevance and quality. Early web search was quite basic, but over time it has evolved into the sophisticated experience we have today using advanced techniques like PageRank and HITS algorithms.

**2. Core Concepts**

We'll examine the following core concepts:

*   **Basic Retrieval:** How search engines initially matched keywords.
*   **Additional Text Features:** Features enhancing retrieval based on more than just term occurrences.
*   **PageRank:** How link structure is used to determine importance.
*   **Hyperlinked-Induced Topic Search (HITS):** How hubs and authorities are found and used to enhance topical results.

**2.1 Basic Retrieval**

**Explanation:** In the early days, web search engines relied heavily on traditional retrieval techniques, similar to how document searches were conducted previously. These methods essentially involved indexing the terms on each webpage and matching these terms to the query terms. A simple example of this would be using a boolean search of "term1 AND term2" to find documents that contain both. Early methods relied heavily on the occurence of terms and didn't have sophisticated ranking models.

**Example:** If you searched for "cat videos," the engine would simply look for pages that contained the words "cat" and "videos". If a page contained these terms, it would be deemed relevant. However, this is a very simplistic approach, and doesn't consider context or quality.

**2.2 Additional Text Features**

**Explanation:**  As web search evolved, simple keyword matching wasn't sufficient.  Search engines started using additional text features to better understand and rank pages. These include:

   *   **Term Proximity:** Instead of just checking if terms occur, the engine checks if they occur closely together on a page. If you search for "white house" a document with the terms "white" and "house" right next to each other would be more relevant than one with them far apart.
   *   **Tag Weighting:** Search engines give more importance to terms that appear in certain HTML tags, such as the `<title>`, `<h1>`, or `<h2>` tags. A term in the title of a document is given more weight than one in a paragraph.
   *   **Anchor Text Boosting:** The text of links ("anchor text") pointing to a page is used as a description of that page. If many external links pointing to a page have the anchor text "University of Basel", then the term "University of Basel" is assigned additional weight on that target page.
   *   **Language and Location Awareness:** Search engines detect the language of a document and the user's location to provide localized results. If you are in Paris searching for "best pizza" the results will prioritize restaurants close to you.
   *   **Penalties and Blocks:** Search engines attempt to penalize or block low quality content to improve the search result quality. Low quality sites or sites employing "black hat" techniques will be de-prioritized.
    
**Example:** If you search for "University of Basel" the search engine won't just look for pages that have these words. It'll prioritize the pages where those words appear close together, in the title, and in the anchor text of other links pointing to the page, whilst also considering your location.

**2.3 PageRank**

**Explanation:**  PageRank is a way to measure the importance of web pages based on their link structure. It treats the web as a graph, where pages are nodes and links are edges. The key idea is that links from important pages count more towards a page's importance than links from less important pages. It is also a global measure, meaning it is not dependent on any specific query. PageRank is computed once using a crawl of the web graph and then applied to each query. It uses an iterative calculation based on random walks, where at each step the random "user" follows a link with probability alpha, or teleports to a random page with probability 1-alpha.

**Step-by-Step Calculation:**

1.  **Initialization:** Every page starts with a small PageRank value.
2.  **Iteration:**
    *   Each page distributes its PageRank to the pages it links to.
    *   The page collects PageRank from all the pages that link to it.
    *   The PageRank of each page is updated, reflecting this new contribution.
    *   This update is calculated as a mixture between a teleport step to a random page, and contributions from other pages weighted by the amount of outgoing links each page has.
3.  **Convergence:** The iteration stops when the PageRank values stabilize, and don't change much between iterations.

**Example:** If a webpage like the "University of Oxford" website has links from many reputable sources (e.g., BBC, CNN), it will have a high PageRank value. If a random blog has a link from only one page, its PageRank will be lower. The page rank is determined by both the quantity of incoming links and their quality (PageRank).

**2.4 Hyperlinked-Induced Topic Search (HITS)**

**Explanation:** HITS builds on the concept of hubs and authorities within a *specific topic*. Unlike PageRank, which is query-agnostic, HITS is query-specific. It attempts to identify:

    *   **Hubs:** Pages that link to many authorities on a specific topic. Think of a list of references on a Wikipedia article, it points to many authoritative sources of information.
    *   **Authorities:** Pages that are linked to by many hubs on a specific topic. Think of the Wikipedia articles themselves, they are referenced by many hubs.

**Step-by-Step Calculation:**

1. **Root Set Creation**: A query is used to obtain an initial top set of results.
2. **Base Set Expansion:** The initial results are expanded to include the pages that link to them, and the pages that they link to.
3. **Authority and Hub Score Calculation:** This process involves an iterative calculation:
    *  Each page starts with equal authority and hub values.
    * The hub values are updated by summing the authority scores of all the pages they link to.
    * The authority scores are updated by summing the hub scores of all the pages that link to them.
    * The authority and hub values are normalized in each iteration
    * This iteration continues until the hub and authority values converge.

**Example:** If you search for "climate change," a hub page might be a website with a list of links to well-known climate science organizations. The linked pages would be the authorities (e.g., NASA's climate change page, the IPCC website).

**3. Relationships**

*   **Basic Retrieval vs. Text Features:** Basic retrieval focuses only on matching terms, while text features enhance this matching by considering term proximity, tag importance, etc.
*   **PageRank vs. HITS:**  PageRank measures overall importance, while HITS is topic-specific and tries to identify the best hubs and authorities *within* that topic. PageRank is used as a global signal, whilst HITS is used to refine the topical relevance for specific queries. HITS will also generate different results for different queries, where PageRank won't.
*   **Text Features + PageRank + HITS:** These methods work together. Text features are used to find pages containing the search terms. PageRank is then used as a general importance boost, and finally HITS can be used to refine the top results based on topical authority.

**4. Applications and Use Cases**

These concepts are applied in:

*   **Web Search Engines:** Google, Bing, and other search engines use these (or very similar) techniques to rank and return results.
*   **Social Media Search:** Platforms like Twitter and Facebook use similar approaches for content retrieval.
*   **Recommendation Systems:** PageRank-like algorithms can be used to rank content or suggest relevant products.

**5. Challenges and Limitations**

*   **Spam and Manipulation:** Search engines need mechanisms to prevent manipulation of these scores via spamming or link farming.
*   **Query Ambiguity:**  It can be challenging to understand a user's intent with short, ambiguous queries. For example, the query "apple" may refer to the fruit or the technology company.
*   **Scalability**: Web data is constantly changing and constantly growing, search engines need to process the information efficiently and fast.

**6. Conclusion**

Web retrieval has come a long way from simple keyword searches. Modern systems utilize a variety of methods from proximity, tag-weighting, PageRank, and HITS to rank pages and provide more relevant results to users. These concepts, though complex, work together to enable us to easily access information on the web.

