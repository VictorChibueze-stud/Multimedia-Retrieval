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

Let's walk through a simple scenario of how a web search engine might process a query, bringing together all the techniques we've discussed. We'll use a real-world example to make it clear.

**Scenario: Searching for "Best Pizza in Paris"**

Let's say you're in Paris and you're craving pizza. You open your browser and type "Best Pizza in Paris" into the search bar. Here's how the search engine might handle this query, step-by-step:

**1. Initial Query Processing**

*   **Input:** You type "Best Pizza in Paris".
*   **Basic Retrieval:**
    *   The search engine starts by looking for web pages that contain the words "best," "pizza," and "Paris."
    *   It creates an initial list of pages that contain these keywords. Let's say it finds 10,000 pages, all containing the terms, but in various combinations and contexts.
*   **Output:** A large list of web pages with at least some mention of the query terms.

**2. Applying Additional Text Features**

Now, the search engine uses more advanced text features to refine that list of 10,000 pages to something more manageable and relevant.

*   **Term Proximity:**
    *   The engine checks how close the words "best," "pizza," and "Paris" are on each page.
    *   It gives a higher score to pages where these words are close together. For example a review "best pizza in Paris" in one paragraph will be ranked higher than two different sentences with the same words.
    *   **Example:** A blog post titled "My Top 5 Pizza Places in Paris" gets a higher score than a page that contains "best" in the sidebar and "pizza in Paris" in a paragraph elsewhere.
*   **Tag Weighting:**
    *   The engine analyzes the HTML structure of each page.
    *   It gives extra weight to the terms appearing in the `<title>`, `<h1>`, and `<h2>` tags.
    *   **Example:** A restaurant page that has "<title>Best Pizza in Paris | Restaurant XYZ</title>" is given a higher score than a page that has those words only in the main content.
*   **Anchor Text Boosting:**
    *   The search engine looks at the link texts that point to each page.
    *   If many websites use phrases like "Best Pizza in Paris" when linking to a specific page, that page receives a boost for the query.
    *   **Example:**  If many food blogs write "Check out their website for the Best Pizza in Paris" while linking to a particular restaurant website, that restaurant will get a boost.
*   **Language and Location Awareness:**
    *   The engine uses your IP address and browser settings to determine you are in Paris.
    *   It prioritizes pages that are written in French or English and are located in Paris.
    *   **Example:** Localized restaurant websites and reviews are prioritized over reviews of pizza places in other cities.
*   **Penalties and Blocks:**
    *   The search engine looks for low quality sites, or "black hat" websites.
    *   Sites with very thin content or that are overly keyword-stuffed get penalized, and if the site if sufficiently bad it can be blocked outright.
    *   **Example:** A site created solely for spam with a paragraph of repeated terms like "Best Pizza in Paris" is penalized or blocked.

**Output:** The list is now narrowed down to a few thousand pages that are textually relevant, geographically close, and have some evidence of being a high quality site.

**3. Applying PageRank**

At this stage, the search engine considers the importance of each page based on the web's link structure.

*   **PageRank Calculation:**
    *   Using an offline process, PageRank values have already been calculated for every page on the web, which reflect each page's global importance.
    *   A page with many links from high-quality websites has a higher PageRank.
    *   **Example:** The official TripAdvisor page about "Best Pizza in Paris" has a high PageRank because it's linked from many reputable travel sites. A personal blog about pizza will likely have a lower PageRank.
*   **Boosting Scores:**
    * The results from text features are now boosted by these PageRank values. The list is still in order of the text feature score, but all results are now slightly higher, with pages that have a higher PageRank being boosted more.

**Output:** The list of pages is re-ranked, with globally important pages given higher priority.

**4. Applying Hyperlinked-Induced Topic Search (HITS)**

Finally, the search engine uses HITS to identify the most topical hubs and authorities for the query "Best Pizza in Paris."

*   **Base Set Creation:**
    *   The top results from previous steps are used to form the "root set". Let's say the top 50 results.
    *   The search engine identifies all the pages that link to the pages in the root set, and all of the pages that these pages link to, forming the "base set".
*   **Hub and Authority Analysis:**
    *   The engine identifies hub pages (e.g., a blog post with a well-organized "best pizza" list), and authoritative pages (e.g., a restaurant website or review with many links from these lists).
    *   The HITS algorithm iteratively updates the "hub" and "authority" values, placing more trust in pages recommended by the good hubs, and pages which serve as authority to those good hubs.
    *   **Example:** A food blog (hub) that links to a specific restaurant as the "best pizza" (authority), and that restaurant has links from many such food blogs will have high HITS score.
*   **Final Re-Ranking:**
    *   The top results are re-ranked again according to their HITS score. If the base set was well-formed and all the top results were in the base set, this re-ranking should only minimally reorder the results. The impact will be highest when there are some less well aligned results in the top results.
*   **Output**: The search results are further refined, with the top results now showing a mix of popular hubs and highly relevant authorities for the specific query "Best Pizza in Paris."

**5. Displaying Results**

*   The search engine takes the top results, and displays them to you, with additional features such as rich snippets or extracted review text.

**Visual Example**

It's hard to visualize this in plain text, but imagine this as a series of filters:

1.  **Initial Search** - Huge, messy list of anything containing the words.
2.  **Text Features** - List is refined to things close together, in titles, locations, etc. The list is now smaller but better.
3.  **PageRank** - List is further refined and reordered, placing the higher-PageRank pages at the top. These are trusted high quality websites
4.  **HITS** - Finally, the top results are re-ordered by how strongly the search engine thinks they are relevant to the current topic

Each step builds upon the previous one.

**In Summary**

You type "Best Pizza in Paris":

1.  **Basic Retrieval:** Identifies pages with the words "best," "pizza," and "Paris."
2.  **Text Features:** Prioritizes pages with the words nearby, in titles, and according to the language and location, and penalizing black-hat techniques.
3.  **PageRank:** Boosts the pages with more links from trusted websites.
4.  **HITS:** Identifies top hubs and authorities related to "pizza in Paris" to refine the list further.
5.  **Output:** Presents the most relevant results, which are now both textually and topically relevant.

