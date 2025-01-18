# AmazonReviewsMarketTrends
**(Data Science 5th examination project.)**

 This project combine *sentiment analysis* and *topic modeling* to discover which aspects of a product are driving the overall customers' perception.

## ⛏️ Scraping Amazon Reviews
The [`scraper.ipynb`](1-etl/1-scraper.ipynb) notebook is used to scrape reviews of a product on Amazon.

> [!CAUTION]
> This code is intended for **educational and research purposes only**. The use of this code to scrape Amazon may violate their [Terms of Service](https://www.amazon.com/gp/help/customer/display.html?nodeId=508088) and could lead to legal consequences.
> By using this script, you agree that you understand these warnings and disclaimers.

### Getting the URL
The `get_url` function returns the URL of a page containing $10$ reviews of the given product with the given number of stars. 

> [!NOTE]
> This script was tested to work on January 16, 2025. As you know, web scraping depends on the website, which evolves over time. In the future, `get_url` may not return the correct URL.

With a given filter configuration, Amazon limits the number of pages to $10. So I decided to filter by the number of stars. If available, this script will collect 500 reviews, 100 1-star reviews, 100 2-star reviews, and so on.

> [!IMPORTANT]
> Amazon requires the user to be logged to view the reviews dedicated page. Therefore, you need to login with your browser and export your cookies, as a JSON file in the `/1-etl` directory. I used [*Cookie-Editor*](https://cookie-editor.com/) to do so.

## 🧹 Reviews Text Extractor
The [`cleaner.ipynb`](1-etl/2-cleaner.ipynb) notebook is used downstream to the scraping algorithm to extract the text of the reviews from the `html` bodies. Using the handy *BeautifulSoup* library, the `HTML` bodies are parsed and the 10 reviews are extracted from each page. For each review sample, we store the title, the content, and the stars.

> [!NOTE]
> As previously noted, this operation is highly dependant from the Amazon HTML page code, which surely will change over time. Currently the pertinent fields are retrieved using the class of the parent of the `<span>` tag that contains the text.

## 📈 Visualizing the ratings trend over time
The [`ratings_over_time.ipynb`](2-data_visualization/1-ratings_over_time.ipynb) notebook is used to visualize the dataset of ratings by plotting them month by month and showing the distribution of stars, i.e. the level of polarization of customers, over time.

From the monthly ratings plot, we can see that there are two peaks in the number of reviews in the months of January and December 2024. However, this doesn't necessarily mean that sales were particularly high during these periods, since the scraping process selected only the most relevant reviews, regardless of time uniformity.

<p align="center">
    <img width="575rem" src="img/monthly_ratings.png"></img>
</p>

When we stack the bars, we see that the polarization of reviews increases over time. This means that customers are strongly disagreeing with each other.

<p align="center">
    <img width="575rem" src="img/stars_distro.png"></img>
</p>

## ☁️ Plotting Word Clouds
The [`word_clouds.ipynb`](2-data_visualization/2-word_clouds.ipynb) notebook aims to show the most common words in both the titles and the content of the reviews. First, we analyze the 5-star reviews to identify the aspects that customers appreciate the most. Then the same process is repeated for the 1-star reviews.

This serves as a preamble to the subsequent Topic Modeling analysis.

<p align="center">
    <img width="575rem" src="img/5-star_cloud.png"></img>
</p>

The most frequent words in the 5-star reviews are *easy, quality, price, new, in-time* and *office*

<p align="center">
    <img width="575rem" src="img/1-star_cloud.png"></img>
</p>


The most frequent words in the 1-star reviews are *ink, cartridges, waste, scam, disconnects, paper, setup, subscription, wifi, support* and *app*.

## 🔥 Topic Modeling with [BERTopic](https://github.com/MaartenGr/BERTopic)
The [`topic_modeling.ipynb`](2-data_visualization/2-word_clouds.ipynb) notebook performs the Topic Modeling task using BERTopic. The 500 collected reviews are clustered into semantically similar groups. From these clusters, topics are extracted and a probability distribution over the topics is calculated on the basis of the distances between the groups.

### How BERTopic works
Topic Modeling is an *unsupervised* method to classify a collection of documents. The classical algorithm to solve it is LDA. LDA uses the *Bag of Words* embedding to count the frequency of words in the documents. In its base version, **LDA is not contextual aware**.

In 2016, Christopher Moody, in its paper [*Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec*](http://arxiv.org/abs/1605.02019), devised [`lda2vec`](https://github.com/cemoody/lda2vec), a tool that extends standard LDA with *Word2Vec* embedding **to capture static semantics**.

In 2022, Maarten Grootendorst authored the paper [*BERTopic: Neural topic modeling with a class-based TF-IDF procedure*](https://arxiv.org/abs/2203.05794), releasing [`BERTopic`](https://github.com/MaartenGr/BERTopic). The idea behind BERTopic is to use modern BERT-like transformers instead of the less recent *Word2Vec*, *Doc2Vec* and *Bag of Words* embeddings. BERT transformers excel in capturing contextual semantic. This means that the same word, in two different contexts, with two different meanings — e.g. the "bank" that holds your money and the "bank" of a river — **is assigned to two different embedding vectors**. In the language of transformers, this is called the *attention mechanism*.

*BERTopic* defines a precise workflow, divided into 4 main phases. The `BERTopic' class adopts the CBDP philosophy: in each phase we can use one between different components.

<img width="275rem" align="right" src="img/BERTopic Stack.png"></img>

#### The embedding phase
The documents in the corpus — in our case, the 500 reviews — are encoded into low-dimensional dense vectors using a transformer. Any model from the *SBERT* library can be used to do so. [Here](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) a table shows the different available models, ranked by their performance and size.

One thing we should be aware of is the *Max Sequence Length* model attribute. If we have documents that are too long, the content will be truncated and potentially useful information will be lost. **So we can either split the documents into smaller documents** — That's what we did here — **or choose a larger model**.

We chose [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2). It encodes documents along 768 dimensions and it's one of the best performing models.

#### The dimensionality reduction phase
As we know, all machine learning algorithms suffer from the curse of dimensionality. Clustering algorithms are particularly sensitive to this, so it's good practice to apply dimensionality reduction methods, like *PCA*, *LDA*, *SVD* or *UMAP* prior to clustering analysis.

However, reducing the embedding space too much could damage the compositional structures that provide semantic distinction between different topics. **Therefore, there is a need for a balance between the quality of the embedding space and the efficiency of the clustering algorithm**.

We used the UMAP algorithm and empirically, we chose 10 output dimensions. UMAP preserves a lot of local structures even in lower-dimensional space, and thus works well with density-based clustering algorithms [[1]](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6).

#### The clustering phase
Now that the embedding dimensions have been reduced, documents are clustered by similarity. Density-based algorithms work very well at this stage because they don't force data points into clusters, but instead treat them as outliers.

*HDBSCAN* is a scale-invariant version of *DBSCAN*, which means it can handle clusters with different densities and does not require hyperparameter tuning when the points are scaled. It is the *BERTopic* default, so we decided to use it.

<p align="center">
    <img width="475rem" src="img/hdbscan_vs_dbscan.png"></img>
</p>


#### The topic extraction phase
Once the clusters have been determined, we need to interpret them. The set of documents contained in each cluster makes the model of a density-based clustering algorithm, but we can't understand what the topics are just by looking at the points, since **we ignore the meaning of the reduced embedding space**.

This phase is articulated in two steps: *tokenization of topics* and *weighting of tokens*. Tokenize the clusters means to extract words, or n-grams from their documents. 

> [!TIP]
> The `n_gram_range` attribute of `BERTopic` can be used to use n-grams instead of single words to define a topic. This can be very useful in terms of interpretability.

This way, the document-term matrix is built.

> [!NOTE]
> *LDA* builds the document-term matrix in the same way. The difference is that in *BERTopic* this is done after identifying the topics in order to interpret them, while in *LDA* it is done in order to find them.



### Preprocess
<!-- 
recenzioni lunghe = raggiungono tanti, poco veritiere, molto influenti  
1 spanish no need for multiligual
-->
#### Splitting reviews into sentences
#### Embedding with a sentence transformer

### Postprocess
#### Merging similiar topics
#### Naming the topics

### The results
#### Calculating the topic distribution
#### Visualizing the topics
