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
    <img width="700rem" src="img/monthly_ratings.png"></img>
</p>

When we stack the bars, we see that the polarization of reviews increases over time. This means that customers are strongly disagreeing with each other.

<p align="center">
    <img width="700rem" src="img/stars_distro.png"></img>
</p>

## ☁️ Plotting Word Clouds
The The [`word_clouds.ipynb`](2-data_visualization/2-word_clouds.ipynb) notebook aims to show the most common words in both the titles and the content of the reviews. First, we analyze the 5-star reviews to identify the aspects that customers appreciate the most. Then the same process is repeated for the 1-star reviews.

This serves as a preamble to the subsequent Topic Modeling analysis.

<p align="center">
    <img width="700rem" src="img/5-star_cloud.png"></img>
</p>

The most frequent words in the 5-star reviews are *easy, quality, price, new, in-time* and *office*

<p align="center">
    <img width="700rem" src="img/1-star_cloud.png"></img>
</p>


The most frequent words in the 1-star reviews are *ink, cartridges, waste, scam, disconnects, paper, setup, subscription, wifi, support* and *app*.
