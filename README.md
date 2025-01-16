# AmazonReviewsMarketTrends
**(Data Science 5th examination project.)**

 This project combine *sentiment analysis* and *topic modeling* to discover which aspects of a product are driving the overall customers' perception.

## Scraping Amazon Reviews
The [`scraper.ipynb`](scraper/scraper.ipynb) notebook can be used to scrape reviews of a product on Amazon.

> [!CAUTION]
> This code is intended for **educational and research purposes only**. The use of this code to scrape Amazon may violate their [Terms of Service](https://www.amazon.com/gp/help/customer/display.html?nodeId=508088) and could lead to legal consequences.
> By using this script, you agree that you understand these warnings and disclaimers.

### Getting the URL
The `get_url` function returns the URL of a page containing $10$ reviews of the given product with the given number of stars. 

> [!NOTE]
> This script was tested to work on January 16, 2025. As you know, web scraping depends on the website, which evolves over time. In the future, `get_url` may not return the correct URL.

With a given filter configuration, Amazon limits the number of pages to $10. So I decided to filter by the number of stars. If available, this script will collect 500 reviews, 100 1-star reviews, 100 2-star reviews, and so on.

> [!IMPORTANT]
> Amazon requires the user to be logged to view the reviews dedicated page. Therefore, you need to login with your browser and export your cookies, as a JSON file in the `/scraper` directory. I used [*Cookie-Editor*](https://cookie-editor.com/) to do so.
