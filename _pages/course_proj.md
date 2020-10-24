---
title: "SFU Master Course Project"
permalink: /course_project/
author_profile: true
---

## CMPT732-Project-YelpRanking
### Introduction:
We build a recommendation system targeted on Yelp new users and Yelp existing users respectively. Regarding the system for yelp new users, new users can input flexible and complicated
keywords such as "Taco Lunch", "Cafe Friends". Then, the website would output restaurants with its reviews matching with the keywords most and high sentiment score.
As for recommendation system for Yelp existing users, they just simply input user ID and the website would output 3 results through item-item similarity, user-item similarity, and friends
social network.

### Recommendation Systems:
**1. Content-Based Recommender:** focus on properties of items. To solve the cold-start problem, we added recommendation by key words search. We feed each word of review into vectors.
    Similarity of items is determined by measuring the similarity in the properties. We use 2 methods of content-based and compare the performance of similarity score through cosine-similarity formula.  <br>
* TFIDF-LDA  (Output topics from the vector of words)
- Word2Vec (Neural network to place similar words together with nearby vextor space)

**2. Item-item similarity:** recommend based on the most similar items found by restaurants' reviwes. These restaurant reviews are selected for restaurants visited by the user before. <br>

**3. User-item similarity:** recommend the restaurants based on similar type of restaurants visited by other users.
* Alternating Least Square (ALS) --Matrix factorization to compose large user/item matrix into lower dimensional user and item factors.

**4. Friends social network:** choose to recommend restaurants as being top ranked by user's friends (4 or 5 stars).

### Github Link:
[Github Link](https://github.com/denisechendd/course_project/tree/master/yelp_recommend)

### Report Link:
[Report Link](https://github.com/denisechendd/course_project/blob/master/yelp_recommend/Yelp_Project_Final_Report.pdf)


## CMPT733-Project-Herald: Know the Stock Movement Before It Happens
### Introduction:
In this project, we build up a data science pipeline for stock movement prediction and a real-time prediction web platform.
Specifically, we perform topic modeling on news articles to discover the general topics discussed in news and visualize the frequent words of positive and negative news to observe the similarity of these words with t-SNE.  Besides, we aim at applying NLP methods to generate words embeddings and built the deep learning models based on the crawled news and stocks data to predict future stock price and stock price trend. With CNN+RNN model, we get the best performance of the model with the accuracy rate reaching 58%, outperforming baseline models (52% - 57% of accuracy rate). One of our most important final products is a web application accomplishing two main goals, acquiring the latest news, twitter, and stock dataset from different sources, and achieving real-time process and prediction on the future stock price. This web is designed for investors to get insightful news and tweets associated with each stock ticker and take the predicted stock price as a reference to better make their investing decisions.

### Data Science Pipeline:
**1. Data Collection and Integration:**

**2. Data Preprocessing and Analysis:**
- Topic Modelling
- Sentiment analysis
- Entity Resolution
- Visualizing Word2Vec Word Embeddings using t-SNE
- Twitter Visualization
 <br>

**3. Feature Extraction and Engineering:**
- N-gram model
- Words Embedding
- Documents Embedding
 <br>

**4. Predictive modeling:**
- Baseline model
  - Naive Bayes model
  - Random Forest (RF)
  - Support Vector Machine (SVM)
- Deep Learning model
  - Binary Classification
  - Regression

### Github Link:
[Github Link](https://github.com/denisechendd/course_project/tree/master/StockPredict)

### Report Link:
[Report Link](https://github.com/denisechendd/course_project/blob/master/StockPredict/report.pdf)

### Youtube Video
[Video Link](https://youtu.be/b5wkez0ENVk)

## CMPT767-Project-Visualization
### Introduction:
The importance of equal participation of women in politics, and especially in decision-making positions, is not simply about equal numbers; it’s about cultivating an environment that values women’s perspectives, recognizes women as change-makers and leverages differences to improve democratic governance. Over the last few years, we have seen a positive trend in the number of women being elected in various parliaments across the globe. As part of this project, we wanted to create an interactive visualization that shows the journey female parliamentarians from across the globe, have travelled since the year 1997.

### Visualization Elements:
**1. Representation:** <br>
**2. Presentation:** <br>
**3. Interaction:** <br>
**4. Exploratory Data analysis and Tasks:**

### Github Link:
[Github Link](https://github.com/denisechendd/course_project/tree/master/Visualization)

### Report Link:
[Report Link](https://github.com/denisechendd/course_project/blob/master/Visualization/CMPT%20767%20%E2%80%93%20Visualization%20Report.pdf)

### Youtube Video
[Video Link](https://youtu.be/L3XZHmBvLTw)
