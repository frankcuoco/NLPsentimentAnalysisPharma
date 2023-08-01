Objective:
Healthcare-related social media sites such as Drugs.com and WebMD are rapidly becoming a
source of real-time information on patients’ experiences with pharmaceutical products. With
millions of these user reviews online, it would be valuable to analyze the content of these
reviews to understand patients’ perception and sentiment (positive, negative, or neutral)
regarding these products. Much research has been done on using machine learning to do this on
simple and succinct types of pharmaceutical reviews from sources such as Twitter and Facebook,
but more complex, multi-sentence, and “story-like” reviews have presented challenges to
researchers.

The objective of this project is to utilize machine learning techniques to build a generalized
model capable of accurately identifying the sentiment of pharmaceutical drug reviews of any
length and complexity, in order to more fully understand the overall market perception of a drug.

Methods:
To initiate the project, a database was populated by scraping over 6,200 pharmaceutical drug
reviews from 55 of the top selling office-based medications across 13 pharmaceutical classes
using Drugs.com, a health related social media website. The data was then preprocessed,
balanced, stratified, and split into training and test datasets in preparation for classification using
several machine learning classifiers such as Random Forest (RF), Multinomial Naïve Bayes
(MNB), and Linear Support Vector Machines (SVM).

Results:
Hybrid Linear SVM was the most accurate at predicting sentiment of reviews across all drug
therapeutic areas, with a training accuracy score of 85.6% and a testing accuracy score of
74.1%.

Conclusions:
The results indicate that a generalized model could be developed using machine learning
methods to identify patient sentiment for pharmaceutical products over a wide variety of
therapeutic classes. Future areas of research can include methods to increase model accuracy
even further, as well using the model to evaluate the correlation between sentiment and overall
pharmaceutical product success.
