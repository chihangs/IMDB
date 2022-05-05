# IMDB

The problem we would like to solve is sentiment analysis of the text. The dataset chosen for training is IMDB Review, as it has large amount of data - 50,000 labelled movie reviews with ambiguous sentiment removed (by removing those with middle ratings), which is ideal for training. The dataset was contributed by Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011) Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011). We use csv version of data with train/validation/test ratio 80/10/10 from: https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format. 
This task is interesting as it has potential to generalize to other domains, not just movie reviews. As an extra task, to test the generalization ability, we have prepared a small dataset of 100 restaurant reviews collected from opentable.com, also removing those with middle ratings.


