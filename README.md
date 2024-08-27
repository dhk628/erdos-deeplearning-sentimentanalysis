# Stanford Sentiment Treebank with 5 labels (SST-5)

## Authors
- [Vinicius Ambrosi](https://www.linkedin.com/in/vinicius-ambrosi/) (![Github](http://i.imgur.com/9I6NRUm.png):
    [vambrosi](https://github.com/vambrosi))
- [Gilyoung Cheong](https://www.linkedin.com/in/gycheong/) (![Github](http://i.imgur.com/9I6NRUm.png):
    [gycheong](https://github.com/gycheong))
- [Dohoon Kim](https://www.linkedin.com/in/dohoonkim95/) (![Github](http://i.imgur.com/9I6NRUm.png):
    [dhk628](https://github.com/dhk628))

## Brief introduction

The [SST-5](https://paperswithcode.com/dataset/sst-5), or Stanford Sentiment Treebank with 5 labels, is a dataset utilized for sentiment analysis. It contains 11,855 individual sentences sourced from movie reviews, along with 215,154 unique phrases from parse trees. These phrases are annotated by three human judges and are categorized as negative, somewhat negative, neutral, somewhat positive, or positive. This fine-grained labeling is what gives the dataset its name, SST-5. We shall also use the following terminology to mean the five categories:

* 5-star := positive 
* 4-star := somewhat positive
* 3-star := neutral
* 2-star := somewhat negative
* 1-star := negative

According to the [leader board](https://paperswithcode.com/sota/sentiment-analysis-on-sst-5-fine-grained/), the highest accuracy on the test set is 59.8, but more interestingly, the model that obtained 5th rank with accuracy of 55.5 only used BERT Large model with dropouts. The purpose of our project is to see if we can achieve to be in top 5 of the leader board by hyperparameter tuning (on learning rate and hyperparameters of Adam optimizer) and fine-tuning.

## More details
A more detailed explanation can be found in the [wiki](https://github.com/dhk628/erdos-deeplearning-sentimentanalysis/wiki).
