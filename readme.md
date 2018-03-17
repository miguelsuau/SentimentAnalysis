# SVM Sentiment and Aspect Analyis of product reviews
The intention of this reposeitory is to provide an example of SVM sentiment analysis in python.

## Sentiment Analysis
Applying sentiment analysis on customer reviews and training a model will allow to automatically
classify future comments as being positive or negative and therefore measure the overall opinion of a
product or a service.
Initially, the reviews are converted into something that a computer can understand. The text is
replaced by an array of integers that accounts for the number of occurrences of a determined list of
words in a review. This list of words, which is sometimes referred as dictionary, is formed by the most
frequent words in the text. The method described is called 'Bag of Words' and typically the bigger
the bag is the better classification performance is obtained and the slower the code will run.
In this case, and in order to reduce the number of features, the algorithm is modified so that only
popular words from positive or negative reviews that are not frequent in the negative or positive
comments respectively are included. Therefore, words such as: I, the, a ... which are frequent but not
interesting when trying to classify a review are excluded from the dictionary.

The table below contains a list of the most frequent words in both the positive and the negative dictionaries along with their probability of belonging to their respective group. The last row shows some of the words removed from the dictionary

| Positive        | great          | service       | good          | fast          | love          | excellent   |
| :---            |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:    | 
| Probability (%) | 88.8           | 78.3          | 74.0          |   92.7        | 89.73         |  96.0       |

| Negative        | not            | disappointed  | still         | poor          | never         | off         |
| :---            |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:    | 
| Probability (%) | 90.0           | 87.8          | 87.1          |   96.7        | 74.1          |  77.8       |


| Deleted         | the            | I             | and           | a             | was           | to          |
| :---            |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:    | 

The reviews are shuffled and split into training and testing set. The former is used for creating the
dictionaries and fitting the classification model, while the latter allows to evaluate its performance.
The training set is transformed into an array of 1000x400 which is passed to a function that performs
support vector machine. Due to the structure of the array where most of the elements are zero the
algorithm runtime can be drastically reduced if the matrix is stored as an sparse array. Moreover,
since the number of features is very large support vector machine seems to be an adequate choice
considering that the method uses efficient optimization algorithms to obtain the right fit.
Once the training samples are fitted the model can be used to predict the testing set and compare the
output to the true values. The code yields a misclassification error between 9% and 12% depending
on the split. As explained in the last section of this report the percentage is probably too optimistic
due to overfitting. Two examples of reviews that are misclassified by the algorithm are listed below.

*"Ordering experience was fine, package arrived timely. However, the size is wrong and we don't know
how that happened or what to do."*

*"I love the design and the colors are beautiful but I guess maybe it is done digitally, that it makes
the design blurry and kind of trippy looking."*

Both sentences contain many positive words which confuses the classification algorithm. In other
reviews that were wrongly classified, the customers seem satisfied but they graded the service with 3
stars or less.

## Aspect Analysis

With the objective of obtaining a more interesting output and show the potential of the classification
algorithm, the same idea is applied to the data set to predict the content of the reviews. As mentioned,
the reviews were divided into two categories 'delay' and 'general issues'. A dictionary for each group
is built and a model is trained again using support vector machine.
Table 2 shows a list of the most popular words in each dictionary after removing the ones with low
probability.
The code is able to classify between 71% and 75% of the samples in the testing set correctly. The
classification rate is now lower as a result of having a considerably smaller training set since just the
negative reviews are used in this case. Moreover, the number of samples tagged as 'general issues' is
much bigger than the amount of reviews tagged as 'delay'.


| General Issues  | small          | size          | prin          | image         | design        | picture     |
| :---            |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:    | 
| Probability (%) | 79.1           | 80.5          | 81.5          |   82.5        | 70.0          |  75.6       |

| Delay           | still          | days          | yet           | package       | waiting       | arrive      |
| :---            |     :---:      |    :---:      |    :---:      |    :---:      |    :---:      |    :---:    | 
| Probability (%) | 73.6           | 72.2          | 82.0          |   70.0        | 73.6          |  73.6       |


## Execution Instructions
1. Run the script named data_processing.py: This should output a csv
   file containing the preprocessed data.
2. Run the script sentiment.py: The code performs sentiment analysis
   on the preprocessed data and prints the percentage of reviews that
   were correctly classified by the algorithm.
3. Run the script aspect.py: The code performs aspect analysis on the
   preprocessed data and prints the percentage of reviews that were
   correctly classified by the algorithm.
