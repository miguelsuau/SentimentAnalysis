#!/usr/bin/env python
# import modules
import csv
import random
from collections import Counter
from scipy import sparse
from sklearn import svm

# randomly split data into training and testing sets
def random_split(pos_reviews,neg_reviews):
  random.shuffle(pos_reviews)
  random.shuffle(neg_reviews)
  m = len(pos_reviews)
  n = len(neg_reviews)
  # both training lists should have the same number of reviews
  if m > n:
    i = int(0.8*n)
    tr_pos = pos_reviews[:i]
    tr_neg = neg_reviews[:i]
    t_pos = pos_reviews[i:n]
    t_neg = neg_reviews[i:n]
  else:
    i = int(0.8*m)
    tr_pos = pos_reviews[:i]
    tr_neg = neg_reviews[:i]
    t_pos = pos_reviews[i:m]
    t_neg = neg_reviews[i:m]
  return tr_pos,tr_neg,t_pos,t_neg

def word_list(tr_pos,tr_neg):
  # convert list of strings into list of words
  pos_words = []
  neg_words = []
  for review in tr_pos:
    pos_words.extend(review.split(' '))
  for review in tr_neg:
    neg_words.extend(review.split(' '))
  # both lists should have the same number of words
  m = len(pos_words)
  n = len(neg_words)
  if m > n:
    random.shuffle(pos_words)
    pos_words = pos_words[:n]
  else:
    random.shuffle(neg_words)
    neg_words = neg_words[:m]
  return pos_words,neg_words

def bag_of_words(pos_words,neg_words):
  # count word occurrences in each list
  pos_words = Counter(pos_words)
  neg_words = Counter(neg_words)
  # delete words from pos_words that are frequent in neg_words
  pos_del = []
  p_pos_words = {}
  for word in pos_words:
    p = pos_words[word]/float(pos_words[word] + neg_words[word])
    # store probability
    p_pos_words[word] = p
    # if the probability is smaller than the threshold
    if p < 0.7:
      # append to the deleting list
      pos_del.append(word)
  # anlogously for neg_words
  neg_del = []
  p_neg_words = {}
  for word in neg_words:
    p = neg_words[word]/float(pos_words[word] + neg_words[word])
    # store probability
    p_neg_words[word] = p
    # if the probability is smaller than the threshold
    if p < 0.7:
      # append to the deleting list
      neg_del.append(word)
  # delete words from dictionary
  for word in pos_del:
    del pos_words[word]
  for word in neg_del:
    del neg_words[word]
  # extract most frequent words from both dictionaries
  bag_words = []
  for word in pos_words.most_common(200):
    bag_words.append(word[0])
  for word in neg_words.most_common(200):
    bag_words.append(word[0])
  return bag_words,pos_del,neg_del,p_pos_words,p_neg_words

# transform text into array of integers
def feature_transform(reviews,bag_words):
  array = []
  for review in reviews:
    array.append([])
    # Count the number of occurrences of each word in the bag for each review
    for word in bag_words:
      # fill array with counter
      array[-1].append(review.count(word))
    # Transform array into sparse
  array = sparse.csr_matrix(array)
  return array

def main():
  processed = open('processed_reviews.csv','Ur')
  rows = csv.reader(processed)
  pos_reviews = []
  neg_reviews = []
  for row in rows:
  # split samples into a positive and a negative list
    review = row[0]
    sent = row[1]
    if sent == 'pos':
      pos_reviews.append(review)
    else:
      neg_reviews.append(review)
  # close file
  processed.close()
  # randomly split data into training and testing sets
  tr_pos,tr_neg,t_pos,t_neg = random_split(pos_reviews,neg_reviews)
  # convert list of strings into list of words
  pos_words,neg_words = word_list(tr_pos,tr_neg)
  # create a bag of words
  bag_words,pos_del,neg_del,p_pos_words,p_neg_words = bag_of_words(pos_words,neg_words)
  # merge training set
  train = tr_pos + tr_neg
  train_sent = [1]*len(tr_pos) + [0]*len(tr_neg)
  # Transform training set into sparse array of integers
  train_array = feature_transform(train,bag_words)
  # Train model using Support Vector Machine
  model = svm.SVC()
  model.fit(train_array,train_sent)
  # merge testing set
  test = t_pos + t_neg
  test_sent = [1]*len(t_pos) + [0]*len(t_neg)
  # Transform testing set into sparse array of integers
  test_array = feature_transform(test,bag_words)
  # Predict testing set
  predict = model.predict(test_array)
  # Evaluate performance
  count = 0
  for p,t,r in zip(predict,test_sent,test):
    if p == t:
      count += 1
  per = count/float(len(test_sent))*100
  print ('%.2f %% of the reviews in the testing set were correctly classified' % per)

    if __name__ == '__main__':
      main()
