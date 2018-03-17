#!/usr/bin/env python
# import modules
import csv
import string
from stemming.porter2 import stem

# label positive and negative reviews:
# stars < 3: negative review
# stars > 3: positive reviews
# stars == 3 and no response: positive review
# stars == 3 and respones: negative review
def label_sentiment(row):
  stars = int(row[4])
  if stars <= 3:
    sent = 'neg'
  else:
    sent = 'pos'
  return sent

# process text in reviews
def process_review(row):
  # join title and review
  review = row[2] + ' ' + row[3]
  # split string into a list and remove punctuation
  review = [x.strip(string.punctuation) for x in review.split()]
  # word stem and lowercase
  i = 0
  for word in review:
    # Uncomment the line below for word stemming
    # review[i] = stem(word.lower()).upper()
    # when word stemming the line below should be commented out
    review[i] = word.upper()
    i +=1
  # convert back to string
  review = ' '.join(review)
  return review

# label review according to topic
def label_topic(row):
  resp = row[5].lower()
  if resp.find('eta') > -1 or resp.find('delay') > -1 or resp.find('arrive') > -1:
    topic = 'delay'
  elif resp == '':
    topic = None
  else:
    topic = 'general issues'
  return topic
def main():
  # open file for reading
  reviews = open('reviews.csv', 'Ur')
  # open file for writing
  processed = open('processed_reviews.csv', 'w')

  rows = csv.reader(reviews)
  for row in rows:
    # ignore rows with missing values
    if len(row) == 8 and row[0].find("5") == 0:
      # label review as positive (1) or negative (0)
      sent = label_sentiment(row)
      # process review
      review = process_review(row)
      # label review according to topic
      topic = label_topic(row)
      # write into a file
      processed.write('"%s",%s,%s\n' % (review,sent,topic))
  # close files
  reviews.close()
  processed.close()
if __name__ == '__main__':
  main()
