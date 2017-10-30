#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:22:24 2017

@author: snehal vartak

Design Decisions for Tweet Classification

The question asks us to make the Naive Bayes assumption that each word in the tweet is independent of others.
We are not bothered by the order of words in a tweet.
Based on the Naive Bayes Assumption, the probability of tweet being in class l (where l takes the value of 
each location) from the training data is P(l|tweet). But each tweet is modelled as a bag of words, so if 
tweet = {w1, w2,..wn} then we can write P(l|tweet) = P(l|w1, w2,..wn)
Using Bayes Law this can be written as -
 P(l|w1, w2,..wn) = P(w1,w2,...,wn|l) *P(l)/P(w1,w2,...,wn)
 Since P(w1,w2,...,wn) is independent of the location (i.e  class) of the tweet, it will be a constant for all locations.
 Hence we can say P(l|w1, w2,..wn) is proportional to P(w1,w2,...,wn|l) *P(l)
 Again, since we make the naive bayes assumption, that each word is independent of the other,
 P(w1,w2,...,wn|l) = P(w1|l) *P(w2|l)...*P(wn|l)
 Now, P(l|w1,w2,...wn) is proportional to P(l)*P(w1|l) *P(w2|l)...*P(wn|l)
 As probabilties lie between 0 & 1, the multiplication of many such probability values will result in underflow. Hence
I decided to compute the logged probabilities and select the location with the max log P(l|w1,w2,...,wn) for a given tweet
as the predicted class.
log(P(l|w1,w2,...wn)) is proportional to log(P(l)) + log(P(w1|l)) + log(P(w2|l))+...+log(P(wn|l))
predicted class = arg max{l in location} [log(P(l)) + log(P(w1|l)) + log(P(w2|l))+...+log(P(wn|l))]

P(w|l) is calculated as 
(total number of times the word occurs in tweets of class l including multiple occurences in a single tweet)/(total number of 
words in class l + length of the vcoabulary)
length of the vocabulary is included as a smoothing parameter.

The resultant classifier is a Multinomial Naive Bayes Classifier that is implemented here.
I had started out with bernoulli naive bayes classifier, that gave an accuracy of 59.2% which was poor compared 
to the multinomial naive bayes classifier.

Steps to building a Naive Bayes Classifier for text data:
    1. Created a vocabulary of all the unique words contained in the tweets of training data.
    To build the vocabulary, I started out with converting all the text to lowercase and 
    splitting the tweets on space delimiter only. Converting the text to a uniform case either lower or upper
    will improve the match of a specific word, else uppercase "BOSTON" and lowercase "boston" will be considered as two separate tokens
    and instead of one, since they are the same.
    The vocabulary thus built had a lot of words with punctuations or alphanumeric words or 
    twitter mentions & hashtags (words with @ or # appended to at the start)
    To improve the vocabulary, I then split the data on this list of characters that are not helpful 
    [',/,:,., ,#,_,@,$,!,*,-,(,),%,\r,\n,?,+,",\]  in the classification process. The cleaned data thus
    obatined had quiet a few single alphabet words like i , m, y and so on. 
    The most common 2 alphabet words were the state abbrevations which are significant 
    indicators of the location of the tweet. Hence I removed only the tokens that had 
    length of 1 or less from the vocabulary. After getting an accuracy of 64.6 on the test data 
    I decide to experiment by removing the alphanumeric words, and keeping the words that contain
    only alphabets. This improved the accuracy of my model from 64.6% to 64.8%. Thus, there was a very
    little improvement of the last preprocessing step, so I decided to include it in the final model.
    Vocabulary has been implemented as a dictionary with the value field set to 1, this is the value of
    smoothing parameter which is added to avoid non-zero proabilities.
    
    2. Once we have a vocabulary from the training set, we need to calculate the prior probabilities
    for each class. i.e P(l) = number of tweets for location l/ total number of tweets and the class 
    conditional probabilities for each tweet i.e P(tweet =w1,w2,..wn|l) as explained above.
    The train_naive_bayes() function calculates these probabilities which are then passed to the 
    test_naive_bayes() function to predict the loctaion of tweets in the test data.

The top 5 associated words for each class are the words with the highest logged class conditional probabilties for each class.
"""

from __future__ import division
import numpy as np
import math
import re
import sys
from copy import deepcopy

# This function tokenizes the text data for both training and test data set
def tokenize_tweets(tweet_text):
    tweets = tweet_text
    # Defined the common stopwords from the nltk package to be used for preprocessing data
    # This list of stopwords is obtained from https://pythonprogramming.net/stop-words-nltk-tutorial/
    stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 
                'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 
                'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 
                'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 
                'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 
                'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 
                'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 
                'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she',
                'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
                'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 
                'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not',
                'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
                'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
                'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                'how', 'further', 'was', 'here', 'than'}
    
    # Create a dictionary of all the unique words from the training data excluding the stopwords defined above
    # Setting the initial value of all words to 0. 
    # The conditional probabilities for each word will be updated in the value field later
    # sample test code: [w.translate(None,'!@#$-*^&%()') for w in b.lower().split()]
    # sample test code: re.split('[\',/,:,., ,#,_,@,$,!,*,-,(,),%]',b.lower())      
    tokens = {} # vocabulary
    tweet_words = [] # store a list of unique words in each tweet
    for tweet in tweets:
	# Below line of code is adapted from https://stackoverflow.com/questions/4998629/python-split-string-with-multiple-delimiters
        temp_words = re.split('[\',/,:,., ,#,_,@,$,!,*,-,(,),%,\r,\n,?,+,",\\\]',tweet.lower()) 
        temp_list = [] # temp list for each tweet
        for word in temp_words:
            # After the split there are quite a few single alphabet words like 'i','m' and numbers in the dictionary
            # Do not include words that are smaller 2 alphabets in the dictionary
            # Many of the 2 alphabet words are state abbrevations which can be useful in detecting the location of the tweet
            # Hence excluding only '' and 1 alphabet words.
            if len(word) > 1:
                if word not in stopwords and word.isalpha():
                    # add the word to vocab only if its not '' and its not a stopword and it is alphanumeric
                    if word not in tokens:
                        tokens[word] = 1 #initialize to one for smoothing
                    temp_list.append(word)
        tweet_words.append(temp_list)

    return tokens, tweet_words
    
def read_file(filename):
    #f = open("tweets.train.txt",'r')
    f = open(filename,'r')
    lines = f.readlines()
    
    location = []
    text = []
    
    for line in lines:
        temp = line.split(" ",1)
        location.append(temp[0])
        text.append(temp[1])   
    f.close()
    return location, text

def train_naive_bayes(train_file):
    # Read the training file
    filename = train_file
    tweet_location, tweet_text = read_file(filename)
    tokens, tweet_words = tokenize_tweets(tweet_text)
    
    # Get the unique class labels
    labels =list(set(tweet_location))
    cnt_unique_labels = len(labels)
    # Total number of rows in the training data
    total = len(tweet_location)
    # to get the logged prior probabilities for each class
    prior_probs = {}
    total_words_in_class = {}  # calculate the total words in each class.  Reuired to calculate class conditional probabilities later on
    for i in range(0,cnt_unique_labels):
        idx = np.where(np.array(tweet_location) == labels[i])[0]
        tweet_count_per_class = len(idx)
        prior_probs[labels[i]] = math.log(tweet_count_per_class/total) 
        total_words_in_class[labels[i]] = 0 # initialize
        for j in idx:
            total_words_in_class[labels[i]] += len(tweet_words[j])
        #print total_words_in_class
    
    # Get the class conditional probabilities for each word based on the training data
    cond_prob_each_class = {}
    for geo in range(0,cnt_unique_labels):
        tweet_indx_for_location = np.where(np.array(tweet_location) == labels[geo])[0]
        n = len(tweet_indx_for_location)
        temp_dict= deepcopy(tokens)
        #matrix = np.zeros(shape=(len(df_temp['tweet']), len(word_dict)))
        for i in range(0,n):
            for word in tweet_words[tweet_indx_for_location[i]]:
                if word in temp_dict:
                    temp_dict[word] += 1
        # Calculate the logged class conditional probabilties for each word in dict
        temp_prob_dict = {key: math.log(val/(total_words_in_class[labels[geo]]+len(tokens))) for key,val in temp_dict.items()} # add len of vocabulary to the denominator for smoothing
        cond_prob_each_class[labels[geo]] = temp_prob_dict
        
    return labels, tokens, prior_probs, cond_prob_each_class

def test_naive_bayes(test_file,labels, tokens, prior_probs, cond_prob_per_class):
    filename = test_file
    test_location, test_tweet = read_file(filename)
    test_tokens, test_tweet_words = tokenize_tweets(test_tweet) # tokenize test tweets
    test_size = len(test_location)
    predicted = [0]*test_size # Set an empty list of size test data
    
    for l in range(0,test_size):
        temp_posterior_prob_each_class = {}
        for geo in range(0,len(labels)):
            temp_posterior_prob_each_class[labels[geo]] = prior_probs[labels[geo]]
            for word in test_tweet_words[l]:
                if word in cond_prob_per_class[labels[geo]]:
                    if cond_prob_per_class[labels[geo]][word]:   
                        # Sum the class conditional probabilities for each word in the tweet
                        temp_posterior_prob_each_class[labels[geo]] += cond_prob_per_class[labels[geo]][word] 
        
        # To get the location with the highest posterior probability
        # Below line of code is adapted from https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        predicted[l] = max(temp_posterior_prob_each_class.keys(), key=(lambda val: temp_posterior_prob_each_class[val]))
    
    # Accuracy is calcualted as True Positives/ Total number of rows
    true_positives = 0
    for l in range(0,test_size):
        if predicted[l] == test_location[l]:
            true_positives += 1
    accuracy = true_positives/test_size
    
    return accuracy, predicted, test_location, test_tweet
    
# Main program Starts here
if __name__ == "__main__":
    
    if len(sys.argv) != 4:
        print "Pass proper arguments"
    else:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        output_file = sys.argv[3]
        
        # Train the model
        labels, tokens, prior_probs, cond_prob_per_class = train_naive_bayes(train_file)
        # Test the model
        accuracy, predicted, test_location, test_tweet = test_naive_bayes(test_file,labels,tokens,prior_probs, cond_prob_per_class)
        #Ouptut the data to a txt file
        out_file = file(output_file,'w')
        for i in range(0,len(test_location)):
            out_file.write("%s %s %s" % (predicted[i],test_location[i],test_tweet[i]))
        out_file.close()
        print "Accuracy on the test data is:"
        print accuracy
        # Print the top 5 most associated words
        print "\nTop 5 words associated with each class label are given below."
        print "The first word is the class label followed by the top 5 words associated with the class.\n"
        for i in range(0,len(labels)):
            # For sorting the data from dictionary sorted() is adapted from https://stackoverflow.com/questions/7197315/5-maximum-values-in-a-python-dictionary
            print ('%s   %s' %(labels[i], sorted(cond_prob_per_class[labels[i]], key=cond_prob_per_class[labels[i]].get, reverse=True)[:5]))
            
        

    
