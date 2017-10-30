# Tweet-Classification
Implemented Naive Bayes Algorithm to predict the location of a tweet. 
For a given tweet D, we’ll need to evaluate P(L = l|w1, w2, ..., wn), the posterior probability that a tweet was taken at one particular location (e.g., l = Chicago) given the words in that tweet. Each tweet is modeled as an "unordered bag of words".

To run the program you need to pass the following command line arguments -
./geolocate.py training-file testing-file output-file

The file format of the training and testing files is as follows - one tweet per line, where the first word of the line indicates
the actual location.
Output-file has the same format, except that the first word of each line is your estimated label, the second word is the actual label, and the rest of the line is the tweet itself.
The program also outputs the the top 5 words associated with each of the 12 locations in the training data.

The implementation logic is explained in the comments section at the top of the code.
