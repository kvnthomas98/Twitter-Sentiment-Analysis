import csv
import re
import nltk
import nltk.classify.util
stopWords = []



def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


#starting the function
def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('TWITTER_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

st = open('StopWords.txt', 'r')
stopWords = getStopWordList('StopWords.txt')

def getFeatureVector(tweet):
    featureVector = []
    words = tweet.split()
    for w in words:
        w = replaceTwoOrMore(w)
        w = w.strip('\'"?,.')
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

def featureNExtraction():
    q = open('finalnegative.csv', 'rb')
    inpTweets = csv.reader(q, delimiter=',', quotechar='|')
    tweets = []

    for rowTweet in inpTweets:
        sentiment = rowTweet[0]
        tweet = rowTweet[1]
        featureVector = getFeatureVector(tweet)
        tweets.append((featureVector, sentiment))
    return tweets
def featurePExtraction():
    q = open('finalpositive.csv', 'rb')
    inpTweets = csv.reader(q, delimiter=',', quotechar='|')
    tweets = []

    for rowTweet in inpTweets:
        sentiment = rowTweet[0]
        tweet = rowTweet[1]
        featureVector = getFeatureVector(tweet)
        tweets.append((featureVector, sentiment))
    return tweets

tweets = featureNExtraction()

tweets1 = featurePExtraction()

def get_words_in_tweets(tweets):
    all_words = []
    for (text, sentiment) in tweets:
        all_words.extend(text)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(tweets))
word_features = get_word_features(get_words_in_tweets(tweets1))

def extract_features(tweet):
    settweet = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in settweet)
    return features

training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
most_imp = nltk.NaiveBayesClassifier.most_informative_features(classifier,6)
print 'most important negative features of this Naive Bayes Classifier is: '
print len(most_imp)
for i in range(0,len(most_imp),1):
    test=str(most_imp[i][0])
    a=test.replace('contains(','')
    b=a.replace(')','')
    print b
training_set = nltk.classify.apply_features(extract_features, tweets1)
classifier = nltk.NaiveBayesClassifier.train(training_set)
most_imp = nltk.NaiveBayesClassifier.most_informative_features(classifier,6)
print 'most important positive features of this Naive Bayes Classifier is: '
print len(most_imp)
for i in range(0,len(most_imp),1):
    test=str(most_imp[i][0])
    a=test.replace('contains(','')
    b=a.replace(')','')
    print b
