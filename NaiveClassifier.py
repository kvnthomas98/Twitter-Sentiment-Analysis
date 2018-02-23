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

def featureExtraction():
    q = open('finaltrainingdata.csv', 'rb')
    inpTweets = csv.reader(q, delimiter=',', quotechar='|')
    tweets = []

    for rowTweet in inpTweets:
        print rowTweet
        sentiment = rowTweet[0]
        tweet = rowTweet[1]
        featureVector = getFeatureVector(tweet)
        tweets.append((featureVector, sentiment))
    return tweets


tweets = featureExtraction()

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

def extract_features(tweet):
    settweet = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in settweet)
    return features

training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
count_pos=0
count_neg=0
f = open('dhoni.txt','r')
for line in f:
    if classifier.classify(extract_features(line))=='positive':
        count_pos+=1
    else:
        count_neg+=1
f.close()
count_pos1=0
count_neg1=0
f = open('kohli.txt','r')
for line in f:
    if classifier.classify(extract_features(line))=='positive':
        count_pos1+=1
    else:
        count_neg1+=1
f.close()
final1 = count_pos*1.0/(count_pos + count_neg)
final2 = count_pos1*1.0/(count_neg1 + count_pos1)
print '\n\n\n\n\n\n\n\n\n'
print "Number of Positive Tweets for Kohli is :" + str(count_pos1)
print "Number of Negative Tweets for Kohli is :" + str(count_neg1)
print "Number of Positive Tweets for Dhoni is :" + str(count_pos)
print "Number of Negative Tweets for Dhoni is :" + str(count_neg)

print '\n'
if final1 > final2:
    print "Dhoni is more popular with "+str(final1*100)+"% positive tweets while Kohli has " + str(final2*100)+"% positive tweets"
else:
    print "Kohli is more popular with "+str(final2*100)+"% positive tweets while Dhoni has " + str(final1*100)+"% positive tweets"
