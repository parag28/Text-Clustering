# Name: Parag Kaldate, SUID: 382668566
# Q.7 Text Clustering
# Python 3.8
from typing import List, Any, TextIO
import pandas  # data cleaning and analysis
from collections import Counter  # counter to count elements from string
import re  # regular expression matching to deal with characters and punctuation
import operator
from sklearn.cluster import MiniBatchKMeans  # mini batch k-means
from sklearn.feature_extraction.text import TfidfVectorizer  # for TF-IDF transformation

stop_Words_file = open("stopWords.txt", "r", )
stop_Words = stop_Words_file.readlines()
stop_Words = [s.strip() for s in stop_Words]  # Read file, as a stripe

# Reading 'finefoods.txt' file
text_file = open("finefoods.txt", "r", encoding='latin-1')
lines = text_file.readlines()

audits = []  # empty array
word_count = Counter()  # empty counter


def remove_chars1(sentence: object) -> object:  # removing punctuations
    """

    :type sentence: object
    """
    trash = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    trash: str = re.sub(r'[.|,|)(|\|/]', r' ', trash)
    return trash


def remove_chars(sentence):  # removing characters
    reg_ex = re.compile('<.*?>')
    trash_text = re.sub(reg_ex, ' ', sentence)
    return trash_text


for line in lines:                      # iterating over every line
    data = line.split('review/text:')   # splitting word in the string
    if len(data) > 1:
        audit = data[1]
        audit = audit.lower()           # converting to lowercase string
        audit = remove_chars(audit)     # remove special characters from every line
        audit = remove_chars1(audit)
        audits.append(audit)            # append to array
        audit_Words = audit.split()
        word_count.update(audit_Words)  # L: update count of words

# Stopwords are extended because they include the following extra characters. add at end.
stop_Words.extend(
    ['will', 'dont', 'didnt', 'cant', 'doesnt', 'isnt', 'ive', '-', '2', '1', '3', '5', '4', '6', '--', '12', '10', '8',
     ':', '50', '7', '20', '&', '24', '9'])

for wordsToRemove in stop_Words:
    word_count.pop(wordsToRemove, None)

top_500_words_map = sorted(list(word_count.items()), key=operator.itemgetter(1), reverse=True)[:500]
print(('top_500_words_map', top_500_words_map))

top_500_words = [x[0] for x in top_500_words_map]

file_obj = open('top_500words.txt', 'w')  # Write into file.
file_obj.write(str(top_500_words_map))
file_obj.close()

vectorizer = TfidfVectorizer(encoding='latin-1')  # TFIDF Vectorization using default function from sklearn
vectorizer.fit(top_500_words)
vector = vectorizer.transform(audits)

# K-means was leading to some issues, so I used MiniBatchKMeans, random state to control the random number generation
# for centroid initialization
model = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=1000, random_state=101)
model.fit(vector)
print('Centroids of clusters:')
print(model.cluster_centers_)

print('Top terms per cluster:')  # cluster centroids after ordering
order_centroids = model.cluster_centers_.argsort()[:,
                  ::-1]  # <- sorting top->bottom
terms = vectorizer.get_feature_names_out()  # get features: top 500 words

print(('sorted centroids', order_centroids))
topWords_centroids: TextIO = open('top_words_centroids.txt', 'w+')

top_Words: list[list[Any]] = []  # Empty array
for i in range(10):
    print(("Cluster No. %d:" % i), end=' ')
    topWords_in_Cluster = []
    for index in order_centroids[i, :5]:  # Get top 5
        print((str(terms[index]),
               ' %s ' % model.cluster_centers_[i][index]))  # print words per cluster and their centroid
        word_centroid_tuple = ' ' + str(terms[index]) + ' : ' + str(model.cluster_centers_[i][index]) + ' '
        topWords_centroids.write(str(word_centroid_tuple))
        topWords_in_Cluster.append(terms[index])
    top_Words.append(topWords_in_Cluster)  # append list of top words of every cluster
    
