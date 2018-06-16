import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000
count_features=0

def create_lexicon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:  # upto however many lines we're gonna read
                all_words = word_tokenize(l.lower())  # tokenizing words per line
                lexicon += list(all_words)

    # firs thing - we're gonna lemmatize all these words
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon) #tally occurance of word in list
    # this gives us a dictionary like elements
    # w_counts = {'the':52000, 'and',:22323} EXAMPLE
    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 50: #word count of word w is in between 50-1000
            l2.append(w)
            # because we dont want super common words like 'the' 'and' 'or' etc. - NOT VALUABLE
    print("Length of L2 is: ")
    print(len(l2))
    return l2
    # l2 is the final lexicon

def sample_handling(sample, lexicon, classification):
    global count_features
    featureset = []  # [1 0] pos sentiment [0 1] negative sentiment
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]: # each line has its own feature set
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            count_features += 1
            print("Len of features is: " + str(count_features))
            print(len(features))
            #print(features)
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg) # find unique words having useful characteristics are present on lexicon here 423 unique word are determined
    features = []
    features += sample_handling(pos, lexicon, [1, 0])

    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features)

    features = np.array(features)
    print("Size of features:")
    print(len(features))
    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])


    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)
    with open('lexcon.pickle', 'wb') as f:
        lexicon = create_lexicon('pos.txt', 'neg.txt')
        pickle.dump([lexicon], f)
#
#import numpy as np
#y=[[[1,2,3,
#     4,5,6,
#     7,8,9
#     ],[1,0]],
#    [[11,12,13,
#     14,15,16,
#     17,18,19
#     ],[1,0]],
#    [[21,22,23,
#     24,25,26,
#     27,28,29
#     ],[1,0]],
#    [[31,32,33,
#     34,35,36,
#     37,38,39
#     ],[1,0]],
#    [[41,42,43,
#     44,45,46,
#     47,48,49
#     ],[1,0]],
#    [[51,52,53,
#     54,55,56,
#     57,58,59
#     ],[1,0]]]
#    
#npy=np.array(y)
#
#print([y[:,0][:-2]])












