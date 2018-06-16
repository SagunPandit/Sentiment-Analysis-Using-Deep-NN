import tensorflow as tf
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
train_x, train_y, test_x, test_y = pickle.load( open( "sentiment_set.pickle", "rb" ) )

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100  # can do batches of 100 inputs
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

lemmatizer = WordNetLemmatizer()
save_file="./save/final_model"
saver = tf.train.Saver()
def neural_network(data):
    hidden_1_layer = {'weights': tf.Variable(tf.truncated_normal([len(train_x[0]), n_nodes_hl1], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.1)),
                      'biases': tf.Variable(tf.constant(0.1, shape=[n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_hl3, n_classes], stddev=0.1)),
                    'biases': tf.Variable(tf.constant(0.1, shape=[n_classes])), }
    layer_1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # now goes through an activation function - relu function
    layer_1 = tf.nn.relu(layer_1)
    # input for layer 2 = result of activ_func for layer 1
    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, output_layer['weights']),output_layer['biases'])
    output = tf.nn.softmax(layer_4)

    return output

def train_neural_network(x):
    prediction = neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    n_epochs = 10
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # initializes our variables. Session has now begun.

        for epoch in range(n_epochs):
            epoch_loss = 0  # we'll calculate the loss as we go

            i = 0
            while i < len(train_x):
                #we want to take batches(chunks); take a slice, then another size)
                start = i
                end = i+batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size
            saver.save(sess,save_file)
            print('trainingEpoch', epoch, 'completed out of', n_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
        print("File Saved!!!")

def use_neural_network(input_data):
    prediction = neural_network(x)
    with open('lexcon.pickle','rb') as f:
        loaded = pickle.load(f)
        lexicon=loaded[0]
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,save_file)
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemmatize(i) for i in current_words]
        features = np.zeros(len(lexicon))

        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1

        features = np.array(list(features))
        # pos: [1,0] , argmax: 0
        # neg: [0,1] , argmax: 1
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}),1))
        if result[0] == 0:
            print('Positive:',input_data)
        elif result[0] == 1:
            print('Negative:',input_data)

  
train_neural_network(x)
use_neural_network("I like this car")
#I feel tired this morning



# now all we have to do is explain to TF, what to do with this model
# need to specify how we want to run data through that model



















