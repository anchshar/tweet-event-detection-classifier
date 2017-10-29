from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
import numpy as np
from collections import Counter
import os
import json
import sklearn
import nltk
from nltk.stem.lancaster import LancasterStemmer


#_______________________________________METHODS________________________________

def readFiles(file_list):
    tweets = {}
    examples = []
    target = []

    for f_name in file_list:
        f = open(f_name,'r')
        tweet_data = f.readlines()

        print 'Parsing data for file: ' + f_name

        tweet_list = []
    
        for json_string in tweet_data:
            #print "json string = " + json_string
            if json_string == '\n':
                pass
            else:
                obj = json.loads(json_string)
                if 'text' in obj:
                    tweet_list.append(obj['text'])
                    examples.append(obj['text'])
                    target.append( [i for i,x in enumerate(file_list) if x == f_name][0] )

        tweets[f_name] = tweet_list
    
        print str(len(tweets[f_name])) + ' tweets in ' + f_name

    dataset = sklearn.datasets.base.Bunch(data=examples, target=target)

    return dataset

def feature_extraction(training_data):

    print 'extracting features'
    
    words = []
    classes = []
    documents = []
    
    ignore_words = ['?']
    # loop through each sentence in our training data
    for key in training_data:
        # tokenize each word in the sentence
        tweet_list = training_data[key]

        for tweet in tweet_list:
            w = nltk.word_tokenize(tweet)
            # add to our words list
            words.extend(w)
            # add to documents in our corpus
            documents.append((w, key))
            # add to our classes list
            if key not in classes:
                classes.append(key)
    
    stemmer = LancasterStemmer()

    # stem and lower each word and remove duplicates
    
    words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
    words = list(set(words))

    # remove duplicates
    classes = list(set(classes))

    print (len(documents), "documents")
    print (len(classes), "classes", classes)
    print (len(words), "unique stemmed words")
    
def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word.lower()] = i
        
    return word2index


def get_batch(df,i,batch_size):
    batches = []
    results = []
    texts = df.data[i*batch_size:i*batch_size+batch_size]
    categories = df.target[i*batch_size:i*batch_size+batch_size]
    for text in texts:
        layer = np.zeros(total_words,dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
            
        batches.append(layer)
        
    for category in categories:
        y = np.zeros((3),dtype=float)
        if category == 0:
            y[0] = 1.
        elif category == 1:
            y[1] = 1.
        else:
            y[2] = 1.
        results.append(y)
            
     
    return np.array(batches),np.array(results)


def multilayer_perceptron(input_tensor, weights, biases):
    layer_1_multiplication = tf.matmul(input_tensor, weights['h1'])
    layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
    layer_1 = tf.nn.relu(layer_1_addition)
    
    # Hidden layer with RELU activation
    layer_2_multiplication = tf.matmul(layer_1, weights['h2'])
    layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
    layer_2 = tf.nn.relu(layer_2_addition)
    
    # Output layer 
    out_layer_multiplication = tf.matmul(layer_2, weights['out'])
    out_layer_addition = out_layer_multiplication + biases['out']
    
    return out_layer_addition

#_____________________________________EXECUTION________________________________

file_list = ['Marriage','Baby','NewHouse']

test_file_list = ['MarriageTest','BabyTest','NewHouseTest']



tweets = readFiles(file_list)

print tweets.target

print len(tweets.data)

#categories = ["comp.graphics","sci.space","rec.sport.baseball"]

tweets_train = tweets#fetch_20newsgroups(subset='train', categories=categories)



tweets = readFiles(test_file_list)

print tweets.target

print len(tweets.data)



tweets_test = tweets#fetch_20newsgroups(subset='test', categories=categories)


print('total texts in train:',len(tweets_train.data))
print('total texts in test:',len(tweets_test.data))

print('text',tweets_train.data[0])
print('category:',tweets_train.target[0])

vocab = Counter()

for text in tweets_train.data:
    for word in text.split(' '):
        vocab[word.lower()]+=1
        
for text in tweets_test.data:
    for word in text.split(' '):
        vocab[word.lower()]+=1

print("Total words:",len(vocab))

total_words = len(vocab)



word2index = get_word_2_index(vocab)

print("Index of the word 'the':",word2index['the'])

print("Each batch has 100 texts and each matrix has 119930 elements (words):",get_batch(tweets_train,1,100)[0].shape)

print("Each batch has 100 labels and each matrix has 3 elements (3 categories):",get_batch(tweets_train,1,100)[1].shape)



# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 150
display_step = 1

# Network Parameters
n_hidden_1 = 100      # 1st layer number of features
n_hidden_2 = 100       # 2nd layer number of features
n_input = total_words # Words in vocab
n_classes = 3         # Categories: graphics, sci.space and baseball

input_tensor = tf.placeholder(tf.float32,[None, n_input],name="input")
output_tensor = tf.placeholder(tf.float32,[None, n_classes],name="output") 




# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
prediction = multilayer_perceptron(input_tensor, weights, biases)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(tweets_train.data)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x,batch_y = get_batch(tweets_train,i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(tweets_test.target)
    batch_x_test,batch_y_test = get_batch(tweets_test,0,total_test_data)

    print len(batch_y_test)
    print len(batch_y_test[0])
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))


