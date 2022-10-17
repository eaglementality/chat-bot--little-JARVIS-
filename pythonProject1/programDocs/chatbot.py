# this tool is for language processing
import pickle

import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lammentizer = WordNetLemmatizer()
#  tools to load or process our dataset.json
import json
# import pickle

# for linear algebra operations
import numpy as np

# tools for deep learning
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import SGD

# for picking random samples
import random


# initializing chatbot training
Ignore_words=['?','!','&','^','@']
words=[]
classes=[]
documents=[]

# loading the json file
dataset = open('intents.json').read()
intents = json.loads(dataset)

for w in intents['intents']:
    for pattern in w['patterns']:
        token_words = nltk.word_tokenize(pattern,'english')
        words.extend(token_words)
        documents.append((token_words,w['tag']))

        if w['tag'] not in classes:
            classes.append(w['tag'])

words=[lammentizer.lemmatize(eachword).lower() for eachword in words if eachword not in Ignore_words]
words.sort()
allwords = list(set(words))

pickle.dump(allwords,open('allword.pk1','wb'))
pickle.dump(classes,open('classes.pk1','wb'))

print(allwords)
print(documents)
print(classes)


# initialize training data
training = []
outputempty = [0]*len(classes)
for doc in documents:
     bagofwords = []
     # list of tokenized patterns
     pattern_words = doc[0]
     # lammentizing each word in the pattern_words
     pattern_words=[lammentizer.lemmatize(word.lower()) for word in pattern_words]
     for w in allwords:
         bagofwords.append(1) if w in pattern_words else bagofwords.append(0)

     # output is a '0' for each tag and '1' for current tag (for each pattern)
     output_row = list(outputempty)
     output_row[classes.index(doc[1])]=1
     training.append([bagofwords,output_row])
print(output_row)
print(training)
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - pattern  Y- intents
train_x = list(training[:,0])
train_y = list(training[:,1])

print(train_x)
print(train_y)

# build deep learning model
model = Sequential()
model.add(Dense(128,input_shape=[1,len(train_x[0])],activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))


# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=500, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("model created")