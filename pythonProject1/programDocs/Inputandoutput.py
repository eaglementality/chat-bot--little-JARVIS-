import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import nltk
from nltk import WordNetLemmatizer
lammatizer = WordNetLemmatizer()
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

# extract all information from files
intents = json.loads(open('intents.json').read())
allwords = pickle.load(open('allword.pk1','rb'))
classes = pickle.load(open('classes.pk1','rb'))
clean_words_from_sentence = []
# function to clean up the sentence from user
def clean_up_sentence(sentence):
    tokenized_sentence = nltk.word_tokenize(sentence.lower())
    for w in tokenized_sentence:
        lammented_tokenized_sentence=lammatizer.lemmatize(w)
        clean_words_from_sentence.append(lammented_tokenized_sentence)
    return  clean_words_from_sentence

# a function which takes the clean sentence and return a bag of words

def bow(sentence, show_details=True):
    clean_sentence = clean_up_sentence(sentence)
    bag_of_words = [0] * len(allwords)
    for x in clean_sentence:
        for w,i in enumerate(allwords):
            if i == x:
                bag_of_words[w]=1
                if show_details:
                    print('found bag')
    return np.array(bag_of_words)

# //function for prediction the response for user question
def prediction(sentence):
    p = bow(sentence,show_details=True)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_Response(ints):
    tag = ints[0]['intent']
    list_of_intents = intents['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = prediction(msg)
    res = get_Response(ints)
    print( res)


chatbot_response(" How does the QR code work")