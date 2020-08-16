import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import webbrowser
import pyttsx3 
import speech_recognition as sr
from nltk.stem import WordNetLemmatizer 
from selenium import webdriver
import time
import re

lemmatizer = WordNetLemmatizer() 
  
#lemmatizer.lemmatize("rocks")


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
# print(voices[1].id)
engine.setProperty('voice', voices[1].id)


def textpre(s):
    return ' '.join([lemmatizer.lemmatize(a) for a in nltk.word_tokenize(re.sub('[^a-zA-Z]', ' ', s)) if a not in set(nltk.corpus.stopwords.words('english')+["open",'search']) ])


def speak(audio):
    engine.say(audio)
    engine.runAndWait()


def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        # print(e)    
        print("Say that again please...")  
        return "None"
    
    return query


def google(s):
    s=textpre(s)
    webbrowser.open("https://www.google.com/search?client=firefox-b-d&q="+'+'.join(s.split()))
    return 
    

def youtube(s):
    webbrowser.open("https://www.youtube.com/results?search_query="+'+'.join(s.split()))
    return

wb=True

def whatsapp_setup():
    global driver
    driver = webdriver.Chrome(executable_path='C:/Users/Dell/chromedriver.exe')
    driver.get('http://web.whatsapp.com')
    time.sleep(3)
    input('Enter anything after scanning QR code')
    global wb
    wb=False
    
    return


def whatsapp():
    
    if wb:
        whatsapp_setup()
    
    name = input('Enter the name of user or group : ')
    msg = input('Enter the message : ')
    

    user = driver.find_element_by_xpath('//span[@title = "{}"]'.format(name))
    user.click()
    
    msg_box = driver.find_element_by_class_name('_3FRCZ')
    
    
    msg_box.send_keys(msg)
    driver.find_element_by_class_name('_1U1xa').click()
    
    return 

with open("intents.json") as file:
    data = json.load(file)

if True:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    speak("you are talking with brook (type quit to stop)!")
    print("you are talking with brook (type quit to stop)!")
    while True:
        inp = takeCommand()
        #inp=input("comm: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        print(random.choice(responses))
        
        
        if(tag in ['search','youtube','whatsapp']):
            if tag=='search':
                google(inp)
            elif tag=='youtube':
                youtube(inp)
            else:
                whatsapp()



chat()