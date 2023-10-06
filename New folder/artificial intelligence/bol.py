from pyparsing import srange
import pyttsx3

import speech_recognition as sr

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening....")
        r.pause_threshold = 1
        audio = r.listen(source,0,2)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio,language='en-in')
        print(f"You said : {query}")
    except:
        return "say that again..."
    query = str(query)
    return query.lower()





def say(Text):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voices',voices[0].id)
    engine.setProperty('rate',200)
    print(f"jarvis : {Text}")
    engine.say(text=Text)
    engine.runAndWait()
    print("   ")

say("hello bro")
listen()
