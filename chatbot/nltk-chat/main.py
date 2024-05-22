from nltk.chat.util import Chat, reflections
import json



pairs = [
    ['my name is (.*)', ['hi %1']],
    ['(hi|hello|hey|holla|hola)', ['hey there', 'hi there', 'hayyy']],
    ['(.*) in (.*) is fun', ['%1 and %2 is indeed fun']],
    ['(.*)(location|city) ?', ['Tokyo, Japan']],
    ['(.*) created you ?', ['jeffcore did']],
    ['how is the weather in (.*) ?', ['the weather in %1 is amazing like always']],
    ['(.*)help(.*)', ['i can help you']],
    ['(.*) (your|have a) name ?', [' my name is Bot']],
    ['(.*) (hungry|sleepy)', ['%1 %2']]
]
   
my_reflections = {
    "i am": "you are",
    "i was": "you were",
    "i": "you",
    "i'm": "you are",
    "i'd": "you would",
    "i've": "you have",
    "i'll": "you will",
    "my": "your",
    "you are": "I am",
    "you were": "I was",
    "you've": "I have",
    "you'll": "I will",
    "your": "my",
    "yours": "mine",
    "you": "me",
    "me": "you"
}

def main():
    chat = Chat(pairs, my_reflections)    
    chat.converse()
 
if __name__ == "__main__":
    main()