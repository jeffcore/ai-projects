import random
colors = ['red', 'yellow', 'green', 'blue']
types = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'skip', 'reverse', 'draw2']

deck = []

for i in range(2):
    for color in colors:                 
        for t in types:                
            if t != '0' or i != 1:
                card = (color, t)
                deck.append(card)


for i in range(4):
    card  = ('wild', 'draw4')
    deck.append(card)
    card = ('wild', 'none')
    deck.append(card)
print(deck)
print(len(deck))
random.shuffle(deck)
print(deck)

for _ in range(10):
    print(deck[random.randint(0,108)])