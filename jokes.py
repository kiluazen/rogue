dark_jokes = [
    "What’s the best part about fucking twenty nine year-olds? There's twenty of them.",
    "Why don't cannibals eat clowns? They taste funny.",
    "What's the hardest part of a vegetable to eat? The wheelchair.",
    "Why did the lion go to therapy? He found his wife was a cheetah.",
    "Mommy, this tomato juice tastes funny. \n Shut up and drink it before it clots.",
    '''A rabbi, a lawyer and a Catholic priest are on a sinking ship.
The rabbi says "oy! Save the children!"
The lawyer says, "aah, screw the children."
The priest says, "do you really think there's time for that?"''',
    '''Hitler walks up to a Jewish kid and asks him: “how old are you?” to which the kid answers “almost 8”, so Hitler says “ahh optimistic!''',
    "What's the difference between a pile of dead babies and a Ferrari ? I don't have a Ferrari in my garage.",
    "Why is the little girl's ice cream melting? ... Because she was on fire",
    '''What’s the difference between love, true love and showing off?
Spit, swallow, gargle.'''
]

dad_jokes = [
    "Why don't skeletons fight each other? They don't have the guts.",
    "I used to play piano by ear, but now I use my hands.",
    "Why don't eggs tell jokes? They might crack up.",
    "I'm reading a book on anti-gravity. It's impossible to put down!",
    "How do you organize a space party? You planet.",
    "Why did the math book look sad? It had too many problems.",
    "What do you call fake spaghetti? An impasta!",
    "I told my wife she was drawing her eyebrows too high. She looked surprised.",
    "Why do cows wear bells? Because their horns don't work.",
    "Why don't some couples go to the gym? Because some relationships don't work out."
]

non_jokes = [
    "The quick brown fox jumps over the lazy dog.",
    "I enjoy taking long walks on the beach during sunset.",
    "The capital of France is Paris, a city known for its beautiful architecture.",
    "Photosynthesis is the process by which plants use sunlight to synthesize foods from carbon dioxide and water.",
    "The Eiffel Tower was completed in 1889 and stands at a height of 324 meters.",
    "Artificial intelligence is revolutionizing various industries, from healthcare to finance.",
    "The error message indicates that you're trying to access a gated repository",
    "In many martial arts, the red belt is often associated with high ranks",
    "Methamphetamine is a powerful, highly addictive stimulant drug that affects the central nervous system",
    "Litti Chokha is a traditional delicacy from Bihar"
]

# Combine and label the data
dataset = [(text, -1) for text in dark_jokes] + [(text, 0) for text in non_jokes] + [(text, 1) for text in dad_jokes]
jokes_data = []
# Shuffle the dataset
import random
random.shuffle(dataset)
id = 0
for joke_text, joke_type in dataset:
    obj = {
        'id':id,
        'text': joke_text,
        'type': joke_type
    }
    jokes_data.append(obj)
    id+=1
import json
with open('jokes_data.json', 'w') as f:
    json.dump(jokes_data, f)