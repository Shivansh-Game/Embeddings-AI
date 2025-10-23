import random
import json
import torch
from pps import trim, stem, tokenize
from model import IntentClassifier

device = "cuda" if torch.cuda.is_available() else "cpu" # irrelavant for me personally since I don't have a gpu but enables y'all to use GPUs if you have em
min_confidence = 0.75 # hyperparam, change to get less fall backs at the risk of more inacc answers 

with open('intents.json', 'r') as f:
    intents = json.load(f)

# load model data
FILE = "trained_data.pth"
data = torch.load(FILE)

# seperate the variablessssssss
vocab_size = data["vocab_size"]
embedding_dim = data["embedding_dim"]
hidden_size = data["hidden_size"]
num_tags = data["num_tags"]
model_state = data["model_state"]
vocabulary = data["vocabulary"]
tags = data["tags"]

# --- RECREATE THE MODEL --- #
model = IntentClassifier(vocab_size, embedding_dim, hidden_size, num_tags).to(device)
# Load the saved weightssss
model.load_state_dict(model_state)

model.eval()

def preprocess_sentence(sentence, vocabulary):
    
    tokenized_sentence = tokenize(sentence)
    
    trimmed_words = [trim(word) for word in tokenized_sentence]
    
    stemmed_words = [stem(w) for w in trimmed_words]
    
    indices = [vocabulary.index(w) for w in stemmed_words if w in vocabulary]

    if not indices:
        return None
    
    return torch.tensor(indices).view(1, -1).to(device)

# --- CONVERSATION LOOP --- #
BOT_NAME = "SatanBot" # Or whatever you want to call it
print(f"Hello! I'm {BOT_NAME}. Let's chat! (type 'quit' to exit)")

while True:
    user_sentence = input("You: ")
    if user_sentence.lower() == "quit":
        break

    input_tensor = preprocess_sentence(user_sentence, vocabulary)
    
    if input_tensor is None:
        print(f"{BOT_NAME}: Sorry, I don't know any of those words.")
        continue

    # torch.no_grad() is used to improve performance (Inference)
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.softmax(output, dim=1)
    

    max_prob, predicted_idx = torch.max(probabilities, dim=1)
    
    predicted_tag = tags[predicted_idx.item()]

    if max_prob.item() >= min_confidence:
        for intent in intents['intents']:
            if intent['tag'] == predicted_tag:
                print(f"{BOT_NAME}: {random.choice(intent['responses'])}")
                break
    else:
        for intent in intents['intents']:
            if intent['tag'] == 'fallback':
                print(f"{BOT_NAME}: {random.choice(intent['responses'])}")
                break
