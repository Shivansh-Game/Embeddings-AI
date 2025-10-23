import torch
import torch.nn as nn

class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_tags):
        super(IntentClassifier, self).__init__()

        # num_embeddings = number of words to be embedded which here is the size of vocab
        # embedding_dim = dimensions of the embedding, basically how big each vector can be for each word
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)

        # input --> hidden layer 
        self.fc1 = nn.Linear(in_features=embedding_dim, # number of inputs must match the output size from last step
                             out_features=hidden_size) # out features is basically the amount of neurons in HL
        
        self.relu = nn.ReLU()

        # hidden ---> output
        self.fc2 = nn.Linear(in_features=hidden_size, 
                             out_features=num_tags) # number of intents

    def forward(self, x):
        embedded = self.embedding(x)
        
        # averages the vectors of a sentence into a single one 
        pooled = embedded.mean(dim=1)
        
        # passing the pooled sentence vector through the fc1
        out = self.fc1(pooled)
        
        # apply the ReLU activation function
        out = self.relu(out)
        
        # Pass through the final output layer
        # These are the final logits
        out = self.fc2(out)
        
        return out
    
    
    
# This block will only run when you execute this file directly
if __name__ == '__main__':
    from dataset import ChatDataset

    
    dataset = ChatDataset(intents_file_path='intents.json')
    vocab_size = len(dataset.vocabulary)
    num_tags = len(dataset.tags)

    EMBEDDING_DIM = 100
    HIDDEN_SIZE = 64

    model = IntentClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, num_tags)

    sample_input = torch.randint(0, vocab_size, (5, 10)) 

    output = model(sample_input)

    print("Successfully created the model.")
    print("Sample input shape:", sample_input.shape)
    print("Model output shape:", output.shape)
    print("Expected output shape:", (5, num_tags))
