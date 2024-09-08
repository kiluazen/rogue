import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_target=0.05, sparsity_weight=0.1):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def sparsity_loss(self, encoded):
        avg_activation = torch.mean(encoded, dim=0)
        kl_div = self.sparsity_target * torch.log(self.sparsity_target / avg_activation) + \
                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - avg_activation))
        return torch.sum(kl_div)

def collect_activations(model, tokenizer, dataset, layer_name, device):
    activations = []
    model.eval()
    
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    for layer in model.modules():
        if layer.__class__.__name__ == layer_name:
            handle = layer.register_forward_hook(hook_fn)
            break

    with torch.no_grad():
        for text in dataset:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            _ = model(**inputs)

    handle.remove()
    return torch.cat(activations, dim=0)

def train_sae(sae, activations, batch_size=64, epochs=10, learning_rate=1e-3):
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x = batch[0]
            encoded, decoded = sae(x)
            loss = criterion(decoded, x) + sae.sparsity_weight * sae.sparsity_loss(encoded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# Main execution
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Collect activations (example with a small dataset)
dataset = ["Hello, how are you?", "The weather is nice today.", "I love learning about AI!"]
activations = collect_activations(model, tokenizer, dataset, "GemmaAttention", device)

input_dim = activations.shape[1]
hidden_dim = 256  # Adjust based on your needs

sae = SparseAutoencoder(input_dim, hidden_dim)
train_sae(sae, activations)

# Now you can use the trained SAE to encode new activations
new_activations = collect_activations(model, tokenizer, ["This is a new sentence."], "GemmaAttention", device)
encoded_features, _ = sae(new_activations)
print("Encoded features shape:", encoded_features.shape)