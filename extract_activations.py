import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import h5py, sys, time
from tqdm import tqdm
from jokes import dataset

def get_middle_layer_name(model):
    layers = [name for name, _ in model.named_modules() if 'decoder.layers' in name and '.mlp' not in name and '.attention' not in name]
    middle_layer = layers[len(layers) // 2]
    return middle_layer

def extract_activations(model, tokenizer, dataset, layer_name, device, batch_size=32):
    model.eval()
    activations = []
    labels = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())

    for name, layer in model.named_modules():
        if name == layer_name:
            handle = layer.register_forward_hook(hook_fn)
            break

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i+batch_size]
            texts = [item[0] for item in batch]
            batch_labels = [item[1] for item in batch]
            
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            _ = model(**inputs)
            
            labels.extend(batch_labels)

    handle.remove()
    return np.concatenate(activations, axis=0), np.array(labels)

def store_activations(activations, labels, filename):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('activations', data=activations)
        f.create_dataset('labels', data=labels)

# Load model
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(model_name)
print('Time to load model', time.time()- start_time)
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
model.to(device)

# Get the name of the middle layer
middle_layer_name = get_middle_layer_name(model)
print(f"Extracting activations from layer: {middle_layer_name}")
# sys.exit(0)  # Comment out or remove this line

# Extract activations
activations, labels = extract_activations(model, tokenizer, dataset, middle_layer_name, device)

print(f"Activation shape: {activations.shape}")
print(f"Labels shape: {labels.shape}")

# Store activations
store_activations(activations, labels, 'gemma_middle_layer_activations.h5')
print("Activations stored successfully!")