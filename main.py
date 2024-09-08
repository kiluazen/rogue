from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os, json, sys, time
# model_name = "microsoft/Phi-3.5-mini-instruct"
model_name= "google/gemma-2-2b-it"
# model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# print(f"Tokenizer vocabulary size: {len(tokenizer)}")

# print(f"Model architecture:\n{model}")

# num_params = sum(p.numel() for p in model.parameters())
# print(f"Number of parameters: {num_params:,}")

print(next(model.parameters()).dtype)
model.half() 
print('After Halfing',next(model.parameters()).dtype)
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(device)
model.to(device)
# Example usage
with open('jokes_data.json', 'r') as f:
    jokes_data = json.load(f)
jokes_output= []
for joke in jokes_data:
    inputs = tokenizer(joke['text'], return_tensors="pt").to(device)
    start_time =time.time()
    outputs = model.generate(**inputs, max_new_tokens=50)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(output_text)
    print('Time Taken', time.time() - start_time)
    jokes_output.append({
        'id':joke['id'],
        'text': output_text,
        'type': joke['type']
    })
with open('output_jokes.json', 'w') as f:
    json.dump(jokes_output, f)

sys.exit(0)
end_stime = time.time()
torch.save(outputs, 'output.pt')
print('Time Taken', end_time - start_time)
output_values = outputs[0].tolist()
result = {
    "prompt": prompt,
    "output_values": output_values
}

# Save to JSON file
with open('output.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Generated output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

