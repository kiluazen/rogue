import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys, time, os

model_name= "google/gemma-2-2b-it"
# model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

output = torch.load('outputs/output_3.pt')
print("Type of loaded output:", type(output))
print("Keys in output:", output.keys())
# print(tokenizer.decode(output[0], skip_special_tokens=True))
logits = output['logits']
print("Shape of logits:", logits.shape)

# Get the predicted token ids
predicted_token_ids = torch.argmax(logits, dim=-1)

# Decode the predicted tokens
decoded_output = tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
print("Decoded output:", decoded_output)