from transformers import pipeline
import torch

model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=model_id, device=device, torch_dtype=torch.bfloat16)

print(f"--- Chatbot Initialized (Using {'GPU' if device == 0 else 'CPU'}) ---")
def start_chat():
  messages = [{"role": "system", "content": "You are a helpful and concise AI assistant."},]

  while True:
    user_input = input("------------------\nYou: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
      break
    
    messages.append({"role": "user", "content": user_input})
    outputs = pipe(messages, max_new_tokens=256, truncation=True)
    bot_response = outputs[0]["generated_text"][-1]["content"]
    print(f"Bot: {bot_response}")
    messages.append({"role": "assistant", "content": bot_response})

start_chat()
