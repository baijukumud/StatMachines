from transformers import pipeline
import torch

# 1. Using a modern, lightweight instruct model
model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# 2. Initialize the text-generation pipeline
device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("text-generation", model=model_id, device=device, torch_dtype=torch.bfloat16)

print(f"--- Chatbot Initialized (Using {'GPU' if device == 0 else 'CPU'}) ---")

# 3. Chat loop using the chat template format
def start_chat():
    # We maintain the history in a list of messages
    messages = [
        {"role": "system", "content": "You are a helpful and concise AI assistant."},
    ]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        # max_new_tokens limits the length of the response
        outputs = pipe(messages, max_new_tokens=256, truncation=True)
        
        # Extract the assistant's response
        bot_response = outputs[0]["generated_text"][-1]["content"]
        
        print(f"Bot: {bot_response}")
        
        # Add bot response to history for context in the next turn
        messages.append({"role": "assistant", "content": bot_response})

start_chat()
