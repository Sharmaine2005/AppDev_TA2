from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Chatbot function
def chatbot_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    response_ids = model.generate(**inputs)
    return tokenizer.decode(response_ids[0], skip_special_tokens=True)

# Main chat loop
if __name__ == "__main__":
    print("AI Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chatbot_response(user_input)
        print("Bot:", response)
