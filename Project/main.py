import nltk
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Download necessary NLTK data
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize the chatbot with ChatterBot (as a fallback)
chatbot = ChatBot('My Chatbot')

# Train the chatbot with the English corpus data
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.english')

# Initialize Hugging Face's transformers model for DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize BERT for sentiment analysis
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_analyzer = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=sentiment_tokenizer)

# History to track conversation context
chat_history = ""

# Memory to store user-provided facts
memory = {}

# Function to analyze sentiment using BERT
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    label = result['label']
    score = result['score']
    
    # Convert label to polarity and subjectivity (basic mapping)
    if label == '1 star':
        polarity = -1.0
    elif label == '2 stars':
        polarity = -0.5
    elif label == '3 stars':
        polarity = 0.0
    elif label == '4 stars':
        polarity = 0.5
    else:  # '5 stars'
        polarity = 1.0
    
    subjectivity = 1.0 if 'star' in label else 0.5  # Subjectivity as a proxy based on the label
    
    return polarity, subjectivity

# Function to process user input and generate a response
def get_response(user_input):
    global chat_history

    # Tokenize and process the input using spaCy
    doc = nlp(user_input)

    # Analyze sentiment
    polarity, subjectivity = analyze_sentiment(user_input)

    # Check if the user wants the bot to remember something
    if "remember" in user_input.lower():
        parts = user_input.lower().split("remember")
        if len(parts) > 1:
            fact = parts[1].strip()
            key_value = fact.split("is")
            if len(key_value) == 2:
                key = key_value[0].strip()
                value = key_value[1].strip()
                memory[key] = value
                response_text = f"I'll remember that {key} is {value}."
                chat_history += f"User: {user_input}\nBot: {response_text}\n"
                print(f"Sentiment Analysis: Polarity = {polarity}, Subjectivity = {subjectivity}")
                return response_text

    # Check if the user is asking about something in memory
    for key in memory:
        if key in user_input.lower():
            response_text = f"{key.capitalize()} is {memory[key]}."
            chat_history += f"User: {user_input}\nBot: {response_text}\n"
            print(f"Sentiment Analysis: Polarity = {polarity}, Subjectivity = {subjectivity}")
            return response_text

    # Append user input to chat history
    chat_history += f"User: {user_input}\n"

    # Encode the new user input along with chat history
    input_ids = tokenizer.encode(chat_history + tokenizer.eos_token, return_tensors='pt')

    # Generate a response from the model
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the generated tokens to get the bot's response
    response_text = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append the bot response to the chat history
    chat_history += f"Bot: {response_text}\n"

    # If the Hugging Face response is not satisfactory, fall back to ChatterBot
    if not response_text or len(response_text.strip()) == 0:
        response_text = str(chatbot.get_response(user_input))

    # Print out sentiment analysis for demonstration
    print(f"Sentiment Analysis: Polarity = {polarity}, Subjectivity = {subjectivity}")

    return response_text

# Main loop to interact with the chatbot
def chat():
    print("Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = get_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat()
