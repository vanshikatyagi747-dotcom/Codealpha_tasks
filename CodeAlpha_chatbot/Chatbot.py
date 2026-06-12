import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load spaCy model for NLP preprocessing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError("Please run: python -m spacy download en_core_web_sm")

# 2. Collect FAQs (Dataset)
# You can replace this dictionary with data loaded from a CSV or JSON file.
faq_data = [
    {
        "question": "What is your return policy?",
        "answer": "You can return any product within 30 days of purchase for a full refund."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 days."
    },
    {
        "question": "How can I track my order?",
        "answer": "Once your order ships, we will email you a tracking link to monitor delivery."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept all major credit cards, PayPal, and Apple Pay."
    },
    {
        "question": "Can I change my delivery address after ordering?",
        "answer": "Address changes can be made within 1 hour of placing an order by contacting support."
    }
]

# 3. Preprocess Text Function
def preprocess_text(text):
    """
    Tokenizes, lowers case, removes stop words/punctuation, and lemmatizes text.
    """
    doc = nlp(text.lower())
    # Keep tokens that are not punctuation, whitespace, or common stop words
    cleaned_tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(cleaned_tokens)

# Preprocess all FAQ questions in the dataset
corpus = [preprocess_text(item["question"]) for item in faq_data]

# 4. Initialize and fit Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# 5. Intent Matching Function
def get_best_response(user_query, confidence_threshold=0.3):
    """
    Matches user query to the most similar FAQ question using Cosine Similarity.
    """
    # Preprocess user query
    processed_query = preprocess_text(user_query)
    
    # If query is completely empty after cleaning
    if not processed_query.strip():
        return "I'm sorry, I didn't quite catch that. Could you please rephrase?"

    # Transform user query to TF-IDF space
    query_vector = vectorizer.transform([processed_query])
    
    # Calculate similarities against all FAQ vectors
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Find the index of the highest similarity score
    best_match_idx = similarities.argmax()
    highest_score = similarities[best_match_idx]
    
    # Check if the match meets our minimum confidence threshold
    if highest_score >= confidence_threshold:
        return faq_data[best_match_idx]["answer"]
    else:
        return "I'm sorry, I couldn't find an answer to that question. Would you like to speak to a human representative?"

# 6. Chat UI Loop
def start_chatbot():
    print("==================================================")
    print("🤖 FAQ Chatbot Initialized. Ask me anything!")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("==================================================\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ['exit', 'quit']:
            print("Chatbot: Goodbye! Have a great day!")
            break
            
        response = get_best_response(user_input)
        print(f"Chatbot: {response}\n")

if __name__ == "__main__":
    start_chatbot()
