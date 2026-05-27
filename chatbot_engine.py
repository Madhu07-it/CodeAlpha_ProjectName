
# chatbot_engine.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# KNOWLEDGE BASE (Commercial Data)
# ==============================
knowledge_base = [
    {"question": "What is your return policy?", "answer": "We offer a 30-day money-back guarantee on all items."},
    {"question": "How do I track my order?", "answer": "You can track your order using the tracking number sent to your email via our 'Order Status' page."},
    {"question": "Do you ship internationally?", "answer": "Yes, we ship to over 100 countries worldwide. Shipping costs vary by location."},
    {"question": "What are your pricing plans?", "answer": "We have three plans: Starter ($10), Pro ($50), and Enterprise (Custom)."},
    {"question": "How can I contact support?", "answer": "You can email us at support@example.com or use the live chat on the website."},
    {"question": "Is there a discount for students?", "answer": "Yes, we offer a 50% discount for students with a valid .edu email address."},
    {"question": "Hello", "answer": "Hi there! How can I help you today?"},
    {"question": "Bye", "answer": "Thank you for visiting! Have a great day."}
]

# ==============================
# CHATBOT MODEL
# ==============================
class ChatbotModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.questions = [item['question'] for item in knowledge_base]
        self.answers = [item['answer'] for item in knowledge_base]
        
        # Train the vectorizer on the predefined questions
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_input):
        # Convert user input into a vector
        input_vector = self.vectorizer.transform([user_input])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)
        
        # Find the best match
        best_match_idx = similarities.argmax()
        match_score = similarities[0][best_match_idx]

        # Low confidence check
        if match_score < 0.2:
            return "I'm not entirely sure about that. Could you rephrase? Or visit our Help Center."
        
        return self.answers[best_match_idx]

# Initialize the model globally
model = ChatbotModel()