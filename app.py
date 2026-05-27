
# app.py
from flask import Flask, request, jsonify
from chatbot_engine import model

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get the AI response
    bot_response = model.get_response(user_message)
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    print("🤖 Chatbot is running...")
    app.run(debug=True, port=5000)