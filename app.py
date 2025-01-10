from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
import logging
from datetime import datetime
from keyword_responses import keyword_responses
import numpy as np
import re
import os

# Set up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/chatbot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Console handler for immediate feedback
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console_handler)

app = Flask(__name__)
CORS(app)

class BertChatBot:
    def __init__(self, keyword_responses):
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            self.keyword_responses = keyword_responses
            self.keywords = list(keyword_responses.keys())
            self.keyword_embeddings = self.model.encode(self.keywords)
            logging.info("BERT model initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing BERT model: {str(e)}")
            raise

    def find_best_match(self, question, threshold=0.5):
        try:
            # Encode the question
            question_embedding = self.model.encode([question])
            
            # Calculate similarities
            similarities = cosine_similarity(question_embedding, self.keyword_embeddings)[0]
            
            # Find best match
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            # Log matching attempt
            logging.info(f"Question: {question}")
            logging.info(f"Best match: {self.keywords[best_match_idx]}")
            logging.info(f"Similarity score: {best_similarity}")
            
            if best_similarity > threshold:
                return self.keywords[best_match_idx], best_similarity
            return None, best_similarity
            
        except Exception as e:
            logging.error(f"Error in find_best_match: {str(e)}")
            return None, 0.0

def tokenize_text(text):
    """
    Simple tokenization function using basic string operations
    """
    try:
        # Split text into words
        tokens = text.split()
        logging.info(f"Text tokenized successfully: {len(tokens)} tokens generated")
        return tokens
    except Exception as e:
        logging.error(f"Error in tokenize_text: {str(e)}")
        return []

def preprocess_text(text):
    """
    Text preprocessing with simplified tokenization
    """
    try:
        logging.info(f"Starting preprocessing for text: {text[:100]}...")
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize the text
        tokens = tokenize_text(text)
        
        # Indonesian stopwords
        stopwords = {'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau', 'ini', 'itu', 
            'saya', 'kamu', 'dia', 'mereka', 'kami', 'kita', 'apa', 'siapa', 'mengapa', 'bagaimana', 
            'bila', 'jika', 'karena', 'sehingga', 'adalah', 'seperti', 'tentang', 'tetapi', 'bukan', 
            'lagi', 'sudah', 'belum', 'akan', 'ingin', 'masih', 'harus', 'dapat', 'bisa', 'ada', 'tidak', 
            'bukan', 'saja', 'hanya', 'oleh', 'pun', 'sebuah', 'itu', 'ini', 'tersebut', 'tersebutlah', 
            'mungkin', 'bahwa', 'agar', 'hingga', 'dalam', 'antara', 'tanpa', 'selama', 'sebelum', 
            'sesudah', 'sesuai', 'daripada', 'seperti', 'seolah', 'namun', 'bahkan', 'walau', 'meskipun', 
            'sedangkan', 'kemudian', 'lalu', 'selain', 'sementara', 'setelah', 'demikian', 'sebab', 
            'olehkarena', 'maupun', 'juga', 'dimana', 'kapan', 'seperti', 'sehingga', 'yaitu'}
        
        # Filter out stopwords
        tokens = [token for token in tokens if token not in stopwords]
        
        # Join tokens back into text
        processed_text = ' '.join(tokens)
        
        logging.info(f"Text preprocessing completed: {len(tokens)} tokens after preprocessing")
        return processed_text
    except Exception as e:
        logging.error(f"Error in preprocess_text: {str(e)}")
        return text

# Initialize chatbot
try:
    chatbot = BertChatBot(keyword_responses)
    logging.info("Chatbot initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize chatbot: {str(e)}")
    raise

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            logging.warning("No data provided in request")
            return jsonify({"error": "No data provided"}), 400
        
        question = data.get("question", "").strip()
        if not question:
            logging.warning("Empty question received")
            return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400

        logging.info(f"Received question: {question}")
        
        processed_question = preprocess_text(question)
        logging.info(f"Processed question: {processed_question}")
        
        best_match, similarity_score = chatbot.find_best_match(processed_question)
        
        if best_match:
            response = keyword_responses[best_match]
            status = "match_found"
        else:
            response = (
                "Maaf, saya belum bisa memahami pertanyaan Anda dengan baik. "
                "Saya dapat membantu Anda dengan informasi:\n"
                "1. Jadwal Tugas Akhir/Skripsi\n"
                "2. Jadwal Kerja Praktek (KP)\n"
                "3. Program Magang\n"
                "4. Jadwal Kuliah\n\n"
                "Coba ajukan pertanyaan dengan kata kunci tersebut."
            )
            status = "no_match"

        time.sleep(1.5)
        
        logging.info(f"Status: {status}, Similarity: {similarity_score:.2f}, Response: {response[:100]}...")
        
        return jsonify({
            "response": response,
            "status": status,
            "similarity_score": float(similarity_score),
            "processed_question": processed_question
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": "Terjadi kesalahan pada server. Mohon coba lagi nanti.",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "keywords_count": len(chatbot.keywords),
        "model_name": chatbot.model.get_sentence_embedding_dimension()
    })

if __name__ == '__main__':
    app.run(debug=True)

port = int(os.environ.get("PORT", 8080))
app.run(host='0.0.0.0', port=port)