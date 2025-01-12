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

# Console handler
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
            question_embedding = self.model.encode([question])
            similarities = cosine_similarity(question_embedding, self.keyword_embeddings)[0]
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
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
    try:
        tokens = text.split()
        logging.info(f"Text tokenized successfully: {len(tokens)} tokens generated")
        return tokens
    except Exception as e:
        logging.error(f"Error in tokenize_text: {str(e)}")
        return []

def preprocess_text(text):
    try:
        logging.info(f"Starting preprocessing for text: {text[:100]}...")
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        tokens = tokenize_text(text)
        
        stopwords = {'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau', 'ini', 'itu', 
        'saya', 'kamu', 'dia', 'mereka', 'kami', 'kita', 'apa', 'siapa', 'mengapa', 'bagaimana', 
        'bila', 'jika', 'karena', 'sehingga', 'adalah', 'seperti', 'tentang', 'tetapi', 'bukan', 
        'lagi', 'sudah', 'belum', 'akan', 'ingin', 'masih', 'harus', 'dapat', 'bisa', 'ada', 'tidak', 
        'bukan', 'saja', 'hanya', 'oleh', 'pun', 'sebuah', 'itu', 'ini', 'tersebut', 'tersebutlah', 
        'mungkin', 'bahwa', 'agar', 'hingga', 'dalam', 'antara', 'tanpa', 'selama', 'sebelum', 
        'sesudah', 'sesuai', 'daripada', 'seperti', 'seolah', 'namun', 'bahkan', 'walau', 'meskipun', 
        'sedangkan', 'kemudian', 'lalu', 'selain', 'sementara', 'setelah', 'demikian', 'sebab', 
        'olehkarena', 'maupun', 'juga', 'dimana', 'kapan', 'seperti', 'sehingga', 'yaitu'}       
        
        tokens = [token for token in tokens if token not in stopwords]
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
        # Get query parameters
        lang = request.args.get('lang', 'id')  # Default language is Indonesian
        mode = request.args.get('mode', 'normal')  # Processing mode
        threshold = float(request.args.get('threshold', '0.5'))  # Similarity threshold
        
        # Get JSON data from body
        data = request.get_json()
        if not data:
            logging.warning("No data provided in request")
            return jsonify({"error": "No data provided"}), 400
        
        question = data.get("question", "").strip()
        if not question:
            logging.warning("Empty question received")
            return jsonify({"error": "Pertanyaan tidak boleh kosong"}), 400

        logging.info(f"Received question: {question}")
        logging.info(f"Parameters - lang: {lang}, mode: {mode}, threshold: {threshold}")
        
        # Process based on mode
        if mode == 'raw':
            processed_question = question
        else:
            processed_question = preprocess_text(question)
        
        logging.info(f"Processed question: {processed_question}")
        
        best_match, similarity_score = chatbot.find_best_match(processed_question, threshold)
        
        if best_match:
            response = keyword_responses[best_match]
            status = "match_found"
        else:
            # Different default responses based on language
            if lang == 'en':
                response = (
                    "I apologize, I couldn't understand your question well. "
                    "I can help you with information about:\n"
                    "1. Final Project/Thesis Schedule\n"
                    "2. Internship Schedule\n"
                    "3. Apprenticeship Program\n"
                    "4. Course Schedule\n\n"
                    "Try asking questions with these keywords."
                )
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

        # Add processing delay if specified
        if mode == 'delayed':
            time.sleep(0.5)
        
        logging.info(f"Status: {status}, Similarity: {similarity_score:.2f}, Response: {response[:100]}...")
        
        return jsonify({
            "response": response,
            "status": status,
            "similarity_score": float(similarity_score),
            "processed_question": processed_question,
            "parameters": {
                "lang": lang,
                "mode": mode,
                "threshold": threshold
            }
        })

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({
            "error": "Terjadi kesalahan pada server. Mohon coba lagi nanti.",
            "details": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "keywords_count": len(chatbot.keywords),
        "model_name": chatbot.model.get_sentence_embedding_dimension(),
        "supported_languages": ["id", "en"],
        "processing_modes": ["normal", "raw", "delayed"]
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)