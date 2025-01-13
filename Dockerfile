# Backend Dockerfile (chatbot-backend/Dockerfile)
FROM python:3.12.3-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]