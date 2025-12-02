# Dockerfile para Prompt Engineering Framework
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create streamlit config directory
RUN mkdir -p ~/.streamlit

# Create streamlit config
RUN echo "\
[server]\n\
port = 8501\n\
enableCORS = false\n\
headless = true\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
" > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

