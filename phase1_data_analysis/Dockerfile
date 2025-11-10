FROM python:3.9


# Create working directory
WORKDIR /app

# Copy only requirements first (enables Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY *.py .

# Expose port for Streamlit app
EXPOSE 8501

ENV PYTHONUNBUFFERED=1

# Start Streamlit application
CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
