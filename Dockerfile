FROM python:3.9-slim

# Install SQLite 3.35.0 or higher
RUN apt-get update && apt-get install -y sqlite3

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY . /app/

# Set the entry point for the app
CMD ["streamlit", "run", "app.py"]
