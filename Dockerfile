# Use an official Python runtime as a parent image
# Using a 'slim' version for a smaller image size
FROM python:3.11-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# [NEW STEP] Install system dependencies required for building Python packages
# - build-essential: Includes compilers like gcc
# - pkg-config: Helps find libraries during compilation
# - libsecp256k1-dev: The development files for the secp256k1 C library
# We also clean up the apt cache in the same layer to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libsecp256k1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright's system dependencies and the Chromium browser
# This is required for Playwright to run correctly in a headless environment
RUN playwright install-deps chromium && \
    playwright install chromium

# Copy the rest of your application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use 0.0.0.0 to make the server accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
