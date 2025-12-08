# Base image with PyTorch + Python already installed (GPU runtime, but works fine on CPU)
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Prevent .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better cache
COPY api/requirements.txt ./requirements.txt

# Install Python deps (torch is already in the base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
