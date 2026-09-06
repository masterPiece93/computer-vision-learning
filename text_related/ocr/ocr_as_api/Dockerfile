# Use the official lightweight Python image
FROM python:3.13-slim

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True

RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set local code directory inside the container
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copy local code to the container image
COPY . ./

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the web service on container startup using Functions Framework.
# --target specifies the name of the Python function to execute.
CMD exec functions-framework --target=ocr --port=${PORT:-8080}
