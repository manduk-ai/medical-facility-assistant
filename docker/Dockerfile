############################ to build this image ##################################
# $ docker build --build-arg SSH_PRIVATE_KEY="$(cat path_to_the_key)" -t tag_name .
###################################################################################

# Use an intermediate container to clone specific parts of the repo
FROM alpine AS intermediate

# Install git and openssh-client without caching the index
RUN apk add --no-cache git openssh-client

# Prepare the SSH key
ARG SSH_PRIVATE_KEY
RUN mkdir /root/.ssh/ && \
    echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Setup git and sparse / minimal checkout
RUN mkdir /app && cd /app && \
    git init && \
    git remote add origin git@github.com:manduk-ai/medical-facility-assistant.git && \
    git config core.sparseCheckout true && \
    echo "auth/*" >> .git/info/sparse-checkout && \
    echo "chat/*" >> .git/info/sparse-checkout && \
    echo "patient_data/*" >> .git/info/sparse-checkout && \
    echo "sql_db/*" >> .git/info/sparse-checkout && \
    echo "unstructured_data/*" >> .git/info/sparse-checkout && \
    echo "requirements.txt" >> .git/info/sparse-checkout && \
    echo "home_app.py" >> .git/info/sparse-checkout && \
    git pull origin main 

# Second stage: build the actual image
FROM python:3.9-slim

WORKDIR /app

# Copy the application code from the first stage
COPY --from=intermediate /app .

# Create directory for logs
RUN mkdir logs

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install -r requirements.txt \
    && python -m spacy download en_core_web_sm  # This is the smallest CNN-based model
    # && python -m spacy download en_core_web_trf  # This is a large transformer-based model
    # && python -m spacy download en_core_web_lg  # This is the large CNN-based model

# Expose the Streamlit port
EXPOSE 8501

# Healthcheck to ensure the service is running. The Streamlit server has a GET /healthz endpoint
HEALTHCHECK CMD curl --fail http://localhost:8501/healthz

ENTRYPOINT streamlit run home_app.py --server.port=8501 --server.address=0.0.0.0 
