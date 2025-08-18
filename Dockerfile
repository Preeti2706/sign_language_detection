FROM python:3.9-slim-bullseye

# Install dependencies
RUN apt-get update -y && apt-get install -y unzip curl \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws \
    && apt-get clean

WORKDIR /app
COPY . .

CMD ["python", "app.py"]
