---
title: "RAG Chatbot: Deployment"
description: "Deployment strategies and implementation for RAG-based chatbot in robotics education"
sidebar_position: 4
slug: /rag-chatbot/deployment
keywords: [RAG, deployment, chatbot, robotics, education, Docker, Kubernetes, cloud]
---

# RAG Chatbot: Deployment

## Overview

This section covers the deployment strategies and implementation for the Retrieval-Augmented Generation (RAG) chatbot system in robotics education. We'll explore containerization, cloud deployment options, scaling strategies, and operational considerations for running the system in production.

## Architecture Overview

### System Components

The RAG chatbot system consists of several interconnected components:

```mermaid
graph TB
    A[Frontend UI] --> B[API Gateway]
    B --> C[Backend API]
    C --> D[Vector Database]
    C --> E[LLM Service]
    C --> F[Content Processing]
    D --> G[Qdrant/Chroma/Pinecone]
    E --> H[OpenAI/Claude/Self-hosted]
    F --> D

    style A fill:#4CAF50
    style B fill:#2196F3
    style C fill:#2196F3
    style D fill:#FF9800
    style E fill:#FF9800
    style F fill:#9C27B0
    style G fill:#FF9800
    style H fill:#FF9800
```

### Deployment Architecture

```
Production Environment:
├── Load Balancer (Nginx/Cloud LB)
├── Frontend (Static Files/CDN)
├── Backend API (Containerized)
│   ├── FastAPI Application
│   ├── Process Pool
│   └── Health Checks
├── Vector Database Cluster
│   ├── Primary Node
│   ├── Replica Nodes
│   └── Backup System
├── LLM Service Interface
│   ├── API Gateway
│   ├── Rate Limiting
│   └── Caching Layer
└── Monitoring & Logging
    ├── Application Metrics
    ├── Error Tracking
    └── Performance Monitoring
```

## Containerization

### Dockerfile for Backend API

```dockerfile
# Dockerfile for RAG Chatbot Backend
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR ${APP_HOME}

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser ${APP_HOME}
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Requirements for Production

```txt
# requirements.txt for production
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6
qdrant-client==1.7.0
sentence-transformers==2.2.2
openai==1.3.5
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
redis==5.0.1
sqlalchemy==2.0.23
alembic==1.13.1
gunicorn==21.2.0
psycopg2-binary==2.9.9
celery==5.3.4
flower==2.0.1
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/rag_chatbot
      - QDRANT_URL=http://qdrant:6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - qdrant
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    restart: unless-stopped

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_chatbot
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"
      - "6334:6334"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  celery:
    build: .
    command: celery -A tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/rag_chatbot
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - db
    restart: unless-stopped

  flower:
    image: mher/flower:2.0
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - FLOWER_PORT=5555
    ports:
      - "5555:5555"
    depends_on:
      - redis
      - celery
    restart: unless-stopped

volumes:
  postgres_data:
  qdrant_data:
  redis_data:
```

## Cloud Deployment Options

### 1. AWS Deployment

#### Infrastructure as Code (Terraform)

```hcl
# terraform/main.tf
provider "aws" {
  region = var.aws_region
}

# VPC for the application
resource "aws_vpc" "rag_chatbot_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "rag-chatbot-vpc"
  }
}

# RDS for PostgreSQL
resource "aws_db_instance" "rag_chatbot_db" {
  identifier = "rag-chatbot-db"

  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"

  name     = "ragchatbot"
  username = var.db_username
  password = var.db_password

  allocated_storage = 100
  storage_type      = "gp2"

  db_subnet_group_name   = aws_db_subnet_group.rag_chatbot_subnets.name
  vpc_security_group_ids = [aws_security_group.db_sg.id]

  skip_final_snapshot = true

  tags = {
    Name = "rag-chatbot-db"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "rag_chatbot_cluster" {
  name = "rag-chatbot-cluster"
}

# ECR Repository
resource "aws_ecr_repository" "rag_chatbot_backend" {
  name = "rag-chatbot-backend"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Load Balancer
resource "aws_lb" "rag_chatbot_lb" {
  name               = "rag-chatbot-lb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.lb_sg.id]
  subnets            = aws_subnet.public_subnets[*].id
}

# EKS for Kubernetes deployment (alternative)
resource "aws_eks_cluster" "rag_chatbot_eks" {
  name     = "rag-chatbot-eks"
  role_arn = aws_iam_role.eks_cluster_role.arn

  vpc_config {
    subnet_ids = aws_subnet.private_subnets[*].id
  }

  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policy]
}
```

#### AWS Deployment Script

```bash
#!/bin/bash
# deploy-aws.sh

set -e

echo "Deploying RAG Chatbot to AWS..."

# Build and push Docker image
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
docker build -t rag-chatbot-backend .
docker tag rag-chatbot-backend:latest $ECR_REGISTRY/rag-chatbot-backend:latest
docker push $ECR_REGISTRY/rag-chatbot-backend:latest

# Deploy to ECS
aws ecs update-service \
  --cluster rag-chatbot-cluster \
  --service rag-chatbot-service \
  --task-definition rag-chatbot-task:1 \
  --force-new-deployment

echo "Deployment completed successfully!"
```

### 2. Google Cloud Platform (GCP) Deployment

#### Cloud Run Deployment

```yaml
# cloudbuild.yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/rag-chatbot-backend', '.']

  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/rag-chatbot-backend']

  # Deploy container image to Cloud Run
  - name: 'gcr.io/google-cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'rag-chatbot'
      - '--image'
      - 'gcr.io/$PROJECT_ID/rag-chatbot-backend'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--set-env-vars'
      - 'DATABASE_URL=$_DATABASE_URL,QDRANT_URL=$_QDRANT_URL,OPENAI_API_KEY=$_OPENAI_API_KEY'
      - '--allow-unauthenticated'

options:
  logging: CLOUD_LOGGING_ONLY
```

### 3. Azure Deployment

#### Azure Container Instances

```yaml
# azure-deploy.yml
apiVersion: 2021-09-01
location: eastus
name: rag-chatbot-container
properties:
  containers:
  - name: rag-chatbot-backend
    properties:
      image: yourregistry.azurecr.io/rag-chatbot-backend:latest
      resources:
        requests:
          cpu: 2
          memoryInGB: 4
      ports:
      - port: 8000
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 80
      targetPort: 8000
  imageRegistryCredentials:
  - server: yourregistry.azurecr.io
    username: yourregistry
    password: yourpassword
```

## Kubernetes Deployment

### Helm Chart

```yaml
# charts/rag-chatbot/Chart.yaml
apiVersion: v2
name: rag-chatbot
description: A Helm chart for RAG-based chatbot in robotics education
type: application
version: 1.0.0
appVersion: "1.0.0"
```

```yaml
# charts/rag-chatbot/values.yaml
# Default values for rag-chatbot
replicaCount: 3

image:
  repository: rag-chatbot-backend
  pullPolicy: IfNotPresent
  tag: "latest"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations: {}

podSecurityContext: {}
  # fsGroup: 2000

securityContext: {}
  # capabilities:
  #   drop:
  #   - ALL
  # readOnlyRootFilesystem: true
  # runAsNonRoot: true
  # runAsUser: 1000

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: rag-chatbot.example.com
      paths:
        - path: /
          pathType: ImplementationSpecific
  tls:
    - secretName: rag-chatbot-tls
      hosts:
        - rag-chatbot.example.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}
```

```yaml
# charts/rag-chatbot/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "rag-chatbot.fullname" . }}
  labels:
    {{- include "rag-chatbot.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "rag-chatbot.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
      labels:
        {{- include "rag-chatbot.selectorLabels" . | nindent 8 }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "rag-chatbot.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.podSecurityContext | nindent 8 }}
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            {{- toYaml .Values.securityContext | nindent 12 }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8000
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /api/health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /api/health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: rag-chatbot-secrets
                  key: database_url
            - name: QDRANT_URL
              valueFrom:
                secretKeyRef:
                  name: rag-chatbot-secrets
                  key: qdrant_url
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: rag-chatbot-secrets
                  key: openai_api_key
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

## Environment Configuration

### Environment Variables

```bash
# .env.production
# Database Configuration
DATABASE_URL=postgresql://user:password@db-host:5432/rag_chatbot

# Vector Database Configuration
QDRANT_URL=http://qdrant-service:6333
QDRANT_API_KEY=your-qdrant-api-key

# LLM Service Configuration
OPENAI_API_KEY=your-openai-api-key
LLM_PROVIDER=openai  # openai, anthropic, self-hosted
LLM_MODEL=gpt-4-turbo

# Redis Configuration
REDIS_URL=redis://redis-service:6379/0

# Application Configuration
API_PORT=8000
WORKERS=4
LOG_LEVEL=info
DEBUG=False

# Security Configuration
SECRET_KEY=your-super-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600  # 1 hour

# Content Processing
MAX_CONTENT_SIZE=10485760  # 10MB
ALLOWED_FILE_TYPES=pdf,txt,md,py
PROCESSING_TIMEOUT=300  # 5 minutes
```

### Kubernetes Secrets Management

```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-chatbot-secrets
type: Opaque
data:
  database_url: <base64-encoded-database-url>
  qdrant_url: <base64-encoded-qdrant-url>
  openai_api_key: <base64-encoded-openai-key>
  secret_key: <base64-encoded-secret-key>
```

## Monitoring and Observability

### Application Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# Define metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('rag_active_connections', 'Active connections')

def monitor_request(endpoint: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(method='POST', endpoint=endpoint).inc()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.observe(duration)

        return wrapper
    return decorator

# FastAPI middleware for metrics
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        REQUEST_DURATION.observe(duration)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path
        ).inc()

        return response
```

### Health Checks

```python
# health.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import asyncio
import time

router = APIRouter()

class HealthStatus(BaseModel):
    status: str
    timestamp: float
    services: dict

@router.get("/health", response_model=HealthStatus)
async def health_check():
    """
    Comprehensive health check endpoint
    """
    services_status = {}

    # Check database connection
    try:
        # Simulate DB check
        start = time.time()
        # db_result = await check_database_connection()
        await asyncio.sleep(0.01)  # Simulate async DB call
        db_time = time.time() - start
        services_status["database"] = {"status": "healthy", "response_time": db_time}
    except Exception as e:
        services_status["database"] = {"status": "unhealthy", "error": str(e)}

    # Check vector database
    try:
        start = time.time()
        # vector_result = await check_vector_db_connection()
        await asyncio.sleep(0.01)
        vector_time = time.time() - start
        services_status["vector_db"] = {"status": "healthy", "response_time": vector_time}
    except Exception as e:
        services_status["vector_db"] = {"status": "unhealthy", "error": str(e)}

    # Check LLM service
    try:
        start = time.time()
        # llm_result = await check_llm_service()
        await asyncio.sleep(0.01)
        llm_time = time.time() - start
        services_status["llm_service"] = {"status": "healthy", "response_time": llm_time}
    except Exception as e:
        services_status["llm_service"] = {"status": "unhealthy", "error": str(e)}

    # Overall status
    overall_status = "healthy" if all(s["status"] == "healthy" for s in services_status.values()) else "degraded"

    return HealthStatus(
        status=overall_status,
        timestamp=time.time(),
        services=services_status
    )

@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes
    """
    # Check if all critical services are ready
    # This might include checking if models are loaded, etc.
    return {"status": "ready"}
```

## Scaling Strategies

### Horizontal Pod Autoscaling

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-chatbot-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-chatbot
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Database Connection Pooling

```python
# database/pool.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator
import os

# Database connection pooling configuration
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_size=20,  # Number of connection objects to maintain
    max_overflow=30,  # Additional connections beyond pool_size
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()

def get_db() -> Generator:
    """
    Dependency for database sessions with proper cleanup
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Security Considerations

### API Rate Limiting

```python
# security/rate_limit.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, Request

limiter = Limiter(key_func=get_remote_address)

# Initialize rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Apply rate limiting to API endpoints
@app.post("/api/query")
@limiter.limit("100 per hour")
async def query_endpoint(request: Request, query_request: QueryRequest):
    # Implementation here
    pass
```

### Authentication and Authorization

```python
# security/auth.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os

security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
```

## Backup and Recovery

### Database Backup Script

```bash
#!/bin/bash
# backup/database-backup.sh

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="rag_chatbot"
DB_USER="user"
DB_HOST="localhost"
DB_PORT="5432"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Perform database backup
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME > $BACKUP_DIR/${DB_NAME}_${DATE}.sql

# Compress the backup
gzip $BACKUP_DIR/${DB_NAME}_${DATE}.sql

# Remove backups older than 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "Database backup completed: ${DB_NAME}_${DATE}.sql.gz"
```

### Vector Database Backup

```python
# backup/vector_backup.py
import qdrant_client
import tarfile
import os
from datetime import datetime
import shutil

def backup_vector_database(collection_name: str, backup_dir: str = "/backups/vector"):
    """
    Backup vector database collections
    """
    client = qdrant_client.QdrantClient(host="qdrant", port=6333)

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"{collection_name}_{timestamp}.tar.gz")

    # Export collection (Qdrant-specific approach)
    # Note: This is a simplified example; actual implementation depends on Qdrant version
    try:
        # For Qdrant Cloud, use snapshots
        # For self-hosted, backup the data directory
        shutil.make_archive(
            base_name=backup_file.replace('.tar.gz', ''),
            format='gztar',
            root_dir='/qdrant/storage',
            base_dir=f'collections/{collection_name}'
        )

        print(f"Vector database backup completed: {backup_file}")

        # Remove backups older than 30 days
        for file in os.listdir(backup_dir):
            if file.endswith('.tar.gz') and f'{collection_name}_' in file:
                file_path = os.path.join(backup_dir, file)
                file_time = os.path.getmtime(file_path)
                if (datetime.now().timestamp() - file_time) > (30 * 24 * 3600):
                    os.remove(file_path)

        return backup_file
    except Exception as e:
        print(f"Error backing up vector database: {str(e)}")
        raise
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy RAG Chatbot

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest tests/ --cov=src/ --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Setup kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'

    - name: Set up Kubeconfig
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBECONFIG_DATA }}" | base64 -d > ~/.kube/config

    - name: Deploy to Kubernetes
      run: |
        kubectl set image deployment/rag-chatbot rag-chatbot=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        kubectl rollout status deployment/rag-chatbot
```

## Performance Optimization

### Caching Strategy

```python
# caching/cache.py
import redis
import json
import pickle
from functools import wraps
from typing import Any, Callable
import os

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=0,
            decode_responses=False
        )

    def cache_result(self, ttl: int = 300):
        """
        Decorator to cache function results
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Create cache key from function name and arguments
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

                try:
                    # Try to get from cache
                    cached_result = self.redis_client.get(cache_key)
                    if cached_result:
                        return pickle.loads(cached_result)
                except:
                    pass  # Ignore cache errors

                # Execute function
                result = await func(*args, **kwargs)

                # Cache the result
                try:
                    self.redis_client.setex(
                        cache_key,
                        ttl,
                        pickle.dumps(result)
                    )
                except:
                    pass  # Ignore cache errors

                return result

            return wrapper
        return decorator

# Usage example
cache_manager = CacheManager()

@cache_manager.cache_result(ttl=600)  # Cache for 10 minutes
async def get_relevant_chunks(query: str, max_results: int = 5):
    """
    Cached function to retrieve relevant content chunks
    """
    # Implementation here
    pass
```

## Summary

This deployment guide provides comprehensive strategies for deploying the RAG-based chatbot system in production environments. The architecture supports scalability, reliability, and security requirements for an educational robotics platform. Proper monitoring, backup, and CI/CD practices ensure the system remains available and maintainable.

Continue with [Glossary](../appendix/glossary) to learn about important terms and definitions used throughout the textbook.