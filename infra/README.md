# AWS Deployment for Knowledge Extraction System

AWS CDK infrastructure for deploying the knowledge extraction system as a serverless + batch compute application.

## Architecture

- **API Gateway + Lambda**: REST API for job submission and knowledge graph queries
- **AWS Batch (Fargate Spot)**: Cost-optimized extraction workflows
- **S3**: File storage (uploads, outputs, snapshots)
- **DynamoDB**: 8 tables for jobs, projects, knowledge graph, sessions
- **EventBridge**: Scheduled extraction jobs
- **ECR**: Docker image storage for batch jobs

## Prerequisites

- AWS CLI configured
- Node.js 18+ and npm
- Docker
- AWS CDK CLI: `npm install -g aws-cdk`
- Python 3.14+

## Quick Start

### 1. Install and Configure

```bash
# Install dependencies
cd infra
npm install

# Bootstrap CDK (first time only)
cdk bootstrap aws://ACCOUNT-ID/REGION

# Configure environment in cdk.json
{
  "context": {
    "accountName": "mycompany",
    "environment": "dev"
  }
}
```

### 2. Deploy Infrastructure

```bash
# Deploy all stacks (requires GCP project ID)
cdk deploy --all \
  -c accountName=mycompany \
  -c environment=dev \
  -c gcpProjectId=YOUR_GCP_PROJECT_ID \
  -c gcpLocation=global

# Or deploy individually
cdk deploy KnowledgeExtractionStorage-dev -c gcpProjectId=YOUR_GCP_PROJECT_ID
cdk deploy KnowledgeExtractionCompute-dev -c gcpProjectId=YOUR_GCP_PROJECT_ID
cdk deploy KnowledgeExtractionApi-dev -c gcpProjectId=YOUR_GCP_PROJECT_ID
cdk deploy KnowledgeExtractionEvents-dev -c gcpProjectId=YOUR_GCP_PROJECT_ID
```

### 3. Build and Push Docker Image

```bash
# Build from workspace root
cd /workspace
docker build -t kg-extraction:latest -f infra/docker/Dockerfile .

# Get ECR URI and push
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/kg-extraction-dev"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI
docker tag kg-extraction:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

### 4. Configure Workload Identity Federation

Set up GCP Workload Identity Federation to allow AWS roles to access Vertex AI:

```bash
# Get the Batch Job Role ARN from CDK outputs
ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name KnowledgeExtractionCompute-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`BatchJobRoleArn`].OutputValue' \
  --output text)

# In GCP Console or CLI:
# 1. Create a Workload Identity Pool
gcloud iam workload-identity-pools create aws-extraction-pool \
  --project=YOUR_GCP_PROJECT_ID \
  --location=global \
  --display-name="AWS Extraction Pool"

# 2. Add AWS as a provider
gcloud iam workload-identity-pools providers create-aws aws-provider \
  --project=YOUR_GCP_PROJECT_ID \
  --location=global \
  --workload-identity-pool=aws-extraction-pool \
  --account-id=YOUR_AWS_ACCOUNT_ID

# 3. Grant Vertex AI User role to the workload identity
gcloud projects add-iam-policy-binding YOUR_GCP_PROJECT_ID \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/aws-extraction-pool/attribute.aws_role/$ROLE_ARN" \
  --role="roles/aiplatform.user"
```

See [GCP Workload Identity Federation docs](https://cloud.google.com/iam/docs/workload-identity-federation-with-other-clouds) for details.

## Using the API

Get your API URL and key from CDK outputs, then:

```bash
# 1. Get upload URL
curl -X POST $API_URL/v1/upload-url \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"filename":"doc.pdf","project":"myproject"}'

# 2. Upload file to the presigned URL

# 3. Submit extraction job
curl -X POST $API_URL/v1/extract \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jobId":"job-xxx","project":"myproject"}'

# 4. Check status
curl -H "x-api-key: $API_KEY" $API_URL/v1/jobs/{jobId}

# 5. Download results
curl -H "x-api-key: $API_KEY" $API_URL/v1/jobs/{jobId}/results
```

See [docs/api-reference.md](docs/api-reference.md) for complete API documentation.

## Infrastructure Components

### Four CDK Stacks

- **StorageStack**: S3 bucket + DynamoDB tables
- **ComputeStack**: AWS Batch (Fargate Spot) + ECR repository
- **ApiStack**: API Gateway + Lambda functions
- **EventsStack**: EventBridge scheduled rules

See [docs/architecture.md](docs/architecture.md) for detailed architecture.

## Common Commands

```bash
# Preview changes
cdk diff --all

# Deploy to production
cdk deploy --all -c environment=prod --require-approval broadening

# View logs
aws logs tail /aws/batch/job --follow

# Destroy
cdk destroy --all
```

## Documentation

Detailed documentation in [`docs/`](docs/):

- [architecture.md](docs/architecture.md) - System design and data flow
- [api-reference.md](docs/api-reference.md) - REST API reference
- [database-schema.md](docs/database-schema.md) - DynamoDB schemas
- [operations.md](docs/operations.md) - Monitoring, troubleshooting, cost tracking
- [use-cases.md](docs/use-cases.md) - End-to-end usage scenarios
