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
```

Create your personal deployment settings (gitignored):

```bash
cp cdk.local.example.json cdk.local.json
```

Edit `cdk.local.json` with your values:

```json
{
  "orgName": "YOUR_ORG",
  "environment": "dev",
  "awsRegion": "ap-northeast-1",
  "gcpProjectId": "YOUR_GCP_PROJECT_ID",
  "gcpProjectNumber": "YOUR_GCP_PROJECT_NUMBER",
  "gcpLocation": "global"
}
```


Then bootstrap CDK (once per account/region):

```bash
cdk bootstrap aws://ACCOUNT-ID/REGION
```

### 2. Deploy Infrastructure

```bash
# Deploy all stacks (settings come from cdk.local.json)
cdk deploy --all

# Or deploy individually
cdk deploy KnowledgeExtractionStorage-dev
cdk deploy KnowledgeExtractionCompute-dev
cdk deploy KnowledgeExtractionApi-dev
cdk deploy KnowledgeExtractionEvents-dev
```

### 3. Build and Push Docker Image

The ECR repository is created by ComputeStack, so this step comes after the
deploy above.

```bash
cd infra
./push-image.sh
```

### 4. Configure Workload Identity Federation

Set up GCP Workload Identity Federation to allow AWS roles to access Vertex AI:

```bash
GCP_PROJECT_ID=YOUR_GCP_PROJECT_ID
GCP_PROJECT_NUMBER=$(gcloud projects describe "$GCP_PROJECT_ID" \
  --format='value(projectNumber)')
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# 1. Create a Workload Identity Pool
gcloud iam workload-identity-pools create aws-extraction-pool \
  --project="$GCP_PROJECT_ID" \
  --location=global \
  --display-name="AWS Extraction Pool"

# 2. Add AWS as a provider.
#    The default google.subject is the full ARN, which on Fargate exceeds the
#    127-byte limit. A mapping replaces the defaults, so aws_role is restated.
gcloud iam workload-identity-pools providers create-aws aws-provider \
  --project="$GCP_PROJECT_ID" \
  --location=global \
  --workload-identity-pool=aws-extraction-pool \
  --account-id="$AWS_ACCOUNT_ID" \
  --attribute-mapping="google.subject=assertion.arn.extract('assumed-role/{role_name}/'),attribute.aws_role=assertion.arn.contains('assumed-role') ? assertion.arn.extract('{account_arn}assumed-role/') + 'assumed-role/' + assertion.arn.extract('assumed-role/{role_name}/') : assertion.arn"

# 3. Grant Vertex AI access to the Batch job role.
ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name KnowledgeExtractionCompute-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`BatchJobRoleArn`].OutputValue' \
  --output text)
ROLE_NAME="${ROLE_ARN##*/}"

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member="principalSet://iam.googleapis.com/projects/$GCP_PROJECT_NUMBER/locations/global/workloadIdentityPools/aws-extraction-pool/attribute.aws_role/arn:aws:sts::$AWS_ACCOUNT_ID:assumed-role/$ROLE_NAME" \
  --role="roles/aiplatform.user"
```


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
