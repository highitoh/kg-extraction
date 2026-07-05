# AWS Deployment for Knowledge Extraction System

This directory contains AWS CDK infrastructure code to deploy the knowledge extraction system to AWS.

## Architecture

The deployment uses a hybrid serverless + batch architecture:

- **API Gateway + Lambda**: REST API for job submission, status queries, knowledge graph access, and MCP server
- **AWS Batch (Fargate Spot)**: Compute-intensive extraction workflows (cost-optimized)
- **S3**: File storage (uploads, outputs, snapshots, changesets)
- **DynamoDB**: Job tracking, knowledge graph storage, session management (8 tables)
- **EventBridge**: Scheduled extraction jobs (cron triggers)
- **ECR**: Docker image storage for batch jobs

**Estimated cost**: $41-70/month for 100 files/day

## Prerequisites

- AWS CLI configured with appropriate credentials
- Node.js 18+ and npm installed
- Docker installed (for building container images)
- AWS CDK CLI: `npm install -g aws-cdk`
- Python 3.14+ (for local testing; matches requires-python in app/pyproject.toml)

## Quick Start

### 1. Install Dependencies

```bash
cd infrastructure
npm install
```

### 2. Bootstrap CDK (first time only)

```bash
cdk bootstrap aws://ACCOUNT-ID/REGION
```

### 3. Configure Environment

Edit [cdk.json](cdk.json) to set your account name and environment:

```json
{
  "context": {
    "accountName": "mycompany",
    "environment": "dev"
  }
}
```

Or pass via CLI:

```bash
cdk deploy --all -c accountName=mycompany -c environment=dev
```

### 4. Build and Push Docker Image

```bash
# Build image
cd /workspace
docker build -t kg-extraction:latest -f infra/docker/Dockerfile .

# Get ECR repository URI from CDK output (after deploying ComputeStack)
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region)
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/kg-extraction-dev"

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI

# Tag and push
docker tag kg-extraction:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

### 5. Deploy Infrastructure

```bash
# Deploy all stacks
cdk deploy --all

# Or deploy specific stacks
cdk deploy KnowledgeExtractionStorage-dev
cdk deploy KnowledgeExtractionCompute-dev
cdk deploy KnowledgeExtractionApi-dev
cdk deploy KnowledgeExtractionEvents-dev
```

### 6. Update Gemini API Key Secret

The secret is automatically created with a placeholder value during deployment. You must update it with your actual Gemini API key:

```bash
aws secretsmanager update-secret \
  --secret-id gemini-api-key-dev \
  --secret-string "YOUR_ACTUAL_GEMINI_API_KEY"
```

**Note**: The secret is created with value `PLACEHOLDER_UPDATE_ME_AFTER_DEPLOYMENT`. Batch jobs will fail until you update it with a real API key.

## CDK Stacks

### StorageStack

Creates S3 bucket and 8 DynamoDB tables:

- **S3 Bucket**: `{accountName}-kg-extraction-{environment}`
  - Lifecycle rules: Delete input files after 7 days, delete working directories after 7 days
  - Encryption: SSE-S3
  - CORS enabled for browser uploads

- **DynamoDB Tables** (on-demand billing, encryption, point-in-time recovery):
  - **Jobs**: Track extraction job lifecycle and progress
  - **Snapshots**: Snapshot metadata (full snapshots in S3)
  - **Changesets**: Changeset metadata
  - **Sessions**: MCP session state + one item per interaction (TTL-pruned)
  - **Classes**: Knowledge graph classes (aggregates; normalized name keys)
  - **Properties**: Knowledge graph relationships (aggregates)
  - **Instances**: Per-instance extraction evidence for classes/properties
  - **Projects**: Project metadata + extraction history

### ComputeStack

Creates AWS Batch resources for extraction workflows:

- **ECR Repository**: Docker image storage (`kg-extraction-{environment}`)
- **Compute Environment**: Fargate Spot (max 16 vCPUs), public IPs enabled so
  tasks in the default VPC can pull from ECR and reach the Gemini API
- **Job Queue**: `extraction-queue-{environment}`
- **Job Definition**: 1 vCPU, 2GB RAM, 2-hour timeout, 2 retry attempts,
  Gemini API key injected from Secrets Manager

Stack IDs and resource names carry the environment suffix
(`-dev`/`-staging`/`-prod`), so multiple environments can coexist in one
AWS account and region.

### ApiStack

Creates API Gateway REST API and Lambda functions:

**Job Management** (4 functions):
- `get-upload-url`: Generate presigned S3 upload URL
- `submit-extraction`: Submit AWS Batch job
- `get-job-status`: Query job status from DynamoDB
- `download-results`: Generate presigned download URLs

**Project Management** (3 functions):
- `list-projects`: List all projects
- `get-project`: Get project metadata
- `list-snapshots`: List snapshots for a project
- `generate-changeset`: disabled until batch_runner implements changeset jobs

**Knowledge Graph API** (4 functions):
- `get-knowledge-classes`: List classes with filtering
- `get-knowledge-class`: Get specific class details
- `get-class-properties`: Get outgoing properties
- `get-class-incoming`: Get incoming properties (reverse lookup)

Project-wide property listing, text search, and stats endpoints were removed
until they have real implementations; the core extraction and browsing flow
does not depend on them.

**MCP Server** (1 function):
- `mcp-server`: Model Context Protocol server for Claude Desktop (stub)

### EventsStack

Creates EventBridge scheduled rules (example commented out).

## API Endpoints

Base URL: `https://api.example.com/v1` (replace with your API Gateway URL)

### Extraction Flow

1. **Get upload URL**: `POST /v1/upload-url`
   ```json
   {"filename": "doc.pdf", "project": "backend"}
   ```

2. **Upload file directly to S3** using presigned URL

3. **Submit extraction**: `POST /v1/extract`
   ```json
   {"jobId": "job-xxx", "project": "backend"}
   ```

4. **Poll status**: `GET /v1/jobs/{jobId}`

5. **Download results**: `GET /v1/jobs/{jobId}/results`

See [docs/api-reference.md](docs/api-reference.md) for complete API documentation.

## Environment Variables

Set in ComputeStack for Batch jobs:

- `STORAGE_BACKEND=s3`
- `LLM_PROVIDER=gemini_api`: Provider selector required by llm_factory
- `BUCKET_NAME`: S3 bucket name
- `JOBS_TABLE`: DynamoDB Jobs table name
- `CLASSES_TABLE`: DynamoDB Classes table name
- `PROPERTIES_TABLE`: DynamoDB Properties table name
- `INSTANCES_TABLE`: DynamoDB Instances table name
- `SNAPSHOTS_TABLE`: DynamoDB Snapshots table name
- `PROJECTS_TABLE`: DynamoDB Projects table name
- `GEMINI_API_KEY`: Injected from the Secrets Manager secret at task start

Batch jobs also receive at runtime (container overrides):

- `JOB_ID`: Unique job identifier (omitted for scheduled jobs; batch_runner
  derives one from AWS_BATCH_JOB_ID and creates the Jobs record itself)
- `PROJECT`: Project name
- `SOURCE`: S3 path to input file, or an external URL
- `SOURCE_TYPE`: Optional source type hint (inferred when omitted)
- `LOCK_MAX_WAIT_SECONDS`: Optional wait for the per-project lock (default 1800)

## Deployment Commands Reference

```bash
# Preview changes
cdk diff --all

# Deploy to development
cdk deploy --all -c accountName=mycompany -c environment=dev

# Deploy to production
cdk deploy --all -c accountName=mycompany -c environment=prod --require-approval broadening

# Deploy specific stack
cdk deploy KnowledgeExtractionStorage-dev -c accountName=mycompany -c environment=dev

# Synthesize CloudFormation template
cdk synth KnowledgeExtractionStorage-dev

# Destroy all resources (⚠️ irreversible)
cdk destroy --all -c accountName=mycompany -c environment=dev
```

## Testing

### Test Extraction via API

```bash
# Set API URL
API_URL="https://xxxxx.execute-api.us-east-1.amazonaws.com/v1"

# Fetch the API key (the key ID is in the ApiStack outputs)
API_KEY=$(aws apigateway get-api-key --api-key API_KEY_ID --include-value --query 'value' --output text)

# 1. Get upload URL
RESPONSE=$(curl -X POST $API_URL/upload-url \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"filename":"test.pdf","project":"test"}')

JOB_ID=$(echo $RESPONSE | jq -r '.jobId')
UPLOAD_URL=$(echo $RESPONSE | jq -r '.uploadUrl')

# 2. Upload file (presigned URL, no API key needed)
curl -X PUT "$UPLOAD_URL" \
  -H "Content-Type: application/pdf" \
  --upload-file /path/to/test.pdf

# 3. Submit extraction
curl -X POST $API_URL/extract \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"jobId\":\"$JOB_ID\",\"project\":\"test\"}"

# 4. Poll status (repeat until COMPLETED)
curl -H "x-api-key: $API_KEY" $API_URL/jobs/$JOB_ID

# 5. Download results
curl -H "x-api-key: $API_KEY" $API_URL/jobs/$JOB_ID/results
```

## Monitoring

### CloudWatch Logs

Batch job logs automatically stream to CloudWatch:

```bash
# Find latest job
JOB_ARN=$(aws batch list-jobs --job-queue extraction-queue-dev --job-status RUNNING --query 'jobSummaryList[0].jobArn' --output text)

# Get log stream
LOG_STREAM=$(aws batch describe-jobs --jobs $JOB_ARN --query 'jobs[0].container.logStreamName' --output text)

# Tail logs
aws logs tail /aws/batch/job --follow --log-stream-names $LOG_STREAM
```

### CloudWatch Metrics

Monitor job performance:

```bash
# View job duration
aws cloudwatch get-metric-statistics \
  --namespace AWS/Batch \
  --metric-name JobDuration \
  --dimensions Name=JobQueue,Value=extraction-queue-dev \
  --start-time $(date -u -d '24 hours ago' --iso-8601=seconds) \
  --end-time $(date -u --iso-8601=seconds) \
  --period 3600 \
  --statistics Average
```

See [docs/operations.md](docs/operations.md) for complete monitoring guide.

## Cost Optimization

Current setup is optimized for cost:

- ✅ **Fargate Spot**: 70% cheaper than on-demand
- ✅ **DynamoDB On-Demand**: Pay per request, no idle charges
- ✅ **Lambda ARM64**: 20% cheaper than x86
- ✅ **S3 Lifecycle**: Auto-delete old input files and working directories
- ✅ **Gemini Flash Lite**: Cheapest LLM model

**Estimated monthly cost** (100 files/day):
- Gemini API: $25-35
- Fargate Spot: $15-25
- Lambda: $5-10
- DynamoDB: $10-15
- S3: $3-5
- CloudWatch: $2-3
- **Total: $41-70/month**

## Troubleshooting

### Job stuck in PENDING

**Cause**: No Fargate capacity available

**Solution**:
```bash
# Check compute environment
aws batch describe-compute-environments --compute-environments extraction-compute-env-dev

# Increase max vCPUs in ComputeStack if needed
```

### Extraction fails with Gemini API error

**Cause**: API key expired or invalid

**Solution**:
```bash
# Rotate API key in Secrets Manager
aws secretsmanager update-secret \
  --secret-id gemini-api-key-dev \
  --secret-string "NEW_API_KEY"
```

### Docker image pull fails

**Cause**: Image not in ECR or IAM permissions missing

**Solution**:
```bash
# Verify image exists
aws ecr describe-images --repository-name kg-extraction-dev

# Re-push image if missing (see Quick Start step 4)
```

## Security

- All API endpoints require an API key (`x-api-key` header); the key is
  created by the ApiStack with a per-key usage plan (100 req/s, burst 200)
- All S3 buckets are private (no public access)
- DynamoDB tables encrypted at rest
- Secrets stored in AWS Secrets Manager, injected into containers at task start
- Known gap: Lambda IAM grants are broader than needed (all functions share
  one grant set); tightening to per-function least privilege is planned

## Documentation

Complete documentation available in [`docs/`](docs/):

- [use-cases.md](docs/use-cases.md) - 5 use cases with flows
- [architecture.md](docs/architecture.md) - System architecture and data flow
- [database-schema.md](docs/database-schema.md) - DynamoDB table schemas
- [api-reference.md](docs/api-reference.md) - REST API and MCP server reference
- [operations.md](docs/operations.md) - Deployment, monitoring, cost tracking

## Next Steps

1. ✅ Deploy infrastructure with CDK
2. ✅ Build and push Docker image to ECR
3. ✅ Create Gemini API key secret
4. ⏭️ Test extraction via API
5. ⏭️ Configure scheduled jobs (EventBridge)
6. ⏭️ Implement full MCP server (currently stub)
7. ⏭️ Add CloudWatch alarms for monitoring
8. ⏭️ Set up CI/CD pipeline

## Support

For issues or questions:

- Check [docs/operations.md](docs/operations.md) troubleshooting section
- Review CloudWatch Logs for batch job errors
- Check DynamoDB Jobs table for error messages
- Review API Gateway logs for endpoint issues

## License

See main project LICENSE file.
