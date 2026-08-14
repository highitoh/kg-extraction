# Operations Guide

## Deployment

### Prerequisites

- AWS CLI configured with appropriate credentials
- Node.js 18+ and npm installed
- Docker installed (for building container images)
- AWS CDK CLI: `npm install -g aws-cdk`

### Initial Setup

1. **Clone repository and install dependencies**:
   ```bash
   cd infrastructure
   npm install
   ```

2. **Bootstrap CDK** (first time only per AWS account/region):
   ```bash
   cdk bootstrap aws://ACCOUNT-ID/REGION
   ```

3. **Configure deployment settings** in `cdk.local.json` (copy
   `cdk.local.example.json`):
   ```json
   {
     "orgName": "mycompany",
     "environment": "dev",
     "awsRegion": "ap-northeast-1",
     "gcpProjectId": "my-gcp-project",
     "gcpLocation": "global"
   }
   ```

   Individual values can be overridden per invocation, which takes precedence:
   ```bash
   cdk deploy --all -c orgName=mycompany -c environment=dev
   ```

### Deployment Commands

**Deploy all stacks** (recommended for first deployment):
```bash
cdk deploy --all
```

**Deploy specific stack**:
```bash
# Storage only (S3 + DynamoDB)
cdk deploy KnowledgeExtractionStorage-dev

# Compute only (AWS Batch)
cdk deploy KnowledgeExtractionCompute-dev

# API only (API Gateway + Lambda + MCP)
cdk deploy KnowledgeExtractionApi-dev

# Events only (EventBridge scheduled jobs)
cdk deploy KnowledgeExtractionEvents-dev
```

**Preview changes before deployment** (CloudFormation changeset):
```bash
cdk diff --all
```

**Synthesize CloudFormation template** (without deploying):
```bash
cdk synth KnowledgeExtractionStorage-dev
```

**Destroy all resources** (⚠️ irreversible):
```bash
cdk destroy --all
```

### Build and Push Docker Image

ComputeStack creates the ECR repository, so deploy it first, then push the
extraction workflow image. Batch jobs fail at run time until the image exists.

```bash
cd infra
./push-image.sh
```

| Option | Default | Purpose |
|---|---|---|
| `--environment ENV` | `environment` from `cdk.local.json` | Which stack to read the repository from |
| `--tag TAG` | `latest` | Tag to push |
| `--skip-build` | off | Re-push the existing local image |

To run the steps manually:

```bash
# Resolve the repository URI from the deployed stack
ECR_URI=$(aws cloudformation describe-stacks \
  --stack-name KnowledgeExtractionCompute-dev \
  --query 'Stacks[0].Outputs[?OutputKey==`EcrRepositoryUri`].OutputValue' \
  --output text)

# Derive the registry host and its region from that URI
REGISTRY="${ECR_URI%%/*}"
REGION=$(echo "$REGISTRY" | cut -d. -f4)

# Build, tag and push
cd /workspace
docker build -t kg-extraction:latest -f infra/docker/Dockerfile .

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

docker tag kg-extraction:latest "$ECR_URI:latest"
docker push "$ECR_URI:latest"
```

**Automated in CDK**: ComputeStack creates ECR repository automatically. Use CDK Docker asset for integrated build/push:

```typescript
const image = ecs.ContainerImage.fromAsset('../app', {
  file: 'Dockerfile',
  buildArgs: {
    PYTHON_VERSION: '3.14'
  }
});
```

### Environment-Specific Deployments

**Development**:
```bash
cdk deploy --all
# Creates: mycompany-kg-extraction-dev
```

**Staging**:
```bash
cdk deploy --all -c environment=staging
# Creates: mycompany-kg-extraction-staging
```

**Production**:
```bash
cdk deploy --all -c environment=prod --require-approval broadening
# Creates: mycompany-kg-extraction-prod
# --require-approval: Ask before creating IAM resources
```

---

## S3 Bucket Naming

Buckets follow the pattern: `{orgName}-kg-extraction-{environment}`

**Why this pattern?**
- **Global uniqueness**: S3 bucket names must be globally unique across all AWS accounts
- **Org name prefix**: Prevents collisions (e.g., "mycompany", "acme-corp")
- **Service identifier**: "kg-extraction" clearly identifies purpose
- **Environment suffix**: Separates dev/staging/prod buckets

**Examples**:
- `mycompany-kg-extraction-dev`
- `mycompany-kg-extraction-staging`
- `mycompany-kg-extraction-prod`
- `acme-corp-kg-extraction-prod`

**Configuration**:

Set `orgName` in `cdk.local.json`:

```json
{
  "orgName": "mycompany"
}
```

Or via CLI:
```bash
cdk deploy --all -c orgName=mycompany -c environment=prod
```

**CDK Implementation** (in StorageStack):

```typescript
const bucketName = `${orgName}-kg-extraction-${environment}`;

const bucket = new s3.Bucket(this, 'ExtractionBucket', {
  bucketName,
  encryption: s3.BucketEncryption.S3_MANAGED,
  // ... other settings
});
```

---

## Monitoring

### CloudWatch Logs

**View logs in AWS Console**:
1. Go to **CloudWatch → Log Groups**
2. Find `/aws/batch/kg-extraction-dev`
3. Filter by job name or search keywords

**View logs via CLI** (recommended for debugging):

```bash
# Find latest running job
JOB_ARN=$(aws batch list-jobs \
  --job-queue extraction-queue-dev \
  --job-status RUNNING \
  --query 'jobSummaryList[0].jobArn' \
  --output text)

# Get log stream name
LOG_STREAM=$(aws batch describe-jobs \
  --jobs $JOB_ARN \
  --query 'jobs[0].container.logStreamName' \
  --output text)

# Tail logs in real-time
aws logs tail /aws/batch/kg-extraction-dev \
  --follow \
  --log-stream-names $LOG_STREAM \
  --format short
```

**Search logs for errors**:
```bash
aws logs filter-log-events \
  --log-group-name /aws/batch/kg-extraction-dev \
  --filter-pattern "ERROR" \
  --start-time $(date -u -d '1 hour ago' +%s)000
```

### CloudWatch Metrics

**View metrics in AWS Console**:
1. Go to **CloudWatch → Metrics**
2. Select namespace: `AWS/Batch` or `KnowledgeExtraction` (custom)
3. View job duration, queue depth, running jobs

**Query metrics via CLI**:

```bash
# Average job duration (last 24 hours)
aws cloudwatch get-metric-statistics \
  --namespace AWS/Batch \
  --metric-name JobDuration \
  --dimensions Name=JobQueue,Value=extraction-queue-dev \
  --start-time $(date -u -d '24 hours ago' --iso-8601=seconds) \
  --end-time $(date -u --iso-8601=seconds) \
  --period 3600 \
  --statistics Average

# Running jobs (real-time)
aws cloudwatch get-metric-statistics \
  --namespace AWS/Batch \
  --metric-name RunningJobs \
  --dimensions Name=JobQueue,Value=extraction-queue-dev \
  --start-time $(date -u -d '5 minutes ago' --iso-8601=seconds) \
  --end-time $(date -u --iso-8601=seconds) \
  --period 60 \
  --statistics Average
```

### CloudWatch Dashboard

**Create custom dashboard** (via CDK or console):

```typescript
const dashboard = new cloudwatch.Dashboard(this, 'ExtractionDashboard', {
  dashboardName: 'knowledge-extraction-monitoring',
});

dashboard.addWidgets(
  new cloudwatch.GraphWidget({
    title: 'Job Queue Depth',
    left: [
      new cloudwatch.Metric({
        namespace: 'AWS/Batch',
        metricName: 'QueueDepth',
        dimensionsMap: { JobQueue: 'extraction-queue-dev' },
      })
    ]
  }),
  new cloudwatch.GraphWidget({
    title: 'Daily Extraction Cost',
    left: [
      new cloudwatch.Metric({
        namespace: 'KnowledgeExtraction',
        metricName: 'ExtractionCost',
        statistic: 'Sum',
        period: cdk.Duration.days(1),
      })
    ]
  })
);
```

**Access dashboard**: CloudWatch → Dashboards → knowledge-extraction-monitoring

### CloudWatch Alarms

**Example alarm** (failed jobs):

```typescript
const failedJobsAlarm = new cloudwatch.Alarm(this, 'FailedJobsAlarm', {
  metric: new cloudwatch.Metric({
    namespace: 'AWS/Batch',
    metricName: 'FailedJobs',
    dimensionsMap: { JobQueue: 'extraction-queue-dev' },
    statistic: 'Sum',
    period: cdk.Duration.minutes(5),
  }),
  threshold: 1,
  evaluationPeriods: 1,
  comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
  alarmDescription: 'Alert when batch jobs fail',
});

// Send email notification
const topic = new sns.Topic(this, 'AlertTopic');
topic.addSubscription(new sns_subscriptions.EmailSubscription('team@example.com'));

failedJobsAlarm.addAlarmAction(new cloudwatch_actions.SnsAction(topic));
```

**List active alarms**:
```bash
aws cloudwatch describe-alarms --state-value ALARM
```

### X-Ray Tracing

**Enable X-Ray** (add to batch_runner.py):

```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch AWS SDK and HTTP libraries
patch_all()

@xray_recorder.capture('extraction_workflow')
def main():
    # Automatically traced
    pass
```

**View traces**:
1. Go to **X-Ray → Traces**
2. Filter by service: `kg-extraction`
3. View service map and trace timelines

---

## Progress Tracking

Progress is tracked in DynamoDB (Jobs table) and updated throughout extraction.

### How It Works

1. **batch_runner.py updates DynamoDB** at each stage:
   - 3%: Waiting for project lock
   - 5%: Preparing input
   - 10%: Downloading knowledge.json
   - 15-85%: Extraction workflow (text, classes, properties)
   - 85%: Uploading results
   - 90%: Creating snapshot
   - 93%: Writing knowledge graph to DynamoDB
   - 97%: Updating project records
   - 100%: Complete

2. **Users poll job status** via API:
   ```bash
   curl https://api.example.com/v1/jobs/job-abc123
   ```

3. **MCP server streams progress** to Claude Desktop:
   - Polls DynamoDB every 5 seconds
   - Sends SSE updates to client

### Implementation (batch_runner.py)

```python
import boto3
import os
from datetime import datetime

dynamodb = boto3.client('dynamodb')

def update_progress(job_id: str, progress: int, message: str):
    """Update job progress in DynamoDB"""
    try:
        dynamodb.update_item(
            TableName=os.environ['JOBS_TABLE'],
            Key={'jobId': {'S': job_id}},
            UpdateExpression='SET progress = :p, progressMessage = :m, lastUpdated = :t',
            ExpressionAttributeValues={
                ':p': {'N': str(progress)},
                ':m': {'S': message},
                ':t': {'S': datetime.utcnow().isoformat() + 'Z'}
            }
        )
    except Exception as e:
        # Continue execution even if update fails
        print(f"Failed to update progress: {e}")

# Usage in main()
update_progress(job_id, 5, "Downloading input file from S3")
# ... download file
update_progress(job_id, 15, "Extracting text from PDF")
# ... extract text
update_progress(job_id, 45, f"Extracting classes ({class_count} found)")
```

### Query Progress (Lambda API)

```python
def handler(event, context):
    job_id = event['pathParameters']['jobId']
    
    response = dynamodb.get_item(
        TableName=os.environ['JOBS_TABLE'],
        Key={'jobId': {'S': job_id}}
    )
    
    item = response.get('Item', {})
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'jobId': job_id,
            'status': item.get('status', {}).get('S'),
            'progress': int(item.get('progress', {}).get('N', 0)),
            'progressMessage': item.get('progressMessage', {}).get('S', '')
        })
    }
```

---

## Cost Tracking

### Estimated Monthly Cost (100 files/day)

| Service | Usage | Cost |
|---------|-------|------|
| **Gemini API** | 50K tokens/file × 100 files × 30 days = 150M tokens | $25-35 |
| **Fargate Spot** | 1 vCPU × 3 min × 100 jobs × 30 days = 9,000 min | $15-25 |
| **Lambda** | 10K requests × 200ms avg × 256 MB | $5-10 |
| **DynamoDB** | 300 writes/day, 500 reads/day, 10 GB storage | $10-15 |
| **S3** | 100 uploads, 200 downloads, 50 GB storage | $3-5 |
| **CloudWatch** | 10 GB logs, 50 metrics, 5 dashboards | $2-3 |
| **API Gateway** | 10K requests/day | $1-2 |
| **Total** | | **$41-70/month** |

### Cost Optimization Strategies

1. **Use Fargate Spot** (70% savings vs on-demand):
   ```typescript
   const computeEnv = new batch.ComputeEnvironment(this, 'ComputeEnv', {
     computeResources: {
       type: batch.ComputeResourceType.FARGATE_SPOT,
       maxvCpus: 16,
     }
   });
   ```

2. **DynamoDB On-Demand** (no idle charges):
   - Only pay for actual read/write requests
   - Auto-scales to demand
   - No need to provision capacity

3. **Lambda ARM64** (20% cheaper than x86):
   ```typescript
   new lambda.Function(this, 'ApiFunction', {
     architecture: lambda.Architecture.ARM_64,
     // ...
   });
   ```

4. **S3 Lifecycle Policies** (auto-delete old files):
   ```typescript
   bucket.addLifecycleRule({
     id: 'DeleteOldInputFiles',
     prefix: 'input/',
     expiration: cdk.Duration.days(7),
   });
   ```

5. **CloudWatch Logs Retention**: ComputeStack expires job logs after one
   month. Shorten it in `lib/stacks/compute-stack.ts` to cut storage costs
   further:
   ```typescript
   retention: logs.RetentionDays.ONE_WEEK,
   ```

6. **Use Gemini Flash Lite** (cheapest model):
   - Input: $0.075/1M tokens (vs $0.15 for Flash)
   - Output: $0.30/1M tokens (vs $0.60 for Flash)
   - 50% cost savings

### Cost Monitoring

**Enable Cost Explorer**:
1. Go to **AWS Billing → Cost Explorer**
2. Filter by tag: `Project: knowledge-extraction`
3. Group by service

**Set budget alerts**:
```bash
aws budgets create-budget \
  --account-id ACCOUNT-ID \
  --budget file://budget.json \
  --notifications-with-subscribers file://notifications.json
```

**budget.json**:
```json
{
  "BudgetName": "KnowledgeExtractionBudget",
  "BudgetLimit": {
    "Amount": "100",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
```

---

## Troubleshooting

### Common Issues

#### 1. Job stuck in PENDING status

**Symptom**: Job submitted but never starts (status=PENDING for > 5 min)

**Diagnosis**:
```bash
# Check Fargate capacity
aws batch describe-compute-environments \
  --compute-environments extraction-compute-env-dev

# Check job queue
aws batch describe-job-queues --job-queues extraction-queue-dev
```

**Solutions**:
- Increase max vCPUs in compute environment
- Check Fargate Spot availability (may fall back to on-demand)
- Verify IAM roles have correct permissions

#### 2. Extraction fails with Gemini API error

**Symptom**: Job status=FAILED, errorMessage contains "Gemini API"

**Diagnosis**:
```bash
# Check logs
aws logs tail /aws/batch/kg-extraction-dev --follow --filter-pattern "Gemini"
```

**Common causes**:
- API key expired or invalid (check Secrets Manager)
- Rate limit exceeded (Gemini API throttling)
- Network timeout (increase retry backoff)

**Solutions**:
- Rotate API key in Secrets Manager
- Add exponential backoff in llm_retry.py
- Check Gemini API quota in GCP console

#### 3. S3 presigned URL expired

**Symptom**: User uploads file but gets 403 Forbidden

**Diagnosis**: Presigned URL expires in 15 minutes

**Solutions**:
- Increase expiry time in Lambda (uploadUrlExpiry)
- User should upload immediately after getting URL
- Implement retry logic in client

#### 4. DynamoDB throttling

**Symptom**: Job status updates fail, logs show "ProvisionedThroughputExceededException"

**Diagnosis**:
```bash
# Check throttled requests
aws cloudwatch get-metric-statistics \
  --namespace AWS/DynamoDB \
  --metric-name UserErrors \
  --dimensions Name=TableName,Value=Jobs
```

**Solutions**:
- Tables use on-demand billing by default (no throttling)
- If using provisioned capacity, increase WCU/RCU
- Implement exponential backoff in batch_runner.py

#### 5. Docker image pull fails

**Symptom**: Batch job fails immediately, logs show "CannotPullContainerError"

**Diagnosis**:
```bash
# Check ECR repository
aws ecr describe-images --repository-name kg-extraction-dev
```

**Solutions**:
- Verify image exists in ECR
- Check IAM permissions for Batch job role (ecr:GetAuthorizationToken, ecr:BatchCheckLayerAvailability)
- Re-push Docker image

---

## Maintenance

### Update Docker Image

1. **Make code changes** in `/workspace/app/src/`
2. **Rebuild and push the image**:
   ```bash
   cd infra
   ./push-image.sh
   ```
3. **Update job definition** (CDK automatically updates on next deploy)
4. **Redeploy compute stack**:
   ```bash
   cdk deploy KnowledgeExtractionCompute-dev
   ```

### Rotate GCP API Key

1. **Create new API key** in GCP Console
2. **Update Secrets Manager**:
   ```bash
   aws secretsmanager update-secret \
     --secret-id gemini-api-key-dev \
     --secret-string "new-api-key-value"
   ```
3. **No redeployment needed** (Batch jobs fetch secret at runtime)

### Clean Up Old Data

**Delete old input files** (automated via S3 lifecycle):
- Input files deleted after 7 days
- Working directories deleted after 7 days

**Delete old CloudWatch logs**:
```bash
aws logs delete-log-group --log-group-name /aws/batch/kg-extraction-dev
```

### Backup DynamoDB

**Enable point-in-time recovery** (already enabled in CDK):
- Automatic continuous backups
- Restore to any point in last 35 days

**Create on-demand backup**:
```bash
aws dynamodb create-backup \
  --table-name Jobs \
  --backup-name jobs-backup-$(date +%Y%m%d)
```

---

## Security Best Practices

1. **Enable MFA** on AWS accounts
2. **Use IAM roles** (not access keys) for services
3. **Rotate secrets** regularly (API keys, passwords)
4. **Enable CloudTrail** for audit logging
5. **Encrypt all data** (S3, DynamoDB, Secrets Manager)
6. **API Gateway requires API keys** by default; add Cognito for per-user identity
7. **Review IAM policies** quarterly (least privilege)
8. **Enable GuardDuty** for threat detection
9. **Tag all resources** for cost tracking and access control

---

## Scaling Beyond 100 Files/Day

If daily volume exceeds 100 files:

1. **Increase Fargate capacity**:
   ```typescript
   const computeEnv = new batch.ComputeEnvironment(this, 'ComputeEnv', {
     computeResources: {
       maxvCpus: 64, // Increase from 16
     }
   });
   ```

2. **Add compute environments in multiple AZs** for higher availability

3. **Consider SQS + Worker Pool** for 1000+ files/day:
   - S3 upload → SQS message
   - ECS Fargate workers poll SQS
   - Parallel processing across files
   - Cost: ~$300-400/month

4. **Use DynamoDB Streams** for real-time progress updates (avoid polling)

5. **Enable S3 Transfer Acceleration** for faster uploads from distant locations
