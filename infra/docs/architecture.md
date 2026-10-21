# Architecture

## Overview

The knowledge extraction system deploys as a hybrid serverless architecture on AWS, combining API Gateway + Lambda for lightweight operations with AWS Batch + Fargate for compute-intensive extraction workflows. All extractions use GCP Vertex AI (Gemini models) via Workload Identity Federation.

## Core AWS Services

- **API Gateway**: REST API for job submission, status queries, knowledge graph access
- **Lambda**: Lightweight API handlers, presigned URL generation, MCP server
- **AWS Batch**: Orchestrates extraction workflows on Fargate compute
- **Fargate Spot**: Cost-optimized container execution
- **S3**: File storage for uploads, extraction outputs, snapshots, and changesets
- **DynamoDB**: Job tracking, knowledge graph storage, session management
- **EventBridge**: Scheduled extraction jobs
- **CloudWatch**: Logs, metrics, alarms, dashboards
- **ECR**: Docker image storage

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    API Clients                                │
│         (Web apps, curl, MCP/Claude Desktop)                  │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  API Gateway   │ REST API + MCP endpoint
              └────────┬───────┘
                       │
         ┌─────────────┴──────────────┐
         │                            │
         ▼                            ▼
┌─────────────────┐         ┌──────────────────┐
│ Lambda Functions│         │  MCP Server      │
│                 │         │  (Lambda + SSE)  │
│ • Upload URLs   │         │                  │
│ • Job submit    │         │ • Session mgmt   │
│ • Status query  │         │ • Tool routing   │
│ • KG API        │         │ • Progress stream│
└────────┬────────┘         └────────┬─────────┘
         │                           │
         │    ┌──────────────────────┘
         │    │
         ▼    ▼
┌──────────────────────┐         ┌─────────────────┐
│  DynamoDB Tables     │         │  S3 Buckets     │
├──────────────────────┤         ├─────────────────┤
│ • Jobs               │         │ • Input files   │
│ • Sessions           │         │ • Outputs       │
│ • Classes            │         │ • Snapshots     │
│ • Properties         │         │ • Changesets    │
│ • Instances          │         │ • knowledge.json│
│ • Projects           │         └─────────────────┘
│ • Snapshots metadata │
│ • Changesets metadata│
└──────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│              Extraction Workflow                              │
└──────────────────────────────────────────────────────────────┘

  ┌──────────────┐                ┌────────────────┐
  │ EventBridge  │                │  API Gateway   │
  │ (Scheduled)  │                │  (On-demand)   │
  └──────┬───────┘                └───────┬────────┘
         │                                │
         └────────────┬───────────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │   AWS Batch      │
            │  (Job Queue)     │
            └─────────┬────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Fargate Spot Task     │
         │  (Docker Container)    │
         ├────────────────────────┤
         │ batch_runner.py        │
         │   ↓                    │
         │ extract_workflow.py    │
         │   ↓                    │
         │ LangGraph workflow     │
         └──────┬────────┬────────┘
                │        │
                ▼        ▼
         ┌──────────┐  ┌──────────────┐
         │ S3 I/O   │  │ GCP Vertex AI│
         └──────────┘  │ (Gemini 3.1  │
                       │  Flash Lite) │
                       └──────────────┘
```

## Data Flow

### On-Demand Extraction

1. **User uploads file**:
   - POST /v1/upload-url → Lambda generates presigned S3 URL
   - User uploads directly to S3 (bypasses API Gateway)
   - POST /v1/extract → Lambda submits AWS Batch job

2. **AWS Batch provisions container**:
   - Fargate Spot allocates resources
   - Pulls Docker image from ECR
   - Starts container with batch_runner.py as entrypoint

3. **batch_runner.py orchestrates**:
   - Acquires the per-project lock (serializes writers of knowledge.json)
   - Downloads file from S3 (URLs are passed through to the workflow)
   - Downloads knowledge.json (if exists)
   - Downloads project_profile.json (if exists; enables relevance screening)
   - Calls extract_workflow.py
   - Uploads outputs and updated knowledge.json to S3
   - Creates snapshot (S3 + Snapshots table)
   - Writes knowledge graph to DynamoDB (Classes/Properties aggregates +
     Instances evidence, deduplicated by content-hash IDs)
   - Updates project metadata and extraction history, releases the lock

4. **User polls status**:
   - GET /v1/jobs/{id} → Lambda queries DynamoDB
   - Returns: status, progress (0-100%), progressMessage
   - When status=COMPLETED, user downloads via presigned URLs

### Scheduled Extraction

1. **EventBridge rule triggers** (e.g., daily at 2 AM UTC)
2. **Directly submits AWS Batch job** (no Lambda needed)
3. **batch_runner creates its own Jobs record** (ID derived from
   AWS_BATCH_JOB_ID, stable across retries), so scheduled runs are tracked
   like on-demand ones
4. **Knowledge carry-over**: existing knowledge.json is loaded and merged, so
   unchanged knowledge is preserved and deduplicated across runs

## Storage Architecture

### S3 Bucket Structure

```
s3://{account_name}-kg-extraction-{env}/
├── input/
│   └── {jobId}/{filename}          # User uploads (deleted after 7 days)
├── projects/
│   └── {project}/
│       ├── knowledge.json          # Current accumulated knowledge
│       ├── project_profile.json    # Optional, user-provided: what the project is about
│       ├── outputs/                # Timestamped outputs
│       │   ├── knowledge_graph_*.ttl
│       │   ├── nodes.csv
│       │   └── relationships.csv
│       ├── snapshots/
│       │   └── {snapshotId}/
│       │       ├── knowledge.json
│       │       └── snapshot.json
│       └── changesets/
│           └── {changesetId}/
│               └── changeset.json
└── working/                        # Temp job directories (deleted after 7 days)
```

**Lifecycle Policies**:
- Input files: Delete after 7 days
- Working directories: Delete after 7 days

### DynamoDB Schema

> 📖 See [database-schema.md](database-schema.md) for complete details

## MCP Server

- **HTTP Streaming**: Server-Sent Events (SSE) for real-time progress updates
- **Session Persistence**: DynamoDB-backed sessions survive Claude Desktop restarts
- **Tool Abstraction**: Hides 3-step presigned URL flow from users
- **Token Tracking**: Records Gemini API usage per interaction
- **Conversation History**: Stores interactions for one year with metadata

## Cross-Cloud Integration

### AWS -> GCP Vertex AI

All LLM inference uses GCP Vertex AI, authenticated with Workload Identity
Federation. The Batch job role is the principal GCP trusts, so no LLM
credential is stored on the AWS side.

## Deployment with AWS CDK

Infrastructure as code using TypeScript.

### Four CDK Stacks

1. **StorageStack**: S3 bucket + DynamoDB tables
2. **ComputeStack**: ECR + AWS Batch
3. **ApiStack**: API Gateway + Lambda functions + MCP server
4. **EventsStack**: EventBridge scheduled rules

### Environment Configuration

Context variables control environment-specific settings:

- `orgName`: S3 bucket prefix (e.g., "mycompany")
- `environment`: dev, staging, prod
- Bucket name format: `{orgName}-kg-extraction-{environment}`

> 📋 For deployment instructions, see [README.md](../README.md) (quick start) or [operations.md](operations.md) (complete guide)

## Security

1. **API Key Authentication**: Every API method requires an API Gateway API
   key (`x-api-key` header) with a per-key usage plan (100 req/s, burst 200).
   The key is created by the ApiStack; see [api-reference.md](api-reference.md)
   for how to fetch its value. For per-user identity and authorization, add
   Cognito on top.

2. **Miro API Key Secret**: The secret (`miro-api-key-{environment}`) is created
   by the ComputeStack with placeholder value
   `PLACEHOLDER_UPDATE_ME_AFTER_DEPLOYMENT` and injected into Batch containers
   as `MIRO_API_KEY`, read by the Batch **execution** role at task start. Only
   Miro sources need it; until it is set they exit with code 2. Set it once
   (no redeployment needed):
   ```bash
   aws secretsmanager put-secret-value \
     --secret-id miro-api-key-dev \
     --secret-string "YOUR_MIRO_API_TOKEN"
   ```
   The secret is retained on stack deletion; `delete-secret
   --force-delete-without-recovery` is required before the same name can be
   recreated.

3. **IP Allowlist**: With `allowed-ips.txt` (see [README.md](../README.md)),
   the API Gateway resource policy and the S3 bucket policy reject API calls
   and presigned URL requests from unlisted addresses. Batch tasks accept no
   inbound traffic.

4. **Known gap**: Lambda IAM grants are broader than needed (all functions
   share one grant set); tightening to per-function least privilege is
   planned.
