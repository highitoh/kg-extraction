# API Reference

## REST API

Base URL: `https://api.example.com/v1`

All endpoints return JSON. Errors follow standard HTTP status codes.

### Authentication

All endpoints require an API Gateway API key in the `x-api-key` header:

```
x-api-key: YOUR_API_KEY
```

The key is created at deployment (ApiStack output `ApiKeyId`); fetch its
value with:

```bash
aws apigateway get-api-key --api-key API_KEY_ID --include-value --query 'value' --output text
```

Requests without a valid key receive `403 Forbidden`. CORS preflight
(OPTIONS) requests are exempt. Presigned S3 URLs (upload/download) are
authorized by their signature and need no API key.

### Rate Limiting

Throttling is enforced by the API Gateway usage plan attached to the key:

- Rate: 100 requests/second per key
- Burst: 200 requests
- Exceeding the limit returns `429 Too Many Requests` (API Gateway does not
  emit `X-RateLimit-*` response headers)

---

## Extraction & Jobs

### POST /v1/upload-url

Generate presigned S3 URL for file upload (step 1 of file upload flow).

**Request**:
```json
{
  "filename": "architecture.pdf",
  "project": "backend"
}
```

**Response** (200 OK):
```json
{
  "jobId": "job-abc123",
  "uploadUrl": "https://s3.amazonaws.com/bucket/input/job-abc123/architecture.pdf?X-Amz-...",
  "expiresIn": 900
}
```

**Errors**:
- `400 Bad Request`: Invalid filename or project name
- `429 Too Many Requests`: Rate limit exceeded

---

### POST /v1/extract

Submit extraction job after file upload (step 2 of file upload flow).

**Request**:
```json
{
  "jobId": "job-abc123",
  "project": "backend"
}
```

**Response** (200 OK):
```json
{
  "jobId": "job-abc123",
  "status": "PENDING",
  "message": "Extraction job submitted"
}
```

**Errors**:
- `400 Bad Request`: Missing jobId or project
- `404 Not Found`: File not uploaded to S3 yet
- `500 Internal Server Error`: Batch job submission failed

---

### GET /v1/jobs/{jobId}

Get job status and progress.

**Response** (200 OK):
```json
{
  "jobId": "job-abc123",
  "project": "backend",
  "status": "RUNNING",
  "progress": 45,
  "progressMessage": "Extracting classes (47 found)",
  "sourceType": "pdf",
  "filename": "architecture.pdf",
  "createdAt": "2026-06-28T10:00:00Z",
  "startedAt": "2026-06-28T10:01:30Z",
  "batchJobId": "arn:aws:batch:...",
  "tokenUsage": {
    "inputTokens": 23700,
    "outputTokens": 40700,
    "totalTokens": 64400
  }
}
```

**Status Values**:
- `PENDING_UPLOAD`: Waiting for file upload
- `PENDING`: Queued, waiting for worker
- `RUNNING`: Extraction in progress
- `COMPLETED`: Extraction successful
- `FAILED`: Extraction failed (see errorMessage)

**Errors**:
- `404 Not Found`: Job ID does not exist

---

### GET /v1/jobs/{jobId}/results

Download extraction results (presigned URLs).

**Response** (200 OK):
```json
{
  "jobId": "job-abc123",
  "status": "COMPLETED",
  "outputs": {
    "ttl": "https://s3.amazonaws.com/.../knowledge_graph_20260628_100430.ttl?X-Amz-...",
    "nodesCsv": "https://s3.amazonaws.com/.../nodes.csv?X-Amz-...",
    "relationshipsCsv": "https://s3.amazonaws.com/.../relationships.csv?X-Amz-...",
    "knowledgeJson": "https://s3.amazonaws.com/.../knowledge.json?X-Amz-..."
  },
  "expiresIn": 900
}
```

**Errors**:
- `404 Not Found`: Job not found
- `409 Conflict`: Job not completed yet

---

## Projects

### GET /v1/projects

List all projects with statistics.

**Query Parameters**:
- `limit` (optional): Number of results (default: 20, max: 100)
- `nextToken` (optional): Pagination token

**Response** (200 OK):
```json
{
  "projects": [
    {
      "project": "backend",
      "projectName": "Backend System Architecture",
      "description": "Knowledge graph for backend services",
      "totalExtractions": 23,
      "totalSnapshots": 15,
      "lastExtractionAt": "2026-06-28T10:04:35Z",
      "sources": ["pdf", "confluence", "miro"],
      "tags": ["architecture", "backend"]
    }
  ],
  "nextToken": "eyJ..."
}
```

---

### GET /v1/projects/{project}

Get project metadata and statistics.

**Response** (200 OK):
```json
{
  "project": "backend",
  "projectName": "Backend System Architecture",
  "description": "Knowledge graph for backend services and APIs",
  "createdAt": "2026-06-01T00:00:00Z",
  "lastExtractionAt": "2026-06-28T10:04:35Z",
  "totalExtractions": 23,
  "totalSnapshots": 15,
  "currentSnapshotId": "20260628100430",
  "sources": ["pdf", "confluence", "miro"],
  "tags": ["architecture", "backend", "services"]
}
```

**Errors**:
- `404 Not Found`: Project does not exist

---

### GET /v1/projects/{project}/extractions

Get extraction history for a project.

**Query Parameters**:
- `limit` (optional): Number of results (default: 10, max: 100)

**Response** (200 OK):
```json
{
  "project": "backend",
  "extractions": [
    {
      "extractedAt": "2026-06-28T10:04:35Z",
      "sourceFile": "architecture.pdf",
      "classesExtracted": 12,
      "propertiesExtracted": 34,
      "tokenUsage": 64400,
      "cost": 0.0140,
      "durationSeconds": 180
    }
  ],
  "totalReturned": 10
}
```

---

## Snapshots & Changesets

### GET /v1/projects/{project}/snapshots

List all snapshots for a project.

**Query Parameters**:
- `limit` (optional): Number of results (default: 20, max: 100)
- `nextToken` (optional): Pagination token

**Response** (200 OK):
```json
{
  "project": "backend",
  "snapshots": [
    {
      "snapshotId": "20260628100430",
      "triggerType": "extraction",
      "sourceType": "pdf",
      "classCount": 47,
      "propertyCount": 156,
      "createdAt": "2026-06-28T10:04:30Z",
      "downloadUrl": "https://s3.amazonaws.com/.../snapshot.json?X-Amz-..."
    }
  ],
  "nextToken": null
}
```

---

### POST /v1/projects/{project}/changesets

> ⚠️ **Currently disabled.** The route is not deployed because changeset
> generation is not yet implemented in the Batch container; requests would
> accept and then always fail. The specification below describes the intended
> behavior once implemented.

Generate changeset between two snapshots.

**Request**:
```json
{
  "beforeSnapshotId": "20260627140000",
  "afterSnapshotId": "20260628100430"
}
```

**Response** (202 Accepted):
```json
{
  "changesetId": "cs_20260628_150000_a1b2c3",
  "status": "PENDING",
  "message": "Changeset generation job submitted",
  "statusUrl": "/v1/projects/backend/changesets/cs_20260628_150000_a1b2c3"
}
```

**Errors**:
- `404 Not Found`: Snapshot does not exist
- `400 Bad Request`: Invalid snapshot IDs

---

### GET /v1/projects/{project}/changesets/{changesetId}

Get changeset details.

**Response** (200 OK):
```json
{
  "changesetId": "cs_20260628_150000_a1b2c3",
  "project": "backend",
  "beforeSnapshotId": "20260627140000",
  "afterSnapshotId": "20260628100430",
  "changeCount": 23,
  "driverCount": 5,
  "createdAt": "2026-06-28T15:00:00Z",
  "downloadUrl": "https://s3.amazonaws.com/.../changeset.json?X-Amz-..."
}
```

---

## Knowledge Graph API

### GET /v1/projects/{project}/knowledge/classes

List all classes in the knowledge graph.

**Query Parameters**:
- `limit` (optional): Number of results (default: 50, max: 200)
- `nextToken` (optional): Pagination token
- `classType` (optional): Filter by ACCEPT | REJECT | PENDING

**Response** (200 OK):
```json
{
  "project": "backend",
  "classes": [
    {
      "className": "AWS Lambda",
      "classIri": "https://github.com/highitoh/kg-extraction#system_component",
      "classType": "ACCEPT",
      "definition": "Serverless compute service for running code",
      "instanceCount": 2,
      "sourceDocuments": ["architecture.pdf", "design-doc.pdf"],
      "createdAt": "2026-06-28T10:04:35Z"
    }
  ],
  "totalReturned": 47,
  "nextToken": null
}
```

---

### GET /v1/projects/{project}/knowledge/classes/{className}

Get specific class details by class name (URL-encoded). Lookup is
case-insensitive: names are normalized before matching (see
[database-schema.md](database-schema.md), "Key Normalization"). The class IRI
cannot be used here - it identifies the metamodel *type* shared by many
classes, not a single class.

**Example**: `/v1/projects/backend/knowledge/classes/AWS%20Lambda`

**Response** (200 OK). `instances` contains the most recent 50 items of
evidence; `instanceCount` is the total:
```json
{
  "className": "AWS Lambda",
  "classIri": "https://github.com/highitoh/kg-extraction#system_component",
  "classType": "ACCEPT",
  "definition": "Serverless compute service that runs code in response to events",
  "instanceCount": 1,
  "instances": [
    {
      "instanceId": "abc123def456",
      "label": "AWS Lambda",
      "source": "Lambda is a serverless compute service...",
      "strength": 0.95,
      "extractedAt": "2026-06-28T10:04:35Z",
      "snapshotId": "20260628100430",
      "sourceDocument": "architecture.pdf"
    }
  ],
  "sourceDocuments": ["architecture.pdf", "design-doc.pdf"]
}
```

**Errors**:
- `404 Not Found`: Class not found

---

### GET /v1/projects/{project}/knowledge/classes/{className}/properties

Get all properties FROM a specific class (outgoing relationships).

**Query Parameters**:
- `limit` (optional): Number of results (default: 50, max: 200)

**Response** (200 OK):
```json
{
  "sourceClass": "API Gateway",
  "properties": [
    {
      "propertyName": "invokes",
      "targetClassName": "AWS Lambda",
      "propertyIri": "https://github.com/highitoh/kg-extraction#invokes",
      "averageStrength": 0.89,
      "instanceCount": 2,
      "definition": "API Gateway invokes Lambda functions to handle requests"
    }
  ],
  "totalReturned": 5
}
```

---

### GET /v1/projects/{project}/knowledge/classes/{className}/incoming

Get all properties TO a specific class (incoming relationships, reverse lookup).

**Query Parameters**:
- `limit` (optional): Number of results (default: 50, max: 200)

**Response** (200 OK):
```json
{
  "targetClass": "AWS Lambda",
  "incomingProperties": [
    {
      "propertyName": "invokes",
      "sourceClassName": "API Gateway",
      "propertyIri": "https://github.com/highitoh/kg-extraction#invokes",
      "averageStrength": 0.89,
      "instanceCount": 2
    },
    {
      "propertyName": "uses",
      "sourceClassName": "Microservice A",
      "averageStrength": 0.92,
      "instanceCount": 1
    }
  ],
  "totalReturned": 2
}
```

---

## MCP Server

The MCP server exposes the extraction system via Model Context Protocol, allowing Claude Desktop and other MCP clients to interact through natural language.

### Endpoint

**URL**: `https://api.example.com/v1/mcp`

**Transport**: HTTP with Server-Sent Events (SSE)

**Headers**:
- `Content-Type`: `application/json` (request)
- `Accept`: `text/event-stream` (response)
- `x-session-id`: Session ID for persistence (optional, returned by server if omitted)

### Configuration (Claude Desktop)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "knowledge-extraction": {
      "url": "https://api.example.com/v1/mcp",
      "transport": "http",
      "headers": {
        "x-api-key": "YOUR_API_KEY",
        "x-session-id": "550e8400-e29b-41d4-a716-446655440000"
      }
    }
  }
}
```

**Session Management**:
- First connection: Server generates `sessionId`, returns in `x-session-id` response header
- Subsequent requests: Include `x-session-id` header to resume session
- Session expires after 30 days of inactivity (DynamoDB TTL)

---

### MCP Tools

#### 1. extract_knowledge

Upload a file or URL and extract knowledge graph.

**Input Schema**:
```json
{
  "file": "base64-encoded-file-or-url",
  "filename": "architecture.pdf",
  "project": "backend"
}
```

**Response** (Streaming SSE):
```
data: {"type":"progress","jobId":"job-xyz789","status":"SUBMITTED","progress":0}

data: {"type":"progress","progress":15,"message":"Extracting text from PDF"}

data: {"type":"progress","progress":45,"message":"Extracting classes (47 found)"}

data: {"type":"result","content":[{"type":"text","text":"Extraction complete! Found 47 classes and 156 properties.\n\nTokens used: 64,400 (Cost: $0.0140)"}]}
```

---

#### 2. list_projects

List all available projects.

**Input Schema**:
```json
{}
```

**Response**:
```json
{
  "projects": [
    {
      "project": "backend",
      "projectName": "Backend System Architecture",
      "totalExtractions": 23
    }
  ]
}
```

---

#### 3. get_extraction_status

Check status of an ongoing extraction job.

**Input Schema**:
```json
{
  "jobId": "job-abc123"
}
```

**Response**:
```json
{
  "jobId": "job-abc123",
  "status": "RUNNING",
  "progress": 45,
  "progressMessage": "Extracting classes (47 found)"
}
```

---

#### 4. query_knowledge

Query the knowledge graph (search, get class details, explore relationships).

**Input Schema**:
```json
{
  "project": "backend",
  "operation": "search",
  "query": "AWS Lambda"
}
```

**Operations**:
- `search`: Text search across classes and properties
- `getClass`: Get specific class details by IRI
- `getProperties`: Get outgoing relationships from a class
- `getIncoming`: Get incoming relationships to a class
- `getStats`: Get knowledge graph statistics

**Response** (for search operation):
```json
{
  "results": {
    "classes": [
      {
        "className": "AWS Lambda",
        "definition": "Serverless compute service",
        "relevance": 0.95
      }
    ]
  }
}
```

---

#### 5. list_snapshots

View snapshot history for a project.

**Input Schema**:
```json
{
  "project": "backend"
}
```

**Response**:
```json
{
  "snapshots": [
    {
      "snapshotId": "20260628100430",
      "classCount": 47,
      "propertyCount": 156,
      "createdAt": "2026-06-28T10:04:30Z"
    }
  ]
}
```

---

#### 6. compare_snapshots

Generate changeset showing differences between two snapshots.

**Input Schema**:
```json
{
  "project": "backend",
  "beforeSnapshotId": "20260627140000",
  "afterSnapshotId": "20260628100430"
}
```

**Response**:
```json
{
  "changesetId": "cs_20260628_150000_a1b2c3",
  "status": "PENDING",
  "message": "Changeset generation started"
}
```

---

#### 7. get_session_stats

View session statistics (tokens, cost, conversation history).

**Input Schema**:
```json
{
  "includeHistory": true,
  "limit": 10,
  "maxAgeDays": 365
}
```

**Parameters**:
- `includeHistory`: Include conversation history in response (default: false)
- `limit`: Number of recent interactions to return (default: 10, max: 1000)
- `maxAgeDays`: Maximum age of interactions to include in days (default: 365 for one year)

**Response**:
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "totalTokensUsed": 128800,
  "totalCost": 0.0280,
  "interactionCount": 2,
  "conversationHistory": [
    {
      "timestamp": "2026-06-28T10:15:30Z",
      "tool": "extract_knowledge",
      "tokens": 64400,
      "cost": 0.0140,
      "durationMs": 180000
    }
  ]
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid project name: must be alphanumeric with hyphens",
    "details": {
      "field": "project",
      "value": "backend@123"
    }
  }
}
```

### Error Codes

- `VALIDATION_ERROR`: Invalid request parameters (400)
- `NOT_FOUND`: Resource does not exist (404)
- `CONFLICT`: Resource in invalid state (409)
- `RATE_LIMIT_EXCEEDED`: Too many requests (429)
- `INTERNAL_ERROR`: Server error (500)
- `SERVICE_UNAVAILABLE`: Downstream service unavailable (503)

---

## Pagination

Endpoints that return lists support pagination:

**Request**:
```
GET /v1/projects?limit=20&nextToken=eyJ...
```

**Response**:
```json
{
  "items": [...],
  "nextToken": "eyJsYXN0RXZhbHVhdGVkS2V5Ijp7InByb2plY3QiOiJiYWNrZW5kIn19"
}
```

**Notes**:
- `nextToken` is opaque, do not parse or modify
- `nextToken=null` means no more results
- Default limit: 20-50 (varies by endpoint)
- Max limit: 100-200 (varies by endpoint)
