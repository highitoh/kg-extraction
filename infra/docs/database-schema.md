# Database Schema

All DynamoDB tables use on-demand billing, encryption at rest, and point-in-time recovery.

## Key Normalization

Class names are extracted by an LLM from natural language, so the same concept can
appear as "AWS Lambda", "aws lambda", or "Lambda #1". To keep identities stable,
every key that contains a class name stores the **normalized** form:

1. `#` characters are removed (reserved as the composite-key delimiter)
2. Whitespace runs collapse to a single space, leading/trailing whitespace is trimmed
3. The result is casefolded (identity is case-insensitive)

The original spelling is preserved in the `displayName` attribute and returned by
the API. Write paths (batch_runner.py) and read paths (Lambda handlers) apply the
same rule — see `lambda/knowledge_api/normalize.py`.

**Normalized keys are immutable identities.** Renaming or merging a class requires
rewriting every Classes, Properties, and Instances item whose key embeds the name.
This is a deliberate trade-off for human-readable keys and cheap lookups.

---

## Table 1: Jobs

**Purpose**: Track extraction job lifecycle, progress, and results

**Primary Key**:
- **PK**: `jobId` (String) - UUID generated on job creation

`jobId` is globally unique, so no sort key is needed. Writers create the record
with `ConditionExpression: attribute_not_exists(jobId)` so retries can never
produce duplicate records, and every reader/updater addresses the item with
`jobId` alone (batch_runner only receives `JOB_ID`, not a timestamp).

Scheduled EventBridge jobs receive no `JOB_ID`; batch_runner derives one from
`AWS_BATCH_JOB_ID` (stable across retry attempts) and creates the record
itself, so scheduled runs are tracked exactly like on-demand ones.

**Attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | Number | Job creation time (Unix timestamp); sort key of project-timestamp-index |
| `project` | String | Project name (e.g., "backend", "confluence-daily") |
| `jobType` | String | extraction \| scheduled \| changeset (set for batch_runner-created records) |
| `status` | String | PENDING_UPLOAD \| PENDING \| RUNNING \| COMPLETED \| FAILED |
| `sourceType` | String | pdf \| docx \| pptx \| txt \| markdown \| miro \| github \| confluence \| jira |
| `sourceLocation` | String | S3 path or external URL |
| `filename` | String | Original filename |
| `uploadUrl` | String | Presigned S3 URL (only for PENDING_UPLOAD status) |
| `uploadUrlExpiry` | Number | Unix timestamp when presigned URL expires |
| `batchJobId` | String | AWS Batch job ARN |
| `createdAt` | String | ISO timestamp when job was created |
| `uploadedAt` | String | ISO timestamp when file upload completed |
| `startedAt` | String | ISO timestamp when extraction started |
| `completedAt` | String | ISO timestamp when extraction completed |
| `outputLocation` | String | S3 project root (no trailing slash); readers append `/outputs/` |
| `errorMessage` | String | Error details if status=FAILED |
| `metadata` | Map | `{fileSize: 5242880, pageCount: 12}` |
| `tokenUsage` | Map | `{inputTokens: 23700, outputTokens: 40700, totalTokens: 64400}` |
| `progress` | Number | 0-100 (updated during extraction) |
| `progressMessage` | String | Current stage (e.g., "Extracting properties (15/47 classes)") |

**GSI: project-timestamp-index**
- **PK**: `project` (String)
- **SK**: `timestamp` (Number)
- **Projection**: ALL
- **Purpose**: Query all jobs for a project, sorted by time

**Status Flow**:
1. `PENDING_UPLOAD`: Upload URL generated, waiting for file
2. `PENDING`: File uploaded, waiting for Batch job to start
3. `RUNNING`: Batch job executing
4. `COMPLETED`: Extraction successful
5. `FAILED`: Extraction failed

**Example Record**:
```json
{
  "jobId": "job-abc123",
  "timestamp": 1719820800,
  "project": "backend",
  "status": "COMPLETED",
  "sourceType": "pdf",
  "sourceLocation": "s3://mycompany-kg-extraction-prod/input/job-abc123/architecture.pdf",
  "filename": "architecture.pdf",
  "batchJobId": "arn:aws:batch:us-east-1:123456789012:job/extraction-abc123",
  "createdAt": "2026-06-28T10:00:00Z",
  "uploadedAt": "2026-06-28T10:00:08Z",
  "startedAt": "2026-06-28T10:01:31Z",
  "completedAt": "2026-06-28T10:04:35Z",
  "outputLocation": "s3://mycompany-kg-extraction-prod/projects/backend",
  "tokenUsage": {
    "inputTokens": 23700,
    "outputTokens": 40700,
    "totalTokens": 64400
  },
  "progress": 100,
  "progressMessage": "Extraction completed successfully",
  "metadata": {
    "fileSize": 5242880,
    "pageCount": 12
  }
}
```

---

## Table 2: Snapshots

**Purpose**: Track knowledge graph snapshots for versioning and rollback

**Primary Key**:
- **PK**: `project` (String)
- **SK**: `snapshotId` (String) - Format: `YYYYMMDDHHMMSS` (e.g., "20260628100430")

The fixed-width timestamp keeps lexicographic order chronological. Because the
resolution is one second, writers must use
`ConditionExpression: attribute_not_exists(snapshotId)` and, on collision,
append a short random suffix (`YYYYMMDDHHMMSS_xxxxxx`) rather than silently
overwriting an existing snapshot. Timestamps are UTC.

**Attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `triggerType` | String | extraction \| optimization \| manual |
| `sourceType` | String | pdf \| confluence \| etc. |
| `s3Location` | String | S3 path to snapshot.json |
| `classCount` | Number | Number of classes in this snapshot |
| `propertyCount` | Number | Number of properties in this snapshot |
| `createdAt` | String | ISO timestamp |

**No GSI** (query by project, sort by snapshotId desc for latest)

**Example Record**:
```json
{
  "project": "backend",
  "snapshotId": "20260628100430",
  "triggerType": "extraction",
  "sourceType": "pdf",
  "s3Location": "s3://mycompany-kg-extraction-prod/projects/backend/snapshots/20260628100430/snapshot.json",
  "classCount": 47,
  "propertyCount": 156,
  "createdAt": "2026-06-28T10:04:30Z"
}
```

---

## Table 3: Changesets

**Purpose**: Track differences between snapshots with business drivers

**Primary Key**:
- **PK**: `project` (String)
- **SK**: `changesetId` (String) - Format: `cs_YYYYMMDD_HHMMSS_xxxxxx`

The random suffix prevents same-second collisions; the timestamp prefix keeps
lexicographic order chronological. Writers use
`ConditionExpression: attribute_not_exists(changesetId)`.

**Attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `beforeSnapshotId` | String | Source snapshot ID |
| `afterSnapshotId` | String | Target snapshot ID |
| `status` | String | PENDING \| RUNNING \| COMPLETED \| FAILED (generation is async) |
| `batchJobId` | String | AWS Batch job ARN of the generation job |
| `errorMessage` | String | Error details if status=FAILED |
| `s3Location` | String | S3 path to changeset.json (set when COMPLETED) |
| `totalChanges` | Number | Total changes (added + updated + deleted, nodes and relationships) |
| `driversCount` | Number | Number of change drivers extracted |
| `createdAt` | String | ISO timestamp (record creation by the API) |
| `startedAt` | String | ISO timestamp (Batch job start) |
| `completedAt` | String | ISO timestamp (Batch job completion) |

**No GSI** (query by project, sort by changesetId desc)

**Example Record**:
```json
{
  "project": "backend",
  "changesetId": "cs_20260628_150000_a1b2c3",
  "beforeSnapshotId": "20260627140000",
  "afterSnapshotId": "20260628100430",
  "status": "COMPLETED",
  "batchJobId": "arn:aws:batch:us-east-1:123456789012:job/changeset-abc123",
  "s3Location": "s3://mycompany-kg-extraction-prod/projects/backend/changesets/cs_20260628_150000_a1b2c3/changeset.json",
  "totalChanges": 23,
  "driversCount": 5,
  "createdAt": "2026-06-28T15:00:00Z",
  "startedAt": "2026-06-28T15:00:31Z",
  "completedAt": "2026-06-28T15:04:12Z"
}
```

---

## Table 4: Sessions

**Purpose**: MCP session persistence for Claude Desktop, conversation history tracking

**Primary Key**:
- **PK**: `sessionId` (String) - UUID generated on first connection
- **SK**: `sk` (String) - `"STATE"` or `"INTERACTION#{timestamp}#{id}"`

Each session is an item collection: one STATE item plus one item per tool
interaction. This keeps every item small and bounded (a session never
approaches the 400KB item limit no matter how active it is), makes appends
cheap (write cost is the interaction size, not the whole history), and lets
DynamoDB TTL prune old interactions natively instead of the application
rewriting a list attribute.

**STATE item attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `state` | Map | Current session state (last project, recent jobs, last activity) |
| `totalTokensUsed` | Number | Cumulative Gemini API tokens across all interactions |
| `totalCost` | Number | Cumulative cost in USD |
| `interactionCount` | Number | Total number of tool calls |
| `createdAt` | String | ISO timestamp when session created |
| `expiresAt` | Number | Unix timestamp for TTL (30 days from last activity, sliding) |

**INTERACTION item attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | String | ISO timestamp of the interaction |
| `tool` | String | MCP tool name (e.g., "extract_knowledge") |
| `request` | String | JSON-encoded request summary |
| `response` | String | JSON-encoded response summary |
| `tokens` | Map | `{inputTokens, outputTokens, totalTokens}` |
| `cost` | Number | Interaction cost in USD |
| `durationMs` | Number | Interaction duration |
| `expiresAt` | Number | Unix timestamp for TTL (365 days after the interaction) |

**TTL**: Enabled on `expiresAt` (applies per item: STATE slides with activity,
interactions expire after one year)

**No GSI** (direct lookup by sessionId; history is a sort-key range query)

**Example Records**:
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "sk": "STATE",
  "state": {
    "currentProject": "backend",
    "recentJobs": ["job-abc123", "job-xyz789"],
    "lastActivity": "2026-06-28T10:30:00Z"
  },
  "totalTokensUsed": 64400,
  "totalCost": 0.0140,
  "interactionCount": 1,
  "createdAt": "2026-06-28T10:00:00Z",
  "expiresAt": 1722259200
}
```
```json
{
  "sessionId": "550e8400-e29b-41d4-a716-446655440000",
  "sk": "INTERACTION#2026-06-28T10:15:30Z#3f2a1b9c",
  "timestamp": "2026-06-28T10:15:30Z",
  "tool": "extract_knowledge",
  "request": "{\"filename\": \"architecture.pdf\", \"project\": \"backend\"}",
  "response": "{\"status\": \"COMPLETED\", \"summary\": \"Extracted 47 classes, 156 properties\"}",
  "tokens": {
    "inputTokens": 23700,
    "outputTokens": 40700,
    "totalTokens": 64400
  },
  "durationMs": 180000,
  "cost": 0.0140,
  "expiresAt": 1782259200
}
```

---

## Table 5: Classes

**Purpose**: Store knowledge graph classes (entities) with aggregate statistics.
Per-instance evidence lives in the Instances table.

**Primary Key**:
- **PK**: `project` (String)
- **SK**: `className` (String) - Normalized class name (see Key Normalization)

**Attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `displayName` | String | Original (non-normalized) spelling, e.g., "AWS Lambda" |
| `classIri` | String | Metamodel type IRI (e.g., "https://github.com/highitoh/kg-extraction#system_component"). Shared by all classes of that type - it is NOT a per-class identity |
| `classType` | String | ACCEPT \| REJECT \| PENDING (the extraction pipeline's judgment) |
| `definition` | String | Class definition (from the reviewer's justification; may be empty) |
| `instanceCount` | Number | Number of instances in the Instances table |
| `latestExtractedAt` | String | ISO timestamp of the most recent instance |
| `sourceDocuments` | String Set | Unique documents where this class was found (ADD gives an atomic union) |
| `createdAt` | String | ISO timestamp when first extracted |
| `updatedAt` | String | ISO timestamp when last updated |

**GSI: classType-index**
- **PK**: `project` (String)
- **SK**: `classType` (String)
- **Projection**: INCLUDE (summary attributes)
- **Purpose**: Filter by ACCEPT/REJECT/PENDING

**GSI: classIri-index**
- **PK**: `project` (String)
- **SK**: `classIri` (String)
- **Projection**: INCLUDE (summary attributes)
- **Purpose**: List all classes belonging to a metamodel type

**Design Rationale**:
- **SK = normalized className**: The class name is the identity users address
  classes by; normalization makes it collision-safe and case-insensitive
- **classIri as attribute**: Links to the metamodel *type*; many classes share
  one IRI, so it cannot identify a single class
- **Aggregates only**: Repeated extraction updates counters instead of
  appending to an unbounded list, keeping items small and writes cheap
- **INCLUDE projections**: GSIs carry only the summary attributes handlers
  read; extend the projection lists in StorageStack when adding fields

**Example Record**:
```json
{
  "project": "backend",
  "className": "aws lambda",
  "displayName": "AWS Lambda",
  "classIri": "https://github.com/highitoh/kg-extraction#system_component",
  "classType": "ACCEPT",
  "definition": "Serverless compute service for running code without managing servers",
  "instanceCount": 2,
  "latestExtractedAt": "2026-06-29T14:22:10Z",
  "sourceDocuments": ["architecture.pdf", "design-doc.pdf"],
  "createdAt": "2026-06-28T10:04:35Z",
  "updatedAt": "2026-06-29T14:22:10Z"
}
```

---

## Table 6: Properties

**Purpose**: Store knowledge graph relationships (properties) with aggregate
statistics. Per-instance evidence lives in the Instances table.

**Primary Key**:
- **PK**: `projectSourceClass` (String) - Composite: `"{project}#{sourceClassName}"`
- **SK**: `relationship` (String) - Composite: `"{propertyName}#{targetClassName}"`

Class name segments use the normalized form; `#` cannot appear inside them
(stripped during normalization), so the delimiter is unambiguous. Property
names come from the metamodel's controlled vocabulary and must not contain `#`.

**Attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `project` | String | Project name |
| `sourceClassName` | String | Display form of source class (e.g., "API Gateway") |
| `targetClassName` | String | Display form of target class (e.g., "AWS Lambda") |
| `propertyName` | String | Relationship name (e.g., "uses", "invokes", "dependsOn") |
| `propertyIri` | String | IRI from metamodel |
| `sourceClassIri` | String | Source class type IRI from metamodel |
| `targetClassIri` | String | Target class type IRI from metamodel |
| `projectTargetClass` | String | Composite: `"{project}#{targetClassName}"` (partition key of target-class-index) |
| `instanceCount` | Number | Number of instances in the Instances table |
| `strengthSum` | Number | Sum of instance strengths (maintained with ADD) |
| `averageStrength` | Number | strengthSum / instanceCount, recomputed on each new instance |
| `definition` | String | Property definition |
| `sourceDocuments` | String Set | Unique documents (ADD gives an atomic union) |
| `createdAt` | String | ISO timestamp |
| `updatedAt` | String | ISO timestamp |

Only properties judged ACCEPT with strength > 0 are written - the same "valid
properties" rule knowledge.json uses.

**GSI: target-class-index**
- **PK**: `projectTargetClass` (String) - Composite: `"{project}#{targetClassName}"`
- **SK**: `relationship` (String) - reuses the base-table attribute
  `"{propertyName}#{targetClassName}"`. A GSI cannot store a different value
  for the same attribute; within a partition the target class is constant, so
  items still sort by propertyName.
- **Projection**: INCLUDE (summary attributes)
- **Purpose**: Reverse lookup - "What depends on AWS Lambda?"

**GSI: propertyName-index**
- **PK**: `project` (String)
- **SK**: `propertyName` (String)
- **Projection**: INCLUDE (summary attributes)
- **Purpose**: Find all "uses" relationships across the project

**Design Rationale**:
- **PK = project#sourceClassName**: Efficient query for "all relationships FROM API Gateway"
- **SK = propertyName#targetClassName**: Unique key like "invokes#aws lambda"
- **Normalized name segments**: Collision-safe, case-insensitive, delimiter-safe
- **Aggregates only**: Same reasoning as the Classes table

**Example Record**:
```json
{
  "projectSourceClass": "backend#api gateway",
  "relationship": "invokes#aws lambda",
  "project": "backend",
  "sourceClassName": "API Gateway",
  "targetClassName": "AWS Lambda",
  "propertyName": "invokes",
  "propertyIri": "https://github.com/highitoh/kg-extraction#invokes",
  "sourceClassIri": "https://github.com/highitoh/kg-extraction#api_gateway",
  "targetClassIri": "https://github.com/highitoh/kg-extraction#system_component",
  "projectTargetClass": "backend#aws lambda",
  "instanceCount": 2,
  "averageStrength": 0.89,
  "definition": "API Gateway invokes Lambda functions to handle requests",
  "sourceDocuments": ["architecture.pdf", "design-doc.pdf"],
  "createdAt": "2026-06-28T10:04:35Z",
  "updatedAt": "2026-06-29T14:22:10Z"
}
```

---

## Table 7: Instances

**Purpose**: One item per extracted instance (evidence) of a class or property.
Keeps Classes/Properties items small and bounded regardless of how many times
an entity is re-extracted.

**Primary Key**:
- **PK**: `entityKey` (String) - Owning entity:
  - Class: `"{project}#CLASS#{className}"`
  - Property: `"{project}#PROP#{sourceClassName}#{propertyName}#{targetClassName}"`
- **SK**: `instanceKey` (String) - the deterministic `instanceId`

Class name segments use the normalized form.

**Why the SK is the instanceId**: the extraction pipeline derives IDs as
content hashes (`src/utils/id_generator.py`), so the same evidence always
produces the same key. batch_runner writes instances with
`ConditionExpression: attribute_not_exists(entityKey)` and increments the
aggregate counters only when the put succeeds - Batch retries and knowledge
carried over into later runs are deduplicated for free, and counters are
exact. The trade-off is that the key carries no chronological order; readers
sort by the `extractedAt` attribute (instance counts per entity are small).

**Attributes**:

| Field | Type | Description |
|-------|------|-------------|
| `instanceId` | String | Deterministic content-hash ID of the extracted instance |
| `label` | String | Extracted label (class instances only) |
| `source` | String | Source text excerpt (class instances only, truncated to 1000 chars) |
| `strength` | Number | Extraction strength (property instances only) |
| `extractedAt` | String | ISO timestamp |
| `snapshotId` | String | Snapshot created by the extraction |
| `sourceDocument` | String | Original filename or URL |

**No GSI** (instances are always fetched for a known entity)

**Example Record**:
```json
{
  "entityKey": "backend#CLASS#aws lambda",
  "instanceKey": "abc123def456abc123def456abc123de",
  "instanceId": "abc123def456abc123def456abc123de",
  "label": "AWS Lambda",
  "source": "Lambda is a serverless compute service that runs your code in response to events...",
  "extractedAt": "2026-06-28T10:04:35Z",
  "snapshotId": "20260628100430",
  "sourceDocument": "architecture.pdf"
}
```

---

## Table 8: Projects

**Purpose**: Project metadata + extraction history (composite key design)

**Primary Key**:
- **PK**: `project` (String)
- **SK**: `recordType` (String) - "METADATA" for project info,
  "EXTRACTION#{timestamp}" for history, "LOCK" for the per-project mutex

**Attributes for METADATA record**:

| Field | Type | Description |
|-------|------|-------------|
| `entityType` | String | Always "PROJECT". Set ONLY on METADATA records - this is the sparse partition key of entityType-index |
| `projectName` | String | Display name |
| `description` | String | Project description |
| `createdAt` | String | ISO timestamp when project was created |
| `lastExtractionAt` | String | ISO timestamp of most recent extraction |
| `totalSnapshots` | Number | Total number of snapshots |
| `currentSnapshotId` | String | Latest snapshot ID |
| `totalExtractions` | Number | Total number of extractions performed |
| `sources` | String Set | Source types seen (e.g., {"pdf", "confluence", "miro"}); ADD gives an atomic union |
| `tags` | List<String> | User-defined tags |

**Note**: `totalClasses` and `totalProperties` are NOT stored (query Classes/Properties tables for current counts)

**Attributes for EXTRACTION#{timestamp} records**:

| Field | Type | Description |
|-------|------|-------------|
| `jobId` | String | Job ID that performed this extraction |
| `extractedAt` | String | ISO timestamp |
| `snapshotId` | String | Snapshot ID created by this extraction |
| `sourceType` | String | pdf \| docx \| miro \| confluence \| etc. |
| `sourceFile` | String | Original filename or URL |
| `classesExtracted` | Number | New class instances written in this run |
| `propertiesExtracted` | Number | New property instances written in this run |
| `tokenUsage` | Map | `{inputTokens, outputTokens, totalTokens}` |
| `cost` | Number | Extraction cost in USD |
| `durationSeconds` | Number | How long extraction took |
| `status` | String | COMPLETED \| FAILED |
| `errorMessage` | String | If failed, the error |

**Attributes for the LOCK record** (per-project mutex):

| Field | Type | Description |
|-------|------|-------------|
| `jobId` | String | Job currently holding the lock |
| `acquiredAt` | String | ISO timestamp |
| `expiresAt` | Number | Unix timestamp; a stale lock (killed container) is stealable after this |

knowledge.json is read-modify-written by every extraction, so batch_runner
serializes writers per project: acquire with a conditional put
(`attribute_not_exists OR expired OR same jobId` - re-entrant for Batch
retries), release with a conditional delete. The LOCK record has no
`entityType` attribute, so it never appears in the listing index.

**GSI: entityType-index** (sparse)
- **PK**: `entityType` (String)
- **SK**: `project` (String)
- **Projection**: ALL
- **Purpose**: List all projects with a query. Only METADATA records carry
  `entityType`, so the index contains exactly one item per project and
  Limit/pagination count real results. A filtered scan (the alternative)
  applies Limit *before* the filter and degrades as history grows.

Extraction history needs no GSI: the base-table sort key
`EXTRACTION#{ISO timestamp}` already sorts chronologically.

**Example Records**:

**Metadata Record**:
```json
{
  "project": "backend",
  "recordType": "METADATA",
  "entityType": "PROJECT",
  "projectName": "Backend System Architecture",
  "description": "Knowledge graph for backend services and APIs",
  "createdAt": "2026-06-01T00:00:00Z",
  "lastExtractionAt": "2026-06-28T10:04:35Z",
  "totalSnapshots": 15,
  "currentSnapshotId": "20260628100430",
  "totalExtractions": 23,
  "sources": ["pdf", "confluence", "miro"],
  "tags": ["architecture", "backend", "services"]
}
```

**Extraction History Record**:
```json
{
  "project": "backend",
  "recordType": "EXTRACTION#2026-06-28T10:04:35Z",
  "jobId": "job-abc123",
  "extractedAt": "2026-06-28T10:04:35Z",
  "snapshotId": "20260628100430",
  "sourceType": "pdf",
  "sourceFile": "architecture.pdf",
  "classesExtracted": 12,
  "propertiesExtracted": 34,
  "tokenUsage": {
    "inputTokens": 23700,
    "outputTokens": 40700,
    "totalTokens": 64400
  },
  "cost": 0.0140,
  "durationSeconds": 180,
  "status": "COMPLETED"
}
```

---

## Query Patterns

### Jobs Table

```python
# Get specific job
get_item(PK="job-abc123")

# List all jobs for a project (sorted by time)
query(GSI="project-timestamp-index", PK="backend", ScanIndexForward=False)
```

### Sessions Table

```python
# Get session state
get_item(PK=session_id, SK="STATE")

# Get recent interactions (newest first, paginated)
query(PK=session_id, SK begins_with "INTERACTION#", ScanIndexForward=False, Limit=10)
```

### Classes Table

```python
# Get all classes for a project
query(PK="backend")

# Get specific class by name (normalize first)
get_item(PK="backend", SK=normalize_class_name("AWS Lambda"))  # "aws lambda"

# List classes of a metamodel type
query(GSI="classIri-index", PK="backend", SK="https://...")

# Filter ACCEPT classes only
query(GSI="classType-index", PK="backend", SK="ACCEPT")
```

### Properties Table

```python
# Get all properties FROM "API Gateway"
query(PK="backend#api gateway")

# Get all properties TO "AWS Lambda" (reverse lookup)
query(GSI="target-class-index", PK="backend#aws lambda")

# Find all "uses" relationships
query(GSI="propertyName-index", PK="backend", SK="uses")
```

### Instances Table

```python
# Evidence for a class (sort by the extractedAt attribute client-side;
# the key is the deterministic instanceId, not a timestamp)
query(PK="backend#CLASS#aws lambda", Limit=50)

# Evidence for a property
query(PK="backend#PROP#api gateway#invokes#aws lambda", Limit=50)
```

### Projects Table

```python
# Get project metadata
get_item(PK="backend", SK="METADATA")

# List all projects (sparse index: one item per project)
query(GSI="entityType-index", PK="PROJECT")

# Get last 10 extractions
query(PK="backend", SK begins_with "EXTRACTION#", ScanIndexForward=False, Limit=10)

# Get extractions by date range (base table, no GSI needed)
query(PK="backend", SK between "EXTRACTION#2026-06-01" and "EXTRACTION#2026-06-30")
```

---

## Data Consistency

- **Eventual consistency**: DynamoDB reads are eventually consistent by default (can request strongly consistent reads)
- **Atomic updates**: Each table write is atomic (job status, progress updates)
- **Cross-table consistency**: No transactions across tables (batch_runner.py writes sequentially)
- **Per-project serialization**: batch_runner holds the project LOCK record for
  the whole run, so knowledge.json and the aggregate counters have a single
  writer per project at any time
- **Idempotency**: Job records use conditional puts (`attribute_not_exists`), so
  duplicate submissions fail fast; instance items are keyed by deterministic
  content-hash IDs and written conditionally, so Batch retries and re-extracted
  evidence never double-count aggregates

## Backup & Recovery

- **Point-in-time recovery**: Enabled on all tables (35-day retention)
- **On-demand backups**: Manual snapshots before major operations
- **DynamoDB Streams**: Optional, for real-time replication or audit logging
