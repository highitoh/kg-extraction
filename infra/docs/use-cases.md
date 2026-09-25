# Use Cases

The AWS deployment supports five use cases organized into three categories.

## Categories

### Category A: Knowledge Extraction (Use Cases 1-2)
**Focus**: How knowledge is extracted from sources and ingested into the system  
**Trigger types**: On-demand (API), scheduled (cron)  
**Primary users**: Data analysts, knowledge engineers, automated pipelines

### Category B: Knowledge Management (Use Cases 3-4)
**Focus**: Version control, history tracking, and system integration  
**Operations**: Snapshots, changesets, rollback, MCP integration  
**Primary users**: Knowledge engineers, project managers, AI assistants

### Category C: Knowledge Utilization (Use Case 5)
**Focus**: How end users access and explore extracted knowledge  
**Access methods**: REST API, MCP tools  
**Primary users**: Developers, researchers, business analysts, new team members

---

## Use Case 1: On-Demand Knowledge Extraction (API-Triggered)

**Scenario**: Users upload documents or provide URLs through a web application or API client to extract knowledge graphs on demand.

**Actors**: Data analysts, knowledge engineers, content managers

**Flow**:
1. User uploads a PDF/DOCX file, or provides a Miro board URL, via REST API
   (GitHub and Confluence URLs are engine/CLI-only for now)
2. System validates the input and returns a job ID immediately
3. Extraction runs asynchronously on AWS Batch (Fargate)
4. User polls job status or receives webhook notification upon completion
5. User downloads results (Turtle file, CSV files, knowledge JSON) via presigned URLs
6. User can view snapshot history and compare versions

**Trigger**: files — `POST /v1/upload-url` → user uploads to S3 → `POST /v1/extract`;
Miro boards — `POST /v1/extract` with `sourceUrl` (single step, no upload)

**AWS Components**: API Gateway → Lambda (presigned URL) → S3 (user upload) → Lambda (submit) → AWS Batch → DynamoDB

---

## Use Case 2: Scheduled Knowledge Updates (Cron-Triggered)

**Scenario**: Automatically extract knowledge from living documents (Confluence spaces, Jira projects) on a daily/weekly schedule to keep knowledge graphs up-to-date.

**Actors**: System administrators, DevOps teams

**Flow**:
1. EventBridge cron rule triggers at scheduled time (e.g., daily at 2 AM UTC)
2. AWS Batch job launches automatically with pre-configured source URLs
3. Incremental extraction detects changes since last run (using watermarks/snapshots)
4. Only modified content is re-extracted and merged into knowledge graph
5. New snapshot is created automatically
6. Changeset is optionally generated to track what changed

**Trigger**: EventBridge scheduled rule

**AWS Components**: EventBridge → AWS Batch → S3 → DynamoDB

---

## Use Case 3: Knowledge Version Management (Snapshot & Changeset APIs)

**Scenario**: Track knowledge graph evolution over time, compare versions, and rollback to previous states if needed.

**Actors**: Knowledge engineers, project managers, auditors

**Flow**:
1. User lists all snapshots for a project via API
2. User selects two snapshots to compare
3. System generates changeset showing added/deleted/modified classes and properties
4. Changeset includes business drivers (extracted by LLM) explaining why changes occurred
5. User can rollback to a previous snapshot if needed
6. Rollback creates a new snapshot (preserving audit trail)

**Trigger**: `GET /v1/projects/{project}/snapshots` and `POST /v1/projects/{project}/changesets` API calls

**AWS Components**: API Gateway → Lambda → DynamoDB + S3 (for snapshot listing) or AWS Batch (for changeset generation)

---

## Use Case 4: MCP Server Integration (Claude Desktop & AI Apps)

**Scenario**: Expose the knowledge extraction system as an MCP (Model Context Protocol) server, allowing Claude Desktop users and AI applications to extract knowledge from documents directly within their conversations.

**Actors**: Claude Desktop users, AI application developers, LLM-based agents

**Flow**:
1. User connects to MCP server via Claude Desktop or custom AI application
2. User invokes MCP tools to interact with the extraction system:
   - `extract_knowledge`: Upload a file or provide URL to extract knowledge
   - `list_projects`: List all available projects
   - `get_extraction_status`: Check status of ongoing extraction jobs
   - `query_knowledge`: Query the knowledge graph for a specific project
   - `list_snapshots`: View snapshot history for a project
   - `compare_snapshots`: Generate changeset between two snapshots
   - `get_session_stats`: View session statistics (tokens, cost, history)
3. MCP server translates tool calls to REST API requests
4. Responses are formatted and returned to the AI assistant
5. User can iteratively refine extractions or query results within conversation

**Trigger**: MCP tool invocation from Claude Desktop or MCP-compatible client

**AWS Components**: API Gateway REST API (/mcp route) → Lambda (streaming) → MCP Server Lambda → DynamoDB (Sessions, Jobs) → AWS Batch

**Benefits**:
- Conversational interface: Extract knowledge within Claude Desktop chat
- Session persistence: Context preserved across restarts with conversation history
- Token & cost tracking: Automatic tracking of Gemini API usage per interaction
- No separate UI needed: Users interact via natural language

---

## Use Case 5: Knowledge Access and Utilization (End-User Queries)

**Scenario**: End users (analysts, researchers, developers) query and explore the extracted knowledge graphs to understand domain concepts, discover relationships, and answer questions without needing to read original documents.

**Actors**: Business analysts, researchers, developers, product managers, new team members

**Flow**:
1. User accesses knowledge graph via REST API or AI assistant (MCP)
2. User performs different types of queries:
   - **Search**: "Find all classes related to 'authentication'"
   - **Browse**: "List all entities in the system architecture project"
   - **Explore**: "Show me all relationships from AWS Lambda"
   - **Discover**: "What concepts connect microservices and API Gateway?"
3. System returns structured knowledge with definitions, relationships, and source documents
4. User drills down into specific classes to see:
   - Detailed definitions
   - Properties (outgoing relationships)
   - Dependencies (incoming relationships)
   - Source documents where concept was extracted
5. User can navigate the knowledge graph by following relationships
6. Results can be exported or integrated into documentation

**Trigger**: 
- REST API: Direct HTTP calls to knowledge endpoints
- MCP: `query_knowledge` tool invocations

**Example Queries Supported**:

1. **Onboarding New Team Members**:
   - "What are the main components in our system?"
   - "Show me the architecture of the payment service"

2. **Impact Analysis**:
   - "What services depend on the User Database?"
   - "If I change the Authentication API, what will be affected?"

3. **Documentation Generation**:
   - "List all microservices and their descriptions"
   - "Export all API endpoints and their relationships"

4. **Research and Discovery**:
   - "What concepts are related to both 'caching' and 'performance'?"
   - "Find orphaned concepts (no relationships)"

**AWS Components**: API Gateway → Lambda (Knowledge Graph API) → DynamoDB (Classes, Properties, Projects tables)

**Benefits**:
- Fast knowledge discovery: No need to read entire documents
- Relationship exploration: Follow connections between concepts
- Source traceability: Link back to original documents
- Living documentation: Always reflects latest extracted knowledge
- AI-powered insights: Use MCP for natural language queries
- Integration-ready: REST API for custom applications, tools, workflows

---

## Cross-Cutting Capabilities

All use cases share these capabilities:

- **Incremental Extraction**: For API sources (Miro, GitHub, Jira, Confluence), only changed content is re-extracted (CLI only; AWS runs are always full re-extractions)
- **Automatic Snapshots**: Each successful extraction creates a timestamped snapshot
- **Multi-Project Support**: Projects are isolated with separate knowledge graphs and snapshots
- **Fault Tolerance**: AWS Batch retries failed jobs automatically with exponential backoff
- **Cross-Cloud LLM**: All extractions use Gemini 3.1 Flash Lite in GCP (cost-optimized)
- **Audit Trail**: All jobs, snapshots, and changesets tracked in DynamoDB with timestamps
