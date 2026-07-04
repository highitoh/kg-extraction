import json
import os
import time
import uuid

import boto3

dynamodb = boto3.client("dynamodb")
s3 = boto3.client("s3")
batch = boto3.client("batch")

# Environment variables
SESSIONS_TABLE = os.environ["SESSIONS_TABLE"]
JOBS_TABLE = os.environ["JOBS_TABLE"]
BUCKET_NAME = os.environ["BUCKET_NAME"]
JOB_QUEUE = os.environ["JOB_QUEUE"]
JOB_DEFINITION = os.environ["JOB_DEFINITION"]


# Session item model (see docs/database-schema.md, Sessions table):
# - PK sessionId, SK "STATE": session state + aggregate stats (30-day sliding TTL)
# - PK sessionId, SK "INTERACTION#{timestamp}#{id}": one item per tool call
#   with its own TTL, so history is pruned by DynamoDB instead of rewriting
#   a growing list attribute (which would hit the 400KB item limit)
STATE_TTL_DAYS = 30
INTERACTION_TTL_DAYS = 365


def put_interaction(session_id, tool, request, response, tokens, cost, duration_ms):
    """Store one tool interaction as its own item with per-item TTL."""
    now = time.gmtime()
    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", now)
    sort_key = f"INTERACTION#{timestamp_iso}#{uuid.uuid4().hex[:8]}"

    dynamodb.put_item(
        TableName=SESSIONS_TABLE,
        Item={
            "sessionId": {"S": session_id},
            "sk": {"S": sort_key},
            "timestamp": {"S": timestamp_iso},
            "tool": {"S": tool},
            "request": {"S": json.dumps(request)},
            "response": {"S": json.dumps(response)},
            "tokens": {"M": {k: {"N": str(v)} for k, v in tokens.items()}},
            "cost": {"N": str(cost)},
            "durationMs": {"N": str(duration_ms)},
            "expiresAt": {
                "N": str(int(time.time()) + INTERACTION_TTL_DAYS * 24 * 60 * 60)
            },
        },
    )

    # Roll aggregate stats up into the STATE item
    dynamodb.update_item(
        TableName=SESSIONS_TABLE,
        Key={"sessionId": {"S": session_id}, "sk": {"S": "STATE"}},
        UpdateExpression=(
            "ADD totalTokensUsed :tokens, totalCost :cost, interactionCount :one SET expiresAt = :expiresAt"
        ),
        ExpressionAttributeValues={
            ":tokens": {"N": str(tokens.get("totalTokens", 0))},
            ":cost": {"N": str(cost)},
            ":one": {"N": "1"},
            ":expiresAt": {"N": str(int(time.time()) + STATE_TTL_DAYS * 24 * 60 * 60)},
        },
    )


def lambda_handler(event, context):
    """
    MCP Server Lambda function with HTTP streaming support.

    This is a simplified stub. Full implementation would include:
    - MCP protocol JSON-RPC handling
    - Server-Sent Events (SSE) streaming
    - Session management with DynamoDB
    - Tool invocation routing
    - Progress streaming for long-running jobs

    Conversation History Strategy:
    - One DynamoDB item per interaction (SK "INTERACTION#{timestamp}#{id}")
      with a 365-day per-item TTL, so DynamoDB prunes history natively
    - The STATE item holds session state and aggregate stats only, keeping
      every item small and every append O(interaction size), regardless of
      how active the session is
    - History reads are a sort-key range query (begins_with "INTERACTION#"),
      newest first, paginated

    For now, returns basic response indicating MCP endpoint is available.
    """
    try:
        # Parse request
        body = json.loads(event.get("body", "{}"))

        # Get or create session
        session_id = event.get("headers", {}).get("x-session-id")
        if not session_id:
            session_id = str(uuid.uuid4())
            # Create the session STATE item; interactions are stored as
            # separate INTERACTION# items via put_interaction()
            timestamp = int(time.time())
            dynamodb.put_item(
                TableName=SESSIONS_TABLE,
                Item={
                    "sessionId": {"S": session_id},
                    "sk": {"S": "STATE"},
                    "state": {
                        "M": {
                            "currentProject": {"S": "default"},
                            "recentJobs": {"L": []},
                            "lastActivity": {
                                "S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            },
                        }
                    },
                    "totalTokensUsed": {"N": "0"},
                    "totalCost": {"N": "0.0"},
                    "interactionCount": {"N": "0"},
                    "createdAt": {
                        "S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    },
                    "expiresAt": {"N": str(timestamp + STATE_TTL_DAYS * 24 * 60 * 60)},
                },
            )

        # MCP protocol response (simplified)
        response_body = {
            "jsonrpc": "2.0",
            "id": body.get("id", 1),
            "result": {
                "type": "info",
                "message": "MCP Server endpoint is available. Full implementation pending.",
                "sessionId": session_id,
                "availableTools": [
                    "extract_knowledge",
                    "list_projects",
                    "get_extraction_status",
                    "query_knowledge",
                    "list_snapshots",
                    "compare_snapshots",
                    "get_session_stats",
                ],
            },
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "x-session-id": session_id,
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(response_body),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"error": {"code": "INTERNAL_ERROR", "message": "MCP server error"}}
            ),
        }
