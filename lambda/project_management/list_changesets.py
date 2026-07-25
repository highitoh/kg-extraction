import json
import os

import boto3

dynamodb = boto3.client("dynamodb")

CHANGESETS_TABLE = os.environ["CHANGESETS_TABLE"]


def handler(event, context):
    """
    List all changesets for a project.

    Request: GET /v1/projects/{project}/changesets?limit=20

    Response: {
        "project": "backend",
        "changesets": [
            {
                "changesetId": "cs_20260628_150000_ab12cd",
                "status": "COMPLETED",
                "totalChanges": 6
            }
        ]
    }
    """
    try:
        project = (event.get("pathParameters", {}) or {}).get("project")

        if not project:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Missing project parameter",
                        }
                    }
                ),
            }

        params = event.get("queryStringParameters") or {}
        limit = int(params.get("limit", 20))

        # Changeset IDs are timestamp-prefixed, so the sort key is
        # chronological; ScanIndexForward=False returns latest first.
        # ("project" is a DynamoDB reserved word, hence the alias.)
        response = dynamodb.query(
            TableName=CHANGESETS_TABLE,
            KeyConditionExpression="#project = :project",
            ExpressionAttributeNames={"#project": "project"},
            ExpressionAttributeValues={":project": {"S": project}},
            Limit=limit,
            ScanIndexForward=False,
        )

        changesets = []
        for item in response.get("Items", []):
            changesets.append(
                {
                    "changesetId": item["changesetId"]["S"],
                    "status": item.get("status", {}).get("S"),
                    "beforeSnapshotId": item.get("beforeSnapshotId", {}).get("S"),
                    "afterSnapshotId": item.get("afterSnapshotId", {}).get("S"),
                    "createdAt": item.get("createdAt", {}).get("S"),
                    "completedAt": item.get("completedAt", {}).get("S"),
                    "totalChanges": int(item["totalChanges"]["N"]) if "totalChanges" in item else None,
                    "driversCount": int(item["driversCount"]["N"]) if "driversCount" in item else None,
                }
            )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "project": project,
                    "changesets": changesets,
                    "nextToken": None,  # TODO: Implement pagination
                }
            ),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to list changesets",
                    }
                }
            ),
        }
