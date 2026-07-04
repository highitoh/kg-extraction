import json
import os

import boto3

dynamodb = boto3.client("dynamodb")

PROJECTS_TABLE = os.environ["PROJECTS_TABLE"]


def handler(event, context):
    """
    Get project metadata and statistics.

    Request: GET /v1/projects/{project}

    Response: {
        "project": "backend",
        "projectName": "Backend System Architecture",
        "totalExtractions": 23,
        ...
    }
    """
    try:
        # Get project from path parameters
        project = event.get("pathParameters", {}).get("project")

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

        # Get METADATA record from DynamoDB
        response = dynamodb.get_item(
            TableName=PROJECTS_TABLE,
            Key={"project": {"S": project}, "recordType": {"S": "METADATA"}},
        )

        if "Item" not in response:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "NOT_FOUND",
                            "message": f"Project {project} not found",
                        }
                    }
                ),
            }

        item = response["Item"]

        # Convert to JSON
        result = {
            "project": project,
            "projectName": item.get("projectName", {}).get("S", ""),
            "description": item.get("description", {}).get("S", ""),
            "createdAt": item.get("createdAt", {}).get("S"),
            "lastExtractionAt": item.get("lastExtractionAt", {}).get("S"),
            "totalExtractions": int(item.get("totalExtractions", {}).get("N", 0)),
            "totalSnapshots": int(item.get("totalSnapshots", {}).get("N", 0)),
            "currentSnapshotId": item.get("currentSnapshotId", {}).get("S"),
            "sources": sorted(item.get("sources", {}).get("SS", [])),
            "tags": [t["S"] for t in item.get("tags", {}).get("L", [])],
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(result),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to get project",
                    }
                }
            ),
        }
