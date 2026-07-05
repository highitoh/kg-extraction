import json
import os

import boto3

dynamodb = boto3.client("dynamodb")

PROJECTS_TABLE = os.environ["PROJECTS_TABLE"]


def handler(event, context):
    """
    List all projects with statistics.

    Request: GET /v1/projects?limit=20

    Response: {
        "projects": [
            {
                "project": "backend",
                "projectName": "Backend System Architecture",
                "totalExtractions": 23,
                ...
            }
        ],
        "nextToken": "..."
    }
    """
    try:
        # Get query parameters
        params = event.get("queryStringParameters") or {}
        limit = int(params.get("limit", 20))
        next_token = params.get("nextToken")

        # Query the sparse entityType-index: only METADATA records carry the
        # entityType attribute, so the index holds exactly one item per
        # project and Limit/pagination count real results (a filtered scan
        # applies Limit before the filter and can return empty pages).
        query_params = {
            "TableName": PROJECTS_TABLE,
            "IndexName": "entityType-index",
            "KeyConditionExpression": "entityType = :entityType",
            "ExpressionAttributeValues": {":entityType": {"S": "PROJECT"}},
            "Limit": limit,
        }

        if next_token:
            query_params["ExclusiveStartKey"] = json.loads(next_token)

        response = dynamodb.query(**query_params)

        # Convert items to JSON
        projects = []
        for item in response.get("Items", []):
            projects.append(
                {
                    "project": item["project"]["S"],
                    "projectName": item.get("projectName", {}).get("S", ""),
                    "description": item.get("description", {}).get("S", ""),
                    "totalExtractions": int(item.get("totalExtractions", {}).get("N", 0)),
                    "totalSnapshots": int(item.get("totalSnapshots", {}).get("N", 0)),
                    "lastExtractionAt": item.get("lastExtractionAt", {}).get("S"),
                    "sources": sorted(item.get("sources", {}).get("SS", [])),
                    "tags": [t["S"] for t in item.get("tags", {}).get("L", [])],
                }
            )

        result = {"projects": projects}

        # Add pagination token if more results exist
        if "LastEvaluatedKey" in response:
            result["nextToken"] = json.dumps(response["LastEvaluatedKey"])

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
                        "message": "Failed to list projects",
                    }
                }
            ),
        }
