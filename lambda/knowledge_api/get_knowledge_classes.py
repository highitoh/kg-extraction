import json
import os
from urllib.parse import unquote

import boto3

dynamodb = boto3.client("dynamodb")

CLASSES_TABLE = os.environ["CLASSES_TABLE"]


def handler(event, context):
    """
    List all classes in the knowledge graph.

    Request: GET /v1/projects/{project}/knowledge/classes?limit=50&classType=ACCEPT

    Response: {
        "project": "backend",
        "classes": [
            {
                "className": "AWS Lambda",
                "classIri": "https://...",
                "classType": "ACCEPT",
                "definition": "...",
                "instanceCount": 2
            }
        ]
    }
    """
    try:
        # Get project from path parameters
        project = event.get("pathParameters", {}).get("project")
        project = unquote(project) if project else None

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

        # Get query parameters
        params = event.get("queryStringParameters") or {}
        limit = int(params.get("limit", 50))
        class_type = params.get("classType")  # Optional filter

        # Query DynamoDB ("project" is a DynamoDB reserved word, hence the alias)
        if class_type:
            # Use GSI to filter by classType
            response = dynamodb.query(
                TableName=CLASSES_TABLE,
                IndexName="classType-index",
                KeyConditionExpression="#project = :project AND classType = :classType",
                ExpressionAttributeNames={"#project": "project"},
                ExpressionAttributeValues={
                    ":project": {"S": project},
                    ":classType": {"S": class_type},
                },
                Limit=limit,
            )
        else:
            # Query all classes for project
            response = dynamodb.query(
                TableName=CLASSES_TABLE,
                KeyConditionExpression="#project = :project",
                ExpressionAttributeNames={"#project": "project"},
                ExpressionAttributeValues={":project": {"S": project}},
                Limit=limit,
            )

        # Convert items to JSON (instanceCount is an aggregate attribute;
        # displayName preserves the original spelling of the normalized key)
        classes = []
        for item in response.get("Items", []):
            source_docs = sorted(item.get("sourceDocuments", {}).get("SS", []))

            classes.append(
                {
                    "className": item.get("displayName", {}).get(
                        "S", item["className"]["S"]
                    ),
                    "classIri": item.get("classIri", {}).get("S"),
                    "classType": item.get("classType", {}).get("S"),
                    "definition": item.get("definition", {}).get("S", ""),
                    "instanceCount": int(item.get("instanceCount", {}).get("N", 0)),
                    "sourceDocuments": source_docs,
                    "createdAt": item.get("createdAt", {}).get("S"),
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
                    "classes": classes,
                    "totalReturned": len(classes),
                    "nextToken": None,  # TODO: Implement pagination
                }
            ),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to get knowledge classes",
                    }
                }
            ),
        }
