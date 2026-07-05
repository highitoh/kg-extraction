import json
import os
from urllib.parse import unquote

import boto3

from knowledge_api.normalize import normalize_class_name

dynamodb = boto3.client("dynamodb")

PROPERTIES_TABLE = os.environ["PROPERTIES_TABLE"]


def handler(event, context):
    """
    Get all properties FROM a specific class (outgoing relationships).

    Request: GET /v1/projects/{project}/knowledge/classes/{className}/properties

    Response: {
        "sourceClass": "API Gateway",
        "properties": [
            {
                "propertyName": "invokes",
                "targetClassName": "AWS Lambda",
                "averageStrength": 0.89,
                "instanceCount": 2
            }
        ]
    }
    """
    try:
        # Get parameters from path
        project = event.get("pathParameters", {}).get("project")
        class_name = event.get("pathParameters", {}).get("className")

        # URL decode
        project = unquote(project) if project else None
        class_name = unquote(class_name) if class_name else None

        if not project or not class_name:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Missing project or className parameter",
                        }
                    }
                ),
            }

        # Query parameters
        params = event.get("queryStringParameters") or {}
        limit = int(params.get("limit", 50))

        # Query DynamoDB (PK = project#sourceClass, normalized class name)
        pk = f"{project}#{normalize_class_name(class_name)}"
        response = dynamodb.query(
            TableName=PROPERTIES_TABLE,
            KeyConditionExpression="projectSourceClass = :pk",
            ExpressionAttributeValues={":pk": {"S": pk}},
            Limit=limit,
        )

        # Convert items to JSON (instanceCount is an aggregate attribute;
        # instance evidence lives in the Instances table)
        properties = []
        for item in response.get("Items", []):
            properties.append(
                {
                    "propertyName": item.get("propertyName", {}).get("S"),
                    "targetClassName": item.get("targetClassName", {}).get("S"),
                    "propertyIri": item.get("propertyIri", {}).get("S"),
                    "averageStrength": float(item.get("averageStrength", {}).get("N", 0)),
                    "instanceCount": int(item.get("instanceCount", {}).get("N", 0)),
                    "definition": item.get("definition", {}).get("S", ""),
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
                    "sourceClass": class_name,
                    "properties": properties,
                    "totalReturned": len(properties),
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
                        "message": "Failed to get class properties",
                    }
                }
            ),
        }
