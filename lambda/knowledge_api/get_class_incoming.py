import json
import os
from urllib.parse import unquote

import boto3

from knowledge_api.normalize import normalize_class_name

dynamodb = boto3.client("dynamodb")

PROPERTIES_TABLE = os.environ["PROPERTIES_TABLE"]


def handler(event, context):
    """
    Get all properties TO a specific class (incoming relationships, reverse lookup).

    Request: GET /v1/projects/{project}/knowledge/classes/{className}/incoming

    Response: {
        "targetClass": "AWS Lambda",
        "incomingProperties": [
            {
                "propertyName": "invokes",
                "sourceClassName": "API Gateway",
                "averageStrength": 0.89
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

        # Query DynamoDB using GSI (PK = project#targetClass, normalized name)
        pk = f"{project}#{normalize_class_name(class_name)}"
        response = dynamodb.query(
            TableName=PROPERTIES_TABLE,
            IndexName="target-class-index",
            KeyConditionExpression="projectTargetClass = :pk",
            ExpressionAttributeValues={":pk": {"S": pk}},
            Limit=limit,
        )

        # Convert items to JSON (instanceCount is an aggregate attribute;
        # instance evidence lives in the Instances table)
        incoming_properties = []
        for item in response.get("Items", []):
            incoming_properties.append(
                {
                    "propertyName": item.get("propertyName", {}).get("S"),
                    "sourceClassName": item.get("sourceClassName", {}).get("S"),
                    "propertyIri": item.get("propertyIri", {}).get("S"),
                    "averageStrength": float(item.get("averageStrength", {}).get("N", 0)),
                    "instanceCount": int(item.get("instanceCount", {}).get("N", 0)),
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
                    "targetClass": class_name,
                    "incomingProperties": incoming_properties,
                    "totalReturned": len(incoming_properties),
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
                        "message": "Failed to get incoming properties",
                    }
                }
            ),
        }
