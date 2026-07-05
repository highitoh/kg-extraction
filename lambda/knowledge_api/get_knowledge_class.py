import json
import os
from urllib.parse import unquote

import boto3

from knowledge_api.normalize import class_entity_key, normalize_class_name

dynamodb = boto3.client("dynamodb")

CLASSES_TABLE = os.environ["CLASSES_TABLE"]
INSTANCES_TABLE = os.environ["INSTANCES_TABLE"]

INSTANCES_LIMIT = 50


def handler(event, context):
    """
    Get specific class details by className.

    Request: GET /v1/projects/{project}/knowledge/classes/{className}

    Response: {
        "className": "AWS Lambda",
        "classIri": "https://...",
        "classType": "ACCEPT",
        "definition": "...",
        "instances": [...]
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

        # Class identity is the normalized name (see normalize.py)
        normalized_name = normalize_class_name(class_name)

        # Get item from DynamoDB
        response = dynamodb.get_item(
            TableName=CLASSES_TABLE,
            Key={"project": {"S": project}, "className": {"S": normalized_name}},
        )

        if "Item" not in response:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "NOT_FOUND",
                            "message": f"Class {class_name} not found",
                        }
                    }
                ),
            }

        item = response["Item"]

        # Instance evidence lives in the Instances table. The sort key is the
        # deterministic instanceId (for idempotent writes), so results are
        # ordered by extractedAt here instead of by key.
        instances_response = dynamodb.query(
            TableName=INSTANCES_TABLE,
            KeyConditionExpression="entityKey = :entityKey",
            ExpressionAttributeValues={":entityKey": {"S": class_entity_key(project, class_name)}},
            Limit=INSTANCES_LIMIT,
        )

        instances = []
        for inst in instances_response.get("Items", []):
            instances.append(
                {
                    "instanceId": inst.get("instanceId", {}).get("S"),
                    "label": inst.get("label", {}).get("S"),
                    "source": inst.get("source", {}).get("S"),
                    "extractedAt": inst.get("extractedAt", {}).get("S"),
                    "snapshotId": inst.get("snapshotId", {}).get("S"),
                    "sourceDocument": inst.get("sourceDocument", {}).get("S"),
                }
            )
        instances.sort(key=lambda i: i.get("extractedAt") or "", reverse=True)

        source_docs = sorted(item.get("sourceDocuments", {}).get("SS", []))

        result = {
            "className": item.get("displayName", {}).get("S", normalized_name),
            "classIri": item.get("classIri", {}).get("S"),
            "classType": item.get("classType", {}).get("S"),
            "definition": item.get("definition", {}).get("S", ""),
            "instanceCount": int(item.get("instanceCount", {}).get("N", 0)),
            "instances": instances,
            "sourceDocuments": source_docs,
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
        import traceback

        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to get knowledge class",
                    }
                }
            ),
        }
