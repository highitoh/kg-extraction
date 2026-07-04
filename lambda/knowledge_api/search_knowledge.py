import json
import os

import boto3

dynamodb = boto3.client("dynamodb")
CLASSES_TABLE = os.environ["CLASSES_TABLE"]


def handler(event, context):
    """Search classes and properties by text query."""
    # TODO: Implement text search (scan with filter or use OpenSearch)
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({"message": "Not yet implemented"}),
    }
