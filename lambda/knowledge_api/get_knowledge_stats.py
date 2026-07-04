import json
import os

import boto3

dynamodb = boto3.client("dynamodb")
CLASSES_TABLE = os.environ["CLASSES_TABLE"]
PROPERTIES_TABLE = os.environ["PROPERTIES_TABLE"]


def handler(event, context):
    """Get knowledge graph statistics."""
    # TODO: Implement stats aggregation
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({"message": "Not yet implemented"}),
    }
