import json
import os

import boto3

dynamodb = boto3.client("dynamodb")
PROPERTIES_TABLE = os.environ["PROPERTIES_TABLE"]


def handler(event, context):
    """List all properties for a project with pagination."""
    # TODO: Implement full pagination support
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps({"message": "Not yet implemented"}),
    }
