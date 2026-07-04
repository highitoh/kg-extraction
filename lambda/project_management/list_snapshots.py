import json
import os

import boto3

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

BUCKET_NAME = os.environ["BUCKET_NAME"]
SNAPSHOTS_TABLE = os.environ["SNAPSHOTS_TABLE"]


def handler(event, context):
    """
    List all snapshots for a project.

    Request: GET /v1/projects/{project}/snapshots?limit=20

    Response: {
        "project": "backend",
        "snapshots": [
            {
                "snapshotId": "20260628100430",
                "classCount": 47,
                "downloadUrl": "https://..."
            }
        ]
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

        # Get query parameters
        params = event.get("queryStringParameters") or {}
        limit = int(params.get("limit", 20))

        # Query DynamoDB ("project" is a DynamoDB reserved word, hence the alias)
        response = dynamodb.query(
            TableName=SNAPSHOTS_TABLE,
            KeyConditionExpression="#project = :project",
            ExpressionAttributeNames={"#project": "project"},
            ExpressionAttributeValues={":project": {"S": project}},
            Limit=limit,
            ScanIndexForward=False,  # Latest first
        )

        # Convert items to JSON with presigned URLs
        snapshots = []
        for item in response.get("Items", []):
            snapshot_id = item["snapshotId"]["S"]
            s3_location = item.get("s3Location", {}).get("S", "")

            # Generate presigned URL for snapshot download
            download_url = None
            if s3_location:
                s3_path = s3_location.replace("s3://", "")
                bucket, key = s3_path.split("/", 1)
                download_url = s3.generate_presigned_url(
                    "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=900
                )

            snapshots.append(
                {
                    "snapshotId": snapshot_id,
                    "triggerType": item.get("triggerType", {}).get("S"),
                    "sourceType": item.get("sourceType", {}).get("S"),
                    "classCount": int(item.get("classCount", {}).get("N", 0)),
                    "propertyCount": int(item.get("propertyCount", {}).get("N", 0)),
                    "createdAt": item.get("createdAt", {}).get("S"),
                    "downloadUrl": download_url,
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
                    "snapshots": snapshots,
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
                        "message": "Failed to list snapshots",
                    }
                }
            ),
        }
