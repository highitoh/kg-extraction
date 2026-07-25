import json
import os

import boto3

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

CHANGESETS_TABLE = os.environ["CHANGESETS_TABLE"]


def handler(event, context):
    """
    Get the status and results of a changeset.

    Request: GET /v1/projects/{project}/changesets/{changesetId}

    Response: {
        "changesetId": "cs_20260628_150000_ab12cd",
        "status": "COMPLETED",
        "beforeSnapshotId": "...",
        "afterSnapshotId": "...",
        "totalChanges": 6,
        "driversCount": 1,
        "downloadUrl": "https://..."
    }
    """
    try:
        path_params = event.get("pathParameters", {}) or {}
        project = path_params.get("project")
        changeset_id = path_params.get("changesetId")

        if not project or not changeset_id:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Missing project or changesetId parameter",
                        }
                    }
                ),
            }

        response = dynamodb.get_item(
            TableName=CHANGESETS_TABLE,
            Key={"project": {"S": project}, "changesetId": {"S": changeset_id}},
        )

        item = response.get("Item")
        if not item:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "NOT_FOUND",
                            "message": f"Changeset {changeset_id} not found",
                        }
                    }
                ),
            }

        status = item.get("status", {}).get("S", "UNKNOWN")

        # Presigned download URL for the changeset JSON once completed
        download_url = None
        s3_location = item.get("s3Location", {}).get("S", "")
        if status == "COMPLETED" and s3_location:
            s3_path = s3_location.replace("s3://", "")
            bucket, key = s3_path.split("/", 1)
            download_url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=900)

        body = {
            "changesetId": changeset_id,
            "project": project,
            "status": status,
            "beforeSnapshotId": item.get("beforeSnapshotId", {}).get("S"),
            "afterSnapshotId": item.get("afterSnapshotId", {}).get("S"),
            "createdAt": item.get("createdAt", {}).get("S"),
            "startedAt": item.get("startedAt", {}).get("S"),
            "completedAt": item.get("completedAt", {}).get("S"),
            "totalChanges": int(item["totalChanges"]["N"]) if "totalChanges" in item else None,
            "driversCount": int(item["driversCount"]["N"]) if "driversCount" in item else None,
            "errorMessage": item.get("errorMessage", {}).get("S"),
            "downloadUrl": download_url,
        }

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(body),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to get changeset",
                    }
                }
            ),
        }
