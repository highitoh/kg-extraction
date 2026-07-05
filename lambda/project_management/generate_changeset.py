import json
import os
import time
import uuid

import boto3

batch = boto3.client("batch")
dynamodb = boto3.client("dynamodb")

SNAPSHOTS_TABLE = os.environ["SNAPSHOTS_TABLE"]
CHANGESETS_TABLE = os.environ["CHANGESETS_TABLE"]
JOB_QUEUE = os.environ["JOB_QUEUE"]
JOB_DEFINITION = os.environ["JOB_DEFINITION"]


def handler(event, context):
    """
    Generate changeset between two snapshots (submitted as Batch job).

    Request: POST /v1/projects/{project}/changesets
    Body: {"beforeSnapshotId": "xxx", "afterSnapshotId": "yyy"}

    Response: {
        "changesetId": "cs_20260628_150000",
        "status": "PENDING",
        "statusUrl": "/v1/projects/backend/changesets/cs_..."
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

        # Parse request body
        body = json.loads(event.get("body", "{}"))
        before_snapshot_id = body.get("beforeSnapshotId")
        after_snapshot_id = body.get("afterSnapshotId")

        if not before_snapshot_id or not after_snapshot_id:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Missing required fields: beforeSnapshotId, afterSnapshotId",
                        }
                    }
                ),
            }

        # Verify snapshots exist
        for snapshot_id in [before_snapshot_id, after_snapshot_id]:
            response = dynamodb.get_item(
                TableName=SNAPSHOTS_TABLE,
                Key={"project": {"S": project}, "snapshotId": {"S": snapshot_id}},
            )
            if "Item" not in response:
                return {
                    "statusCode": 404,
                    "body": json.dumps(
                        {
                            "error": {
                                "code": "NOT_FOUND",
                                "message": f"Snapshot {snapshot_id} not found",
                            }
                        }
                    ),
                }

        # Generate changeset ID. The fixed-width timestamp prefix keeps
        # lexicographic order chronological; the random suffix prevents
        # collisions between requests in the same second.
        changeset_id = f"cs_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}"

        # Submit AWS Batch job for changeset generation
        batch_response = batch.submit_job(
            jobName=f"changeset-{changeset_id}",
            jobQueue=JOB_QUEUE,
            jobDefinition=JOB_DEFINITION,
            containerOverrides={
                "environment": [
                    {"name": "JOB_TYPE", "value": "changeset"},
                    {"name": "PROJECT", "value": project},
                    {"name": "CHANGESET_ID", "value": changeset_id},
                    {"name": "BEFORE_SNAPSHOT_ID", "value": before_snapshot_id},
                    {"name": "AFTER_SNAPSHOT_ID", "value": after_snapshot_id},
                ]
            },
        )

        # Create changeset record in DynamoDB
        dynamodb.put_item(
            TableName=CHANGESETS_TABLE,
            Item={
                "project": {"S": project},
                "changesetId": {"S": changeset_id},
                "beforeSnapshotId": {"S": before_snapshot_id},
                "afterSnapshotId": {"S": after_snapshot_id},
                "status": {"S": "PENDING"},
                "createdAt": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                "batchJobId": {"S": batch_response["jobArn"]},
            },
            ConditionExpression="attribute_not_exists(changesetId)",
        )

        return {
            "statusCode": 202,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "changesetId": changeset_id,
                    "status": "PENDING",
                    "message": "Changeset generation job submitted",
                    "statusUrl": f"/v1/projects/{project}/changesets/{changeset_id}",
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
                        "message": "Failed to generate changeset",
                    }
                }
            ),
        }
