import json
import os

import boto3

dynamodb = boto3.client("dynamodb")

JOBS_TABLE = os.environ["JOBS_TABLE"]


def handler(event, context):
    """
    Get job status and progress.

    Request: GET /v1/jobs/{jobId}

    Response: {
        "jobId": "job-xxx",
        "status": "RUNNING",
        "progress": 45,
        "progressMessage": "Extracting classes (47 found)",
        ...
    }
    """
    try:
        # Get job ID from path parameters
        job_id = event.get("pathParameters", {}).get("jobId")

        if not job_id:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Missing jobId parameter",
                        }
                    }
                ),
            }

        # Get job record from DynamoDB
        response = dynamodb.get_item(TableName=JOBS_TABLE, Key={"jobId": {"S": job_id}})

        if "Item" not in response:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "NOT_FOUND",
                            "message": f"Job {job_id} not found",
                        }
                    }
                ),
            }

        item = response["Item"]

        # Convert DynamoDB item to JSON
        result = {
            "jobId": job_id,
            "project": item.get("project", {}).get("S"),
            "status": item.get("status", {}).get("S"),
            "progress": int(item.get("progress", {}).get("N", 0)),
            "progressMessage": item.get("progressMessage", {}).get("S"),
            "sourceType": item.get("sourceType", {}).get("S"),
            "filename": item.get("filename", {}).get("S"),
            "createdAt": item.get("createdAt", {}).get("S"),
        }

        # Optional fields
        if "startedAt" in item:
            result["startedAt"] = item["startedAt"]["S"]
        if "completedAt" in item:
            result["completedAt"] = item["completedAt"]["S"]
        if "batchJobId" in item:
            result["batchJobId"] = item["batchJobId"]["S"]
        if "errorMessage" in item:
            result["errorMessage"] = item["errorMessage"]["S"]
        if "asOf" in item:
            result["asOf"] = item["asOf"]["S"]
        if "tokenUsage" in item:
            token_usage = item["tokenUsage"]["M"]
            result["tokenUsage"] = {
                "inputTokens": int(token_usage.get("inputTokens", {}).get("N", 0)),
                "outputTokens": int(token_usage.get("outputTokens", {}).get("N", 0)),
                "totalTokens": int(token_usage.get("totalTokens", {}).get("N", 0)),
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
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to get job status",
                    }
                }
            ),
        }
