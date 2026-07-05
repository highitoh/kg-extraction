import json
import os
import time

import boto3

s3 = boto3.client("s3")
batch = boto3.client("batch")
dynamodb = boto3.client("dynamodb")

BUCKET_NAME = os.environ["BUCKET_NAME"]
JOBS_TABLE = os.environ["JOBS_TABLE"]
JOB_QUEUE = os.environ["JOB_QUEUE"]
JOB_DEFINITION = os.environ["JOB_DEFINITION"]


def handler(event, context):
    """
    Submit extraction job after file upload (step 2 of upload flow).

    Request: POST /v1/extract
    Body: {"jobId": "job-xxx", "project": "backend"}

    Response: {"jobId": "job-xxx", "status": "PENDING"}
    """
    try:
        # Parse request body
        body = json.loads(event.get("body", "{}"))
        job_id = body.get("jobId")
        project = body.get("project")

        if not job_id or not project:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Missing required fields: jobId, project",
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

        job_item = response["Item"]

        # Verify file exists in S3
        source_location = job_item.get("sourceLocation", {}).get("S")
        if not source_location or not source_location.startswith("s3://"):
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Invalid source location",
                        }
                    }
                ),
            }

        # Parse S3 path
        s3_path = source_location.replace("s3://", "")
        bucket, key = s3_path.split("/", 1)

        try:
            s3.head_object(Bucket=bucket, Key=key)
        except s3.exceptions.ClientError:
            return {
                "statusCode": 404,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "NOT_FOUND",
                            "message": "File not uploaded to S3 yet",
                        }
                    }
                ),
            }

        # Submit AWS Batch job
        batch_response = batch.submit_job(
            jobName=f"extraction-{job_id}",
            jobQueue=JOB_QUEUE,
            jobDefinition=JOB_DEFINITION,
            containerOverrides={
                "environment": [
                    {"name": "JOB_ID", "value": job_id},
                    {"name": "PROJECT", "value": project},
                    {"name": "SOURCE", "value": source_location},
                ]
            },
        )

        batch_job_id = batch_response["jobId"]
        batch_job_arn = batch_response["jobArn"]

        # Update DynamoDB (status: PENDING)
        dynamodb.update_item(
            TableName=JOBS_TABLE,
            Key={"jobId": {"S": job_id}},
            UpdateExpression=(
                "SET #status = :status, batchJobId = :batchJobId, "
                "uploadedAt = :uploadedAt, progress = :progress, progressMessage = :msg"
            ),
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": {"S": "PENDING"},
                ":batchJobId": {"S": batch_job_arn},
                ":uploadedAt": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                ":progress": {"N": "0"},
                ":msg": {"S": "Waiting for worker to start"},
            },
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "jobId": job_id,
                    "status": "PENDING",
                    "message": "Extraction job submitted",
                    "batchJobId": batch_job_id,
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
                        "message": "Failed to submit extraction job",
                    }
                }
            ),
        }
