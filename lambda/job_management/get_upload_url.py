import json
import os
import time
import uuid

import boto3

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

BUCKET_NAME = os.environ["BUCKET_NAME"]
JOBS_TABLE = os.environ["JOBS_TABLE"]


def handler(event, context):
    """
    Generate presigned S3 upload URL for file upload (step 1 of upload flow).

    Request: POST /v1/upload-url
    Body: {"filename": "document.pdf", "project": "backend"}

    Response: {"jobId": "job-xxx", "uploadUrl": "https://...", "expiresIn": 900}
    """
    try:
        # Parse request body
        body = json.loads(event.get("body", "{}"))
        filename = body.get("filename")
        project = body.get("project")

        if not filename or not project:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "VALIDATION_ERROR",
                            "message": "Missing required fields: filename, project",
                        }
                    }
                ),
            }

        # Generate job ID
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        timestamp = int(time.time())

        # Generate presigned S3 upload URL. ContentType is deliberately not
        # signed: a signed value would force clients to send exactly that
        # Content-Type header or get 403 SignatureDoesNotMatch.
        s3_key = f"input/{job_id}/{filename}"
        upload_url = s3.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": BUCKET_NAME,
                "Key": s3_key,
            },
            ExpiresIn=900,  # 15 minutes
        )

        # Create job record in DynamoDB (status: PENDING_UPLOAD).
        # timestamp is a plain attribute used by the project-timestamp-index GSI.
        # The condition guarantees a single record per jobId even on retries.
        dynamodb.put_item(
            TableName=JOBS_TABLE,
            Item={
                "jobId": {"S": job_id},
                "timestamp": {"N": str(timestamp)},
                "project": {"S": project},
                "status": {"S": "PENDING_UPLOAD"},
                "filename": {"S": filename},
                "sourceLocation": {"S": f"s3://{BUCKET_NAME}/{s3_key}"},
                "uploadUrl": {"S": upload_url},
                "uploadUrlExpiry": {"N": str(timestamp + 900)},
                "createdAt": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                "progress": {"N": "0"},
                "progressMessage": {"S": "Waiting for file upload"},
            },
            ConditionExpression="attribute_not_exists(jobId)",
        )

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"jobId": job_id, "uploadUrl": upload_url, "expiresIn": 900}),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to generate upload URL",
                    }
                }
            ),
        }
