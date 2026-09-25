import json
import os
import re
import time
import uuid
from urllib.parse import urlparse

import boto3

s3 = boto3.client("s3")
batch = boto3.client("batch")
dynamodb = boto3.client("dynamodb")

BUCKET_NAME = os.environ["BUCKET_NAME"]
JOBS_TABLE = os.environ["JOBS_TABLE"]
JOB_QUEUE = os.environ["JOB_QUEUE"]
JOB_DEFINITION = os.environ["JOB_DEFINITION"]

MIRO_BOARD_PATH = re.compile(r"^/app/board/(?P<board_id>[A-Za-z0-9_=-]+)(?:/.*)?$")
MAX_SOURCE_URL_LENGTH = 2048


def error_response(status_code, code, message):
    """Build an error response in the shape every handler in this API uses."""
    return {
        "statusCode": status_code,
        "body": json.dumps({"error": {"code": code, "message": message}}),
    }


def validate_miro_url(source_url):
    """Return (source_type, board_id), or (None, None) when unsupported.

    Mirrors parse_board_id_from_url in app/src/extract/miro/miro_text_extractor.py,
    duplicated because lambda/ cannot import from app/. The host is matched
    exactly to reject look-alikes; trailing paths and query strings are kept
    because Miro's share button emits them.
    """
    parsed = urlparse(source_url)
    if parsed.scheme != "https":
        return None, None
    host = parsed.netloc.split(":")[0].lower()
    if host != "miro.com" and not host.endswith(".miro.com"):
        return None, None
    match = MIRO_BOARD_PATH.match(parsed.path)
    return ("miro", match.group("board_id")) if match else (None, None)


def resolve_url_source(source_url, requested_source_type, project):
    """Validate a URL source and create its Jobs record.

    Returns (job_id, source_type) or an error response dict.
    """
    if len(source_url) > MAX_SOURCE_URL_LENGTH:
        return error_response(
            400,
            "VALIDATION_ERROR",
            f"sourceUrl exceeds {MAX_SOURCE_URL_LENGTH} characters",
        )

    source_type, _board_id = validate_miro_url(source_url)
    if source_type is None:
        return error_response(
            400,
            "VALIDATION_ERROR",
            "Unsupported sourceUrl; expected a Miro board URL like https://miro.com/app/board/BOARD_ID",
        )

    # Whitelist, not passthrough: a caller must not steer the record away
    # from what the URL actually is.
    if requested_source_type and requested_source_type != source_type:
        return error_response(
            400,
            "VALIDATION_ERROR",
            f"sourceType '{requested_source_type}' does not match the sourceUrl (detected '{source_type}')",
        )

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    dynamodb.put_item(
        TableName=JOBS_TABLE,
        Item={
            "jobId": {"S": job_id},
            # Plain attribute used by the project-timestamp-index GSI
            "timestamp": {"N": str(int(time.time()))},
            "project": {"S": project},
            "jobType": {"S": "extraction"},
            # No upload step for a URL, so the job is ready to run
            "status": {"S": "PENDING"},
            "sourceType": {"S": source_type},
            "sourceLocation": {"S": source_url},
            # Matches batch_runner.source_document_name() for non-s3:// sources
            "filename": {"S": source_url},
            "createdAt": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            "progress": {"N": "0"},
            "progressMessage": {"S": "Waiting for worker to start"},
        },
        ConditionExpression="attribute_not_exists(jobId)",
    )

    return job_id, source_type


def resolve_uploaded_source(job_id):
    """Look up an uploaded file's source location.

    Returns the s3:// location, or an error response dict.
    """
    response = dynamodb.get_item(TableName=JOBS_TABLE, Key={"jobId": {"S": job_id}})

    if "Item" not in response:
        return error_response(404, "NOT_FOUND", f"Job {job_id} not found")

    job_item = response["Item"]

    # Verify file exists in S3
    source_location = job_item.get("sourceLocation", {}).get("S")
    if not source_location or not source_location.startswith("s3://"):
        return error_response(400, "VALIDATION_ERROR", "Invalid source location")

    # Parse S3 path
    s3_path = source_location.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)

    try:
        s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError:
        return error_response(404, "NOT_FOUND", "File not uploaded to S3 yet")

    return source_location


def handler(event, context):
    """
    Submit an extraction job.

    Request: POST /v1/extract
    Body (uploaded file, step 2 of the upload flow):
        {"jobId": "job-xxx", "project": "backend"}
    Body (URL source, single step):
        {"sourceUrl": "https://miro.com/app/board/BOARD_ID", "project": "backend"}

    Response: {"jobId": "job-xxx", "status": "PENDING"}
    """
    try:
        # Parse request body
        body = json.loads(event.get("body", "{}"))
        job_id = body.get("jobId")
        source_url = body.get("sourceUrl")
        project = body.get("project")

        if job_id and source_url:
            return error_response(
                400,
                "VALIDATION_ERROR",
                "Provide either jobId or sourceUrl, not both",
            )

        if not project or not (job_id or source_url):
            return error_response(
                400,
                "VALIDATION_ERROR",
                "Missing required fields: project, and one of jobId or sourceUrl",
            )

        if source_url:
            resolved = resolve_url_source(source_url, body.get("sourceType"), project)
            if isinstance(resolved, dict):
                return resolved
            job_id, source_type = resolved
            source_location = source_url
        else:
            resolved = resolve_uploaded_source(job_id)
            if isinstance(resolved, dict):
                return resolved
            source_location = resolved
            # Left unset so batch_runner infers it from the file extension
            source_type = None

        # Submit AWS Batch job
        container_environment = [
            {"name": "JOB_ID", "value": job_id},
            {"name": "PROJECT", "value": project},
            {"name": "SOURCE", "value": source_location},
        ]
        if source_type:
            container_environment.append({"name": "SOURCE_TYPE", "value": source_type})

        try:
            batch_response = batch.submit_job(
                jobName=f"extraction-{job_id}",
                jobQueue=JOB_QUEUE,
                jobDefinition=JOB_DEFINITION,
                containerOverrides={"environment": container_environment},
            )
        except Exception as e:
            # The record already exists; leaving it PENDING would make the
            # client poll forever.
            print(f"Error submitting batch job: {str(e)}")
            dynamodb.update_item(
                TableName=JOBS_TABLE,
                Key={"jobId": {"S": job_id}},
                UpdateExpression="SET #status = :status, errorMessage = :msg",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":status": {"S": "FAILED"},
                    ":msg": {"S": "Failed to submit extraction job to AWS Batch"},
                },
            )
            raise

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
        return error_response(500, "INTERNAL_ERROR", "Failed to submit extraction job")
