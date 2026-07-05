import json
import os

import boto3

s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

BUCKET_NAME = os.environ["BUCKET_NAME"]
JOBS_TABLE = os.environ["JOBS_TABLE"]


def handler(event, context):
    """
    Download extraction results (presigned URLs).

    Request: GET /v1/jobs/{jobId}/results

    Response: {
        "jobId": "job-xxx",
        "status": "COMPLETED",
        "outputs": {
            "ttl": "https://s3.../knowledge_graph_xxx.ttl?...",
            "nodesCsv": "https://s3.../nodes.csv?...",
            "relationshipsCsv": "https://s3.../relationships.csv?...",
            "knowledgeJson": "https://s3.../knowledge.json?..."
        }
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
        status = item.get("status", {}).get("S")

        if status != "COMPLETED":
            return {
                "statusCode": 409,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "CONFLICT",
                            "message": f"Job not completed yet (current status: {status})",
                        }
                    }
                ),
            }

        # Get output location from DynamoDB
        output_location = item.get("outputLocation", {}).get("S")
        project = item.get("project", {}).get("S")

        if not output_location:
            return {
                "statusCode": 500,
                "body": json.dumps(
                    {
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "Output location not found",
                        }
                    }
                ),
            }

        # Parse S3 path (tolerate a trailing slash on the stored location,
        # which would otherwise produce a double slash that matches no keys)
        s3_path = output_location.replace("s3://", "")
        bucket, prefix = s3_path.split("/", 1)
        prefix = prefix.rstrip("/")

        # List output files
        list_response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/outputs/")

        # Find output files
        outputs = {}
        for obj in list_response.get("Contents", []):
            key = obj["Key"]
            filename = key.split("/")[-1]

            # Generate presigned URL (valid for 15 minutes)
            url = s3.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=900)

            if filename.endswith(".ttl"):
                outputs["ttl"] = url
            elif filename == "nodes.csv":
                outputs["nodesCsv"] = url
            elif filename == "relationships.csv":
                outputs["relationshipsCsv"] = url

        # Also include knowledge.json (current accumulated knowledge)
        knowledge_key = f"projects/{project}/knowledge.json"
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=knowledge_key)
            outputs["knowledgeJson"] = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": knowledge_key},
                ExpiresIn=900,
            )
        except Exception:
            pass

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {
                    "jobId": job_id,
                    "status": status,
                    "outputs": outputs,
                    "expiresIn": 900,
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
                        "message": "Failed to get job results",
                    }
                }
            ),
        }
