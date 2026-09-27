import json
import os
import time

import boto3

dynamodb = boto3.client("dynamodb")
s3 = boto3.client("s3")

BUCKET_NAME = os.environ["BUCKET_NAME"]
JOBS_TABLE = os.environ["JOBS_TABLE"]
SNAPSHOTS_TABLE = os.environ["SNAPSHOTS_TABLE"]
CHANGESETS_TABLE = os.environ["CHANGESETS_TABLE"]
CLASSES_TABLE = os.environ["CLASSES_TABLE"]
PROPERTIES_TABLE = os.environ["PROPERTIES_TABLE"]
INSTANCES_TABLE = os.environ["INSTANCES_TABLE"]
PROJECTS_TABLE = os.environ["PROJECTS_TABLE"]

LOCK_EXPIRY_SECONDS = 15 * 60


def _error(status_code, code, message):
    return {
        "statusCode": status_code,
        "body": json.dumps({"error": {"code": code, "message": message}}),
    }


def _query_keys(table, key_names, pk_name, pk_value, index_name=None):
    """Return the base-table keys of every item under one partition."""
    params = {
        "TableName": table,
        "KeyConditionExpression": "#pk = :pk",
        "ExpressionAttributeNames": {"#pk": pk_name},
        "ExpressionAttributeValues": {":pk": {"S": pk_value}},
    }
    if index_name:
        params["IndexName"] = index_name
    keys = []
    for page in dynamodb.get_paginator("query").paginate(**params):
        keys.extend({name: item[name] for name in key_names} for item in page["Items"])
    return keys


def _batch_delete(table, keys):
    for i in range(0, len(keys), 25):
        requests = {table: [{"DeleteRequest": {"Key": key}} for key in keys[i : i + 25]]}
        while requests:
            requests = dynamodb.batch_write_item(RequestItems=requests).get("UnprocessedItems")
    return len(keys)


def _delete_s3_prefix(prefix):
    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": objects, "Quiet": True})
            count += len(objects)
    return count


def _acquire_lock(project, owner):
    now = int(time.time())
    try:
        dynamodb.put_item(
            TableName=PROJECTS_TABLE,
            Item={
                "project": {"S": project},
                "recordType": {"S": "LOCK"},
                "jobId": {"S": owner},
                "acquiredAt": {"S": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))},
                "expiresAt": {"N": str(now + LOCK_EXPIRY_SECONDS)},
            },
            # A retried delete re-takes its own unexpired lock
            ConditionExpression="attribute_not_exists(#project) OR expiresAt < :now OR begins_with(jobId, :delete)",
            ExpressionAttributeNames={"#project": "project"},
            ExpressionAttributeValues={":now": {"N": str(now)}, ":delete": {"S": "delete-"}},
        )
        return True
    except dynamodb.exceptions.ConditionalCheckFailedException:
        return False


def handler(event, context):
    """
    Delete a project's knowledge graph: S3 files and all DynamoDB records.

    Request: DELETE /v1/projects/{project}

    Response: {
        "project": "backend",
        "deleted": {"s3Objects": 42, "classes": 120, ...}
    }
    """
    try:
        project = (event.get("pathParameters") or {}).get("project")
        if not project:
            return _error(400, "VALIDATION_ERROR", "Missing project parameter")

        response = dynamodb.get_item(
            TableName=PROJECTS_TABLE,
            Key={"project": {"S": project}, "recordType": {"S": "METADATA"}},
        )
        if "Item" not in response:
            return _error(404, "NOT_FOUND", f"Project {project} not found")

        if not _acquire_lock(project, f"delete-{context.aws_request_id}"):
            return _error(409, "CONFLICT", f"Project {project} has an extraction in progress")

        # S3 first so a later extraction cannot merge the old knowledge.json back in
        deleted = {"s3Objects": _delete_s3_prefix(f"projects/{project}/")}

        class_keys = _query_keys(CLASSES_TABLE, ["project", "className"], "project", project)
        property_keys = _query_keys(
            PROPERTIES_TABLE,
            ["projectSourceClass", "relationship"],
            "project",
            project,
            index_name="propertyName-index",
        )

        entity_keys = [f"{project}#CLASS#{key['className']['S']}" for key in class_keys]
        entity_keys += [
            f"{project}#PROP#{key['projectSourceClass']['S'][len(project) + 1 :]}#{key['relationship']['S']}"
            for key in property_keys
        ]
        deleted["instances"] = sum(
            _batch_delete(
                INSTANCES_TABLE,
                _query_keys(INSTANCES_TABLE, ["entityKey", "instanceKey"], "entityKey", entity_key),
            )
            for entity_key in entity_keys
        )

        deleted["properties"] = _batch_delete(PROPERTIES_TABLE, property_keys)
        deleted["classes"] = _batch_delete(CLASSES_TABLE, class_keys)
        deleted["snapshots"] = _batch_delete(
            SNAPSHOTS_TABLE, _query_keys(SNAPSHOTS_TABLE, ["project", "snapshotId"], "project", project)
        )
        deleted["changesets"] = _batch_delete(
            CHANGESETS_TABLE, _query_keys(CHANGESETS_TABLE, ["project", "changesetId"], "project", project)
        )
        deleted["jobs"] = _batch_delete(
            JOBS_TABLE,
            _query_keys(JOBS_TABLE, ["jobId"], "project", project, index_name="project-timestamp-index"),
        )

        # Projects last: METADATA must survive until the end so a retry still finds the project
        _batch_delete(PROJECTS_TABLE, _query_keys(PROJECTS_TABLE, ["project", "recordType"], "project", project))

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"project": project, "deleted": deleted}),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return _error(500, "INTERNAL_ERROR", "Failed to delete project")
