#!/usr/bin/env python3
"""
AWS Batch job orchestration layer for knowledge extraction.

This script runs inside AWS Batch (Fargate container) and orchestrates:
1. Resolve the job record (creates one for scheduled runs without JOB_ID)
2. Acquire the per-project lock (serializes writers of knowledge.json)
3. Download input files from S3 (or pass URLs through to the workflow)
4. Call extract_workflow.py
5. Upload outputs to S3 and create a snapshot (S3 + DynamoDB)
6. Write the knowledge graph to DynamoDB (aggregates + instance evidence)
7. Update project metadata / extraction history and job status

Environment variables:
- JOB_ID: Unique job identifier (optional; scheduled runs derive one from
  AWS_BATCH_JOB_ID, which is stable across retry attempts)
- JOB_TYPE: extraction (default) | scheduled
- PROJECT: Project name
- SOURCE: S3 path or external URL
- SOURCE_TYPE: Optional source type hint (pdf, miro, confluence, ...)
- AS_OF: Optional YYYYMMDD; extract a Miro board as of 23:59:59 UTC that day
- STORAGE_BACKEND: 's3' (required for AWS)
- MIRO_API_KEY: Miro REST API token, injected from Secrets Manager by the
  Batch job definition; required for Miro sources
- BUCKET_NAME: S3 bucket name
- JOBS_TABLE: DynamoDB Jobs table name
- CLASSES_TABLE: DynamoDB Classes table name
- PROPERTIES_TABLE: DynamoDB Properties table name
- INSTANCES_TABLE: DynamoDB Instances table name (per-instance evidence)
- SNAPSHOTS_TABLE: DynamoDB Snapshots table name
- PROJECTS_TABLE: DynamoDB Projects table name
- CHANGESETS_TABLE: DynamoDB Changesets table name
- LOCK_MAX_WAIT_SECONDS: How long to wait for the project lock (default 1800)
"""

import logging
import os
import re
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# AWS clients
s3 = boto3.client("s3")
dynamodb = boto3.client("dynamodb")

# Environment variables
JOB_ID_ENV = os.environ.get("JOB_ID")
JOB_TYPE = os.environ.get("JOB_TYPE", "extraction")
PROJECT = os.environ.get("PROJECT")
SOURCE = os.environ.get("SOURCE")
SOURCE_TYPE = os.environ.get("SOURCE_TYPE")
AS_OF = os.environ.get("AS_OF")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
JOBS_TABLE = os.environ.get("JOBS_TABLE")
CLASSES_TABLE = os.environ.get("CLASSES_TABLE")
PROPERTIES_TABLE = os.environ.get("PROPERTIES_TABLE")
INSTANCES_TABLE = os.environ.get("INSTANCES_TABLE")
SNAPSHOTS_TABLE = os.environ.get("SNAPSHOTS_TABLE")
PROJECTS_TABLE = os.environ.get("PROJECTS_TABLE")
CHANGESETS_TABLE = os.environ.get("CHANGESETS_TABLE")
LOCK_MAX_WAIT_SECONDS = int(os.environ.get("LOCK_MAX_WAIT_SECONDS", "1800"))
LOCK_POLL_SECONDS = 20
LOCK_EXPIRY_SECONDS = 3 * 60 * 60  # longer than the 2h Batch job timeout

# Project profile file name, as read by src/utils/project_profile.py
PROFILE_FILENAME = "project_profile.json"

# Working directories
WORK_DIR = Path("/tmp/extraction")
INPUT_DIR = WORK_DIR / "input"

# How much of a source excerpt to keep on an instance item
INSTANCE_SOURCE_MAX_CHARS = 1000

_WHITESPACE_RUN = re.compile(r"\s+")


def now_iso() -> str:
    """Current UTC time as an ISO timestamp with Z suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_class_name(name: str) -> str:
    """Normalize a class name into its DynamoDB key form.

    Must stay in sync with lambda/knowledge_api/normalize.py (see
    infra/docs/database-schema.md, "Key Normalization").
    """
    without_delimiter = name.replace("#", " ")
    collapsed = _WHITESPACE_RUN.sub(" ", without_delimiter).strip()
    return collapsed.casefold()


def property_name_from_iri(property_iri: str) -> str:
    """Extract the relationship name from a metamodel property IRI."""
    return property_iri.rsplit("#", 1)[-1] if "#" in property_iri else property_iri


def infer_source_type(source: str) -> str:
    """Best-effort source type when SOURCE_TYPE is not provided."""
    if source.startswith("miro:"):
        return "miro"
    if source.startswith(("http://", "https://")):
        for marker, source_type in (
            ("miro.com", "miro"),
            ("atlassian.net/wiki", "confluence"),
            ("atlassian.net", "jira"),
            ("github.com", "github"),
        ):
            if marker in source:
                return source_type
        return "url"
    suffix = Path(source).suffix.lower().lstrip(".")
    return {"md": "markdown"}.get(suffix, suffix or "unknown")


# ---------------------------------------------------------------------------
# Jobs table
# ---------------------------------------------------------------------------


def resolve_job_id() -> str:
    """Return the job ID, creating a Jobs record for scheduled runs.

    On-demand jobs receive JOB_ID (record created by the upload Lambda).
    Scheduled EventBridge jobs have no JOB_ID, so one is derived from
    AWS_BATCH_JOB_ID - stable across retry attempts, making the conditional
    put idempotent (a retry reuses the record created by the first attempt).
    """
    if JOB_ID_ENV:
        return JOB_ID_ENV

    batch_job_id = os.environ.get("AWS_BATCH_JOB_ID") or uuid.uuid4().hex[:12]
    job_id = f"job-{batch_job_id}"
    timestamp = int(time.time())
    try:
        dynamodb.put_item(
            TableName=JOBS_TABLE,
            Item={
                "jobId": {"S": job_id},
                "timestamp": {"N": str(timestamp)},
                "project": {"S": PROJECT},
                "status": {"S": "PENDING"},
                "jobType": {"S": JOB_TYPE},
                "sourceType": {"S": SOURCE_TYPE or infer_source_type(SOURCE)},
                "sourceLocation": {"S": SOURCE},
                "filename": {"S": source_document_name(SOURCE)},
                "createdAt": {"S": now_iso()},
                "progress": {"N": "0"},
                "progressMessage": {"S": "Scheduled job starting"},
            },
            ConditionExpression="attribute_not_exists(jobId)",
        )
        logger.info(f"Created Jobs record for scheduled run: {job_id}")
    except dynamodb.exceptions.ConditionalCheckFailedException:
        logger.info(f"Reusing Jobs record from previous attempt: {job_id}")
    return job_id


def update_job(job_id: str, expression: str, names: dict, values: dict) -> None:
    """Update the job record."""
    dynamodb.update_item(
        TableName=JOBS_TABLE,
        Key={"jobId": {"S": job_id}},
        UpdateExpression=expression,
        **({"ExpressionAttributeNames": names} if names else {}),
        ExpressionAttributeValues=values,
    )


def update_progress(job_id: str, progress: int, message: str) -> None:
    """Update job progress in DynamoDB (best effort)."""
    try:
        update_job(
            job_id,
            "SET progress = :p, progressMessage = :m, lastUpdated = :t",
            {},
            {
                ":p": {"N": str(progress)},
                ":m": {"S": message},
                ":t": {"S": now_iso()},
            },
        )
        logger.info(f"Progress: {progress}% - {message}")
    except Exception as e:
        logger.error(f"Failed to update progress: {e}")


# ---------------------------------------------------------------------------
# Per-project lock
# ---------------------------------------------------------------------------
# knowledge.json is read-modify-written by every extraction, so concurrent
# jobs on the same project would silently drop each other's results. The lock
# is a Projects-table item (recordType="LOCK") acquired with a conditional
# put; expiresAt guards against locks leaked by killed containers. It has no
# entityType attribute, so it never appears in the project listing index.


def acquire_project_lock(job_id: str) -> bool:
    """Acquire the per-project lock, waiting up to LOCK_MAX_WAIT_SECONDS.

    Re-entrant for the same job so Batch retries don't deadlock on a lock
    left behind by their own previous attempt.
    """
    deadline = time.time() + LOCK_MAX_WAIT_SECONDS
    while True:
        try:
            dynamodb.put_item(
                TableName=PROJECTS_TABLE,
                Item={
                    "project": {"S": PROJECT},
                    "recordType": {"S": "LOCK"},
                    "jobId": {"S": job_id},
                    "acquiredAt": {"S": now_iso()},
                    "expiresAt": {"N": str(int(time.time()) + LOCK_EXPIRY_SECONDS)},
                },
                ConditionExpression="attribute_not_exists(#project) OR expiresAt < :now OR jobId = :jobId",
                ExpressionAttributeNames={"#project": "project"},
                ExpressionAttributeValues={
                    ":now": {"N": str(int(time.time()))},
                    ":jobId": {"S": job_id},
                },
            )
            logger.info(f"Acquired project lock for '{PROJECT}'")
            return True
        except dynamodb.exceptions.ConditionalCheckFailedException:
            if time.time() >= deadline:
                return False
            logger.info(f"Project '{PROJECT}' is locked by another job, waiting {LOCK_POLL_SECONDS}s")
            time.sleep(LOCK_POLL_SECONDS)


def release_project_lock(job_id: str) -> None:
    """Release the per-project lock (best effort, only if we own it)."""
    try:
        dynamodb.delete_item(
            TableName=PROJECTS_TABLE,
            Key={"project": {"S": PROJECT}, "recordType": {"S": "LOCK"}},
            ConditionExpression="jobId = :jobId",
            ExpressionAttributeValues={":jobId": {"S": job_id}},
        )
        logger.info(f"Released project lock for '{PROJECT}'")
    except Exception as e:
        logger.warning(f"Failed to release project lock: {e}")


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def source_document_name(source: str) -> str:
    """Human-readable source document identifier (filename or URL)."""
    if source.startswith("s3://"):
        return source.rsplit("/", 1)[-1]
    return source


def download_from_s3(s3_path: str, local_path: Path) -> None:
    """Download file from S3 to local path."""
    s3_path = s3_path.replace("s3://", "")
    bucket, key = s3_path.split("/", 1)

    logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")
    s3.download_file(bucket, key, str(local_path))
    logger.info(f"Downloaded {local_path.stat().st_size} bytes")


def upload_to_s3(local_path: Path, s3_key: str) -> None:
    """Upload file from local path to S3."""
    logger.info(f"Uploading {local_path} to s3://{BUCKET_NAME}/{s3_key}")
    s3.upload_file(str(local_path), BUCKET_NAME, s3_key)
    logger.info(f"Uploaded {local_path.stat().st_size} bytes")


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def create_snapshot(
    knowledge_json_path: Path,
    source_type: str,
    class_count: int,
    property_count: int,
    source_raw_path: Path = None,
    snapshot_id: str = None,
    source_identifier: str = None,
) -> str:
    """Create a snapshot: claim the DynamoDB record, then upload files to S3.

    The record is claimed with a conditional put; on a same-second collision
    a random suffix is appended (see database-schema.md, Snapshots).
    """
    import json

    base_id = snapshot_id or time.strftime("%Y%m%d%H%M%S", time.gmtime())
    for snapshot_id in (base_id, f"{base_id}_{uuid.uuid4().hex[:6]}"):
        s3_prefix = f"projects/{PROJECT}/snapshots/{snapshot_id}"
        s3_location = f"s3://{BUCKET_NAME}/{s3_prefix}/snapshot.json"
        item = {
            "project": {"S": PROJECT},
            "snapshotId": {"S": snapshot_id},
            "triggerType": {"S": "extraction"},
            "sourceType": {"S": source_type},
            "s3Location": {"S": s3_location},
            "classCount": {"N": str(class_count)},
            "propertyCount": {"N": str(property_count)},
            "createdAt": {"S": now_iso()},
        }
        if source_identifier:
            item["sourceIdentifier"] = {"S": source_identifier}
        try:
            dynamodb.put_item(
                TableName=SNAPSHOTS_TABLE,
                Item=item,
                ConditionExpression="attribute_not_exists(snapshotId)",
            )
        except dynamodb.exceptions.ConditionalCheckFailedException:
            continue

        snapshot_meta = {
            "snapshot_id": snapshot_id,
            "project": PROJECT,
            "trigger_type": "extraction",
            "source_type": source_type,
            "class_count": class_count,
            "property_count": property_count,
            "created_at": now_iso(),
        }
        meta_path = WORK_DIR / "snapshot.json"
        meta_path.write_text(json.dumps(snapshot_meta, ensure_ascii=False, indent=2))
        upload_to_s3(meta_path, f"{s3_prefix}/snapshot.json")
        upload_to_s3(knowledge_json_path, f"{s3_prefix}/knowledge.json")
        if source_raw_path is not None:
            upload_to_s3(source_raw_path, f"{s3_prefix}/source_raw.json")
        logger.info(f"Created snapshot {snapshot_id}")
        return snapshot_id

    raise RuntimeError("Could not claim a unique snapshot ID")


# ---------------------------------------------------------------------------
# Knowledge graph writes
# ---------------------------------------------------------------------------
# Instance items are keyed by the deterministic content hash the extraction
# pipeline assigns (see src/utils/id_generator.py). The conditional put fails
# for evidence that already exists - from a Batch retry or from knowledge
# carried over into a later run - so aggregate counters are incremented
# exactly once per unique piece of evidence.


def put_instance(entity_key: str, instance_id: str, attributes: dict) -> bool:
    """Write one instance item; returns False if it already existed."""
    item = {
        "entityKey": {"S": entity_key},
        "instanceKey": {"S": instance_id},
        "instanceId": {"S": instance_id},
        **attributes,
    }
    try:
        dynamodb.put_item(
            TableName=INSTANCES_TABLE,
            Item=item,
            ConditionExpression="attribute_not_exists(entityKey)",
        )
        return True
    except dynamodb.exceptions.ConditionalCheckFailedException:
        return False


def write_class(cls: dict, source_document: str, snapshot_id: str, extracted_at: str) -> bool:
    """Upsert one class aggregate and its instance; returns True if new evidence."""
    label = cls.get("label", "")
    if not label:
        return False
    normalized = normalize_class_name(label)
    if not normalized:
        return False

    sources = cls.get("sources") or []
    instance_attributes = {
        "label": {"S": label},
        "source": {"S": (sources[0] if sources else "")[:INSTANCE_SOURCE_MAX_CHARS]},
        "extractedAt": {"S": extracted_at},
        "snapshotId": {"S": snapshot_id},
        "sourceDocument": {"S": source_document},
    }
    is_new = put_instance(f"{PROJECT}#CLASS#{normalized}", cls["id"], instance_attributes)

    justification = cls.get("justification") or ""
    # "definition" is a DynamoDB reserved word, hence the #definition alias
    expression = (
        "SET displayName = :label, classIri = :iri, classType = :type, "
        "updatedAt = :now, createdAt = if_not_exists(createdAt, :now), "
        + ("#definition = :def" if justification else "#definition = if_not_exists(#definition, :def)")
        + (", latestExtractedAt = :now" if is_new else "")
        + " ADD sourceDocuments :doc"
        + (", instanceCount :one" if is_new else "")
    )
    values = {
        ":label": {"S": label},
        ":iri": {"S": cls.get("class_iri", "")},
        ":type": {"S": cls.get("judgment", "PENDING")},
        ":def": {"S": justification},
        ":now": {"S": extracted_at},
        ":doc": {"SS": [source_document]},
    }
    if is_new:
        values[":one"] = {"N": "1"}
    dynamodb.update_item(
        TableName=CLASSES_TABLE,
        Key={"project": {"S": PROJECT}, "className": {"S": normalized}},
        UpdateExpression=expression,
        ExpressionAttributeNames={"#definition": "definition"},
        ExpressionAttributeValues=values,
    )
    return is_new


def write_property(prop: dict, source_document: str, snapshot_id: str, extracted_at: str) -> bool:
    """Upsert one property aggregate and its instance; returns True if new evidence."""
    src_label = prop.get("src_label", "")
    dest_label = prop.get("dest_label", "")
    property_iri = prop.get("property_iri", "")
    if not src_label or not dest_label or not property_iri:
        return False

    src_norm = normalize_class_name(src_label)
    dest_norm = normalize_class_name(dest_label)
    property_name = property_name_from_iri(property_iri)
    strength = prop.get("strength", 0)

    entity_key = f"{PROJECT}#PROP#{src_norm}#{property_name}#{dest_norm}"
    instance_attributes = {
        "strength": {"N": str(strength)},
        "extractedAt": {"S": extracted_at},
        "snapshotId": {"S": snapshot_id},
        "sourceDocument": {"S": source_document},
    }
    is_new = put_instance(entity_key, prop["id"], instance_attributes)

    # "project" and "definition" are DynamoDB reserved words, hence the aliases
    expression = (
        "SET #project = :project, sourceClassName = :src, targetClassName = :dest, "
        "propertyName = :name, propertyIri = :iri, projectTargetClass = :ptc, "
        "updatedAt = :now, createdAt = if_not_exists(createdAt, :now), "
        "#definition = if_not_exists(#definition, :empty) "
        "ADD sourceDocuments :doc" + (", instanceCount :one, strengthSum :strength" if is_new else "")
    )
    values = {
        ":project": {"S": PROJECT},
        ":src": {"S": src_label},
        ":dest": {"S": dest_label},
        ":name": {"S": property_name},
        ":iri": {"S": property_iri},
        ":ptc": {"S": f"{PROJECT}#{dest_norm}"},
        ":now": {"S": extracted_at},
        ":empty": {"S": ""},
        ":doc": {"SS": [source_document]},
    }
    if is_new:
        values[":one"] = {"N": "1"}
        values[":strength"] = {"N": str(strength)}

    key = {
        "projectSourceClass": {"S": f"{PROJECT}#{src_norm}"},
        "relationship": {"S": f"{property_name}#{dest_norm}"},
    }
    response = dynamodb.update_item(
        TableName=PROPERTIES_TABLE,
        Key=key,
        UpdateExpression=expression,
        ExpressionAttributeNames={"#project": "project", "#definition": "definition"},
        ExpressionAttributeValues=values,
        ReturnValues="ALL_NEW",
    )

    if is_new:
        # Recompute the stored average; safe because the project lock
        # guarantees a single writer per project.
        attributes = response.get("Attributes", {})
        count = int(attributes.get("instanceCount", {}).get("N", "1"))
        strength_sum = float(attributes.get("strengthSum", {}).get("N", str(strength)))
        dynamodb.update_item(
            TableName=PROPERTIES_TABLE,
            Key=key,
            UpdateExpression="SET averageStrength = :avg",
            ExpressionAttributeValues={":avg": {"N": str(round(strength_sum / max(count, 1), 4))}},
        )
    return is_new


def write_knowledge_graph(result: dict, source_document: str, snapshot_id: str) -> tuple:
    """Write extracted classes and properties to DynamoDB.

    All classes are written with their judgment (the API filters by
    classType). Properties are written only when accepted with a positive
    strength, matching the "valid properties" semantics of knowledge.json.

    Returns (new_class_instances, new_property_instances).
    """
    extracted_at = now_iso()

    new_classes = 0
    for cls in result.get("classes", []):
        try:
            if write_class(cls, source_document, snapshot_id, extracted_at):
                new_classes += 1
        except Exception as e:
            logger.error(f"Failed to write class '{cls.get('label', '?')}': {e}")

    new_properties = 0
    for prop in result.get("properties", []):
        if prop.get("judgment") != "ACCEPT" or prop.get("strength", 0) <= 0:
            continue
        try:
            if write_property(prop, source_document, snapshot_id, extracted_at):
                new_properties += 1
        except Exception as e:
            logger.error(f"Failed to write property '{prop.get('id', '?')}': {e}")

    logger.info(f"Knowledge graph written: {new_classes} new class instances, {new_properties} new property instances")
    return new_classes, new_properties


# ---------------------------------------------------------------------------
# Projects table
# ---------------------------------------------------------------------------


def update_project_records(
    job_id: str,
    snapshot_id: str,
    source_type: str,
    source_document: str,
    classes_extracted: int,
    properties_extracted: int,
    duration_seconds: int,
) -> None:
    """Update project METADATA and append the extraction history record."""
    now = now_iso()

    # entityType makes the METADATA record visible in entityType-index
    dynamodb.update_item(
        TableName=PROJECTS_TABLE,
        Key={"project": {"S": PROJECT}, "recordType": {"S": "METADATA"}},
        UpdateExpression=(
            "SET entityType = :entityType, projectName = if_not_exists(projectName, :name), "
            "createdAt = if_not_exists(createdAt, :now), lastExtractionAt = :now, "
            "currentSnapshotId = :snapshot "
            "ADD totalExtractions :one, totalSnapshots :one, sources :sourceType"
        ),
        ExpressionAttributeValues={
            ":entityType": {"S": "PROJECT"},
            ":name": {"S": PROJECT},
            ":now": {"S": now},
            ":snapshot": {"S": snapshot_id},
            ":one": {"N": "1"},
            ":sourceType": {"SS": [source_type]},
        },
    )

    dynamodb.put_item(
        TableName=PROJECTS_TABLE,
        Item={
            "project": {"S": PROJECT},
            "recordType": {"S": f"EXTRACTION#{now}"},
            "jobId": {"S": job_id},
            "extractedAt": {"S": now},
            "snapshotId": {"S": snapshot_id},
            "sourceType": {"S": source_type},
            "sourceFile": {"S": source_document},
            "classesExtracted": {"N": str(classes_extracted)},
            "propertiesExtracted": {"N": str(properties_extracted)},
            "durationSeconds": {"N": str(duration_seconds)},
            "status": {"S": "COMPLETED"},
        },
    )


# ---------------------------------------------------------------------------
# Changesets
# ---------------------------------------------------------------------------
# A changeset records what one extraction changed and why. It is generated at
# the end of the extraction, while the source text is still at hand: the
# reason for a change is written in the text and is not kept in the snapshots.


def record_changeset(
    before_snapshot_id: str,
    after_snapshot_id: str,
    knowledge_before: Path,
    knowledge_after: Path,
    result: dict,
) -> str:
    """Generate the changeset for the extraction that just ran and upload it.

    Args:
        before_snapshot_id: Snapshot before the extraction
        after_snapshot_id: Snapshot the extraction produced
        knowledge_before: knowledge.json as it was before the extraction
        knowledge_after: knowledge.json the extraction produced
        result: Extraction result, read for the source sentences

    Returns:
        The changeset ID
    """
    import json
    import shutil

    changeset_id = f"cs_{after_snapshot_id}"
    output_dir = WORK_DIR / "output"

    # generate_changeset reads both sides from the local snapshot layout
    for snapshot_id, knowledge in ((before_snapshot_id, knowledge_before), (after_snapshot_id, knowledge_after)):
        snapshot_dir = output_dir / PROJECT / "snapshots" / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(knowledge, snapshot_dir / "knowledge.json")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils.changeset_generator import generate_changeset
    from src.utils.changeset_inputs import change_source_sentences

    generate_changeset(
        before_snapshot_id=before_snapshot_id,
        after_snapshot_id=after_snapshot_id,
        project=PROJECT,
        output_dir=str(output_dir),
        changeset_id=changeset_id,
        source_sentences=change_source_sentences(result),
        file_id=result.get("file_id", "unknown"),
    )

    changeset_dir = output_dir / PROJECT / "changesets" / changeset_id
    s3_prefix = f"projects/{PROJECT}/changesets/{changeset_id}"
    for output_file in changeset_dir.glob("*"):
        if output_file.is_file():
            upload_to_s3(output_file, f"{s3_prefix}/{output_file.name}")

    summary = json.loads((changeset_dir / "changeset.json").read_text()).get("summary", {})
    dynamodb.put_item(
        TableName=CHANGESETS_TABLE,
        Item={
            "project": {"S": PROJECT},
            "changesetId": {"S": changeset_id},
            "beforeSnapshotId": {"S": before_snapshot_id},
            "afterSnapshotId": {"S": after_snapshot_id},
            "status": {"S": "COMPLETED"},
            "s3Location": {"S": f"s3://{BUCKET_NAME}/{s3_prefix}/changeset.json"},
            "totalChanges": {"N": str(summary.get("total_changes", 0))},
            "driversCount": {"N": str(summary.get("drivers", 0))},
            "createdAt": {"S": now_iso()},
        },
    )
    logger.info(f"Created changeset {changeset_id}")
    return changeset_id


def latest_snapshot_id() -> str:
    """Return the project's most recent snapshot ID, or an empty string."""
    response = dynamodb.query(
        TableName=SNAPSHOTS_TABLE,
        KeyConditionExpression="#project = :project",
        ExpressionAttributeNames={"#project": "project"},
        ExpressionAttributeValues={":project": {"S": PROJECT}},
        Limit=1,
        ScanIndexForward=False,
    )
    items = response.get("Items", [])
    return items[0]["snapshotId"]["S"] if items else ""


def miro_source_identifier(board_url: str) -> str:
    """Source identifier of a Miro board, in the form snapshot_manager records locally."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.extract.miro.miro_text_extractor import parse_board_id_from_url

    return f"miro:{parse_board_id_from_url(board_url)}"


def stage_previous_miro_snapshot(source_identifier: str) -> str:
    """Download the board's latest source_raw.json to where the workflow diffs against it.

    Returns the snapshot ID, or an empty string when there is nothing to diff
    against and the board is extracted in full.
    """
    pages = dynamodb.get_paginator("query").paginate(
        TableName=SNAPSHOTS_TABLE,
        KeyConditionExpression="#project = :project",
        # Applied after each page is read, so keep paging until a match
        FilterExpression="sourceIdentifier = :source",
        ExpressionAttributeNames={"#project": "project"},
        ExpressionAttributeValues={":project": {"S": PROJECT}, ":source": {"S": source_identifier}},
        ScanIndexForward=False,
    )
    for page in pages:
        for item in page.get("Items", []):
            snapshot_id = item["snapshotId"]["S"]
            # The layout load_source_raw reads (src/utils/source_change_detector.py)
            local_path = WORK_DIR / "output" / PROJECT / "snapshots" / snapshot_id / "source_raw.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                download_from_s3(
                    f"s3://{BUCKET_NAME}/projects/{PROJECT}/snapshots/{snapshot_id}/source_raw.json", local_path
                )
            except Exception as e:
                logger.warning(f"Could not download source_raw.json of snapshot {snapshot_id}: {e}")
                return ""
            return snapshot_id

    logger.info(f"No earlier snapshot of {source_identifier}; extracting in full")
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# Credentials a source type cannot run without, injected from Secrets Manager
# by the Batch job definition (infra/lib/stacks/compute-stack.ts).
SOURCE_CREDENTIALS = {
    "miro": ("MIRO_API_KEY", "miro-api-key-<environment>"),
}

# Secrets are created with a placeholder so task startup never fails on a
# missing secret; must match the placeholder in compute-stack.ts.
SECRET_PLACEHOLDER_PREFIX = "PLACEHOLDER"


def preflight_source_config():
    """Verify the credential the source type needs is present and real.

    Without it the job fails at the first API call, minutes into a paid run.
    Exit code 2 marks a configuration error (not retryable).
    """
    source_type = SOURCE_TYPE or infer_source_type(SOURCE)
    required = SOURCE_CREDENTIALS.get(source_type)
    if not required:
        return

    env_name, secret_name = required
    value = (os.environ.get(env_name) or "").strip()
    if not value:
        logger.error(
            f"{env_name} must be set for {source_type} sources. "
            f"Set the Secrets Manager secret '{secret_name}' with: "
            f"aws secretsmanager put-secret-value --secret-id {secret_name} "
            "--secret-string YOUR_TOKEN"
        )
        sys.exit(2)
    if value.startswith(SECRET_PLACEHOLDER_PREFIX):
        logger.error(
            f"{env_name} still holds the placeholder value created at deploy time. "
            f"Set the real token in the Secrets Manager secret '{secret_name}'."
        )
        sys.exit(2)


def preflight_as_of():
    """Verify AS_OF is a real YYYYMMDD date. Exit code 2: not retryable."""
    if not AS_OF:
        return
    try:
        datetime.strptime(AS_OF, "%Y%m%d")
    except ValueError:
        logger.error(f"AS_OF must be a YYYYMMDD date, got: {AS_OF}")
        sys.exit(2)


def preflight_llm_config():
    """Preflight LLM configuration checks.

    Without these the job fails at the first LLM call, minutes into a paid
    Fargate run (for extraction jobs, with the project lock already held).
    Exit code 2 marks a configuration error (not retryable).
    """
    if not os.environ.get("LLM_PROVIDER"):
        logger.error("LLM_PROVIDER must be set (the AWS job definition sets gcp_vertexai)")
        sys.exit(2)
    if os.environ.get("LLM_PROVIDER") == "gcp_vertexai":
        if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
            logger.error("GOOGLE_CLOUD_PROJECT must be set when using Vertex AI")
            sys.exit(2)
        # Wire up Workload Identity Federation before the first LLM call;
        # google-auth cannot reach the Fargate credentials endpoint on its own.
        from src.utils.gcp_wif import setup_workload_identity

        try:
            setup_workload_identity()
        except Exception as e:
            logger.error(f"Failed to configure GCP federated credentials: {e}")
            sys.exit(2)


def main():
    """Main extraction workflow."""
    import json

    started = time.time()

    missing = [
        name
        for name, value in {
            "PROJECT": PROJECT,
            "SOURCE": SOURCE,
            "BUCKET_NAME": BUCKET_NAME,
            "JOBS_TABLE": JOBS_TABLE,
        }.items()
        if not value
    ]
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(2)

    # Cheaper than the LLM check (no WIF token exchange)
    preflight_source_config()
    preflight_as_of()
    preflight_llm_config()

    job_id = resolve_job_id()
    lock_acquired = False

    try:
        logger.info(f"Starting extraction job: {job_id}")
        logger.info(f"Project: {PROJECT}")
        logger.info(f"Source: {SOURCE}")

        update_job(
            job_id,
            "SET #status = :status, startedAt = :time",
            {"#status": "status"},
            {":status": {"S": "RUNNING"}, ":time": {"S": now_iso()}},
        )

        # Serialize writers per project before touching knowledge.json
        update_progress(job_id, 3, "Waiting for project lock")
        lock_acquired = acquire_project_lock(job_id)
        if not lock_acquired:
            raise RuntimeError(f"Timed out waiting for project lock ({LOCK_MAX_WAIT_SECONDS}s)")

        # Prepare working directories
        output_dir = WORK_DIR / "output" / PROJECT
        for directory in (WORK_DIR, INPUT_DIR, output_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # Stage 1: Resolve input (5-10%)
        update_progress(job_id, 5, "Preparing input")
        source_document = source_document_name(SOURCE)
        source_type = SOURCE_TYPE or infer_source_type(SOURCE)

        if SOURCE.startswith("s3://"):
            extraction_source = str(INPUT_DIR / source_document)
            download_from_s3(SOURCE, Path(extraction_source))
        else:
            # URLs (Miro, Confluence, GitHub, ...) go straight to the workflow
            extraction_source = SOURCE

        update_progress(job_id, 10, "Downloading knowledge.json")

        # The workflow reads accumulated knowledge from the project output
        # directory, so place knowledge.json there.
        knowledge_json_path = output_dir / "knowledge.json"
        knowledge_s3_key = f"projects/{PROJECT}/knowledge.json"
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=knowledge_s3_key)
            download_from_s3(f"s3://{BUCKET_NAME}/{knowledge_s3_key}", knowledge_json_path)
            logger.info("Downloaded existing knowledge.json")
        except Exception:
            logger.info("No existing knowledge.json found, starting fresh")
            knowledge_json_path.write_text('{"classes": [], "properties": []}')

        # The workflow reads the project profile from the same directory; a project
        # without one is extracted without relevance screening.
        profile_path = output_dir / PROFILE_FILENAME
        profile_s3_key = f"projects/{PROJECT}/{PROFILE_FILENAME}"
        try:
            s3.head_object(Bucket=BUCKET_NAME, Key=profile_s3_key)
            download_from_s3(f"s3://{BUCKET_NAME}/{profile_s3_key}", profile_path)
            logger.info("Downloaded project profile")
        except Exception:
            logger.info("No project profile found, extracting without relevance screening")

        # Keep the state this extraction starts from, so the changeset below
        # can diff against it once the extraction has overwritten it.
        previous_snapshot_id = latest_snapshot_id()
        knowledge_before_path = WORK_DIR / "knowledge_before.json"
        shutil.copy2(knowledge_json_path, knowledge_before_path)

        # Snapshots must stay in time order; check before any LLM call
        as_of_snapshot_id = f"{AS_OF}235959" if AS_OF else None
        if as_of_snapshot_id and previous_snapshot_id and as_of_snapshot_id <= previous_snapshot_id:
            message = (
                f"asOf {AS_OF} must be later than the latest snapshot {previous_snapshot_id}; "
                "extract earlier dates first"
            )
            logger.error(message)
            update_job(
                job_id,
                "SET #status = :status, errorMessage = :error",
                {"#status": "status"},
                {":status": {"S": "FAILED"}, ":error": {"S": message}},
            )
            # Not retryable; the finally block releases the lock
            sys.exit(2)

        source_identifier = miro_source_identifier(SOURCE) if source_type == "miro" else None
        source_previous_snapshot_id = stage_previous_miro_snapshot(source_identifier) if source_identifier else ""

        # Stage 2-4: Run extraction workflow (15-85%)
        update_progress(job_id, 15, "Starting knowledge extraction")

        sys.path.insert(0, str(Path(__file__).parent))
        from extract_workflow import run_knowledge_extraction

        def progress_callback(progress: int, message: str):
            update_progress(job_id, progress, message)

        # Local snapshotting is disabled: snapshots are written to S3 and
        # DynamoDB below, not to the container's ephemeral disk.
        result = run_knowledge_extraction(
            source=extraction_source,
            project=PROJECT,
            output_dir=str(WORK_DIR / "output"),
            knowledge_json_path=str(knowledge_json_path),
            create_snapshot=False,
            progress_callback=progress_callback,
            as_of=AS_OF,
            previous_snapshot_id=source_previous_snapshot_id or None,
        )

        if not result.get("summary") or result.get("errors"):
            errors = "; ".join(result.get("errors", [])) or "unknown error"
            raise RuntimeError(f"Extraction failed: {errors}")

        # A run that extracted nothing is a failure.
        if result["summary"]["classes_count"] == 0:
            errors = "; ".join(result.get("errors", [])) or "no errors reported"
            raise RuntimeError(f"Extraction produced no classes: {errors}")

        logger.info(f"Extraction completed: {result['summary']}")

        # Stage 5: Upload results (85-90%)
        update_progress(job_id, 85, "Uploading results to S3")

        output_prefix = f"projects/{PROJECT}/outputs"
        # knowledge.json goes to its own key below; the profile is an input, not an output
        for output_file in output_dir.glob("*"):
            if output_file.is_file() and output_file.name not in ("knowledge.json", PROFILE_FILENAME):
                upload_to_s3(output_file, f"{output_prefix}/{output_file.name}")

        if knowledge_json_path.exists():
            upload_to_s3(knowledge_json_path, knowledge_s3_key)

        # Stage 6: Create snapshot (90-93%)
        update_progress(job_id, 90, "Creating snapshot")

        # Persist raw source data next to the snapshot so changeset driver
        # extraction can diff source items between snapshots.
        raw_source_data = (
            result.get("miro_raw_data")
            or result.get("jira_raw_data")
            or result.get("github_raw_data")
            or result.get("confluence_raw_data")
        )
        source_raw_path = None
        if raw_source_data:
            source_raw_path = WORK_DIR / "source_raw.json"
            source_raw_path.write_text(json.dumps(raw_source_data, ensure_ascii=False))

        knowledge = json.loads(knowledge_json_path.read_text())
        snapshot_id = create_snapshot(
            knowledge_json_path,
            source_type,
            class_count=len(knowledge.get("classes", [])),
            property_count=len(knowledge.get("properties", [])),
            source_raw_path=source_raw_path,
            snapshot_id=as_of_snapshot_id,
            source_identifier=source_identifier,
        )

        # Record what this extraction changed and why. Best effort: the
        # extraction itself has already succeeded at this point.
        if previous_snapshot_id:
            update_progress(job_id, 92, "Recording changeset")
            try:
                record_changeset(
                    previous_snapshot_id,
                    snapshot_id,
                    knowledge_before_path,
                    knowledge_json_path,
                    result,
                )
            except Exception as e:
                logger.warning(f"Failed to record changeset: {e}")

        # Stage 7: Write knowledge graph to DynamoDB (93-97%)
        update_progress(job_id, 93, "Writing knowledge graph to DynamoDB")
        new_classes, new_properties = write_knowledge_graph(result, source_document, snapshot_id)

        # Stage 8: Project metadata + history (97-99%)
        update_progress(job_id, 97, "Updating project records")
        update_project_records(
            job_id,
            snapshot_id,
            source_type,
            source_document,
            classes_extracted=new_classes,
            properties_extracted=new_properties,
            duration_seconds=int(time.time() - started),
        )

        # Stage 9: Final status (100%)
        update_progress(job_id, 100, "Extraction completed successfully")
        update_job(
            job_id,
            "SET #status = :status, completedAt = :time, outputLocation = :output",
            {"#status": "status"},
            {
                ":status": {"S": "COMPLETED"},
                ":time": {"S": now_iso()},
                ":output": {"S": f"s3://{BUCKET_NAME}/projects/{PROJECT}"},
            },
        )

        logger.info("Job completed successfully")

    except Exception as e:
        logger.error(f"Job failed: {str(e)}", exc_info=True)
        try:
            update_job(
                job_id,
                "SET #status = :status, errorMessage = :error",
                {"#status": "status"},
                {":status": {"S": "FAILED"}, ":error": {"S": str(e)}},
            )
        except Exception:
            pass
        sys.exit(1)

    finally:
        if lock_acquired:
            release_project_lock(job_id)


if __name__ == "__main__":
    main()
