"""Bridge AWS task credentials to GCP Workload Identity Federation.

google-auth resolves AWS credentials from environment variables or the EC2
instance metadata server. Fargate tasks expose neither: their credentials come
from the ECS container credentials endpoint, which google-auth does not query.
boto3 does understand that endpoint, so this module fetches the task role
credentials through boto3, exports them as the environment variables that
google-auth reads first, and writes the external-account credential
configuration that Application Default Credentials loads.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

CREDENTIAL_FILE = "/tmp/gcp-wif-credentials.json"

# Mirrors the Batch job timeout in infra/lib/stacks/compute-stack.ts. The
# credentials are exported once as a snapshot, so a lifetime shorter than the
# job means GCP token refreshes can start failing partway through the run.
JOB_LIFETIME_SECONDS = 2 * 60 * 60


def setup_workload_identity() -> None:
    """Point Application Default Credentials at the WIF provider.

    No-op when GOOGLE_APPLICATION_CREDENTIALS is already set (an explicit
    credential file wins) or when GOOGLE_WIF_AUDIENCE is absent (nothing to
    federate against, e.g. local runs with user credentials).
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    audience = os.environ.get("GOOGLE_WIF_AUDIENCE")
    if not audience:
        return

    _export_aws_credentials()

    # gcloud iam workload-identity-pools create-cred-config also emits EC2
    # metadata URLs for discovering the region and credentials. They are
    # omitted here: _export_aws_credentials always supplies both through the
    # environment, which google-auth reads first, and the EC2 metadata server
    # is unreachable from Fargate anyway. environment_id and
    # regional_cred_verification_url are validated on load, so both must stay.
    config = {
        "type": "external_account",
        "audience": audience,
        "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
        "token_url": "https://sts.googleapis.com/v1/token",
        "credential_source": {
            "environment_id": "aws1",
            "regional_cred_verification_url": (
                "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
            ),
        },
    }
    with open(CREDENTIAL_FILE, "w") as f:
        json.dump(config, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIAL_FILE
    logger.info("Configured GCP Workload Identity Federation credentials")


def _export_aws_credentials() -> None:
    """Expose the task role credentials where google-auth can find them.

    Freezing takes the three values under one lock: reading them one by one
    can straddle a refresh and pair an access key with the next key's secret.
    The snapshot is not refreshed afterwards, so its lifetime is checked
    against the job's.
    """
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return

    import boto3

    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials available to federate to GCP")
    frozen = credentials.get_frozen_credentials()
    _warn_if_expiring_before_job_ends(credentials)

    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
        if not session.region_name:
            raise RuntimeError("AWS region could not be determined for GCP federation")
        os.environ["AWS_REGION"] = session.region_name


def _warn_if_expiring_before_job_ends(credentials) -> None:
    """Flag a snapshot that may not last the whole job.

    Long extractions would otherwise fail on a GCP token refresh with an
    expired-credentials error, far from its cause. Static credentials carry no
    expiry and expose no refresh_needed, so they are skipped.
    """
    if not hasattr(credentials, "refresh_needed"):
        return
    if credentials.refresh_needed(refresh_in=JOB_LIFETIME_SECONDS):
        logger.warning(
            "AWS credentials expire within %d seconds, which is less than the job "
            "timeout. GCP access may fail mid-run because the federated credentials "
            "are a snapshot taken at startup.",
            JOB_LIFETIME_SECONDS,
        )
