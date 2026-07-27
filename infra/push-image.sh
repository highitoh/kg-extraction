#!/bin/bash

# ECR Image Push Wrapper Script
# Builds the extraction workflow image and pushes it to the ECR repository
# created by ComputeStack. The repository URI comes from the deployed stack
# outputs, so the account and region always match what CDK deployed.
#
# Usage:
#   ./push-image.sh [OPTIONS]
# Options: [--environment ENV] [--tag TAG] [--skip-build]

cd "$(dirname "$0")"

# Default values
ENVIRONMENT=""
TAG="latest"
SKIP_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --environment)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    *)
      echo "Error: Unknown option: $1"
      echo "Usage: ./push-image.sh [--environment ENV] [--tag TAG] [--skip-build]"
      exit 1
      ;;
  esac
done

# Default to the environment configured for deployment
if [ -z "$ENVIRONMENT" ]; then
  ENVIRONMENT=$(node -p "require('./cdk.local.json').environment || 'dev'" 2>/dev/null || echo "dev")
fi

STACK_NAME="KnowledgeExtractionCompute-${ENVIRONMENT}"

# Resolve the repository from the deployed stack
echo "Resolving ECR repository from ${STACK_NAME}..."
ECR_URI=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --query 'Stacks[0].Outputs[?OutputKey==`EcrRepositoryUri`].OutputValue' \
  --output text 2>/dev/null)

if [ -z "${ECR_URI}" ] || [ "${ECR_URI}" = "None" ]; then
  echo "Error: Could not read EcrRepositoryUri from ${STACK_NAME}"
  echo ""
  echo "The compute stack must be deployed before pushing an image:"
  echo "  cdk deploy ${STACK_NAME}"
  exit 1
fi

# Split the URI into its registry host and that host's region
REGISTRY="${ECR_URI%%/*}"
REGION=$(echo "${REGISTRY}" | cut -d. -f4)

echo "Repository: ${ECR_URI}"
echo "Region:     ${REGION}"
echo "Tag:        ${TAG}"

# Build image (context is the workspace root; the Dockerfile copies app/)
if [ "${SKIP_BUILD}" = false ]; then
  echo "Building image..."
  if ! docker build -t kg-extraction:latest -f docker/Dockerfile ..; then
    echo "Error: Docker build failed"
    exit 1
  fi
fi

# Authenticate against the registry
echo "Logging in to ${REGISTRY}..."
if ! aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"; then
  echo "Error: ECR login failed"
  exit 1
fi

# Tag and push
echo "Pushing ${ECR_URI}:${TAG}..."
if ! docker tag kg-extraction:latest "${ECR_URI}:${TAG}"; then
  echo "Error: Failed to tag image (build it first, or omit --skip-build)"
  exit 1
fi

if ! docker push "${ECR_URI}:${TAG}"; then
  echo "Error: Docker push failed"
  exit 1
fi

echo "Push completed successfully!"
