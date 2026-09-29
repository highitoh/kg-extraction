#!/usr/bin/env node
import 'source-map-support/register';
import * as fs from 'fs';
import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { StorageStack } from '../lib/stacks/storage-stack';
import { ComputeStack } from '../lib/stacks/compute-stack';
import { ApiStack } from '../lib/stacks/api-stack';
import { EventsStack } from '../lib/stacks/events-stack';
import { readAllowedIps } from '../lib/allowed-ips';

const app = new cdk.App();

// Per-developer settings live in cdk.local.json (gitignored; see
// cdk.local.example.json). CLI context (-c key=value) takes precedence.
const localConfigPath = path.join(__dirname, '..', 'cdk.local.json');
const localConfig: Record<string, string> = fs.existsSync(localConfigPath)
  ? JSON.parse(fs.readFileSync(localConfigPath, 'utf-8'))
  : {};

const getConfig = (key: string): string | undefined =>
  app.node.tryGetContext(key) ?? localConfig[key];

const orgName = getConfig('orgName');
const environment = getConfig('environment') || 'dev';
const gcpProjectId = getConfig('gcpProjectId');
const gcpProjectNumber = getConfig('gcpProjectNumber');
const gcpLocation = getConfig('gcpLocation') || 'global';
const gcpWifPool = getConfig('gcpWifPool') || 'aws-extraction-pool';
const gcpWifProvider = getConfig('gcpWifProvider') || 'aws-provider';
const llmExtractModel = getConfig('llmExtractModel');
const llmEvalModel = getConfig('llmEvalModel');
const llmReasoningEffort = getConfig('llmReasoningEffort') || 'low';

// Inbound IP allowlist (gitignored; see allowed-ips.example.txt); absent or empty means unrestricted
const allowedIps = readAllowedIps(path.join(__dirname, '..', 'allowed-ips.txt'));

// Resource names carry no region component, so deploying to an unintended
// region silently creates a parallel copy of the whole system instead of
// failing. Require an explicit region rather than falling back to a default.
const account = process.env.CDK_DEFAULT_ACCOUNT;
const region = getConfig('awsRegion') ?? process.env.CDK_DEFAULT_REGION;

// The account is deliberately not configurable: taking it from the credentials
// in use means the deployment target cannot drift from the caller's identity.
if (!account) {
  throw new Error(
    'Could not resolve the target AWS account. The CDK CLI derives it from the ' +
      'AWS credentials in use, so verify them with `aws sts get-caller-identity` ' +
      'and pass --profile if you deploy with a named profile.',
  );
}

if (!region) {
  throw new Error(
    'Could not resolve the target AWS region. Set "awsRegion" in cdk.local.json, ' +
      'or configure a default region for the AWS CLI.',
  );
}

const env = { account, region };

// orgName prefixes the globally-unique S3 bucket name, so a placeholder
// default would either collide or create a wrongly-named bucket; require it.
if (!orgName) {
  throw new Error(
    'orgName must be provided via cdk.local.json or CDK context (-c orgName=YOUR_ORG)',
  );
}

if (!gcpProjectId) {
  throw new Error(
    'gcpProjectId must be provided via cdk.local.json or CDK context (-c gcpProjectId=YOUR_PROJECT_ID)',
  );
}

// The Workload Identity Federation audience embeds the numeric project
// number, which cannot be derived from the project ID at synth time.
if (!gcpProjectNumber) {
  throw new Error(
    'gcpProjectNumber must be provided via cdk.local.json or CDK context. ' +
      "Look it up with: gcloud projects describe PROJECT_ID --format='value(projectNumber)'",
  );
}

// Stack IDs and resource names carry the environment suffix so that
// dev/staging/prod can coexist in one AWS account and region.

// Stack 1: Storage (S3 + DynamoDB)
const storageStack = new StorageStack(app, `KnowledgeExtractionStorage-${environment}`, {
  env,
  orgName,
  environment,
  allowedIps,
  description: 'Storage layer for knowledge extraction system (S3 + DynamoDB)',
});

// Stack 2: Compute (AWS Batch)
const computeStack = new ComputeStack(app, `KnowledgeExtractionCompute-${environment}`, {
  env,
  environment,
  gcpProjectId,
  gcpProjectNumber,
  gcpWifPool,
  gcpWifProvider,
  gcpLocation,
  llmExtractModel,
  llmEvalModel,
  llmReasoningEffort,
  extractionBucket: storageStack.extractionBucket,
  jobsTable: storageStack.jobsTable,
  classesTable: storageStack.classesTable,
  propertiesTable: storageStack.propertiesTable,
  instancesTable: storageStack.instancesTable,
  snapshotsTable: storageStack.snapshotsTable,
  projectsTable: storageStack.projectsTable,
  changesetsTable: storageStack.changesetsTable,
  description: 'Compute layer for knowledge extraction system (AWS Batch + Fargate)',
});
computeStack.addStackDependency(storageStack);

// Stack 3: API (API Gateway + Lambda + MCP Server)
const apiStack = new ApiStack(app, `KnowledgeExtractionApi-${environment}`, {
  env,
  environment,
  allowedIps,
  extractionBucket: storageStack.extractionBucket,
  jobsTable: storageStack.jobsTable,
  snapshotsTable: storageStack.snapshotsTable,
  changesetsTable: storageStack.changesetsTable,
  sessionsTable: storageStack.sessionsTable,
  classesTable: storageStack.classesTable,
  propertiesTable: storageStack.propertiesTable,
  instancesTable: storageStack.instancesTable,
  projectsTable: storageStack.projectsTable,
  jobQueue: computeStack.jobQueue,
  jobDefinition: computeStack.jobDefinition,
  description: 'API layer for knowledge extraction system (API Gateway + Lambda + MCP)',
});
apiStack.addStackDependency(computeStack);

// Stack 4: Events (EventBridge scheduled jobs)
const eventsStack = new EventsStack(app, `KnowledgeExtractionEvents-${environment}`, {
  env,
  jobQueue: computeStack.jobQueue,
  jobDefinition: computeStack.jobDefinition,
  description: 'Event-driven triggers for knowledge extraction system (EventBridge)',
});
eventsStack.addStackDependency(computeStack);

app.synth();
