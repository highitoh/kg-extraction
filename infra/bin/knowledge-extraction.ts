#!/usr/bin/env node
import 'source-map-support/register';
import * as fs from 'fs';
import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { StorageStack } from '../lib/stacks/storage-stack';
import { ComputeStack } from '../lib/stacks/compute-stack';
import { ApiStack } from '../lib/stacks/api-stack';
import { EventsStack } from '../lib/stacks/events-stack';

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
const gcpLocation = getConfig('gcpLocation') || 'global';

// Resource names carry no region component, so deploying to an unintended
// region silently creates a parallel copy of the whole system instead of
// failing. Require an explicit region rather than falling back to a default.
const account = process.env.CDK_DEFAULT_ACCOUNT;
const region = getConfig('awsRegion') ?? process.env.CDK_DEFAULT_REGION;

if (!account || !region) {
  throw new Error(
    'Could not resolve the target AWS account/region. Configure the AWS CLI ' +
      '(aws configure), or set "awsRegion" in cdk.local.json.',
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

// Stack IDs and resource names carry the environment suffix so that
// dev/staging/prod can coexist in one AWS account and region.

// Stack 1: Storage (S3 + DynamoDB)
const storageStack = new StorageStack(app, `KnowledgeExtractionStorage-${environment}`, {
  env,
  orgName,
  environment,
  description: 'Storage layer for knowledge extraction system (S3 + DynamoDB)',
});

// Stack 2: Compute (AWS Batch)
const computeStack = new ComputeStack(app, `KnowledgeExtractionCompute-${environment}`, {
  env,
  environment,
  gcpProjectId,
  gcpLocation,
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
computeStack.addDependency(storageStack);

// Stack 3: API (API Gateway + Lambda + MCP Server)
const apiStack = new ApiStack(app, `KnowledgeExtractionApi-${environment}`, {
  env,
  environment,
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
apiStack.addDependency(computeStack);

// Stack 4: Events (EventBridge scheduled jobs)
const eventsStack = new EventsStack(app, `KnowledgeExtractionEvents-${environment}`, {
  env,
  jobQueue: computeStack.jobQueue,
  jobDefinition: computeStack.jobDefinition,
  description: 'Event-driven triggers for knowledge extraction system (EventBridge)',
});
eventsStack.addDependency(computeStack);

app.synth();
