#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { StorageStack } from '../lib/stacks/storage-stack';
import { ComputeStack } from '../lib/stacks/compute-stack';
import { ApiStack } from '../lib/stacks/api-stack';
import { EventsStack } from '../lib/stacks/events-stack';

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION || 'us-east-1',
};

// Get configuration from context
const accountName = app.node.tryGetContext('accountName');
const environment = app.node.tryGetContext('environment') || 'dev';
const gcpProjectId = app.node.tryGetContext('gcpProjectId');
const gcpLocation = app.node.tryGetContext('gcpLocation') || 'global';

// accountName prefixes the globally-unique S3 bucket name, so a placeholder
// default would either collide or create a wrongly-named bucket; require it.
if (!accountName) {
  throw new Error('accountName must be provided via CDK context (-c accountName=YOUR_ORG)');
}

if (!gcpProjectId) {
  throw new Error('gcpProjectId must be provided via CDK context (-c gcpProjectId=YOUR_PROJECT_ID)');
}

// Stack IDs and resource names carry the environment suffix so that
// dev/staging/prod can coexist in one AWS account and region.

// Stack 1: Storage (S3 + DynamoDB)
const storageStack = new StorageStack(app, `KnowledgeExtractionStorage-${environment}`, {
  env,
  accountName,
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
