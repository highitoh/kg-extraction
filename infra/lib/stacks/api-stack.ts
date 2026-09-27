import * as cdk from 'aws-cdk-lib';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as batch from 'aws-cdk-lib/aws-batch';
import { Construct } from 'constructs';

interface ApiStackProps extends cdk.StackProps {
  environment: string;
  extractionBucket: s3.Bucket;
  jobsTable: dynamodb.Table;
  snapshotsTable: dynamodb.Table;
  changesetsTable: dynamodb.Table;
  sessionsTable: dynamodb.Table;
  classesTable: dynamodb.Table;
  propertiesTable: dynamodb.Table;
  instancesTable: dynamodb.Table;
  projectsTable: dynamodb.Table;
  jobQueue: batch.JobQueue;
  jobDefinition: batch.EcsJobDefinition;
}

export class ApiStack extends cdk.Stack {
  public readonly api: apigateway.RestApi;

  constructor(scope: Construct, id: string, props: ApiStackProps) {
    super(scope, id, props);

    // Common Lambda environment variables
    const commonEnv = {
      BUCKET_NAME: props.extractionBucket.bucketName,
      JOBS_TABLE: props.jobsTable.tableName,
      SNAPSHOTS_TABLE: props.snapshotsTable.tableName,
      CHANGESETS_TABLE: props.changesetsTable.tableName,
      SESSIONS_TABLE: props.sessionsTable.tableName,
      CLASSES_TABLE: props.classesTable.tableName,
      PROPERTIES_TABLE: props.propertiesTable.tableName,
      INSTANCES_TABLE: props.instancesTable.tableName,
      PROJECTS_TABLE: props.projectsTable.tableName,
      JOB_QUEUE: props.jobQueue.jobQueueName,
      JOB_DEFINITION: props.jobDefinition.jobDefinitionName,
    };

    // Lambda code directory
    const lambdaCode = lambda.Code.fromAsset('../lambda');

    // SubmitJob is called with the job definition name so that Batch picks the
    // active revision. IAM then authorises against the unversioned ARN, which a
    // revision-qualified resource never matches, so grant both forms.
    const jobDefinitionArnBase = cdk.Stack.of(this).formatArn({
      service: 'batch',
      resource: 'job-definition',
      resourceName: props.jobDefinition.jobDefinitionName,
    });
    const batchSubmitResources = [
      props.jobQueue.jobQueueArn,
      jobDefinitionArnBase,
      `${jobDefinitionArnBase}:*`,
    ];

    // Tables the API writes directly
    const operationalTables = [props.jobsTable, props.snapshotsTable, props.changesetsTable, props.sessionsTable];
    // Tables written by Batch extraction; the API only reads them, except for project deletion
    const knowledgeTables = [props.classesTable, props.propertiesTable, props.instancesTable, props.projectsTable];

    const grantDataAccess = (fn: lambda.Function, { writeKnowledge = false } = {}) => {
      props.extractionBucket.grantReadWrite(fn);
      operationalTables.forEach((t) => t.grantReadWriteData(fn));
      knowledgeTables.forEach((t) => (writeKnowledge ? t.grantReadWriteData(fn) : t.grantReadData(fn)));
    };

    // Helper function to create Lambda with common settings
    const createLambda = (
      name: string,
      handler: string,
      { timeout = 30, writeKnowledge = false } = {},
    ): lambda.Function => {
      const fn = new lambda.Function(this, name, {
        runtime: lambda.Runtime.PYTHON_3_14,
        code: lambdaCode,
        handler,
        environment: commonEnv,
        timeout: cdk.Duration.seconds(timeout),
        memorySize: 256,
        architecture: lambda.Architecture.ARM_64,
      });

      grantDataAccess(fn, { writeKnowledge });

      // Grant Batch permissions
      fn.addToRolePolicy(new iam.PolicyStatement({
        actions: ['batch:SubmitJob', 'batch:DescribeJobs'],
        resources: batchSubmitResources,
      }));

      return fn;
    };

    // API Gateway. Every method requires an API key (x-api-key header);
    // CORS preflight OPTIONS methods are exempt (CDK sets apiKeyRequired
    // false on them, so browser preflights still work).
    this.api = new apigateway.RestApi(this, 'ExtractionApi', {
      restApiName: `knowledge-extraction-api-${props.environment}`,
      description: 'API for knowledge extraction from documents',
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
      },
      defaultMethodOptions: {
        apiKeyRequired: true,
      },
      deployOptions: {
        stageName: 'v1',
        throttlingRateLimit: 100,
        throttlingBurstLimit: 200,
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: true,
      },
    });

    // Job Management Lambdas
    const getUploadUrl = createLambda('GetUploadUrl', 'job_management.get_upload_url.handler');
    const submitExtraction = createLambda('SubmitExtraction', 'job_management.submit_extraction.handler');
    const getJobStatus = createLambda('GetJobStatus', 'job_management.get_job_status.handler');
    const downloadResults = createLambda('DownloadResults', 'job_management.download_results.handler');

    // Project Management Lambdas
    const listProjects = createLambda('ListProjects', 'project_management.list_projects.handler');
    const getProject = createLambda('GetProject', 'project_management.get_project.handler');
    const listSnapshots = createLambda('ListSnapshots', 'project_management.list_snapshots.handler');
    const getChangeset = createLambda('GetChangeset', 'project_management.get_changeset.handler');
    const listChangesets = createLambda('ListChangesets', 'project_management.list_changesets.handler');
    // Outlives API Gateway's 29s timeout so a large delete still finishes after the client gets a 504
    const deleteProject = createLambda('DeleteProject', 'project_management.delete_project.handler', {
      timeout: 300,
      writeKnowledge: true,
    });

    // Knowledge Graph API Lambdas
    const getKnowledgeClasses = createLambda('GetKnowledgeClasses', 'knowledge_api.get_knowledge_classes.handler');
    const getKnowledgeClass = createLambda('GetKnowledgeClass', 'knowledge_api.get_knowledge_class.handler');
    const getClassProperties = createLambda('GetClassProperties', 'knowledge_api.get_class_properties.handler');
    const getClassIncoming = createLambda('GetClassIncoming', 'knowledge_api.get_class_incoming.handler');

    // API Resources - Job Management
    const uploadUrl = this.api.root.addResource('upload-url');
    uploadUrl.addMethod('POST', new apigateway.LambdaIntegration(getUploadUrl));

    const extract = this.api.root.addResource('extract');
    extract.addMethod('POST', new apigateway.LambdaIntegration(submitExtraction));

    const jobs = this.api.root.addResource('jobs');
    const job = jobs.addResource('{jobId}');
    job.addMethod('GET', new apigateway.LambdaIntegration(getJobStatus));

    const results = job.addResource('results');
    results.addMethod('GET', new apigateway.LambdaIntegration(downloadResults));

    // API Resources - Projects
    const projects = this.api.root.addResource('projects');
    projects.addMethod('GET', new apigateway.LambdaIntegration(listProjects));

    const project = projects.addResource('{project}');
    project.addMethod('GET', new apigateway.LambdaIntegration(getProject));
    project.addMethod('DELETE', new apigateway.LambdaIntegration(deleteProject));

    const snapshots = project.addResource('snapshots');
    snapshots.addMethod('GET', new apigateway.LambdaIntegration(listSnapshots));

    const changesets = project.addResource('changesets');
    changesets.addMethod('GET', new apigateway.LambdaIntegration(listChangesets));

    const changeset = changesets.addResource('{changesetId}');
    changeset.addMethod('GET', new apigateway.LambdaIntegration(getChangeset));

    // API Resources - Knowledge Graph
    const knowledge = project.addResource('knowledge');

    const classes = knowledge.addResource('classes');
    classes.addMethod('GET', new apigateway.LambdaIntegration(getKnowledgeClasses));

    // className (normalized) is the identity of a class; classIri is the
    // shared metamodel type and cannot address a single class.
    const classByName = classes.addResource('{className}');
    classByName.addMethod('GET', new apigateway.LambdaIntegration(getKnowledgeClass));

    const properties = classByName.addResource('properties');
    properties.addMethod('GET', new apigateway.LambdaIntegration(getClassProperties));

    const incoming = classByName.addResource('incoming');
    incoming.addMethod('GET', new apigateway.LambdaIntegration(getClassIncoming));

    // MCP Server Lambda (with extended timeout for streaming)
    const mcpLambda = new lambda.Function(this, 'McpServer', {
      runtime: lambda.Runtime.PYTHON_3_14,
      handler: 'mcp_server.lambda_handler',
      code: lambda.Code.fromAsset('../lambda/mcp_server'),
      timeout: cdk.Duration.minutes(15),
      environment: commonEnv,
      architecture: lambda.Architecture.ARM_64,
      memorySize: 512,
    });

    grantDataAccess(mcpLambda);

    mcpLambda.addToRolePolicy(new iam.PolicyStatement({
      actions: ['batch:SubmitJob', 'batch:DescribeJobs'],
      resources: batchSubmitResources,
    }));

    // Add /mcp endpoint
    const mcpResource = this.api.root.addResource('mcp');
    mcpResource.addMethod('POST', new apigateway.LambdaIntegration(mcpLambda, {
      proxy: true,
    }));

    // API key + usage plan. Without a key the open /extract endpoint lets
    // anyone spend Gemini tokens and Fargate compute; the usage plan also
    // gives per-key throttling.
    const apiKey = this.api.addApiKey('DefaultApiKey', {
      apiKeyName: `kg-extraction-${props.environment}`,
    });

    const usagePlan = this.api.addUsagePlan('DefaultUsagePlan', {
      name: `kg-extraction-${props.environment}`,
      throttle: {
        rateLimit: 100,
        burstLimit: 200,
      },
    });
    usagePlan.addApiStage({ stage: this.api.deploymentStage });
    usagePlan.addApiKey(apiKey);

    // Outputs
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: this.api.url,
      description: 'API Gateway URL',
    });

    new cdk.CfnOutput(this, 'McpEndpoint', {
      value: `${this.api.url}mcp`,
      description: 'MCP Server endpoint for Claude Desktop',
    });

    new cdk.CfnOutput(this, 'ApiKeyId', {
      value: apiKey.keyId,
      description:
        'API key ID; fetch the value with: aws apigateway get-api-key --api-key <id> --include-value',
    });
  }
}
