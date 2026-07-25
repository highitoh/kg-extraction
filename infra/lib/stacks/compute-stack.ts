import * as cdk from 'aws-cdk-lib';
import * as batch from 'aws-cdk-lib/aws-batch';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import { Construct } from 'constructs';

interface ComputeStackProps extends cdk.StackProps {
  environment: string;
  gcpProjectId: string;
  gcpLocation?: string;
  extractionBucket: s3.Bucket;
  jobsTable: dynamodb.Table;
  classesTable: dynamodb.Table;
  propertiesTable: dynamodb.Table;
  instancesTable: dynamodb.Table;
  snapshotsTable: dynamodb.Table;
  projectsTable: dynamodb.Table;
  changesetsTable: dynamodb.Table;
}

export class ComputeStack extends cdk.Stack {
  public readonly jobQueue: batch.JobQueue;
  public readonly jobDefinition: batch.EcsJobDefinition;
  public readonly ecrRepository: ecr.Repository;

  constructor(scope: Construct, id: string, props: ComputeStackProps) {
    super(scope, id, props);

    const { environment } = props;

    // Resource names carry the environment suffix so multiple environments
    // can coexist in one account/region.

    // ECR Repository for Docker images
    this.ecrRepository = new ecr.Repository(this, 'ExtractionRepository', {
      repositoryName: `kg-extraction-${environment}`,
      imageScanOnPush: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // IAM Role for Batch Job
    const jobRole = new iam.Role(this, 'BatchJobRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      description: 'Role for knowledge extraction batch jobs',
    });

    // Grant S3 permissions
    props.extractionBucket.grantReadWrite(jobRole);

    // Grant DynamoDB permissions
    props.jobsTable.grantReadWriteData(jobRole);
    props.classesTable.grantReadWriteData(jobRole);
    props.propertiesTable.grantReadWriteData(jobRole);
    props.instancesTable.grantReadWriteData(jobRole);
    props.snapshotsTable.grantReadWriteData(jobRole);
    props.projectsTable.grantReadWriteData(jobRole);
    props.changesetsTable.grantReadWriteData(jobRole);

    // Vertex AI authentication via Workload Identity Federation
    // Configure GCP to allow this AWS role to access Vertex AI:
    // 1. Create a Workload Identity Pool in your GCP project
    // 2. Add AWS as a provider with the account ID and role ARN
    // 3. Grant the workload identity user the Vertex AI User role
    // See: https://cloud.google.com/iam/docs/workload-identity-federation-with-other-clouds

    // Execution Role for ECS Task
    const executionRole = new iam.Role(this, 'BatchExecutionRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSTaskExecutionRolePolicy'),
      ],
    });

    // Grant ECR pull permissions
    this.ecrRepository.grantPull(executionRole);

    // Compute Environment (Fargate Spot)
    const computeEnvironment = new batch.FargateComputeEnvironment(this, 'ExtractionComputeEnv', {
      computeEnvironmentName: `extraction-compute-env-${environment}`,
      vpc: ec2.Vpc.fromLookup(this, 'DefaultVpc', { isDefault: true }),
      maxvCpus: 16,
      spot: true,
    });

    // Job Queue
    this.jobQueue = new batch.JobQueue(this, 'ExtractionQueue', {
      jobQueueName: `extraction-queue-${environment}`,
      priority: 1,
      computeEnvironments: [
        {
          computeEnvironment,
          order: 1,
        },
      ],
    });

    // Job Definition
    this.jobDefinition = new batch.EcsJobDefinition(this, 'ExtractionJobDefinition', {
      jobDefinitionName: `extraction-job-${environment}`,
      container: new batch.EcsFargateContainerDefinition(this, 'ExtractionContainer', {
        image: ecs.ContainerImage.fromEcrRepository(this.ecrRepository, 'latest'),
        cpu: 1,
        memory: cdk.Size.mebibytes(2048),
        jobRole,
        executionRole,
        // The default VPC has public subnets and no NAT gateway; without a
        // public IP the task cannot pull from ECR or reach the Vertex AI API
        assignPublicIp: true,
        environment: {
          STORAGE_BACKEND: 's3',
          // Required by llm_factory; selects Vertex AI,
          // authenticated via Workload Identity Federation
          LLM_PROVIDER: 'gcp_vertexai',
          GOOGLE_CLOUD_PROJECT: props.gcpProjectId,
          GOOGLE_CLOUD_LOCATION: props.gcpLocation || 'global',
          BUCKET_NAME: props.extractionBucket.bucketName,
          JOBS_TABLE: props.jobsTable.tableName,
          CLASSES_TABLE: props.classesTable.tableName,
          PROPERTIES_TABLE: props.propertiesTable.tableName,
          INSTANCES_TABLE: props.instancesTable.tableName,
          SNAPSHOTS_TABLE: props.snapshotsTable.tableName,
          PROJECTS_TABLE: props.projectsTable.tableName,
          CHANGESETS_TABLE: props.changesetsTable.tableName,
        },
        logging: ecs.LogDriver.awsLogs({
          streamPrefix: 'extraction-job',
        }),
      }),
      timeout: cdk.Duration.hours(2),
      retryAttempts: 2,
      retryStrategies: [
        batch.RetryStrategy.of(batch.Action.EXIT, batch.Reason.CANNOT_PULL_CONTAINER),
        batch.RetryStrategy.of(batch.Action.EXIT, batch.Reason.custom({
          onExitCode: '1',
        })),
        // Exit code 2 = configuration error (missing env vars); retrying
        // cannot succeed
        batch.RetryStrategy.of(batch.Action.EXIT, batch.Reason.custom({
          onExitCode: '2',
        })),
      ],
    });

    // Outputs
    new cdk.CfnOutput(this, 'JobQueueArn', {
      value: this.jobQueue.jobQueueArn,
      description: 'ARN of the batch job queue',
    });

    new cdk.CfnOutput(this, 'JobDefinitionArn', {
      value: this.jobDefinition.jobDefinitionArn,
      description: 'ARN of the batch job definition',
    });

    new cdk.CfnOutput(this, 'EcrRepositoryUri', {
      value: this.ecrRepository.repositoryUri,
      description: 'URI of the ECR repository',
    });

    new cdk.CfnOutput(this, 'GcpProjectId', {
      value: props.gcpProjectId,
      description: 'GCP Project ID for Vertex AI access',
    });

    new cdk.CfnOutput(this, 'BatchJobRoleArn', {
      value: jobRole.roleArn,
      description: 'ARN of the batch job role (configure in GCP Workload Identity Federation)',
    });
  }
}
