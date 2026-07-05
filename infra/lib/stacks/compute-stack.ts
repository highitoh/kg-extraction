import * as cdk from 'aws-cdk-lib';
import * as batch from 'aws-cdk-lib/aws-batch';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

interface ComputeStackProps extends cdk.StackProps {
  environment: string;
  extractionBucket: s3.Bucket;
  jobsTable: dynamodb.Table;
  classesTable: dynamodb.Table;
  propertiesTable: dynamodb.Table;
  instancesTable: dynamodb.Table;
  snapshotsTable: dynamodb.Table;
  projectsTable: dynamodb.Table;
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

    // Gemini API Key secret, injected into the container as GEMINI_API_KEY
    // (created with a placeholder value; update it after deployment)
    const geminiSecret = new secretsmanager.Secret(this, 'GeminiApiKey', {
      secretName: `gemini-api-key-${environment}`,
      description: 'Gemini API key for knowledge extraction - UPDATE THIS VALUE AFTER DEPLOYMENT',
      secretStringValue: cdk.SecretValue.unsafePlainText('PLACEHOLDER_UPDATE_ME_AFTER_DEPLOYMENT'),
    });

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
        // public IP the task cannot pull from ECR or reach the Gemini API
        assignPublicIp: true,
        environment: {
          STORAGE_BACKEND: 's3',
          // Required by llm_factory; selects the Gemini Developer API,
          // authenticated by the injected GEMINI_API_KEY secret
          LLM_PROVIDER: 'gemini_api',
          BUCKET_NAME: props.extractionBucket.bucketName,
          JOBS_TABLE: props.jobsTable.tableName,
          CLASSES_TABLE: props.classesTable.tableName,
          PROPERTIES_TABLE: props.propertiesTable.tableName,
          INSTANCES_TABLE: props.instancesTable.tableName,
          SNAPSHOTS_TABLE: props.snapshotsTable.tableName,
          PROJECTS_TABLE: props.projectsTable.tableName,
        },
        secrets: {
          GEMINI_API_KEY: batch.Secret.fromSecretsManager(geminiSecret),
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
        // Exit code 2 = configuration error (missing env vars, placeholder
        // secret); retrying cannot succeed
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

    new cdk.CfnOutput(this, 'GeminiSecretName', {
      value: geminiSecret.secretName,
      description: 'Gemini API secret name (UPDATE THE VALUE with: aws secretsmanager update-secret --secret-id gemini-api-key --secret-string "YOUR_KEY")',
    });
  }
}
