import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Construct } from 'constructs';

interface StorageStackProps extends cdk.StackProps {
  orgName: string;
  environment: string;
}

export class StorageStack extends cdk.Stack {
  public readonly extractionBucket: s3.Bucket;
  public readonly jobsTable: dynamodb.Table;
  public readonly snapshotsTable: dynamodb.Table;
  public readonly changesetsTable: dynamodb.Table;
  public readonly sessionsTable: dynamodb.Table;
  public readonly classesTable: dynamodb.Table;
  public readonly propertiesTable: dynamodb.Table;
  public readonly instancesTable: dynamodb.Table;
  public readonly projectsTable: dynamodb.Table;

  constructor(scope: Construct, id: string, props: StorageStackProps) {
    super(scope, id, props);

    const { orgName, environment } = props;
    const bucketName = `${orgName}-kg-extraction-${environment}`;

    // S3 Bucket
    this.extractionBucket = new s3.Bucket(this, 'ExtractionBucket', {
      bucketName,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: false,
      lifecycleRules: [
        {
          id: 'DeleteOldInputFiles',
          prefix: 'input/',
          expiration: cdk.Duration.days(7),
        },
        {
          id: 'DeleteOldWorkingDirectories',
          prefix: 'working/',
          expiration: cdk.Duration.days(7),
        },
      ],
      cors: [
        {
          allowedMethods: [s3.HttpMethods.PUT, s3.HttpMethods.POST],
          allowedOrigins: ['*'], // Configure for production
          allowedHeaders: ['*'],
          maxAge: 3000,
          exposedHeaders: ['ETag'],
        },
      ],
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // DynamoDB Tables

    // Jobs Table
    // jobId is a UUID and globally unique, so it is the sole key.
    // timestamp remains a plain attribute used as the sort key of
    // project-timestamp-index for time-ordered listing per project.
    this.jobsTable = new dynamodb.Table(this, 'JobsTable', {
      tableName: `${environment}-kg-extraction-jobs`,
      partitionKey: { name: 'jobId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.jobsTable.addGlobalSecondaryIndex({
      indexName: 'project-timestamp-index',
      partitionKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'timestamp', type: dynamodb.AttributeType.NUMBER },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // Snapshots Table
    this.snapshotsTable = new dynamodb.Table(this, 'SnapshotsTable', {
      tableName: `${environment}-kg-extraction-snapshots`,
      partitionKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'snapshotId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // Changesets Table
    this.changesetsTable = new dynamodb.Table(this, 'ChangesetsTable', {
      tableName: `${environment}-kg-extraction-changesets`,
      partitionKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'changesetId', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // Sessions Table (MCP)
    // Item collection per session: SK "STATE" holds session state and
    // aggregate stats, SK "INTERACTION#{timestamp}#{id}" holds one item per
    // interaction. Per-item TTL on expiresAt prunes interactions by age
    // without rewriting a growing list attribute.
    this.sessionsTable = new dynamodb.Table(this, 'SessionsTable', {
      tableName: `${environment}-kg-extraction-sessions`,
      partitionKey: { name: 'sessionId', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'sk', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      timeToLiveAttribute: 'expiresAt',
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // Classes Table (Knowledge Graph)
    // className is the normalized identity key (see docs/database-schema.md,
    // "Key Normalization"); the original spelling is kept in displayName.
    // Instance evidence lives in the Instances table; items here hold only
    // aggregates (instanceCount, latestExtractedAt).
    this.classesTable = new dynamodb.Table(this, 'ClassesTable', {
      tableName: `${environment}-kg-extraction-classes`,
      partitionKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'className', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // INCLUDE projections keep GSI write amplification bounded; extend the
    // attribute lists when handlers need additional fields from these indexes.
    const classSummaryAttributes = [
      'classIri',
      'classType',
      'displayName',
      'definition',
      'instanceCount',
      'sourceDocuments',
      'createdAt',
    ];

    this.classesTable.addGlobalSecondaryIndex({
      indexName: 'classType-index',
      partitionKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'classType', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.INCLUDE,
      nonKeyAttributes: classSummaryAttributes.filter((a) => a !== 'classType'),
    });

    // Lists classes belonging to a metamodel type (classIri is the type IRI
    // shared by many classes, not a per-class identity).
    this.classesTable.addGlobalSecondaryIndex({
      indexName: 'classIri-index',
      partitionKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'classIri', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.INCLUDE,
      nonKeyAttributes: classSummaryAttributes.filter((a) => a !== 'classIri'),
    });

    // Properties Table (Knowledge Graph)
    // Composite keys use normalized class names ("#" is stripped during
    // normalization, so it is safe as the delimiter). Instance evidence lives
    // in the Instances table; items here hold only aggregates.
    this.propertiesTable = new dynamodb.Table(this, 'PropertiesTable', {
      tableName: `${environment}-kg-extraction-properties`,
      partitionKey: { name: 'projectSourceClass', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'relationship', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    const propertySummaryAttributes = [
      'project',
      'sourceClassName',
      'targetClassName',
      'propertyName',
      'propertyIri',
      'definition',
      'averageStrength',
      'instanceCount',
    ];

    // Reverse lookup. The GSI sort key reuses the base-table relationship
    // attribute ("{propertyName}#{targetClassName}"); within a partition the
    // target class is fixed, so items still sort by propertyName.
    this.propertiesTable.addGlobalSecondaryIndex({
      indexName: 'target-class-index',
      partitionKey: { name: 'projectTargetClass', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'relationship', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.INCLUDE,
      nonKeyAttributes: propertySummaryAttributes,
    });

    this.propertiesTable.addGlobalSecondaryIndex({
      indexName: 'propertyName-index',
      partitionKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'propertyName', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.INCLUDE,
      nonKeyAttributes: propertySummaryAttributes.filter(
        (a) => a !== 'project' && a !== 'propertyName'
      ),
    });

    // Instances Table (Knowledge Graph evidence)
    // One item per extracted instance, keyed by the owning entity:
    //   entityKey = "{project}#CLASS#{className}"
    //             | "{project}#PROP#{sourceClassName}#{propertyName}#{targetClassName}"
    //   instanceKey = "{extractedAt}#{instanceId}" (chronological, unique)
    // Keeps Classes/Properties items small and bounded regardless of how many
    // times an entity is re-extracted.
    this.instancesTable = new dynamodb.Table(this, 'InstancesTable', {
      tableName: `${environment}-kg-extraction-instances`,
      partitionKey: { name: 'entityKey', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'instanceKey', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // Projects Table (Metadata + Extraction History)
    this.projectsTable = new dynamodb.Table(this, 'ProjectsTable', {
      tableName: `${environment}-kg-extraction-projects`,
      partitionKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'recordType', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // Sparse GSI for listing projects: entityType is set only on METADATA
    // records, so the index contains exactly one item per project and
    // "list all projects" is a query instead of a filtered scan.
    // Time-ordered extraction history needs no GSI - the base-table sort key
    // "EXTRACTION#{ISO timestamp}" already sorts chronologically.
    this.projectsTable.addGlobalSecondaryIndex({
      indexName: 'entityType-index',
      partitionKey: { name: 'entityType', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'project', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // Outputs
    new cdk.CfnOutput(this, 'BucketName', {
      value: this.extractionBucket.bucketName,
      description: 'S3 bucket for knowledge extraction files',
    });

    new cdk.CfnOutput(this, 'JobsTableName', {
      value: this.jobsTable.tableName,
      description: 'DynamoDB table for job tracking',
    });
  }
}
