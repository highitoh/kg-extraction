import * as cdk from 'aws-cdk-lib';
import * as events from 'aws-cdk-lib/aws-events';
import * as targets from 'aws-cdk-lib/aws-events-targets';
import * as batch from 'aws-cdk-lib/aws-batch';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

interface EventsStackProps extends cdk.StackProps {
  jobQueue: batch.JobQueue;
  jobDefinition: batch.EcsJobDefinition;
}

export class EventsStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: EventsStackProps) {
    super(scope, id, props);

    // IAM Role for EventBridge to submit Batch jobs
    const eventBridgeRole = new iam.Role(this, 'EventBridgeRole', {
      assumedBy: new iam.ServicePrincipal('events.amazonaws.com'),
      description: 'Role for EventBridge to submit Batch jobs',
    });

    eventBridgeRole.addToPolicy(new iam.PolicyStatement({
      actions: ['batch:SubmitJob'],
      resources: [props.jobQueue.jobQueueArn, props.jobDefinition.jobDefinitionArn],
    }));

    // Example: Daily Confluence extraction at 2 AM UTC (commented out by default)
    // Uncomment and customize for your scheduled extractions.
    // No JOB_ID is passed: batch_runner derives one from AWS_BATCH_JOB_ID
    // (stable across retry attempts) and creates the Jobs record itself, so
    // scheduled runs are tracked like on-demand ones.

    /*
    const confluenceSchedule = new events.Rule(this, 'ConfluenceSchedule', {
      ruleName: 'confluence-daily-extraction',
      schedule: events.Schedule.cron({ hour: '2', minute: '0' }),
      description: 'Daily Confluence extraction',
    });

    confluenceSchedule.addTarget(new targets.BatchJob(
      props.jobQueue.jobQueueArn,
      props.jobQueue,
      props.jobDefinition.jobDefinitionArn,
      props.jobDefinition,
      {
        jobName: 'scheduled-confluence',
        event: events.RuleTargetInput.fromObject({
          containerOverrides: {
            environment: [
              { name: 'JOB_TYPE', value: 'scheduled' },
              { name: 'PROJECT', value: 'confluence-daily' },
              { name: 'SOURCE', value: 'https://company.atlassian.net/wiki/spaces/DOCS' },
              { name: 'SOURCE_TYPE', value: 'confluence' },
              { name: 'INCREMENTAL', value: 'true' },
            ],
          },
        }),
      }
    ));
    */

    // Outputs
    new cdk.CfnOutput(this, 'EventBridgeRoleArn', {
      value: eventBridgeRole.roleArn,
      description: 'ARN of the EventBridge role for Batch job submission',
    });
  }
}
