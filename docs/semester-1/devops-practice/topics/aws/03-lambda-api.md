# Problem 3: Serverless REST API with Lambda, API Gateway & DynamoDB 🟡

## 📋 Problem Statement

Build a serverless REST API that:
- Uses AWS Lambda functions for business logic
- API Gateway for HTTP endpoints
- DynamoDB for data persistence
- Implements CRUD operations
- Includes authentication with API keys
- Has proper error handling and logging
- Uses environment variables for configuration

**Difficulty**: Intermediate 🟡

---

## 🎯 Learning Objectives

- Build serverless applications with Lambda
- Configure API Gateway REST APIs
- Work with DynamoDB NoSQL database
- Implement IAM roles and policies
- Use CloudWatch for monitoring
- Handle Lambda layers and dependencies

---

## 📚 Background

Serverless architecture eliminates server management, automatically scales, and follows a pay-per-use model. This is ideal for APIs with variable traffic and reduces operational overhead.

---

## 💡 Hints

<details>
<summary>Hint 1: Lambda Function Structure</summary>

Basic Lambda handler:

```javascript
exports.handler = async (event) => {
    const response = {
        statusCode: 200,
        body: JSON.stringify({ message: 'Success' })
    };
    return response;
};
```

</details>

<details>
<summary>Hint 2: DynamoDB Operations</summary>

Use AWS SDK v3:

```javascript
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand } from "@aws-sdk/lib-dynamodb";

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);
```

</details>

<details>
<summary>Hint 3: API Gateway Integration</summary>

Map Lambda response to API Gateway:

```yaml
IntegrationResponses:
  - StatusCode: 200
    ResponseTemplates:
      application/json: $input.json('$.body')
```

</details>

---

## 📖 Solution Articles & References

### Official AWS Documentation
- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html)
- [API Gateway REST API Tutorial](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-create-api-as-simple-proxy-for-lambda.html)
- [DynamoDB Getting Started](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/GettingStartedDynamoDB.html)
- [Lambda Environment Variables](https://docs.aws.amazon.com/lambda/latest/dg/configuration-envvars.html)

### Step-by-Step Tutorials
- [Build a Serverless API](https://aws.amazon.com/getting-started/hands-on/build-serverless-web-app-lambda-apigateway-s3-dynamodb-cognito/)
- [Serverless Framework Tutorial](https://www.serverless.com/framework/docs/tutorial)
- [Lambda + DynamoDB CRUD](https://docs.aws.amazon.com/lambda/latest/dg/with-ddb-example.html)

### Best Practices
- [Serverless Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [DynamoDB Best Practices](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html)
- [API Gateway Security](https://docs.aws.amazon.com/apigateway/latest/developerguide/security.html)

---

## 🚀 Challenge Problems

<details>
<summary>Challenge 1: Add Cognito Authentication 🔴</summary>

**Problem**: Implement user authentication using AWS Cognito

**Requirements**:
- Create Cognito User Pool
- Add signup/login endpoints
- Protect API routes with JWT tokens
- Implement password reset flow

**Reference Articles**:
- [Cognito User Pools](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-identity-pools.html)
- [API Gateway Cognito Authorizer](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-integrate-with-cognito.html)
- [JWT Token Validation](https://aws.amazon.com/blogs/mobile/integrating-amazon-cognito-user-pools-with-api-gateway/)

**Key Concepts**:
- User pools vs Identity pools
- JWT token validation
- Custom authorizers
- Refresh token rotation

</details>

<details>
<summary>Challenge 2: Implement Caching with ElastiCache 🟡</summary>

**Problem**: Add Redis caching layer to reduce DynamoDB reads

**Requirements**:
- Set up ElastiCache Redis cluster
- Implement cache-aside pattern
- Add cache invalidation logic
- Monitor cache hit ratio

**Reference Articles**:
- [ElastiCache for Redis](https://docs.aws.amazon.com/AmazonElastiCache/latest/red-ug/WhatIs.html)
- [Caching Strategies](https://aws.amazon.com/caching/best-practices/)
- [Lambda with ElastiCache](https://aws.amazon.com/blogs/database/how-to-use-amazon-elasticache-for-redis-to-cache-data-from-aws-lambda/)

**Key Concepts**:
- Cache-aside vs write-through
- TTL strategies
- Cache warming
- Connection pooling in Lambda

</details>

<details>
<summary>Challenge 3: Add SQS Queue for Async Processing 🟡</summary>

**Problem**: Offload heavy tasks to background jobs using SQS

**Requirements**:
- Create SQS queue
- Trigger Lambda from SQS
- Implement batch processing
- Add dead letter queue (DLQ)

**Reference Articles**:
- [Lambda with SQS](https://docs.aws.amazon.com/lambda/latest/dg/with-sqs.html)
- [SQS Best Practices](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-best-practices.html)
- [Dead Letter Queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-dead-letter-queues.html)

**Key Concepts**:
- Visibility timeout
- Message deduplication
- FIFO vs Standard queues
- Batch processing

</details>

<details>
<summary>Challenge 4: Implement Rate Limiting 🟡</summary>

**Problem**: Add rate limiting to prevent API abuse

**Requirements**:
- Use API Gateway usage plans
- Implement custom rate limiter with DynamoDB
- Add throttling per API key
- Return proper 429 responses

**Reference Articles**:
- [API Gateway Throttling](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-request-throttling.html)
- [Usage Plans and API Keys](https://docs.aws.amazon.com/apigateway/latest/developerguide/api-gateway-api-usage-plans.html)
- [Custom Rate Limiting](https://aws.amazon.com/blogs/architecture/rate-limiting-strategies-for-serverless-applications/)

**Key Concepts**:
- Token bucket algorithm
- Usage plans
- Quota management
- Per-user rate limits

</details>

<details>
<summary>Challenge 5: Add CloudWatch Dashboards & Alarms 🟢</summary>

**Problem**: Create comprehensive monitoring and alerting

**Requirements**:
- Custom CloudWatch dashboard
- Alarms for errors, latency, throttles
- SNS notifications for critical issues
- X-Ray tracing for debugging

**Reference Articles**:
- [CloudWatch Dashboards](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Dashboards.html)
- [Lambda with X-Ray](https://docs.aws.amazon.com/lambda/latest/dg/services-xray.html)
- [CloudWatch Alarms](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/AlarmThatSendsEmail.html)

**Key Concepts**:
- Custom metrics
- Distributed tracing
- Log aggregation
- Alarm actions

</details>

<details>
<summary>Challenge 6: Multi-Region Deployment 🔴</summary>

**Problem**: Deploy API to multiple AWS regions for global availability

**Requirements**:
- Deploy to 2+ regions
- Use Route53 for global routing
- Replicate DynamoDB with Global Tables
- Implement health checks

**Reference Articles**:
- [DynamoDB Global Tables](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/GlobalTables.html)
- [Route53 Latency Routing](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-policy.html)
- [Multi-Region Serverless](https://aws.amazon.com/blogs/compute/building-a-multi-region-serverless-application-with-amazon-api-gateway-and-aws-lambda/)

**Key Concepts**:
- Global tables
- Cross-region replication
- Latency-based routing
- Disaster recovery

</details>

<details>
<summary>Challenge 7: Implement CI/CD with CodePipeline 🟡</summary>

**Problem**: Automate deployment with AWS native tools

**Requirements**:
- Create CodePipeline
- Use CodeBuild for testing
- Deploy with SAM or CDK
- Implement blue/green deployments

**Reference Articles**:
- [CodePipeline Tutorial](https://docs.aws.amazon.com/codepipeline/latest/userguide/tutorials-simple-s3.html)
- [SAM Deployment](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-deploying.html)
- [Lambda Blue/Green](https://docs.aws.amazon.com/lambda/latest/dg/lambda-traffic-shifting-using-aliases.html)

**Key Concepts**:
- Pipeline stages
- Build specifications
- Deployment strategies
- Rollback mechanisms

</details>

<details>
<summary>Challenge 8: Add File Upload to S3 with Presigned URLs 🟡</summary>

**Problem**: Allow secure file uploads directly to S3

**Requirements**:
- Generate presigned URLs
- Validate file types and sizes
- Process uploads with Lambda
- Store metadata in DynamoDB

**Reference Articles**:
- [S3 Presigned URLs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/PresignedUrlUploadObject.html)
- [S3 Event Notifications](https://docs.aws.amazon.com/AmazonS3/latest/userguide/NotificationHowTo.html)
- [Lambda S3 Processing](https://docs.aws.amazon.com/lambda/latest/dg/with-s3-example.html)

**Key Concepts**:
- Presigned URLs
- CORS configuration
- S3 events
- Image processing

</details>

<details>
<summary>Challenge 9: Implement GraphQL API with AppSync 🟡</summary>

**Problem**: Convert REST API to GraphQL using AWS AppSync

**Requirements**:
- Define GraphQL schema
- Create resolvers for DynamoDB
- Implement subscriptions
- Add real-time updates

**Reference Articles**:
- [AWS AppSync Documentation](https://docs.aws.amazon.com/appsync/latest/devguide/what-is-appsync.html)
- [GraphQL Schema Design](https://docs.aws.amazon.com/appsync/latest/devguide/designing-a-graphql-api.html)
- [AppSync Resolvers](https://docs.aws.amazon.com/appsync/latest/devguide/resolver-mapping-template-reference.html)

**Key Concepts**:
- GraphQL schemas
- Resolvers and data sources
- Subscriptions
- Real-time data

</details>

<details>
<summary>Challenge 10: Cost Optimization & Performance 🟡</summary>

**Problem**: Optimize Lambda costs and improve performance

**Requirements**:
- Analyze CloudWatch costs
- Optimize Lambda memory/timeout
- Implement provisioned concurrency
- Use Lambda Layers for dependencies

**Reference Articles**:
- [Lambda Cost Optimization](https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/)
- [Provisioned Concurrency](https://docs.aws.amazon.com/lambda/latest/dg/provisioned-concurrency.html)
- [Lambda Layers](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html)
- [Cost Explorer](https://aws.amazon.com/aws-cost-management/aws-cost-explorer/)

**Key Concepts**:
- Power tuning
- Cold start optimization
- Provisioned concurrency
- Lambda layers

</details>

---

## 💡 Quick Start Code Snippet

<details>
<summary>Sample Lambda Function (Node.js)</summary>

```javascript
// handler.js
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, PutCommand, GetCommand, ScanCommand, DeleteCommand } from "@aws-sdk/lib-dynamodb";

const client = new DynamoDBClient({});
const docClient = DynamoDBDocumentClient.from(client);
const TABLE_NAME = process.env.TABLE_NAME;

export const createItem = async (event) => {
    try {
        const body = JSON.parse(event.body);
        const item = {
            id: Date.now().toString(),
            ...body,
            createdAt: new Date().toISOString()
        };

        await docClient.send(new PutCommand({
            TableName: TABLE_NAME,
            Item: item
        }));

        return {
            statusCode: 201,
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(item)
        };
    } catch (error) {
        console.error('Error:', error);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Could not create item' })
        };
    }
};

export const getItem = async (event) => {
    try {
        const { id } = event.pathParameters;

        const result = await docClient.send(new GetCommand({
            TableName: TABLE_NAME,
            Key: { id }
        }));

        if (!result.Item) {
            return {
                statusCode: 404,
                body: JSON.stringify({ error: 'Item not found' })
            };
        }

        return {
            statusCode: 200,
            body: JSON.stringify(result.Item)
        };
    } catch (error) {
        console.error('Error:', error);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Could not get item' })
        };
    }
};

export const listItems = async (event) => {
    try {
        const result = await docClient.send(new ScanCommand({
            TableName: TABLE_NAME
        }));

        return {
            statusCode: 200,
            body: JSON.stringify({
                items: result.Items,
                count: result.Count
            })
        };
    } catch (error) {
        console.error('Error:', error);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Could not list items' })
        };
    }
};

export const deleteItem = async (event) => {
    try {
        const { id } = event.pathParameters;

        await docClient.send(new DeleteCommand({
            TableName: TABLE_NAME,
            Key: { id }
        }));

        return {
            statusCode: 204,
            body: ''
        };
    } catch (error) {
        console.error('Error:', error);
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Could not delete item' })
        };
    }
};
```

</details>

---

## 🛠️ Infrastructure as Code

<details>
<summary>AWS SAM Template</summary>

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  ItemsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: items-table
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: id
          AttributeType: S
      KeySchema:
        - AttributeName: id
          KeyType: HASH

  ItemsApi:
    Type: AWS::Serverless::Api
    Properties:
      StageName: prod
      Cors:
        AllowOrigin: "'*'"
        AllowHeaders: "'*'"

  CreateItemFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: handler.createItem
      Runtime: nodejs18.x
      Environment:
        Variables:
          TABLE_NAME: !Ref ItemsTable
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref ItemsTable
      Events:
        Create:
          Type: Api
          Properties:
            RestApiId: !Ref ItemsApi
            Path: /items
            Method: POST
```

</details>

---

## ✨ Best Practices

- ✅ Use environment variables for configuration
- ✅ Implement proper error handling
- ✅ Enable X-Ray tracing for debugging
- ✅ Set appropriate Lambda timeout and memory
- ✅ Use Lambda layers for common dependencies
- ✅ Implement idempotency for write operations
- ✅ Monitor with CloudWatch metrics and logs
- ✅ Use VPC endpoints for private access
- ✅ Implement least privilege IAM policies

---

**Previous**: [← EC2 Auto-Scaling](02-ec2-deployment.md) | **Next**: [RDS & Database Management →](04-rds-database.md)
