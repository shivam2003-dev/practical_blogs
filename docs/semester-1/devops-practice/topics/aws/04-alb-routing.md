# Problem 4: Application Load Balancer with Target Groups 🟡

## 📋 Problem Statement

Set up a production-ready Application Load Balancer that:
- Distributes traffic across multiple EC2 instances
- Implements path-based and host-based routing
- Configures health checks for automatic failover
- Enables SSL/TLS termination
- Implements sticky sessions
- Sets up access logs for monitoring

**Difficulty**: Intermediate 🟡

---

## 🎯 Learning Objectives

- Configure Application Load Balancer (ALB)
- Create and manage Target Groups
- Implement routing rules
- Set up SSL certificates with ACM
- Configure health checks
- Enable access logging
- Understand load balancing algorithms

---

## 📚 Background

Application Load Balancers operate at Layer 7 (HTTP/HTTPS) and provide advanced routing capabilities. They're essential for distributing traffic, ensuring high availability, and enabling zero-downtime deployments.

---

## 💡 Hints

<details>
<summary>Hint 1: Target Group Configuration</summary>

```bash
aws elbv2 create-target-group \
    --name my-targets \
    --protocol HTTP \
    --port 80 \
    --vpc-id vpc-xxxxx \
    --health-check-protocol HTTP \
    --health-check-path /health
```

</details>

<details>
<summary>Hint 2: Path-Based Routing</summary>

Route based on URL paths:
- `/api/*` → API servers
- `/images/*` → Image servers
- `/` → Web servers

</details>

<details>
<summary>Hint 3: SSL Certificate</summary>

Request certificate with ACM:

```bash
aws acm request-certificate \
    --domain-name example.com \
    --validation-method DNS
```

</details>

---

## 📖 Solution Articles & References

### Official AWS Documentation
- [Application Load Balancer Guide](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html)
- [Target Groups Overview](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html)
- [Listener Rules](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/listener-update-rules.html)
- [Health Checks](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/target-group-health-checks.html)

### Tutorials & Guides
- [Getting Started with ALB](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/application-load-balancer-getting-started.html)
- [SSL/TLS Certificates](https://docs.aws.amazon.com/acm/latest/userguide/acm-overview.html)
- [Access Logs](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-access-logs.html)

### Best Practices
- [ALB Best Practices](https://aws.amazon.com/blogs/networking-and-content-delivery/best-practices-for-deploying-gateway-load-balancer/)
- [Security Best Practices](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/security-best-practices.html)
- [Performance Optimization](https://aws.amazon.com/premiumsupport/knowledge-center/elb-latency-troubleshooting/)

---

## 🚀 Challenge Problems

<details>
<summary>Challenge 1: Blue-Green Deployment with ALB 🟡</summary>

**Problem**: Implement zero-downtime deployment using ALB target groups

**Requirements**:
- Create two target groups (blue and green)
- Deploy new version to green
- Gradually shift traffic using weighted routing
- Automatic rollback on errors

**Reference Articles**:
- [Blue/Green Deployments](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html)
- [Weighted Target Groups](https://aws.amazon.com/blogs/aws/new-application-load-balancer-simplifies-deployment-with-weighted-target-groups/)
- [CodeDeploy with ALB](https://docs.aws.amazon.com/codedeploy/latest/userguide/tutorial-ecs-deployment.html)

**Key Concepts**:
- Target group weights
- Traffic shifting
- Canary deployments
- Rollback strategies

</details>

<details>
<summary>Challenge 2: Multi-Tier Architecture Routing 🟡</summary>

**Problem**: Route traffic to different application tiers

**Requirements**:
- Frontend servers: `/` and `/static/*`
- API servers: `/api/*`
- Admin panel: `/admin/*`
- Microservices: `/service1/*`, `/service2/*`

**Reference Articles**:
- [Path-Based Routing](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/tutorial-load-balancer-routing.html)
- [Host-Based Routing](https://aws.amazon.com/blogs/aws/new-host-based-routing-support-for-aws-application-load-balancers/)
- [Microservices with ALB](https://aws.amazon.com/blogs/compute/microservice-delivery-with-amazon-ecs-and-application-load-balancers/)

**Key Concepts**:
- Listener rules priority
- Path patterns
- Host headers
- Query string routing

</details>

<details>
<summary>Challenge 3: WAF Integration for Security 🔴</summary>

**Problem**: Add AWS WAF to protect against common attacks

**Requirements**:
- Block SQL injection attempts
- Rate limiting per IP
- Geo-blocking
- Custom rules for your application

**Reference Articles**:
- [AWS WAF Documentation](https://docs.aws.amazon.com/waf/latest/developerguide/waf-chapter.html)
- [WAF with ALB](https://docs.aws.amazon.com/waf/latest/developerguide/web-acl-associating-aws-resource.html)
- [Managed Rules](https://docs.aws.amazon.com/waf/latest/developerguide/aws-managed-rule-groups.html)

**Key Concepts**:
- Web ACLs
- Rule groups
- Rate-based rules
- IP sets

</details>

<details>
<summary>Challenge 4: Cross-Zone Load Balancing 🟢</summary>

**Problem**: Optimize traffic distribution across availability zones

**Requirements**:
- Enable cross-zone load balancing
- Monitor distribution metrics
- Cost analysis
- Performance comparison

**Reference Articles**:
- [Cross-Zone Load Balancing](https://docs.aws.amazon.com/elasticloadbalancing/latest/userguide/how-elastic-load-balancing-works.html#cross-zone-load-balancing)
- [ALB Metrics](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-cloudwatch-metrics.html)

**Key Concepts**:
- Availability zones
- Distribution algorithms
- Cost implications
- Request routing

</details>

<details>
<summary>Challenge 5: Lambda Target Integration 🟡</summary>

**Problem**: Route ALB traffic directly to Lambda functions

**Requirements**:
- Configure Lambda as target
- Handle multi-value headers
- Implement streaming responses
- Performance optimization

**Reference Articles**:
- [Lambda as ALB Target](https://docs.aws.amazon.com/lambda/latest/dg/services-alb.html)
- [Lambda Multi-Value Headers](https://aws.amazon.com/blogs/networking-and-content-delivery/lambda-functions-as-targets-for-application-load-balancers/)

**Key Concepts**:
- Lambda integration
- Event format
- Response format
- Cold start mitigation

</details>

<details>
<summary>Challenge 6: Implement Sticky Sessions 🟢</summary>

**Problem**: Maintain user session affinity to specific targets

**Requirements**:
- Enable session stickiness
- Cookie-based vs application-based
- Duration configuration
- Testing and validation

**Reference Articles**:
- [Sticky Sessions](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/sticky-sessions.html)
- [Session Affinity](https://aws.amazon.com/blogs/aws/new-application-load-balancer-session-affinity/)

**Key Concepts**:
- Session cookies
- Duration settings
- Stickiness algorithms
- Use cases

</details>

<details>
<summary>Challenge 7: HTTP/2 and WebSocket Support 🟡</summary>

**Problem**: Enable modern protocols for better performance

**Requirements**:
- Configure HTTP/2
- Set up WebSocket connections
- Implement real-time features
- Performance testing

**Reference Articles**:
- [HTTP/2 Support](https://aws.amazon.com/about-aws/whats-new/2016/11/application-load-balancers-now-support-http2/)
- [WebSocket Connections](https://aws.amazon.com/blogs/aws/new-aws-application-load-balancer/)

**Key Concepts**:
- Protocol versions
- Persistent connections
- Multiplexing
- Server push

</details>

<details>
<summary>Challenge 8: Global Accelerator Integration 🔴</summary>

**Problem**: Add AWS Global Accelerator for global traffic optimization

**Requirements**:
- Create Global Accelerator
- Add ALB as endpoint
- Configure health checks
- Multi-region setup

**Reference Articles**:
- [AWS Global Accelerator](https://docs.aws.amazon.com/global-accelerator/latest/dg/what-is-global-accelerator.html)
- [Global Accelerator with ALB](https://aws.amazon.com/blogs/networking-and-content-delivery/using-aws-global-accelerator-to-achieve-blue-green-deployments/)

**Key Concepts**:
- Anycast IPs
- Traffic dials
- Endpoint groups
- Health checking

</details>

<details>
<summary>Challenge 9: Advanced Health Check Strategies 🟡</summary>

**Problem**: Implement sophisticated health checking

**Requirements**:
- Custom health check endpoints
- Graceful shutdown handling
- Dependency checking
- Circuit breaker pattern

**Reference Articles**:
- [Health Check Best Practices](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/target-group-health-checks.html)
- [Health Check Configuration](https://aws.amazon.com/premiumsupport/knowledge-center/elb-fix-failing-health-checks-alb/)

**Key Concepts**:
- Health check intervals
- Thresholds
- Custom endpoints
- Dependencies

</details>

<details>
<summary>Challenge 10: Cost Optimization with ALB 🟡</summary>

**Problem**: Optimize ALB costs while maintaining performance

**Requirements**:
- Analyze LCU (Load Balancer Capacity Units)
- Optimize listener rules
- Connection draining strategy
- Right-sizing target groups

**Reference Articles**:
- [ALB Pricing](https://aws.amazon.com/elasticloadbalancing/pricing/)
- [LCU Calculation](https://aws.amazon.com/blogs/networking-and-content-delivery/understanding-load-balancer-capacity-units-and-pricing-for-alb/)
- [Cost Optimization](https://aws.amazon.com/blogs/networking-and-content-delivery/best-practices-for-deploying-gateway-load-balancer/)

**Key Concepts**:
- LCU metrics
- Connection handling
- Idle timeouts
- Request patterns

</details>

---

## 💡 Quick Configuration Script

<details>
<summary>Complete ALB Setup Script</summary>

```bash
#!/bin/bash

# Variables
REGION="us-east-1"
VPC_ID="vpc-xxxxx"
SUBNET_IDS="subnet-xxx subnet-yyy"
CERTIFICATE_ARN="arn:aws:acm:..."

# Create security group for ALB
ALB_SG=$(aws ec2 create-security-group \
    --group-name alb-sg \
    --description "ALB Security Group" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' --output text)

# Allow HTTP and HTTPS
aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG \
    --protocol tcp --port 80 --cidr 0.0.0.0/0 --region $REGION

aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG \
    --protocol tcp --port 443 --cidr 0.0.0.0/0 --region $REGION

# Create target groups
WEB_TG=$(aws elbv2 create-target-group \
    --name web-targets \
    --protocol HTTP --port 80 \
    --vpc-id $VPC_ID \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --region $REGION \
    --query 'TargetGroups[0].TargetGroupArn' --output text)

API_TG=$(aws elbv2 create-target-group \
    --name api-targets \
    --protocol HTTP --port 3000 \
    --vpc-id $VPC_ID \
    --health-check-path /api/health \
    --region $REGION \
    --query 'TargetGroups[0].TargetGroupArn' --output text)

# Create load balancer
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name my-application-lb \
    --subnets $SUBNET_IDS \
    --security-groups $ALB_SG \
    --scheme internet-facing \
    --type application \
    --region $REGION \
    --query 'LoadBalancers[0].LoadBalancerArn' --output text)

# Create HTTPS listener
HTTPS_LISTENER=$(aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTPS --port 443 \
    --certificates CertificateArn=$CERTIFICATE_ARN \
    --default-actions Type=forward,TargetGroupArn=$WEB_TG \
    --region $REGION \
    --query 'Listeners[0].ListenerArn' --output text)

# Add path-based routing rule for API
aws elbv2 create-rule \
    --listener-arn $HTTPS_LISTENER \
    --priority 10 \
    --conditions Field=path-pattern,Values='/api/*' \
    --actions Type=forward,TargetGroupArn=$API_TG \
    --region $REGION

# Create HTTP listener (redirect to HTTPS)
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP --port 80 \
    --default-actions Type=redirect,RedirectConfig='{Protocol=HTTPS,Port=443,StatusCode=HTTP_301}' \
    --region $REGION

# Enable access logs
S3_BUCKET="my-alb-logs-${AWS_ACCOUNT_ID}"
aws s3 mb s3://$S3_BUCKET --region $REGION

aws elbv2 modify-load-balancer-attributes \
    --load-balancer-arn $ALB_ARN \
    --attributes \
        Key=access_logs.s3.enabled,Value=true \
        Key=access_logs.s3.bucket,Value=$S3_BUCKET \
        Key=idle_timeout.timeout_seconds,Value=60 \
    --region $REGION

# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --query 'LoadBalancers[0].DNSName' \
    --output text --region $REGION)

echo "ALB DNS: $ALB_DNS"
echo "Web Target Group: $WEB_TG"
echo "API Target Group: $API_TG"
```

</details>

---

## 🔍 Monitoring & Troubleshooting

### Key CloudWatch Metrics to Monitor

```bash
# Target health
aws cloudwatch get-metric-statistics \
    --namespace AWS/ApplicationELB \
    --metric-name HealthyHostCount \
    --dimensions Name=TargetGroup,Value=targetgroup/web-targets/xxx \
    --start-time 2025-11-12T00:00:00Z \
    --end-time 2025-11-12T23:59:59Z \
    --period 300 \
    --statistics Average

# Request count
aws cloudwatch get-metric-statistics \
    --namespace AWS/ApplicationELB \
    --metric-name RequestCount \
    --dimensions Name=LoadBalancer,Value=app/my-application-lb/xxx \
    --start-time 2025-11-12T00:00:00Z \
    --end-time 2025-11-12T23:59:59Z \
    --period 300 \
    --statistics Sum

# Target response time
aws cloudwatch get-metric-statistics \
    --namespace AWS/ApplicationELB \
    --metric-name TargetResponseTime \
    --dimensions Name=LoadBalancer,Value=app/my-application-lb/xxx \
    --start-time 2025-11-12T00:00:00Z \
    --end-time 2025-11-12T23:59:59Z \
    --period 300 \
    --statistics Average
```

---

## ✨ Best Practices

- ✅ Always use HTTPS with valid SSL certificates
- ✅ Enable cross-zone load balancing for even distribution
- ✅ Configure proper health check intervals and thresholds
- ✅ Use connection draining for graceful instance removal
- ✅ Enable access logs for troubleshooting
- ✅ Implement least privilege security groups
- ✅ Use multiple availability zones
- ✅ Monitor key CloudWatch metrics
- ✅ Set appropriate idle timeout values
- ✅ Use WAF for additional security

---

**Previous**: [← Lambda API](03-lambda-api.md) | **Next**: [RDS & Multi-AZ →](05-rds-database.md)
