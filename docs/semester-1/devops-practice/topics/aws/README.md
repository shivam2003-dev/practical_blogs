# AWS Practice Problems - Complete List 📚

## Overview

This section contains 15+ industry-standard AWS problems covering all Associate-level services. Each problem includes challenge extensions and comprehensive reference articles.

---

## 🟢 Beginner Level (1-5)

### 1. [S3 Static Website Hosting](01-s3-static-website.md)
Deploy static websites with S3 and CloudFront CDN
- **Key Services**: S3, CloudFront, Route53
- **Challenges**: Custom domains, CDN optimization, CI/CD
- **Industry Use**: Marketing sites, documentation, SPAs

### 2. [EC2 with Auto Scaling](02-ec2-deployment.md)
Deploy Node.js apps with auto-scaling and load balancing
- **Key Services**: EC2, ALB, Auto Scaling, CloudWatch
- **Challenges**: Blue-green deployment, monitoring, cost optimization
- **Industry Use**: Web applications, APIs, microservices

### 3. [Serverless API (Lambda + API Gateway)](03-lambda-api.md)
Build REST APIs without managing servers
- **Key Services**: Lambda, API Gateway, DynamoDB
- **Challenges**: Authentication, caching, async processing
- **Industry Use**: Backend APIs, webhooks, event processing

### 4. [Application Load Balancer](04-alb-routing.md)
Advanced traffic routing and distribution
- **Key Services**: ALB, Target Groups, WAF
- **Challenges**: Multi-tier routing, blue-green, Lambda targets
- **Industry Use**: Microservices, multi-tenant apps

### 5. [RDS PostgreSQL with High Availability](05-rds-database.md)
Production database with Multi-AZ and replicas
- **Key Services**: RDS, Secrets Manager, DMS
- **Challenges**: Failover testing, performance tuning, DR
- **Industry Use**: Any production application

---

## 🟡 Intermediate Level (6-10)

### 6. VPC & Network Architecture
Design secure network infrastructure
- **Key Services**: VPC, Subnets, NAT Gateway, VPC Peering
- **Challenges**: Multi-tier architecture, VPN setup, Transit Gateway
- **Industry Use**: Enterprise applications, hybrid cloud

### 7. ECS/Fargate Container Deployment
Deploy containerized applications
- **Key Services**: ECS, Fargate, ECR, Service Discovery
- **Challenges**: Auto-scaling, service mesh, blue-green
- **Industry Use**: Microservices, modern applications

### 8. CloudFormation Infrastructure as Code
Automate infrastructure provisioning
- **Key Services**: CloudFormation, StackSets
- **Challenges**: Nested stacks, custom resources, drift detection
- **Industry Use**: All AWS deployments

### 9. ElastiCache for Performance
Add caching layer for scalability
- **Key Services**: ElastiCache (Redis/Memcached)
- **Challenges**: Cache strategies, clustering, monitoring
- **Industry Use**: High-traffic applications

### 10. CloudWatch Monitoring & Logging
Comprehensive observability setup
- **Key Services**: CloudWatch, X-Ray, EventBridge
- **Challenges**: Custom metrics, distributed tracing, automation
- **Industry Use**: Production monitoring

---

## 🔴 Advanced Level (11-15)

### 11. Multi-Region Architecture
Global application deployment
- **Key Services**: Route53, CloudFront, Global Tables
- **Challenges**: Active-active, disaster recovery, data replication
- **Industry Use**: Global SaaS platforms

### 12. EKS Kubernetes Deployment
Kubernetes on AWS
- **Key Services**: EKS, kubectl, Helm
- **Challenges**: Service mesh, Ingress controllers, monitoring
- **Industry Use**: Cloud-native applications

### 13. Step Functions Orchestration
Serverless workflow automation
- **Key Services**: Step Functions, Lambda, SQS
- **Challenges**: Error handling, parallel processing, callbacks
- **Industry Use**: Data pipelines, order processing

### 14. Security & Compliance Setup
Implement AWS security best practices
- **Key Services**: IAM, GuardDuty, Security Hub, Config
- **Challenges**: MFA, SCPs, audit trails, compliance reporting
- **Industry Use**: All production environments

### 15. Cost Optimization & Governance
Manage AWS costs effectively
- **Key Services**: Cost Explorer, Budgets, Trusted Advisor
- **Challenges**: Reserved instances, savings plans, tagging
- **Industry Use**: All AWS accounts

---

## 📊 Service Coverage Matrix

| Service | Problem # | Difficulty |
|---------|-----------|------------|
| S3 | 1, 3 | 🟢 |
| EC2 | 2 | 🟢 |
| Lambda | 3, 13 | 🟢🔴 |
| RDS | 5 | 🟡 |
| ALB | 4 | 🟡 |
| VPC | 6 | 🟡 |
| ECS/Fargate | 7 | 🟡 |
| CloudFormation | 8 | 🟡 |
| ElastiCache | 9 | 🟡 |
| CloudWatch | 10 | 🟡 |
| Route53 | 1, 11 | 🟢🔴 |
| EKS | 12 | 🔴 |
| DynamoDB | 3 | 🟢 |
| API Gateway | 3 | 🟢 |
| CloudFront | 1 | 🟢 |
| WAF | 4 | 🔴 |
| Secrets Manager | 5 | 🟢 |

---

## 🎯 Learning Path Recommendations

### For Backend Developers
1. Lambda + API Gateway (Problem 3)
2. RDS Setup (Problem 5)
3. ElastiCache (Problem 9)
4. Step Functions (Problem 13)

### For DevOps Engineers
1. EC2 + Auto Scaling (Problem 2)
2. ALB Configuration (Problem 4)
3. CloudFormation (Problem 8)
4. Multi-Region (Problem 11)

### For Full-Stack Developers
1. S3 Static Site (Problem 1)
2. Serverless API (Problem 3)
3. ECS/Fargate (Problem 7)
4. Monitoring (Problem 10)

### For Solutions Architects
1. VPC Architecture (Problem 6)
2. Multi-Region (Problem 11)
3. Security Setup (Problem 14)
4. Cost Optimization (Problem 15)

---

## 📚 Additional Resources

### AWS Certification Paths
- [AWS Certified Solutions Architect - Associate](https://aws.amazon.com/certification/certified-solutions-architect-associate/)
- [AWS Certified Developer - Associate](https://aws.amazon.com/certification/certified-developer-associate/)
- [AWS Certified SysOps Administrator - Associate](https://aws.amazon.com/certification/certified-sysops-admin-associate/)

### Hands-On Labs
- [AWS Well-Architected Labs](https://wellarchitectedlabs.com/)
- [AWS Workshops](https://workshops.aws/)
- [AWS Free Tier Guide](https://aws.amazon.com/free/)

### Community Resources
- [AWS Reddit](https://www.reddit.com/r/aws/)
- [AWS re:Post](https://repost.aws/)
- [AWS User Groups](https://aws.amazon.com/developer/community/usergroups/)

---

## 💡 Pro Tips

1. **Always use IaC**: CloudFormation, CDK, or Terraform
2. **Tag everything**: Cost allocation and resource management
3. **Enable monitoring**: CloudWatch from day one
4. **Security first**: Least privilege, encryption, MFA
5. **Cost awareness**: Use budgets and alerts
6. **Automate**: CI/CD pipelines for all deployments
7. **Test disaster recovery**: Regular DR drills
8. **Document architecture**: Keep diagrams updated
9. **Stay updated**: AWS releases new features weekly
10. **Practice hands-on**: Theory + practice = mastery

---

## 🎓 Interview Preparation

These problems cover common AWS interview questions:
- Design a scalable web application
- Implement high availability
- Optimize costs
- Secure infrastructure
- Handle disaster recovery
- Troubleshoot common issues

---

**Ready to start?** Begin with [Problem 1: S3 Static Website →](01-s3-static-website.md)
