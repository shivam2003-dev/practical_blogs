# DevOps Problem Bank

A single hub for all hands-on DevOps problems across AWS and GitHub Actions. Each problem statement links to a full walkthrough (available now or coming soon) and highlights the key services, skills, and real-world scenarios you will practice.

---

## ☁️ AWS Problem Bank

| # | Service / Topic | Problem Statement | Skills & Focus | Link |
|---|-----------------|-------------------|----------------|------|
| 1 | S3 + CloudFront | Deploy a global static site with S3, CloudFront, HTTPS, custom domains, and automated invalidations. | Static hosting, CDN, DNS, automation | [Full guide](topics/aws/01-s3-static-website.md) |
| 2 | EC2 + Auto Scaling + ALB | Launch a Node.js app behind an Auto Scaling Group and ALB with health checks and CloudWatch alarms. | Compute, scaling, ALB, monitoring | [Full guide](topics/aws/02-ec2-deployment.md) |
| 3 | Lambda + API Gateway + DynamoDB | Build a serverless REST API with CRUD, validation, caching, and asynchronous processing hooks. | Serverless, NoSQL, API design | [Full guide](topics/aws/03-lambda-api.md) |
| 4 | Application Load Balancer | Implement ALB routing (path, host, weighted), SSL offload, WAF security, and blue/green deployments. | L7 load balancing, security, traffic shifting | [Full guide](topics/aws/04-alb-routing.md) |
| 5 | RDS PostgreSQL | Provision a production-grade RDS with Multi-AZ, read replicas, backup strategy, and secrets rotation. | Relational DB, HA/DR, security | [Full guide](topics/aws/05-rds-database.md) |
| 6 | Network Load Balancer | Design a high-throughput, low-latency NLB setup for TCP/UDP workloads with cross-zone balancing and failover. | L4 load balancing, resilience | Coming soon |
| 7 | VPC Architecture | Create a three-tier VPC (public, private, data), NAT gateway, endpoints, and network ACL strategy. | Networking, security, routing | Coming soon |
| 8 | ECS/Fargate Deployments | Ship a containerized microservice with Fargate tasks, service discovery, and blue/green rollouts. | Containers, orchestration, deployment | Coming soon |
| 9 | AWS Step Functions | Orchestrate a data processing workflow using Lambda, SQS, and Step Functions with retries and alerts. | Workflow orchestration, automation | Coming soon |
| 10 | AWS Cost Governance | Implement cost visibility with tagging, budgets, anomaly detection, and cleanup automation. | FinOps, automation, governance | Coming soon |

---

## ⚙️ GitHub Actions Problem Bank

| # | Scenario | Problem Statement | Skills & Focus | Link |
|---|----------|-------------------|----------------|------|
| 1 | Starter Workflow | Trigger CI on pushes and PRs, printing context, environment vars, and actor metadata. | Workflow syntax, triggers | [Full guide](topics/github-actions/01-basic-workflow.md) |
| 2 | Node.js CI Matrix | Run lint/test/build across Node versions with caching, artifacts, and coverage uploads. | Matrix builds, caching, reporting | [Full guide](topics/github-actions/02-nodejs-ci.md) |
| 3 | Docker Build & Push | Build multi-arch images, scan vulnerabilities, and publish to Docker Hub & GHCR with caching. | Containers, registries, security | [Full guide](topics/github-actions/03-docker-workflow.md) |
| 4 | Multi-Environment Deploy | Implement dev → staging → prod pipeline with approvals, blue/green strategy, and rollback. | Environments, approvals, infra deploy | [Full guide](topics/github-actions/04-multi-environment.md) |
| 5 | Monorepo Matrix | Dynamically detect changed packages and run selective builds with reusable workflows. | Monorepos, dynamic matrices | Coming soon |
| 6 | Secrets Rotation | Automate credential rotation and re-encryption using GitHub Actions + AWS Secrets Manager. | Security, automation | Coming soon |
| 7 | Infrastructure Drift Detection | Schedule Terraform plan checks and alert on drift using Actions + Terraform Cloud. | IaC, compliance | Coming soon |
| 8 | Performance Budgets | Fail builds when Lighthouse or bundle-size budgets are exceeded via GitHub Actions. | Web performance, CI gates | Coming soon |
| 9 | Incident Runbooks | Build ChatOps workflow to page on-call, create incident issues, and post to Slack. | Incident response, integrations | Coming soon |
| 10 | Release Automation | Automate semantic versioning, changelog generation, and release artifacts with approvals. | Release engineering, automation | Coming soon |

---

## 🧭 How to Use This Bank

1. **Pick your path:** choose AWS or GitHub Actions scenarios based on your goals.
2. **Start with fundamentals:** complete the “Full guide” problems before tackling “Coming soon” challenges on your own.
3. **Level up with extensions:** each full guide contains advanced challenge blocks with hints and resources.
4. **Track mastery:** revisit the problem bank periodically—new scenarios will unlock over time.
5. **Suggest additions:** have a production scenario you want to practice? Open an issue or drop a note.

> 💡 Tip: Combine AWS infrastructure problems with GitHub Actions automation for end-to-end DevOps practice.
