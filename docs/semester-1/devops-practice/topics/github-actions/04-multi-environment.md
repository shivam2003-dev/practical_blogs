# Problem 4: Multi-Environment Deployment Pipeline 🔴

## 📋 Problem Statement

Create an advanced deployment pipeline that:
- Deploys to multiple environments (dev, staging, production)
- Uses environment-specific configurations
- Requires manual approval for production
- Implements blue-green deployment strategy
- Includes automated rollback on failure
- Sends notifications to Slack/Discord

**Difficulty**: Advanced 🔴

---

## 🎯 Learning Objectives

- Implement environment-based deployments
- Use GitHub Environments and protection rules
- Create reusable workflows
- Implement approval gates
- Handle secrets per environment
- Set up deployment notifications

---

## 📚 Background

Production-grade CI/CD pipelines require careful orchestration across multiple environments. This ensures changes are thoroughly tested before reaching production, with safety mechanisms like manual approvals and automated rollbacks.

---

## 💡 Hints

<details>
<summary>Hint 1: GitHub Environments</summary>

Define environments in repository settings or workflow:

```yaml
jobs:
  deploy:
    environment:
      name: production
      url: https://prod.example.com
```

</details>

<details>
<summary>Hint 2: Approval Gates</summary>

Set up environment protection rules in GitHub:
- Settings → Environments → protection rules
- Required reviewers for production

</details>

<details>
<summary>Hint 3: Reusable Workflows</summary>

Create reusable deployment workflow:

```yaml
# .github/workflows/deploy-reusable.yml
on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string
```

</details>

<details>
<summary>Hint 4: Deployment Strategy</summary>

Use deployment slots or tags for blue-green:

```yaml
- name: Deploy to blue slot
  run: |
    # Deploy new version
    # Test blue slot
    # Swap blue and green
```

</details>

---

## ✅ Solution

<details>
<summary>Click to view the complete solution</summary>

### Step 1: Set Up Repository Environments

Go to GitHub Repository → Settings → Environments and create:

1. **Development**
   - No protection rules
   - Secrets: `DEV_API_URL`, `DEV_DATABASE_URL`

2. **Staging**
   - Required reviewers: 1 reviewer
   - Secrets: `STAGING_API_URL`, `STAGING_DATABASE_URL`

3. **Production**
   - Required reviewers: 2 reviewers
   - Wait timer: 5 minutes
   - Limit to protected branches: `main` only
   - Secrets: `PROD_API_URL`, `PROD_DATABASE_URL`

### Step 2: Create Reusable Deployment Workflow

**.github/workflows/deploy-reusable.yml:**
```yaml
name: Reusable Deploy Workflow

on:
  workflow_call:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: string
      app-version:
        description: 'Application version to deploy'
        required: true
        type: string
    secrets:
      api-url:
        required: true
      database-url:
        required: true
      slack-webhook:
        required: false
    outputs:
      deployment-url:
        description: "URL of the deployed application"
        value: ${{ jobs.deploy.outputs.url }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: ${{ inputs.environment }}
      url: ${{ steps.deploy.outputs.url }}
    
    outputs:
      url: ${{ steps.deploy.outputs.url }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Build application
        env:
          API_URL: ${{ secrets.api-url }}
          DATABASE_URL: ${{ secrets.database-url }}
          VERSION: ${{ inputs.app-version }}
        run: |
          echo "Building for ${{ inputs.environment }}"
          npm run build
          echo "VERSION=${{ inputs.app-version }}" > dist/version.txt

      - name: Run smoke tests
        run: npm run test:smoke

      - name: Deploy to ${{ inputs.environment }}
        id: deploy
        run: |
          # Simulate deployment
          DEPLOY_URL="https://${{ inputs.environment }}.example.com"
          echo "Deploying to $DEPLOY_URL"
          
          # In real scenario, use deployment tools:
          # - AWS CLI, Terraform, Ansible
          # - Kubernetes kubectl
          # - Cloud provider CLIs
          
          echo "url=$DEPLOY_URL" >> $GITHUB_OUTPUT

      - name: Health check
        run: |
          echo "Running health check on ${{ steps.deploy.outputs.url }}"
          # curl -f ${{ steps.deploy.outputs.url }}/health || exit 1

      - name: Notify Slack
        if: always() && secrets.slack-webhook != ''
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: |
            Deployment to ${{ inputs.environment }}
            Version: ${{ inputs.app-version }}
            URL: ${{ steps.deploy.outputs.url }}
          webhook_url: ${{ secrets.slack-webhook }}
```

### Step 3: Create Main Pipeline Workflow

**.github/workflows/pipeline.yml:**
```yaml
name: Multi-Environment Pipeline

on:
  push:
    branches:
      - main
      - develop
  pull_request:
    branches:
      - main

env:
  APP_VERSION: ${{ github.sha }}

jobs:
  # Build and Test
  build:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.version }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Lint code
        run: npm run lint

      - name: Run unit tests
        run: npm run test:unit

      - name: Run integration tests
        run: npm run test:integration

      - name: Build application
        run: npm run build

      - name: Generate version
        id: version
        run: |
          VERSION="${GITHUB_SHA:0:7}"
          echo "version=$VERSION" >> $GITHUB_OUTPUT

      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: build-${{ steps.version.outputs.version }}
          path: |
            dist/
            package.json
          retention-days: 30

  # Deploy to Development
  deploy-dev:
    needs: build
    if: github.ref == 'refs/heads/develop' || github.event_name == 'pull_request'
    uses: ./.github/workflows/deploy-reusable.yml
    with:
      environment: development
      app-version: ${{ needs.build.outputs.version }}
    secrets:
      api-url: ${{ secrets.DEV_API_URL }}
      database-url: ${{ secrets.DEV_DATABASE_URL }}
      slack-webhook: ${{ secrets.SLACK_WEBHOOK }}

  # Deploy to Staging
  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/main'
    uses: ./.github/workflows/deploy-reusable.yml
    with:
      environment: staging
      app-version: ${{ needs.build.outputs.version }}
    secrets:
      api-url: ${{ secrets.STAGING_API_URL }}
      database-url: ${{ secrets.STAGING_DATABASE_URL }}
      slack-webhook: ${{ secrets.SLACK_WEBHOOK }}

  # Run E2E tests on Staging
  e2e-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Run E2E tests
        env:
          BASE_URL: ${{ needs.deploy-staging.outputs.deployment-url }}
        run: |
          npm ci
          npm run test:e2e

  # Deploy to Production (requires approval)
  deploy-production:
    needs: [build, deploy-staging, e2e-tests]
    if: github.ref == 'refs/heads/main' && success()
    uses: ./.github/workflows/deploy-reusable.yml
    with:
      environment: production
      app-version: ${{ needs.build.outputs.version }}
    secrets:
      api-url: ${{ secrets.PROD_API_URL }}
      database-url: ${{ secrets.PROD_DATABASE_URL }}
      slack-webhook: ${{ secrets.SLACK_WEBHOOK }}

  # Post-deployment verification
  verify-production:
    needs: deploy-production
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Production smoke tests
        env:
          BASE_URL: ${{ needs.deploy-production.outputs.deployment-url }}
        run: |
          echo "Running production smoke tests"
          npm ci
          npm run test:smoke:production

      - name: Monitor for 5 minutes
        run: |
          echo "Monitoring application health..."
          for i in {1..10}; do
            echo "Health check $i/10"
            # curl -f ${BASE_URL}/health
            sleep 30
          done

  # Rollback on failure
  rollback:
    needs: [deploy-production, verify-production]
    if: failure()
    runs-on: ubuntu-latest
    environment:
      name: production
    
    steps:
      - name: Notify about failure
        uses: 8398a7/action-slack@v3
        with:
          status: failure
          text: '⚠️ Production deployment failed! Initiating rollback...'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}

      - name: Rollback to previous version
        run: |
          echo "Rolling back production deployment"
          # Implement rollback logic
          # - Revert to previous image tag
          # - Restore from backup
          # - Switch traffic back to previous version

      - name: Verify rollback
        run: |
          echo "Verifying rollback was successful"
          # curl -f https://prod.example.com/health

      - name: Notify rollback complete
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Rollback completed'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Step 4: Blue-Green Deployment Enhancement

**.github/workflows/blue-green-deploy.yml:**
```yaml
name: Blue-Green Deployment

on:
  workflow_call:
    inputs:
      environment:
        required: true
        type: string

jobs:
  deploy-blue-green:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Determine current slot
        id: current
        run: |
          # Check which slot is currently active
          ACTIVE_SLOT=$(curl -s https://${{ inputs.environment }}.example.com/api/slot)
          if [ "$ACTIVE_SLOT" == "blue" ]; then
            TARGET_SLOT="green"
          else
            TARGET_SLOT="blue"
          fi
          echo "target=$TARGET_SLOT" >> $GITHUB_OUTPUT
          echo "Deploying to $TARGET_SLOT slot"

      - name: Deploy to inactive slot
        run: |
          echo "Deploying to ${{ steps.current.outputs.target }} slot"
          # Deploy application to inactive slot
          # Example with AWS:
          # aws deploy create-deployment \
          #   --application-name myapp \
          #   --deployment-group-name ${{ steps.current.outputs.target }}

      - name: Run tests on new slot
        run: |
          TARGET_URL="https://${{ steps.current.outputs.target }}.${{ inputs.environment }}.example.com"
          echo "Testing $TARGET_URL"
          # Run comprehensive tests
          # curl -f $TARGET_URL/health

      - name: Warm up new slot
        run: |
          echo "Warming up ${{ steps.current.outputs.target }} slot"
          # Send requests to warm up caches
          for i in {1..100}; do
            curl -s $TARGET_URL > /dev/null
          done

      - name: Switch traffic to new slot
        run: |
          echo "Switching traffic to ${{ steps.current.outputs.target }}"
          # Update load balancer or DNS
          # Example:
          # aws elbv2 modify-rule \
          #   --rule-arn $RULE_ARN \
          #   --actions Type=forward,TargetGroupArn=$NEW_TARGET_GROUP

      - name: Monitor new slot
        run: |
          echo "Monitoring for 5 minutes"
          for i in {1..10}; do
            # Check error rates, response times
            sleep 30
          done

      - name: Finalize deployment
        run: |
          echo "Deployment successful"
          # Update deployment tags
          # Keep old slot for quick rollback
```

### Step 5: Package.json Scripts

Add these scripts:

```json
{
  "scripts": {
    "build": "webpack --mode production",
    "lint": "eslint . --ext .js",
    "test:unit": "jest --coverage",
    "test:integration": "jest --config jest.integration.config.js",
    "test:e2e": "cypress run",
    "test:smoke": "jest --config jest.smoke.config.js",
    "test:smoke:production": "BASE_URL=https://prod.example.com npm run test:smoke"
  }
}
```

</details>

---

## 🔍 Explanation

### Pipeline Flow:

1. **Build Stage**: Lint, test, and build application
2. **Dev Deployment**: Automatic on feature branches
3. **Staging Deployment**: Automatic on main branch
4. **E2E Tests**: Run comprehensive tests on staging
5. **Production Deployment**: Requires manual approval
6. **Verification**: Monitor production health
7. **Rollback**: Automatic if verification fails

### Key Features:

- **Reusable Workflows**: DRY principle for deployments
- **Environment Protection**: Approval gates for production
- **Blue-Green Strategy**: Zero-downtime deployments
- **Automated Rollback**: Quick recovery from failures
- **Comprehensive Testing**: Unit, integration, E2E, smoke tests
- **Notifications**: Keep team informed via Slack

---

## 📖 Related Articles

- [Using Environments for Deployment](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
- [Reusable Workflows](https://docs.github.com/en/actions/using-workflows/reusing-workflows)
- [Blue-Green Deployments](https://martinfowler.com/bliki/BlueGreenDeployment.html)

---

## ✨ Best Practices

- ✅ Always test in staging before production
- ✅ Use manual approval gates for production
- ✅ Implement automated rollback mechanisms
- ✅ Monitor deployments closely
- ✅ Keep deployment artifacts for quick rollback
- ✅ Send notifications for all deployment events
- ✅ Use environment-specific configurations

---

**Previous**: [← Docker Workflow](03-docker-workflow.md)
