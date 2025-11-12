# Problem 2: Deploy Node.js Application to EC2 with Auto-Scaling 🟡

## 📋 Problem Statement

Deploy a Node.js application to AWS EC2 with:
- Launch Template configuration
- Auto Scaling Group (ASG) for handling traffic
- Application Load Balancer (ALB)
- Security groups for proper access control
- User Data script for automatic application setup
- Health checks and monitoring

**Difficulty**: Intermediate 🟡

---

## 🎯 Learning Objectives

- Launch and configure EC2 instances
- Set up Application Load Balancer
- Create Auto Scaling Groups
- Configure Security Groups
- Use User Data for instance initialization
- Implement health checks

---

## 📚 Background

EC2 (Elastic Compute Cloud) provides scalable computing capacity. Combined with Auto Scaling and Load Balancing, you can create highly available and fault-tolerant applications that automatically scale based on demand.

---

## 💡 Hints

<details>
<summary>Hint 1: Security Group Configuration</summary>

Create security groups for ALB and EC2:

```bash
# ALB security group (allow HTTP/HTTPS)
aws ec2 create-security-group \
    --group-name alb-sg \
    --description "Security group for ALB"

# EC2 security group (allow traffic from ALB)
aws ec2 create-security-group \
    --group-name ec2-sg \
    --description "Security group for EC2 instances"
```

</details>

<details>
<summary>Hint 2: User Data Script</summary>

Automate application setup:

```bash
#!/bin/bash
yum update -y
curl -sL https://rpm.nodesource.com/setup_18.x | bash -
yum install -y nodejs
```

</details>

<details>
<summary>Hint 3: Target Group for Load Balancer</summary>

Create target group for health checks:

```bash
aws elbv2 create-target-group \
    --name my-targets \
    --protocol HTTP \
    --port 3000 \
    --vpc-id vpc-xxxxx \
    --health-check-path /health
```

</details>

---

## ✅ Solution

<details>
<summary>Click to view the complete solution</summary>

### Step 1: Create Sample Node.js Application

**app.js:**
```javascript
const express = require('express');
const os = require('os');

const app = express();
const PORT = process.env.PORT || 3000;

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        hostname: os.hostname()
    });
});

// Main endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'Hello from EC2!',
        instance: os.hostname(),
        platform: os.platform(),
        uptime: os.uptime()
    });
});

// Load test endpoint
app.get('/api/data', (req, res) => {
    const data = Array.from({ length: 1000 }, (_, i) => ({
        id: i,
        value: Math.random()
    }));
    res.json(data);
});

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Instance: ${os.hostname()}`);
});
```

**package.json:**
```json
{
  "name": "ec2-nodejs-app",
  "version": "1.0.0",
  "description": "Node.js app for EC2 deployment",
  "main": "app.js",
  "scripts": {
    "start": "node app.js",
    "dev": "nodemon app.js"
  },
  "dependencies": {
    "express": "^4.18.2"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
```

### Step 2: Create Deployment Package

```bash
# Create deployment directory
mkdir deployment
cp app.js package.json deployment/

# Create startup script
cat > deployment/start-app.sh <<'EOF'
#!/bin/bash
cd /home/ec2-user/app
npm install
npm start
EOF

chmod +x deployment/start-app.sh

# Create systemd service
cat > deployment/nodeapp.service <<'EOF'
[Unit]
Description=Node.js Application
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/app
ExecStart=/usr/bin/node app.js
Restart=on-failure
Environment=NODE_ENV=production
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
EOF
```

### Step 3: Create User Data Script

**user-data.sh:**
```bash
#!/bin/bash
set -e

# Update system
yum update -y

# Install Node.js 18
curl -sL https://rpm.nodesource.com/setup_18.x | bash -
yum install -y nodejs git

# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm

# Create app directory
mkdir -p /home/ec2-user/app
cd /home/ec2-user/app

# Clone or download application
cat > app.js <<'APPEOF'
const express = require('express');
const os = require('os');

const app = express();
const PORT = process.env.PORT || 3000;

app.get('/health', (req, res) => {
    res.status(200).json({ status: 'healthy', hostname: os.hostname() });
});

app.get('/', (req, res) => {
    res.json({
        message: 'Hello from EC2!',
        instance: os.hostname(),
        region: process.env.AWS_REGION
    });
});

app.listen(PORT, () => console.log(`Server on port ${PORT}`));
APPEOF

cat > package.json <<'PKGEOF'
{
  "name": "ec2-app",
  "version": "1.0.0",
  "dependencies": {
    "express": "^4.18.2"
  }
}
PKGEOF

# Install dependencies
npm install

# Create systemd service
cat > /etc/systemd/system/nodeapp.service <<'SVCEOF'
[Unit]
Description=Node.js Application
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/app
ExecStart=/usr/bin/node app.js
Restart=always
Environment=NODE_ENV=production
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
SVCEOF

# Change ownership
chown -R ec2-user:ec2-user /home/ec2-user/app

# Enable and start service
systemctl daemon-reload
systemctl enable nodeapp
systemctl start nodeapp

# Log completion
echo "Application deployed successfully" >> /var/log/user-data.log
```

### Step 4: Set Up AWS Infrastructure

**setup-infrastructure.sh:**
```bash
#!/bin/bash

# Configuration
REGION="us-east-1"
AMI_ID="ami-0c55b159cbfafe1f0"  # Amazon Linux 2 (update for your region)
INSTANCE_TYPE="t3.micro"
KEY_NAME="my-ec2-key"

# Get default VPC
VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" \
    --output text \
    --region $REGION)

echo "Using VPC: $VPC_ID"

# Get subnets
SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$VPC_ID" \
    --query "Subnets[*].SubnetId" \
    --output text \
    --region $REGION)

echo "Using Subnets: $SUBNET_IDS"

# Create Security Group for ALB
ALB_SG=$(aws ec2 create-security-group \
    --group-name alb-security-group \
    --description "Security group for Application Load Balancer" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

echo "Created ALB Security Group: $ALB_SG"

# Allow HTTP traffic to ALB
aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Create Security Group for EC2
EC2_SG=$(aws ec2 create-security-group \
    --group-name ec2-security-group \
    --description "Security group for EC2 instances" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

echo "Created EC2 Security Group: $EC2_SG"

# Allow traffic from ALB to EC2
aws ec2 authorize-security-group-ingress \
    --group-id $EC2_SG \
    --protocol tcp \
    --port 3000 \
    --source-group $ALB_SG \
    --region $REGION

# Allow SSH (optional, for debugging)
aws ec2 authorize-security-group-ingress \
    --group-id $EC2_SG \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Create Target Group
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name nodejs-targets \
    --protocol HTTP \
    --port 3000 \
    --vpc-id $VPC_ID \
    --health-check-enabled \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --region $REGION \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

echo "Created Target Group: $TARGET_GROUP_ARN"

# Create Application Load Balancer
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name nodejs-alb \
    --subnets $SUBNET_IDS \
    --security-groups $ALB_SG \
    --region $REGION \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

echo "Created ALB: $ALB_ARN"

# Create Listener
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN \
    --region $REGION

# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --query 'LoadBalancers[0].DNSName' \
    --output text \
    --region $REGION)

echo "ALB DNS: $ALB_DNS"

# Create Launch Template
LAUNCH_TEMPLATE_ID=$(aws ec2 create-launch-template \
    --launch-template-name nodejs-launch-template \
    --version-description "Initial version" \
    --launch-template-data "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SecurityGroupIds\": [\"$EC2_SG\"],
        \"UserData\": \"$(base64 -i user-data.sh)\",
        \"TagSpecifications\": [{
            \"ResourceType\": \"instance\",
            \"Tags\": [{\"Key\": \"Name\", \"Value\": \"nodejs-app-instance\"}]
        }]
    }" \
    --region $REGION \
    --query 'LaunchTemplate.LaunchTemplateId' \
    --output text)

echo "Created Launch Template: $LAUNCH_TEMPLATE_ID"

# Create Auto Scaling Group
aws autoscaling create-auto-scaling-group \
    --auto-scaling-group-name nodejs-asg \
    --launch-template LaunchTemplateId=$LAUNCH_TEMPLATE_ID \
    --min-size 2 \
    --max-size 4 \
    --desired-capacity 2 \
    --target-group-arns $TARGET_GROUP_ARN \
    --health-check-type ELB \
    --health-check-grace-period 300 \
    --vpc-zone-identifier "${SUBNET_IDS// /,}" \
    --region $REGION

echo "Created Auto Scaling Group"

# Create Scaling Policies
# Scale up policy
SCALE_UP_POLICY=$(aws autoscaling put-scaling-policy \
    --auto-scaling-group-name nodejs-asg \
    --policy-name scale-up \
    --scaling-adjustment 1 \
    --adjustment-type ChangeInCapacity \
    --cooldown 300 \
    --region $REGION \
    --query 'PolicyARN' \
    --output text)

# Scale down policy
SCALE_DOWN_POLICY=$(aws autoscaling put-scaling-policy \
    --auto-scaling-group-name nodejs-asg \
    --policy-name scale-down \
    --scaling-adjustment -1 \
    --adjustment-type ChangeInCapacity \
    --cooldown 300 \
    --region $REGION \
    --query 'PolicyARN' \
    --output text)

# Create CloudWatch Alarms
aws cloudwatch put-metric-alarm \
    --alarm-name high-cpu \
    --alarm-description "Scale up if CPU > 70%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 70 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $SCALE_UP_POLICY \
    --dimensions Name=AutoScalingGroupName,Value=nodejs-asg \
    --region $REGION

aws cloudwatch put-metric-alarm \
    --alarm-name low-cpu \
    --alarm-description "Scale down if CPU < 30%" \
    --metric-name CPUUtilization \
    --namespace AWS/EC2 \
    --statistic Average \
    --period 300 \
    --threshold 30 \
    --comparison-operator LessThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $SCALE_DOWN_POLICY \
    --dimensions Name=AutoScalingGroupName,Value=nodejs-asg \
    --region $REGION

echo "=========================================="
echo "Deployment Complete!"
echo "ALB DNS: http://$ALB_DNS"
echo "Wait 5-10 minutes for instances to be healthy"
echo "=========================================="

# Save configuration
cat > deployment-config.txt <<EOF
VPC_ID=$VPC_ID
ALB_SG=$ALB_SG
EC2_SG=$EC2_SG
TARGET_GROUP_ARN=$TARGET_GROUP_ARN
ALB_ARN=$ALB_ARN
ALB_DNS=$ALB_DNS
LAUNCH_TEMPLATE_ID=$LAUNCH_TEMPLATE_ID
SCALE_UP_POLICY=$SCALE_UP_POLICY
SCALE_DOWN_POLICY=$SCALE_DOWN_POLICY
EOF
```

### Step 5: Test the Deployment

```bash
# Run setup script
chmod +x setup-infrastructure.sh
./setup-infrastructure.sh

# Wait for instances to be healthy (5-10 minutes)
# Then test
ALB_DNS=$(cat deployment-config.txt | grep ALB_DNS | cut -d'=' -f2)

# Test health endpoint
curl http://$ALB_DNS/health

# Test main endpoint
curl http://$ALB_DNS/

# Load test (generate traffic for auto-scaling)
for i in {1..1000}; do
    curl -s http://$ALB_DNS/api/data > /dev/null &
done
```

### Step 6: Cleanup Script

**cleanup.sh:**
```bash
#!/bin/bash

REGION="us-east-1"

# Load configuration
source deployment-config.txt

echo "Deleting Auto Scaling Group..."
aws autoscaling delete-auto-scaling-group \
    --auto-scaling-group-name nodejs-asg \
    --force-delete \
    --region $REGION

echo "Waiting for instances to terminate..."
sleep 60

echo "Deleting Launch Template..."
aws ec2 delete-launch-template \
    --launch-template-id $LAUNCH_TEMPLATE_ID \
    --region $REGION

echo "Deleting ALB..."
aws elbv2 delete-load-balancer \
    --load-balancer-arn $ALB_ARN \
    --region $REGION

sleep 30

echo "Deleting Target Group..."
aws elbv2 delete-target-group \
    --target-group-arn $TARGET_GROUP_ARN \
    --region $REGION

echo "Deleting Security Groups..."
aws ec2 delete-security-group --group-id $EC2_SG --region $REGION
aws ec2 delete-security-group --group-id $ALB_SG --region $REGION

echo "Cleanup complete!"
```

</details>

---

## 🔍 Explanation

### Key Components:

1. **EC2 Instances**: Virtual servers running your application
2. **Launch Template**: Blueprint for EC2 instances
3. **Auto Scaling Group**: Manages instance count based on policies
4. **Application Load Balancer**: Distributes traffic across instances
5. **Target Group**: Routes requests to healthy instances
6. **Security Groups**: Firewall rules for network access
7. **User Data**: Initialization script for instances
8. **CloudWatch Alarms**: Trigger scaling based on metrics

### Auto Scaling Strategy:
- **Min Size**: 2 (always maintain 2 instances)
- **Max Size**: 4 (scale up to 4 under load)
- **Desired**: 2 (normal operating capacity)
- **Scale Up**: When CPU > 70% for 10 minutes
- **Scale Down**: When CPU < 30% for 10 minutes

---

## 📖 Related Articles

- [EC2 User Guide](https://docs.aws.amazon.com/ec2/)
- [Auto Scaling Documentation](https://docs.aws.amazon.com/autoscaling/)
- [Application Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/)

---

## ✨ Best Practices

- ✅ Use Launch Templates instead of Launch Configurations
- ✅ Implement health checks for automatic recovery
- ✅ Enable detailed monitoring
- ✅ Use multiple availability zones
- ✅ Set appropriate scaling policies
- ✅ Use Systems Manager for secure access
- ✅ Tag resources properly

---

**Previous**: [← S3 Static Website](01-s3-static-website.md) | **Next**: [Serverless with Lambda →](03-lambda-api.md)
