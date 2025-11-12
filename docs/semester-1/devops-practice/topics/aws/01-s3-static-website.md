# Problem 1: Deploy Static Website to S3 with CloudFront 🟢

## 📋 Problem Statement

Deploy a static website to AWS S3 and set up CloudFront CDN for global content delivery. Your solution should:
- Create an S3 bucket configured for static website hosting
- Upload website files to S3
- Set up CloudFront distribution
- Configure custom error pages
- Enable HTTPS

**Difficulty**: Beginner 🟢

---

## 🎯 Learning Objectives

- Understand S3 static website hosting
- Configure bucket policies for public access
- Set up CloudFront distribution
- Work with AWS CLI
- Understand CDN concepts

---

## 📚 Background

Amazon S3 can host static websites (HTML, CSS, JavaScript) efficiently and cheaply. CloudFront is AWS's Content Delivery Network (CDN) that caches your content globally, providing faster access to users worldwide.

---

## 💡 Hints

<details>
<summary>Hint 1: S3 Bucket Configuration</summary>

Enable static website hosting:

```bash
aws s3 website s3://your-bucket-name/ \
  --index-document index.html \
  --error-document error.html
```

</details>

<details>
<summary>Hint 2: Bucket Policy for Public Access</summary>

Make bucket publicly readable:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::your-bucket-name/*"
  }]
}
```

</details>

<details>
<summary>Hint 3: Sync Files to S3</summary>

Upload files with AWS CLI:

```bash
aws s3 sync ./dist s3://your-bucket-name --delete
```

</details>

<details>
<summary>Hint 4: Create CloudFront Distribution</summary>

Use AWS Console or CLI to create a distribution pointing to your S3 bucket.

</details>

---

## ✅ Solution

<details>
<summary>Click to view the complete solution</summary>

### Step 1: Create Sample Static Website

**index.html:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My AWS Static Site</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>Welcome to My AWS-Hosted Website! 🚀</h1>
        <p>This site is hosted on S3 and delivered via CloudFront CDN.</p>
        <div class="stats">
            <div class="stat">
                <h3>Fast</h3>
                <p>Global CDN delivery</p>
            </div>
            <div class="stat">
                <h3>Secure</h3>
                <p>HTTPS enabled</p>
            </div>
            <div class="stat">
                <h3>Scalable</h3>
                <p>Auto-scaling infrastructure</p>
            </div>
        </div>
    </div>
    <script src="script.js"></script>
</body>
</html>
```

**style.css:**
```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
}

.container {
    text-align: center;
    padding: 2rem;
    max-width: 800px;
}

h1 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.stats {
    display: flex;
    gap: 2rem;
    margin-top: 3rem;
    justify-content: center;
}

.stat {
    background: rgba(255, 255, 255, 0.1);
    padding: 2rem;
    border-radius: 10px;
    backdrop-filter: blur(10px);
}

.stat h3 {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}
```

**script.js:**
```javascript
console.log('Website loaded successfully!');
document.addEventListener('DOMContentLoaded', () => {
    console.log('AWS S3 + CloudFront deployment');
});
```

**error.html:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>404 - Page Not Found</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 50px;
            background-color: #f0f0f0;
        }
        h1 { color: #e74c3c; }
        a { color: #3498db; text-decoration: none; }
    </style>
</head>
<body>
    <h1>404 - Page Not Found</h1>
    <p>The page you're looking for doesn't exist.</p>
    <a href="/">Go back home</a>
</body>
</html>
```

### Step 2: Configure AWS CLI

Install and configure AWS CLI:

```bash
# Install AWS CLI (macOS)
brew install awscli

# Configure credentials
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (e.g., us-east-1)
# - Default output format (json)
```

### Step 3: Create S3 Bucket

```bash
# Set variables
BUCKET_NAME="my-static-website-$(date +%s)"
REGION="us-east-1"

# Create bucket
aws s3 mb s3://$BUCKET_NAME --region $REGION

# Enable static website hosting
aws s3 website s3://$BUCKET_NAME/ \
    --index-document index.html \
    --error-document error.html
```

### Step 4: Configure Bucket Policy

Create `bucket-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::BUCKET_NAME/*"
    }
  ]
}
```

Replace `BUCKET_NAME` and apply:

```bash
# Replace bucket name in policy
sed "s/BUCKET_NAME/$BUCKET_NAME/g" bucket-policy.json > temp-policy.json

# Apply bucket policy
aws s3api put-bucket-policy \
    --bucket $BUCKET_NAME \
    --policy file://temp-policy.json

# Disable block public access
aws s3api put-public-access-block \
    --bucket $BUCKET_NAME \
    --public-access-block-configuration \
    BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false
```

### Step 5: Upload Website Files

```bash
# Upload files
aws s3 sync . s3://$BUCKET_NAME \
    --exclude "*.json" \
    --exclude ".git/*" \
    --exclude "README.md"

# Set cache control headers
aws s3 cp s3://$BUCKET_NAME s3://$BUCKET_NAME \
    --recursive \
    --metadata-directive REPLACE \
    --cache-control "max-age=86400"
```

### Step 6: Create CloudFront Distribution

Create `cloudfront-config.json`:

```json
{
  "Comment": "Static website distribution",
  "Enabled": true,
  "Origins": {
    "Quantity": 1,
    "Items": [
      {
        "Id": "S3-Website",
        "DomainName": "BUCKET_NAME.s3-website-REGION.amazonaws.com",
        "CustomOriginConfig": {
          "HTTPPort": 80,
          "HTTPSPort": 443,
          "OriginProtocolPolicy": "http-only"
        }
      }
    ]
  },
  "DefaultCacheBehavior": {
    "TargetOriginId": "S3-Website",
    "ViewerProtocolPolicy": "redirect-to-https",
    "AllowedMethods": {
      "Quantity": 2,
      "Items": ["GET", "HEAD"]
    },
    "ForwardedValues": {
      "QueryString": false,
      "Cookies": {
        "Forward": "none"
      }
    },
    "MinTTL": 0,
    "DefaultTTL": 86400,
    "MaxTTL": 31536000
  },
  "DefaultRootObject": "index.html",
  "CustomErrorResponses": {
    "Quantity": 1,
    "Items": [
      {
        "ErrorCode": 404,
        "ResponsePagePath": "/error.html",
        "ResponseCode": "404",
        "ErrorCachingMinTTL": 300
      }
    ]
  }
}
```

Apply configuration:

```bash
# Create distribution (simplified command)
aws cloudfront create-distribution \
    --origin-domain-name $BUCKET_NAME.s3-website-$REGION.amazonaws.com \
    --default-root-object index.html

# Get distribution domain name
DISTRIBUTION_ID=$(aws cloudfront list-distributions \
    --query "DistributionList.Items[0].Id" --output text)

echo "Distribution ID: $DISTRIBUTION_ID"
echo "Wait for deployment (15-20 minutes)"
```

### Step 7: Test Your Website

```bash
# Get CloudFront domain
CLOUDFRONT_DOMAIN=$(aws cloudfront get-distribution \
    --id $DISTRIBUTION_ID \
    --query "Distribution.DomainName" --output text)

echo "Your website is available at: https://$CLOUDFRONT_DOMAIN"

# Test with curl
curl https://$CLOUDFRONT_DOMAIN
```

### Step 8: Automation Script

Create `deploy.sh`:

```bash
#!/bin/bash

BUCKET_NAME="my-static-website-$(date +%s)"
REGION="us-east-1"

echo "Creating S3 bucket: $BUCKET_NAME"
aws s3 mb s3://$BUCKET_NAME --region $REGION

echo "Configuring static website hosting"
aws s3 website s3://$BUCKET_NAME/ \
    --index-document index.html \
    --error-document error.html

echo "Setting bucket policy"
cat > temp-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublicReadGetObject",
    "Effect": "Allow",
    "Principal": "*",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::$BUCKET_NAME/*"
  }]
}
EOF

aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://temp-policy.json

echo "Uploading files"
aws s3 sync . s3://$BUCKET_NAME --exclude "*.sh" --exclude "*.json"

echo "Website URL: http://$BUCKET_NAME.s3-website-$REGION.amazonaws.com"
```

</details>

---

## 🔍 Explanation

### Key Concepts:

1. **S3 Static Website Hosting**: Serves static files with high availability
2. **Bucket Policy**: JSON-based access control for S3 resources
3. **CloudFront Distribution**: CDN that caches content at edge locations worldwide
4. **Origin**: Source of content (S3 bucket in this case)
5. **Cache Behavior**: Rules for caching and serving content
6. **TTL (Time To Live)**: How long content is cached

### Why Use CloudFront?
- **Performance**: Content served from nearest edge location
- **Security**: HTTPS by default, DDoS protection
- **Cost**: Reduced S3 data transfer costs
- **Global**: 400+ edge locations worldwide

---

## 🚀 Extensions & Challenges

<details>
<summary>Challenge 1: Add Custom Domain with Route 53</summary>

```bash
# Create hosted zone
aws route53 create-hosted-zone \
    --name example.com \
    --caller-reference $(date +%s)

# Create record set for CloudFront
aws route53 change-resource-record-sets \
    --hosted-zone-id YOUR_ZONE_ID \
    --change-batch file://dns-records.json
```

</details>

<details>
<summary>Challenge 2: Enable Access Logging</summary>

```bash
# Create log bucket
aws s3 mb s3://logs-$BUCKET_NAME

# Enable logging
aws s3api put-bucket-logging \
    --bucket $BUCKET_NAME \
    --bucket-logging-status file://logging-config.json
```

</details>

<details>
<summary>Challenge 3: Automate with GitHub Actions</summary>

Create `.github/workflows/deploy-s3.yml`:

```yaml
name: Deploy to S3

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Sync to S3
        run: |
          aws s3 sync . s3://${{ secrets.S3_BUCKET }} --delete
          
      - name: Invalidate CloudFront
        run: |
          aws cloudfront create-invalidation \
            --distribution-id ${{ secrets.CLOUDFRONT_ID }} \
            --paths "/*"
```

</details>

---

## 📖 Related Articles

- [Hosting a Static Website on S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/WebsiteHosting.html)
- [CloudFront Documentation](https://docs.aws.amazon.com/cloudfront/)
- [AWS CLI S3 Commands](https://docs.aws.amazon.com/cli/latest/reference/s3/)

---

## ✨ Best Practices

- ✅ Enable versioning on S3 bucket
- ✅ Use CloudFront for better performance and security
- ✅ Set appropriate cache headers
- ✅ Enable access logging
- ✅ Use meaningful bucket names
- ✅ Implement proper IAM policies
- ✅ Enable HTTPS with CloudFront

---

**Next**: [Deploy Node.js App to EC2 →](02-ec2-deployment.md)
