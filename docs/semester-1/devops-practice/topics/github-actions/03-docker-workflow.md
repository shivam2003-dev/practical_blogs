# Problem 3: Docker Build and Push Workflow 🟡

## 📋 Problem Statement

Create a GitHub Actions workflow that:
- Builds a Docker image from your application
- Tags the image with the git commit SHA and 'latest'
- Pushes the image to Docker Hub
- Only runs on pushes to the main branch
- Uses Docker layer caching for faster builds

**Difficulty**: Intermediate 🟡

---

## 🎯 Learning Objectives

- Build Docker images in GitHub Actions
- Authenticate with Docker Hub
- Use GitHub Secrets for credentials
- Implement Docker layer caching
- Work with Docker tags

---

## 📚 Background

Docker containerization is crucial for modern DevOps. GitHub Actions can automate building and pushing Docker images to registries, ensuring your containers are always up-to-date with your code.

---

## 💡 Hints

<details>
<summary>Hint 1: Docker Hub Authentication</summary>

Use secrets for Docker credentials:

```yaml
- name: Login to Docker Hub
  uses: docker/login-action@v2
  with:
    username: ${{ secrets.DOCKERHUB_USERNAME }}
    password: ${{ secrets.DOCKERHUB_TOKEN }}
```

</details>

<details>
<summary>Hint 2: Building Docker Image</summary>

Use `docker/build-push-action@v4`:

```yaml
- name: Build and push
  uses: docker/build-push-action@v4
  with:
    context: .
    push: true
    tags: user/app:latest
```

</details>

<details>
<summary>Hint 3: Multiple Tags</summary>

Tag with commit SHA and latest:

```yaml
tags: |
  ${{ secrets.DOCKERHUB_USERNAME }}/app:latest
  ${{ secrets.DOCKERHUB_USERNAME }}/app:${{ github.sha }}
```

</details>

<details>
<summary>Hint 4: Docker Buildx for Caching</summary>

Set up Docker Buildx for advanced features:

```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v2
```

</details>

---

## ✅ Solution

<details>
<summary>Click to view the complete solution</summary>

### Step 1: Create Sample Application

**Dockerfile:**
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./

RUN npm ci --only=production

COPY . .

EXPOSE 3000

CMD ["node", "index.js"]
```

**index.js:**
```javascript
const http = require('http');

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'text/plain' });
  res.end('Hello from Docker!\n');
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

**package.json:**
```json
{
  "name": "docker-demo",
  "version": "1.0.0",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  },
  "dependencies": {}
}
```

**.dockerignore:**
```
node_modules
npm-debug.log
.git
.gitignore
README.md
.env
.DS_Store
```

### Step 2: Set Up Docker Hub Secrets

1. Go to your GitHub repository → Settings → Secrets and variables → Actions
2. Add two secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Your Docker Hub access token (not password!)

**To create Docker Hub token:**
- Go to Docker Hub → Account Settings → Security → New Access Token

### Step 3: Create Workflow

Create `.github/workflows/docker-build.yml`:

```yaml
name: Docker Build and Push

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

env:
  IMAGE_NAME: ${{ secrets.DOCKERHUB_USERNAME }}/myapp

jobs:
  docker:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Docker Hub
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}

    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=registry,ref=${{ env.IMAGE_NAME }}:buildcache
        cache-to: type=registry,ref=${{ env.IMAGE_NAME }}:buildcache,mode=max

    - name: Image digest
      run: echo ${{ steps.docker_build.outputs.digest }}
```

### Step 4: Test Locally

Before pushing, test your Docker image locally:

```bash
docker build -t myapp:test .
docker run -p 3000:3000 myapp:test
curl http://localhost:3000
```

### Step 5: Commit and Push

```bash
git add .
git commit -m "Add Docker build workflow"
git push origin main
```

</details>

---

## 🔍 Explanation

### Key Concepts:

1. **Docker Buildx**: Advanced builder with multi-platform support and caching
2. **docker/login-action**: Securely authenticates with Docker registries
3. **docker/metadata-action**: Generates tags and labels from Git metadata
4. **docker/build-push-action**: Builds and pushes Docker images
5. **Layer Caching**: Stores layers in registry for faster subsequent builds

### Tag Strategy:
- `latest`: Always points to the latest main branch build
- `main-<sha>`: Tagged with branch and commit SHA
- `pr-<number>`: Pull request builds (not pushed)

### Multi-Platform Builds:
Builds for both AMD64 and ARM64 architectures, enabling deployment across different platforms.

---

## 🚀 Extensions & Challenges

<details>
<summary>Challenge 1: Add Image Scanning</summary>

Scan for vulnerabilities using Trivy:

```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: ${{ env.IMAGE_NAME }}:latest
    format: 'sarif'
    output: 'trivy-results.sarif'

- name: Upload Trivy results to GitHub Security
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: 'trivy-results.sarif'
```

</details>

<details>
<summary>Challenge 2: Multi-Stage Build Optimization</summary>

Optimize Dockerfile with multi-stage builds:

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

# Production stage
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app .
EXPOSE 3000
CMD ["node", "index.js"]
```

</details>

<details>
<summary>Challenge 3: Push to Multiple Registries</summary>

Push to both Docker Hub and GitHub Container Registry:

```yaml
- name: Login to GitHub Container Registry
  uses: docker/login-action@v2
  with:
    registry: ghcr.io
    username: ${{ github.actor }}
    password: ${{ secrets.GITHUB_TOKEN }}

- name: Build and push
  uses: docker/build-push-action@v4
  with:
    context: .
    push: true
    tags: |
      ${{ secrets.DOCKERHUB_USERNAME }}/myapp:latest
      ghcr.io/${{ github.repository }}:latest
```

</details>

<details>
<summary>Challenge 4: Add Health Check</summary>

Add health check to Dockerfile:

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"
```

</details>

---

## 📖 Related Articles

- [Docker Build Push Action](https://github.com/docker/build-push-action)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Multi-Platform Images](https://docs.docker.com/build/building/multi-platform/)
- [Docker Layer Caching](https://docs.docker.com/build/cache/)

---

## ✨ Best Practices

- ✅ Use specific base image versions (not `latest`)
- ✅ Implement multi-stage builds to reduce image size
- ✅ Add .dockerignore to exclude unnecessary files
- ✅ Run containers as non-root user
- ✅ Use access tokens instead of passwords
- ✅ Scan images for vulnerabilities
- ✅ Tag images with commit SHA for traceability
- ✅ Use layer caching to speed up builds
- ✅ Build for multiple platforms when needed

---

## 🐛 Common Issues & Solutions

<details>
<summary>Issue: Authentication failure</summary>

**Solution**: 
- Ensure you're using an access token, not password
- Verify secrets are correctly set in repository settings
- Check token permissions and expiration

</details>

<details>
<summary>Issue: Slow build times</summary>

**Solution**:
- Enable Docker layer caching
- Use multi-stage builds
- Order Dockerfile commands from least to most frequently changing
- Use .dockerignore effectively

</details>

<details>
<summary>Issue: Image size too large</summary>

**Solution**:
- Use alpine base images
- Implement multi-stage builds
- Remove unnecessary dependencies
- Use `npm ci --only=production`

</details>

---

**Previous**: [← Node.js CI](02-nodejs-ci.md) | **Next**: [Deploy to AWS →](04-deploy-aws.md)
