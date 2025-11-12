# Problem 2: CI Pipeline for Node.js Application 🟢

## 📋 Problem Statement

Create a Continuous Integration (CI) pipeline for a Node.js application that:
- Installs dependencies
- Runs linting checks
- Executes unit tests
- Runs on multiple Node.js versions (14.x, 16.x, 18.x)
- Reports test coverage

**Difficulty**: Beginner 🟢

---

## 🎯 Learning Objectives

- Set up Node.js environment in GitHub Actions
- Use matrix strategy for testing multiple versions
- Cache dependencies for faster builds
- Integrate testing and linting tools
- Work with artifacts

---

## 📚 Background

A CI pipeline automatically tests your code every time you push changes. This ensures code quality and catches bugs early. Matrix strategies allow you to test against multiple environments simultaneously.

---

## 💡 Hints

<details>
<summary>Hint 1: Setup Node.js Action</summary>

Use the official `actions/setup-node@v3` action to set up Node.js:

```yaml
- uses: actions/setup-node@v3
  with:
    node-version: '18.x'
```

</details>

<details>
<summary>Hint 2: Checkout Code First</summary>

Always checkout your repository code before other steps:

```yaml
- uses: actions/checkout@v3
```

</details>

<details>
<summary>Hint 3: Matrix Strategy</summary>

Test multiple Node versions:

```yaml
strategy:
  matrix:
    node-version: [14.x, 16.x, 18.x]
```

</details>

<details>
<summary>Hint 4: Caching Dependencies</summary>

Cache npm dependencies to speed up builds:

```yaml
- uses: actions/setup-node@v3
  with:
    node-version: ${{ matrix.node-version }}
    cache: 'npm'
```

</details>

---

## ✅ Solution

<details>
<summary>Click to view the complete solution</summary>

### Step 1: Create Sample Node.js Project

First, create a simple Node.js project structure:

**package.json:**
```json
{
  "name": "nodejs-ci-demo",
  "version": "1.0.0",
  "description": "Demo project for GitHub Actions CI",
  "main": "index.js",
  "scripts": {
    "test": "jest --coverage",
    "lint": "eslint .",
    "start": "node index.js"
  },
  "devDependencies": {
    "jest": "^29.0.0",
    "eslint": "^8.0.0"
  }
}
```

**index.js:**
```javascript
function add(a, b) {
  return a + b;
}

function multiply(a, b) {
  return a * b;
}

module.exports = { add, multiply };
```

**index.test.js:**
```javascript
const { add, multiply } = require('./index');

test('adds 1 + 2 to equal 3', () => {
  expect(add(1, 2)).toBe(3);
});

test('multiplies 2 * 3 to equal 6', () => {
  expect(multiply(2, 3)).toBe(6);
});
```

**.eslintrc.json:**
```json
{
  "env": {
    "node": true,
    "es2021": true,
    "jest": true
  },
  "extends": "eslint:recommended",
  "parserOptions": {
    "ecmaVersion": 12
  }
}
```

### Step 2: Create CI Workflow

Create `.github/workflows/nodejs-ci.yml`:

```yaml
name: Node.js CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [14.x, 16.x, 18.x]

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Setup Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v3
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run linting
      run: npm run lint

    - name: Run tests
      run: npm test

    - name: Upload coverage reports
      uses: actions/upload-artifact@v3
      with:
        name: coverage-report-${{ matrix.node-version }}
        path: coverage/
      if: always()
```

### Step 3: Enhanced Version with Status Badges

Add this to your workflow for status checks:

```yaml
    - name: Test Summary
      if: always()
      run: |
        echo "### Test Results :test_tube:" >> $GITHUB_STEP_SUMMARY
        echo "- Node Version: ${{ matrix.node-version }}" >> $GITHUB_STEP_SUMMARY
        echo "- Status: ${{ job.status }}" >> $GITHUB_STEP_SUMMARY
```

### Step 4: Commit and Push

```bash
git add .
git commit -m "Add Node.js CI pipeline"
git push origin main
```

</details>

---

## 🔍 Explanation

### Key Concepts:

1. **`actions/checkout@v3`**: Checks out your repository code
2. **`actions/setup-node@v3`**: Sets up Node.js environment
3. **`npm ci`**: Clean install of dependencies (faster and more reliable than `npm install`)
4. **`strategy.matrix`**: Runs jobs in parallel for each Node version
5. **`cache: 'npm'`**: Caches npm dependencies between runs
6. **`actions/upload-artifact@v3`**: Saves build artifacts (like coverage reports)

### Why use `npm ci` instead of `npm install`?
- Faster (10-20x in some cases)
- More reliable and deterministic
- Requires package-lock.json
- Removes node_modules before installing

### Matrix Strategy Benefits:
- Tests code against multiple Node versions simultaneously
- Ensures compatibility across versions
- Runs in parallel for faster feedback

---

## 🚀 Extensions & Challenges

<details>
<summary>Challenge 1: Add Code Coverage Badge</summary>

Integrate with Codecov:

```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage/coverage-final.json
    flags: unittests
    name: codecov-umbrella
```

Add to package.json:
```json
"jest": {
  "coverageThreshold": {
    "global": {
      "branches": 80,
      "functions": 80,
      "lines": 80,
      "statements": 80
    }
  }
}
```

</details>

<details>
<summary>Challenge 2: Add Build Step</summary>

If your project needs building:

```yaml
- name: Build application
  run: npm run build

- name: Upload build artifacts
  uses: actions/upload-artifact@v3
  with:
    name: build-${{ matrix.node-version }}
    path: dist/
```

</details>

<details>
<summary>Challenge 3: Add Dependency Caching Strategy</summary>

Optimize caching further:

```yaml
- name: Cache node modules
  uses: actions/cache@v3
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

</details>

<details>
<summary>Challenge 4: Add Slack Notification</summary>

Notify team on failure:

```yaml
- name: Slack Notification
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: 'CI Pipeline completed'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
  if: failure()
```

</details>

---

## 📖 Related Articles

- [Building and Testing Node.js](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-nodejs)
- [Using a Matrix Strategy](https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs)
- [Caching Dependencies](https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows)
- [Jest Documentation](https://jestjs.io/docs/getting-started)

---

## ✨ Best Practices

- ✅ Use `npm ci` for consistent, fast installs
- ✅ Cache dependencies to reduce build time
- ✅ Test on multiple Node versions for compatibility
- ✅ Set coverage thresholds to maintain code quality
- ✅ Use artifacts to preserve test results
- ✅ Add meaningful names to workflow steps
- ✅ Run linting before tests (fail fast)

---

## 🐛 Common Issues & Solutions

<details>
<summary>Issue: Tests fail only in CI</summary>

**Solution**: Ensure all dependencies are in package.json, not globally installed. Check Node version compatibility.

</details>

<details>
<summary>Issue: Slow workflow execution</summary>

**Solution**: 
- Use caching for dependencies
- Consider using `npm ci` instead of `npm install`
- Parallelize jobs when possible

</details>

<details>
<summary>Issue: Coverage reports not generated</summary>

**Solution**: Ensure jest is configured properly and coverage directory exists:

```json
"jest": {
  "coverageDirectory": "coverage"
}
```

</details>

---

**Previous**: [← Basic Workflow](01-basic-workflow.md) | **Next**: [Python Application Testing →](03-python-testing.md)
