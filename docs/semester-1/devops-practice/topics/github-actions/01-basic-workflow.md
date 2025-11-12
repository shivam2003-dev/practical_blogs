# Problem 1: Create Your First GitHub Actions Workflow 🟢

## 📋 Problem Statement

Create a basic GitHub Actions workflow that runs whenever code is pushed to the `main` branch. The workflow should:
- Print "Hello, GitHub Actions!"
- Display the current date and time
- Show the name of the person who triggered the workflow

**Difficulty**: Beginner 🟢

---

## 🎯 Learning Objectives

- Understand basic workflow syntax
- Learn about workflow triggers
- Use environment variables
- Work with GitHub context

---

## 📚 Background

GitHub Actions workflows are defined in YAML files stored in the `.github/workflows/` directory. Each workflow consists of one or more jobs, and each job contains steps that execute commands or actions.

---

## 💡 Hints

<details>
<summary>Hint 1: Workflow File Location</summary>

Create a file at `.github/workflows/hello-world.yml` in your repository root.

</details>

<details>
<summary>Hint 2: Basic Workflow Structure</summary>

```yaml
name: Workflow Name
on: [trigger_event]
jobs:
  job-name:
    runs-on: ubuntu-latest
    steps:
      - name: Step name
        run: command
```

</details>

<details>
<summary>Hint 3: Using GitHub Context</summary>

Access the actor's name using: `${{ github.actor }}`

</details>

<details>
<summary>Hint 4: Running Commands</summary>

Use `run: echo "Your message"` to print messages. For date, use `date` command.

</details>

---

## ✅ Solution

<details>
<summary>Click to view the complete solution</summary>

### Step 1: Create Workflow File

Create `.github/workflows/hello-world.yml`:

```yaml
name: Hello World Workflow

on:
  push:
    branches:
      - main

jobs:
  greet:
    runs-on: ubuntu-latest
    
    steps:
      - name: Print greeting
        run: echo "Hello, GitHub Actions!"
      
      - name: Display current date and time
        run: date
      
      - name: Show who triggered the workflow
        run: echo "This workflow was triggered by ${{ github.actor }}"
```

### Step 2: Commit and Push

```bash
git add .github/workflows/hello-world.yml
git commit -m "Add hello world workflow"
git push origin main
```

### Step 3: Verify Workflow Execution

1. Go to your GitHub repository
2. Click on the "Actions" tab
3. You should see your workflow running
4. Click on the workflow run to see the output

### Expected Output

```
Hello, GitHub Actions!
Wed Nov 12 10:30:45 UTC 2025
This workflow was triggered by your-username
```

</details>

---

## 🔍 Explanation

### Key Concepts:

1. **`name`**: Identifies your workflow in the Actions tab
2. **`on`**: Specifies when the workflow should run (push events on main branch)
3. **`jobs`**: Contains one or more jobs to execute
4. **`runs-on`**: Specifies the runner environment (ubuntu-latest)
5. **`steps`**: Individual tasks within a job
6. **`run`**: Executes shell commands
7. **`${{ }}`**: Expression syntax for accessing context and variables

### Workflow Trigger:
- `on.push.branches`: Triggers only when pushing to specified branches

### GitHub Context:
- `github.actor`: Username of the person who triggered the workflow
- Other useful context: `github.repository`, `github.ref`, `github.sha`

---

## 🚀 Extensions & Challenges

<details>
<summary>Challenge 1: Add Multiple Triggers</summary>

Modify the workflow to also run on pull requests:

```yaml
on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main
```

</details>

<details>
<summary>Challenge 2: Add Environment Variables</summary>

Add custom environment variables and use them:

```yaml
env:
  GREETING: "Welcome to DevOps!"

jobs:
  greet:
    runs-on: ubuntu-latest
    steps:
      - name: Use environment variable
        run: echo $GREETING
```

</details>

<details>
<summary>Challenge 3: Use Multiple Operating Systems</summary>

Test your workflow on different OS:

```yaml
jobs:
  greet:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - name: Print OS
        run: echo "Running on ${{ matrix.os }}"
```

</details>

---

## 📖 Related Articles

- [Understanding GitHub Actions Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Events that Trigger Workflows](https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows)
- [Context and Expression Syntax](https://docs.github.com/en/actions/learn-github-actions/contexts)

---

## ✨ Best Practices

- ✅ Use descriptive names for workflows and steps
- ✅ Keep workflows simple and focused
- ✅ Use comments to explain complex logic
- ✅ Test workflows on feature branches before merging
- ✅ Use specific branch names instead of wildcards when possible

---

**Next Problem**: [CI Pipeline for Node.js Application →](02-nodejs-ci.md)
