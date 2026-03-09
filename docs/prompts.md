# Prompt Examples

Typical prompts to get started with Local CLI. Copy and paste directly.

---

## Code Generation

```
Create a Python Flask REST API with CRUD endpoints for a todo list. Use SQLite for storage.
```

```
Write a React component that displays a sortable, filterable data table with pagination.
```

```
Generate a CLI tool in Python that converts CSV files to JSON, with column filtering and type inference.
```

## Code Editing & Refactoring

```
Refactor src/utils.py to split the 300-line parse() function into smaller, testable functions.
```

```
Add type hints to all functions in local_cli/agent.py.
```

```
Find all TODO comments in the codebase and list them with file paths and line numbers.
```

## Bug Fixing

```
The tests in tests/test_auth.py are failing with "KeyError: token". Find and fix the bug.
```

```
Users report that file uploads over 10MB crash the server. Investigate and fix.
```

```
There's a race condition in the WebSocket handler. Find it and add proper locking.
```

## Code Review & Understanding

```
Explain how the agent loop works in local_cli/agent.py. Walk through the main execution flow.
```

```
Review src/security.py for potential vulnerabilities. Check for injection, path traversal, and auth issues.
```

```
What does the orchestrator do? Show me the key files and explain the architecture.
```

## Project Setup

```
Set up a new Next.js project with TypeScript, Tailwind CSS, and Prisma. Include a basic layout and database schema.
```

```
Create a Dockerfile for this Python project. Include multi-stage build, non-root user, and health check.
```

```
Add GitHub Actions CI/CD that runs tests, lints, and deploys to Vercel on push to main.
```

## Testing

```
Write unit tests for local_cli/security.py. Cover all edge cases including path traversal and command injection.
```

```
Create an integration test that starts the server, sends requests, and verifies responses.
```

```
The test coverage for src/auth/ is 40%. Add tests to bring it above 80%.
```

## Multi-Step Tasks

```
I want to add user authentication to this Express app. Use JWT tokens, bcrypt for passwords, and add login/register/logout endpoints with middleware.
```

```
Migrate the database from SQLite to PostgreSQL. Update the schema, connection config, queries, and tests.
```

```
Add dark mode support to the entire app. Create a theme context, update all components, and persist the preference.
```

## Using Slash Commands

```
/models                    # Browse and install models
/model qwen3:8b            # Switch to a different model
/provider claude            # Switch to Claude API
/checkpoint before-refactor # Save a git snapshot
/rollback                  # Undo to last checkpoint
/plan create auth-system    # Create a structured plan
/knowledge save api-spec    # Save context for later
/agents                    # Check sub-agent status
/skills                    # List available skills
/brain qwen3:30b           # Use a stronger model for orchestration
```

## Advanced: Sub-Agents & Planning

```
Create a plan for implementing a payment system with Stripe. Break it into steps: schema design, API endpoints, webhook handling, and frontend integration.
```

```
I need to refactor 5 modules simultaneously. Use sub-agents to analyze each module in parallel, then give me a unified refactoring plan.
```

```
Save the current API specification as knowledge so you can reference it in future conversations.
```

## Tips

- **Be specific**: "Add error handling to the login endpoint" > "improve the code"
- **Reference files**: "Fix the bug in src/auth/middleware.ts line 42" helps the agent find context faster
- **Use checkpoints**: Run `/checkpoint` before big changes so you can `/rollback` if needed
- **Chain tasks**: The agent handles multi-step tasks well — describe the end goal and let it work
- **Switch models**: Use `/brain qwen3:30b` for complex reasoning, smaller models for quick edits
