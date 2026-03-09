# Skills System

Local CLI supports a **skills** system that automatically injects contextual instructions into the AI's prompt based on trigger keywords in your input.

## How It Works

1. Create a `SKILL.md` file inside `.agents/skills/<skill-name>/`
2. The file has YAML frontmatter with `name`, `triggers`, and `description`
3. When your input matches a trigger keyword, the skill's content is injected into the conversation

## Directory Structure

```
.agents/skills/
├── django-api/
│   └── SKILL.md
├── react-patterns/
│   └── SKILL.md
└── testing/
    └── SKILL.md
```

## SKILL.md Format

```markdown
---
name: django-api
triggers: [django, REST API, DRF, serializer]
description: Django REST Framework conventions and patterns
---

## Django REST Framework Guidelines

- Use ModelSerializer for standard CRUD
- Use ViewSets with routers for URL configuration
- Apply permission classes at the view level
- Use pagination for list endpoints
- Write serializer-level validation, not view-level

### Project Structure
```
app/
├── serializers.py
├── views.py
├── urls.py
├── permissions.py
└── tests/
```
```

## Creating Skills

### Example: Code Review Skill

```bash
mkdir -p .agents/skills/code-review
```

`.agents/skills/code-review/SKILL.md`:
```markdown
---
name: code-review
triggers: [review, PR, pull request, code quality]
description: Code review checklist and standards
---

## Code Review Standards

When reviewing code, check:

1. **Security**: No hardcoded secrets, SQL injection, XSS
2. **Error handling**: Proper try/catch, meaningful error messages
3. **Testing**: New code has tests, edge cases covered
4. **Performance**: No N+1 queries, unnecessary loops
5. **Readability**: Clear naming, no magic numbers
```

### Example: Project Convention Skill

```bash
mkdir -p .agents/skills/conventions
```

`.agents/skills/conventions/SKILL.md`:
```markdown
---
name: project-conventions
triggers: [new file, create, component, module]
description: Project-specific coding conventions
---

## Conventions

- Use snake_case for Python, camelCase for TypeScript
- All API responses follow { data, error, meta } format
- Database models go in models/, not in route files
- Tests mirror src/ structure in tests/
- No default exports in TypeScript
```

## Slash Commands

| Command | Description |
|---------|-------------|
| `/skills` | List all discovered skills |
| `/skills show <name>` | Show a skill's content |

## How Matching Works

- Matching is **case-insensitive substring** matching
- If your input contains "django", the `django-api` skill triggers
- Multiple skills can match a single input
- Matched skill content is added to the system prompt for that turn

## Use Cases

- **Team standards**: Encode coding conventions, PR templates, commit formats
- **Framework guides**: Django, React, FastAPI patterns your team follows
- **Domain knowledge**: Business rules, API specs, data models
- **Checklists**: Deployment steps, review criteria, testing requirements
