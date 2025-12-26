# GitHub Copilot Instructions

## Expertise

You are an expert in:

- **Software Architecture** - Scalable systems, clean architecture, SOLID principles
- **Python/FastAPI** - Async patterns, dependency injection, Pydantic schemas
- **React + shadcn/ui + TailwindCSS** - Modern component-driven frontend
- **Quantitative Finance** - Investment management, market data, financial analysis

---

## Environment & Testing

### Local Development

- **ALWAYS** use `.venv` virtual environment for local tests
- Run tests with: `python -m pytest tests/ -v --tb=short`
- Activate venv: `source .venv/bin/activate`

### Docker Development

- Use `docker-compose.dev.yml` for updates and tests
- **Rebuild if necessary**: `docker compose -f docker-compose.dev.yml up --build`
- Check container logs for runtime errors

### Test Execution Rules

```bash
# Run tests - FAIL ON FIRST ERROR, WARNINGS ARE ERRORS
python -m pytest tests/ -x -W error --tb=short
```

- `-x` = Stop on first failure
- `-W error` = Treat warnings as errors
- **ALL errors are FATAL** - Do not proceed until fixed
- **NEVER just edit tests to make them pass** - Analyze the root cause and fix the actual code
- Re-run tests repeatedly until ALL pass with zero warnings

### Adding Dependencies - NEVER Edit Requirements Directly

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Install latest version
pip install <package-name>

# 3. Freeze to requirements
pip freeze > requirements.txt

# 4. Test everything still works
python -m pytest tests/ -x -W error --tb=short
```

- **NEVER** manually edit `requirements.txt` or `requirements-dev.txt`
- **ALWAYS** install via pip first, then freeze
- Test after adding any new dependency
- For dev-only dependencies: `pip freeze > requirements-dev.txt`

---

## Database & Migrations

### SQLAlchemy - NEVER Raw SQL

- **ALWAYS** use SQLAlchemy ORM models in `app/database/orm.py`
- **NEVER** write raw SQL queries - use SQLAlchemy query builder
- Define models with proper types: `Mapped[type]`, `mapped_column()`

### Alembic Migrations - NEVER Manual SQL

```bash
# Auto-generate migrations from ORM model changes
alembic revision --autogenerate -m "description_of_change"

# Apply migrations
alembic upgrade head
```

- **NEVER** create manual SQL migration files
- **NEVER** put raw SQL in `migrations/` folder
- Let Alembic detect ORM model changes automatically
- Review generated migration before applying

---

## Code Quality

### Python Standards

```python
# GOOD: Clean, typed Python
async def get_user(user_id: int) -> User | None:
    """Fetch user by ID."""
    return await db.get(User, user_id)

# BAD: Untyped, unclear
async def get_user(id):
    return await db.get(User, id)
```

- **Type hints on ALL functions** - parameters and return types
- **Docstrings** for public functions
- **Async/await** for I/O operations
- **Pydantic schemas** for API request/response validation

### DRY Principle - Don't Repeat Yourself

- **Extract common logic** into utility functions
- **Reuse existing services** - check `app/services/` first
- **Keep functions short** - single responsibility
- If you're copying code, you're doing it wrong

### Code Organization

```text
app/
├── api/routes/      # FastAPI route handlers (thin layer)
├── services/        # Business logic (reusable)
├── repositories/    # Database access (SQLAlchemy)
├── schemas/         # Pydantic models
├── database/        # ORM models, connection
└── core/            # Config, security, utilities
```

---

## Frontend Development

### shadcn/ui Components

- **ALWAYS** consult shadcn MCP for component info before implementing
- Use `mcp_shadcn_*` tools to search for components and examples
- Install components: `npx shadcn@latest add <component>`
- Check `frontend/components.json` for project configuration

### Component Patterns

```tsx
// GOOD: Using shadcn components
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardContent } from "@/components/ui/card"

// BAD: Custom implementations of existing components
<button className="...">Click</button>
```

### TailwindCSS

- Use Tailwind utility classes
- Follow existing patterns in the codebase
- Responsive design: `sm:`, `md:`, `lg:` prefixes

---

## Workflow Checklist

Before submitting any change:

1. [ ] Used `.venv` for Python execution
2. [ ] SQLAlchemy ORM, not raw SQL
3. [ ] Alembic autogenerate for migrations
4. [ ] All functions have type hints
5. [ ] No code duplication
6. [ ] Tests pass with `-x -W error`
7. [ ] shadcn components used where applicable
8. [ ] Docker compose works if infrastructure changed

---

## Common Mistakes to Avoid

❌ **Don't** fetch data on-demand when it should be cached/stored
❌ **Don't** write raw SQL migrations
❌ **Don't** skip type hints
❌ **Don't** duplicate existing utility functions
❌ **Don't** edit tests to make them pass - fix the code
❌ **Don't** ignore warnings - they're errors
❌ **Don't** create custom UI components when shadcn has them

✅ **Do** store data locally, fetch on schedule
✅ **Do** use Alembic autogenerate
✅ **Do** type everything
✅ **Do** reuse services and utilities
✅ **Do** fix root causes, not symptoms
✅ **Do** treat warnings as errors
✅ **Do** use shadcn/ui components  
