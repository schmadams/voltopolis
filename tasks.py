from invoke import task


import os
from invoke import task


@task
def req_compile(ctx):
    """
    Compile Python requirements without upgrading.a
    """
    ctx.run('pip-compile requirements/requirements.in')


@task
def req_upgrade(ctx):
    """
    Compile Python requirements with upgrading.
    """
    ctx.run('pip-compile -U requirements/requirements.in')


@task
def build(ctx):
    """
    Install all dependencies.
    """
    ctx.run('pip install -r requirements/requirements.txt')


@task
def rebuild(ctx):
    """
    Compile and rebuild the environment dependencies
    """
    ctx.run('inv req-compile && inv build')


@task
def lint(ctx, path='src'):
    """
    Lint project files
    """
    ctx.run(f'pylint --fail-under=9.0 --rcfile=.pylintrc {path}')


@task(
    help={
        'path': 'Specify files or directories to lint',
        'check': 'Only runs check without reformat (default: False)',
    }
)
def lint_black(ctx, path='src', check=False):
    """
    Runs the black formatter.
    """

    cmd = 'black --line-length=100 --skip-string-normalization {check} {path}'.format(
        check='--check' if check else '', path=path
    )
    ctx.run(cmd)

# ── Environment ────────────────────────────────────────────────

@task
def setup(c):
    """Create venv and install all dependencies."""
    c.run("uv venv")
    c.run("uv pip install -e '.[dev]'")
    print("Setup complete. Activate your venv with: .venv\\Scripts\\activate")


@task
def install(c):
    """Install/sync dependencies (run after updating pyproject.toml)."""
    c.run("uv pip install -e '.[dev]'")


# ── Development server ─────────────────────────────────────────

@task
def dev(c):
    """Start the FastAPI development server with auto-reload."""
    c.run("uv run uvicorn backend.main:app --reload --port 8000")


# ── Utilities ──────────────────────────────────────────────────

@task
def routes(c):
    """Print all registered API routes."""
    c.run("python -c \"from backend.main import app; [print(r.methods, r.path) for r in app.routes]\"")


@task
def check_env(c):
    """Check whether required user-level environment variables are available."""
    c.run(
        "uv run python -c "
        "\"from pathlib import Path; "
        "from dotenv import load_dotenv; "
        "import os; "
        "env_file = Path.home() / '.env'; "
        "print('ENV_FILE:', env_file); "
        "print('EXISTS:', env_file.exists()); "
        "load_dotenv(env_file, override=True); "
        "print('SUPABASE_URL loaded:', bool(os.getenv('SUPABASE_URL'))); "
        "print('SUPABASE_KEY loaded:', bool(os.getenv('SUPABASE_KEY')))\""
    )