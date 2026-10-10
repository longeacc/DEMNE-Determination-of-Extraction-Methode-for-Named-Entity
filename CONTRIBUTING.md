# Contributing to DEMNE

We welcome contributions to DEMNE! 

## Getting Started

1. Fork the repository and create your branch from main.
2. Install the development dependencies:
   ```bash
   pip install -e .[dev,ner]
   ```
3. Create a branch for your feature or bugfix.
4. Ensure you write tests for any new code. Run tests with:
   ```bash
   pytest --cov
   ```
5. Follow code style guidelines: We use black, isort, and 
ruff for formatting and linting.

## Pull Requests
- Format your commit messages using Conventional Commits (e.g., feat: ..., fix: ..., docs: ...).
- Reference issues you are resolving (e.g. Resolves #123).

## Data Governance
If you're contributing validation code, ensure it uses synthetic or fully anonymized datatypes matching docs/DATA_GOVERNANCE.md.
