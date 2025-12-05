# Contributing to legiscope

Thank you for your interest in contributing to legiscope!

## Development Setup

See [AGENTS.md](AGENTS.md) for detailed information about environment setup, LLM provider configuration, development commands, and project structure.

## Code Style and Testing

We use `ruff` for linting and formatting, and `pytest` for testing:

```bash
# Format code
make format

# Fix linting issues
make fix

# Run tests
make test
```

Please ensure your code passes linting checks and tests before submitting a pull request.

## Commit Messages

We loosely follow the [Conventional Commits](https://www.conventionalcommits.org/) standard. This helps maintain a clear commit history.

### Format

```
<type>[optional scope]: <description>

[optional body]
```

### Common Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Test additions or updates
- `chore`: Maintenance tasks, dependency updates

### Examples

```
feat(query): add batch query processing

fix(embeddings): handle empty text inputs

docs: update README with docs/ directory

test(retrieve): add HYDE query rewriting tests
```

### Breaking Changes

Use `!` after the type/scope to indicate breaking changes:

```
feat(llm_config)!: change Config API to use settings objects
```

## Pull Requests

1. Fork the repository and create a new branch from `main`
2. Make your changes
3. Ensure tests pass (`make test`) and linting passes (`make lint`)
4. Submit a pull request

## License

By contributing to legiscope, you agree that your contributions will be licensed under the same license as the project.
