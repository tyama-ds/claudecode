# CLAUDE.md - AI Assistant Guidelines

This document provides guidance for AI assistants working with the `claudecode` repository.

## Project Overview

**Repository**: claudecode
**Status**: Initial setup phase
**Owner**: tyama-ds

This repository is in its early stages of development. As the project evolves, this document should be updated to reflect new conventions, structures, and workflows.

## Repository Structure

```
claudecode/
├── README.md          # Project description and documentation
├── CLAUDE.md          # AI assistant guidelines (this file)
└── .git/              # Git version control
```

### Planned Directory Structure

As the project grows, consider organizing with:

```
claudecode/
├── src/               # Source code
├── tests/             # Test files
├── docs/              # Documentation
├── scripts/           # Build and utility scripts
└── config/            # Configuration files
```

## Development Workflow

### Git Practices

1. **Branch Naming**: Use descriptive branch names
   - Feature branches: `feature/<description>`
   - Bug fixes: `fix/<description>`
   - Claude sessions: `claude/<session-id>`

2. **Commits**: Write clear, descriptive commit messages
   - Use present tense ("Add feature" not "Added feature")
   - Keep the first line under 72 characters
   - Include context in the body when needed

3. **Current Branch**: `claude/claude-md-ml7vo9fdvec9ff47-hYlCt`

### Commands

```bash
# Check repository status
git status

# View recent commits
git log --oneline -10

# Push changes (use the current branch)
git push -u origin <branch-name>
```

## Code Conventions

### General Guidelines

1. **Readability**: Write clear, self-documenting code
2. **Simplicity**: Prefer simple solutions over complex ones
3. **Consistency**: Follow existing patterns in the codebase
4. **Testing**: Write tests for new functionality

### File Organization

- Keep files focused on a single responsibility
- Use meaningful file and directory names
- Group related functionality together

## For AI Assistants

### Approval Required Before Coding

- **コーディングは必ずオーナーの「許可」を得てから開始すること。**
- まず提案（方針・設計・変更内容）を提示し、オーナーが内容を吟味・決定するのを待つ。
- オーナーから明示的な許可が出るまで、コードの作成・編集・ファイル変更を行ってはならない。
- 許可なく勝手にコーディングを進めることは禁止する。

### Before Making Changes

1. **Read First**: Always read relevant files before modifying them
2. **Understand Context**: Explore the codebase structure before implementing
3. **Check Existing Patterns**: Follow established conventions in the codebase

### When Implementing Features

1. **Plan**: Use task tracking to organize multi-step work
2. **Incremental Changes**: Make small, focused commits
3. **Test**: Verify changes work as expected
4. **Document**: Update documentation when adding significant features

### What to Avoid

- Don't introduce security vulnerabilities (XSS, SQL injection, etc.)
- Don't over-engineer solutions
- Don't add unnecessary dependencies
- Don't modify code without reading it first
- Don't create files unless necessary (prefer editing existing files)

### Communication

- Be concise and direct in responses
- Provide file paths and line numbers when referencing code
- Explain the reasoning behind significant decisions

## Configuration Files (To Be Added)

As the project develops, consider adding:

- `package.json` - Node.js dependencies and scripts
- `tsconfig.json` - TypeScript configuration
- `.gitignore` - Files to exclude from version control
- `.eslintrc.js` - Linting rules
- `.prettierrc` - Code formatting rules
- `jest.config.js` - Testing configuration

## Testing

Currently no testing framework is configured. When tests are added:

1. Place test files in `tests/` or alongside source files with `.test.` suffix
2. Run tests before committing changes
3. Maintain test coverage for critical functionality

## Building and Running

*Build and run instructions will be added once the project's technology stack is established.*

## Environment Setup

*Environment setup instructions will be added as the project develops.*

## Key Files Reference

| File | Purpose |
|------|---------|
| `README.md` | Project overview and documentation |
| `CLAUDE.md` | AI assistant guidelines (this file) |

## Updating This Document

This CLAUDE.md should be updated when:

- New conventions are established
- Project structure changes significantly
- Build/test workflows are added
- New tools or dependencies are introduced

---

*Last updated: 2026-02-04*
