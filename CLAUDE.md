# CLAUDE.md - AI Assistant Guidelines

This document provides guidance for AI assistants working with the `claudecode` repository.

## Project Overview

**Repository**: claudecode
**Status**: Initial setup phase
**Owner**: tyama-ds

This repository is in its early stages of development. As the project evolves, this document should be updated to reflect new conventions, structures, and workflows.

## 開発進捗ダッシュボード（必ず更新する）

このリポジトリの開発状況は、for_eigyo（tyama-ds/for_eigyo）の開発ボードに for_eigyo のプロジェクトと一緒に表示する。

- 正本: このリポジトリの `progress/projects/<slug>.json`（1プロジェクト1ファイル。slug はトップレベルのフォルダ名）
- 生成スクリプト・画面・項目の説明: for_eigyo の `progress/`（[progress/README.md](https://github.com/tyama-ds/for_eigyo/blob/main/progress/README.md)）
- 公開先（claude.ai Artifact）: https://claude.ai/artifact/ANrf5LMNjv2ojHr9JHZ3fP

進捗ファイルの更新は、オーナーの許可を得た作業に付随して行う（この更新のために別途許可を求めなくてよい）。

### プロジェクトのコードを変更したとき

コミットの前に次を行う。

1. 変更したプロジェクトの `progress/projects/<slug>.json` を更新する。
   - `log` の先頭に `{"date": "YYYY-MM-DD", "summary": "何をしたか（1行）", "ref": "#PR番号（あれば）"}` を追加する
   - `updated` を今日の日付にする
   - 完了したマイルストーンは `state` を `done` にして `date` を入れる。着手中なら `doing`、新しい予定は `todo` で追加する
   - `phase`・`progress`（0〜100 の目安）・`next`・`risks`・`status` を実態に合わせて見直す
2. for_eigyo を同じ親フォルダに用意し（無ければ `git clone https://github.com/tyama-ds/for_eigyo ../for_eigyo`。リモート環境では add_repo で追加）、
   `python ../for_eigyo/progress/build_dashboard.py --source claudecode=$(pwd)` を実行して検証と生成を行う。
   新しいフォルダがあれば、このリポジトリの `progress/projects/` に下書きが自動作成されるので、記入してコミットする。
3. push したら、生成された `../for_eigyo/progress/dashboard.html` を上記の公開先へ再公開する。
   Artifact ツールで公開先を `action: "read"` してから、`url` に公開先を指定して publish する（新しい URL を作らない）。
   for_eigyo 側の dashboard.html はコミットしなくてよい（for_eigyo での次の作業時に更新される）。
   Artifact ツールが使えない環境では再公開を省略し、その旨をユーザーに伝える。

### 確認

`python ../for_eigyo/progress/build_dashboard.py --source claudecode=$(pwd) --check --data-only` で、
進捗ファイルの欠けと書式の誤りを検出できる。CI（`.github/workflows/progress.yml`）でも同じ確認を行う。

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
