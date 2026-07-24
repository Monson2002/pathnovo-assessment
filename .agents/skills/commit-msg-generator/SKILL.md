---
name: commit-msg-generator
description: Generate high-quality Conventional Commit messages from staged Git changes.
---

# Purpose

Generate a concise, accurate Conventional Commit message based on the currently staged changes.

## Instructions

When invoked:

1. Ensure the current directory is a Git repository.
2. Read the staged changes using:
   - `git diff --cached --stat`
   - `git diff --cached`
3. Do NOT inspect unstaged or untracked files.
4. Determine the primary intent of the change.
5. Generate a Conventional Commit message.

## Conventional Commit Types

Use the most appropriate type:

- feat
- fix
- refactor
- docs
- test
- chore
- ci
- build
- perf
- style
- revert

## Commit Message Rules

- Subject in imperative mood.
- Maximum 72 characters.
- No trailing period.
- Lowercase type.
- Format:

<type>(optional-scope): short summary

Examples:

feat(auth): add JWT refresh token support

fix(api): handle empty request body

refactor(parser): simplify token validation

docs: update installation guide

If the change is substantial, include a body:

- Explain why, not what.
- Wrap lines around 72 characters.
- Omit the body for small changes.

## Ignore

Do not overemphasize:

- formatting-only changes
- whitespace
- import ordering
- generated files
- lock files (unless they are the primary change)

Focus on the functional changes.

## Output

Return ONLY the commit message.

Do not include:

- markdown
- code fences
- explanations
- commentary
- confidence scores

The output must be directly usable with:

git commit -m "<message>"

or

git commit
