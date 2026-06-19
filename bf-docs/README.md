# bf-docs

Architecture Decision Records (ADRs) for `blackfuel-ai/bf-vllm`. Each ADR captures one decision: its context, what we chose, and the consequences.

Format follows [Michael Nygard's template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions). New ADRs use `bf-docs/template.md` as a starting point.

## Index

| # | Title | Status |
| --- | --- | --- |
| [0001](adr/0001-private-mirror-with-public-fork.md) | Private mirror with repurposed public fork | Accepted |
| [0002](adr/0002-two-branch-architecture.md) | Two-branch architecture, tag-gated merge sync | Accepted |
| [0003](adr/0003-change-classification-and-patch-discipline.md) | Change classification via `.bf-paths` manifest; patch discipline | Accepted |
| [0004](adr/0004-build-channels-versioning-labels.md) | Three image channels, versioning, OCI label conventions | Accepted |
| [0005](adr/0005-build-ci-development-velocity.md) | Build CI optimized for development velocity | Accepted |
| [0006](adr/0006-agentic-workflow-boundaries.md) | Agentic workflow boundaries (Day 0) | Accepted |

## Conventions

- ADRs are immutable once accepted. Decisions change via a new ADR that **supersedes** the old one (header updated to `Status: Superseded by ADR-NNNN`).
- File names: `NNNN-kebab-case-title.md`, sequentially numbered from 0001.
- Each ADR is ~60-150 lines. If it grows beyond that, it's probably two decisions and should split.

## What's not in `bf-docs/`

Operational artifacts — the workflow YAML, the sccache configuration, the actual git commands the sync workflow runs — live with the code (`.github/workflows/`, `.bf-paths`, scripts). ADRs explain the **why**; the code is the **how**.
