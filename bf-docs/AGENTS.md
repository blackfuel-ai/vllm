# bf-docs/AGENTS.md — Blackfuel context for `bf-vllm`

This file is the Blackfuel overlay on top of upstream's [`AGENTS.md`](../AGENTS.md) (the public-fork-wide agent rules from `vllm-project/vllm`). Both apply when working in `bf-vllm`; this one adds the BF-only conventions.

Auto-discovered via the `@bf-docs/AGENTS.md` include in root [`CLAUDE.md`](../CLAUDE.md) — Claude Code loads it on every session in this repo. Other tools (Cursor, Codex) pick it up by the `AGENTS.md` filename convention when the agent's cwd is `bf-docs/`.

---

## Project mission

`bf-vllm` is Blackfuel's **private mirror** of `vllm-project/vllm`. Two goals, deliberately in tension and explicitly balanced in the ADRs:

1. **Be among the first to serve a new upstream model.** Image built and deployable within hours of a model landing upstream — not after a full sync+review+test cycle.
2. **Preserve our custom workflows, ROCm tooling, CPU-bench images, kernel patches, and scheduling tweaks** during fast upstream pulls — no losing BF work to a sync.

The structural answer to both is the **two-branch architecture** ([ADR-0002](./adr/0002-two-branch-architecture.md)): `main` carries our additions, `upstream-main` fast-forwards from `vllm-project/vllm:main`, and a daily sync merges upstream into `main` with conflict surface bounded by the `bf-*` filename prefix convention.

`bf-vllm` is also a **playground for autonomous agents** to amortize the maintenance cost of a vendor distribution — see [ADR-0006](./adr/0006-agentic-workflow-boundaries.md).

## Repository layout

| Path | Owner | Description |
| --- | --- | --- |
| `vllm/`, `csrc/`, `tests/`, `docker/`, etc. | Upstream | Pure upstream vLLM. Edits to these files are `[bf-patch]` ([ADR-0003](./adr/0003-change-classification-and-patch-discipline.md)), subject to the 14-day upstreaming deadline. |
| `bf-docs/` | Blackfuel | ADRs, agent context, BF-only documentation. Never conflicts with upstream. |
| `bf-tools/` | Blackfuel | Helper scripts and modules (version helper, release-notes generator, agent-log, etc.). |
| `.github/workflows/bf-*.yml` | Blackfuel | BF-authored CI workflows. The sync workflow deletes any non-`bf-*.yml` workflow on every merge. |
| `.github/workflows/matchers/*.json` | Upstream | Problem matchers used by `bf-precommit`. Inherited via sync. |
| `Makefile` | Blackfuel | BF-additive (upstream has none). Lint and dev shortcuts. |
| `.pre-commit-config.yaml` | Upstream | Lint config — single source of truth for `bf-precommit`. |

The `bf-` filename prefix is **mandatory for workflows** and **strongly preferred for other BF-authored files** ([ADR-0003](./adr/0003-change-classification-and-patch-discipline.md)). It makes "is this ours or upstream's?" a zero-cost question.

## Change classification ([ADR-0003](./adr/0003-change-classification-and-patch-discipline.md))

Every change classifies into one of three buckets at authoring time:

| Bucket | Where it lives | Conflict surface | Enforcement |
| --- | --- | --- | --- |
| **BF-additive files** | Paths listed in `.bf-paths` (`bf-docs/`, `bf-tools/`, `.github/workflows/bf-*.yml`, `Makefile`, ...) | Zero — upstream will never create these paths | `bf-classification-lint` (planned) |
| **BF patches** | Edits to existing upstream files | Bounded — managed via the `[bf-patch]` commit convention | `bf-patches-trailer-lint` (planned) |
| **Pure upstream sync** | Touches no BF surface; arrives via the sync workflow | N/A | — |

Neither the `.bf-paths` manifest nor its lint workflows ship in this PR — they land together in a follow-up. `bf-lint.yml` (current) validates `.bf-paths` syntax once the file exists and skips cleanly while it doesn't. ADR-0003 is the source of truth for the classification rule; this table mirrors it.

Rule of thumb: prefer additive files. Reach for `[bf-patch]` only when there's no extension point. If you keep patching the same upstream file repeatedly, the right move is often to **upstream a hook/extension point** so the next time it can be additive.

Every `[bf-patch]` commit carries:

- `[bf-patch]` prefix in the subject line.
- `Upstream-status:` trailer — one of `candidate | submitted-#NNNN | pending-upstream-#NNNN | rejected | bf-only`.
- 14-day deadline to move from `candidate` to `submitted-#NNNN` (or be reclassified to `bf-only` / `rejected`).

## Test discipline

Write the test first; run it, then fix the *code* — never edit a test to fit the code. If a test and the code genuinely disagree about intended behaviour, stop and ask the operator; do not resolve it yourself. Every BF feature (a `[bf-feature]`, or a `[bf-patch]` that adds behaviour) **must** ship its test in the **same PR**.

BF tests live in **`tests/bf/`** — a single Blackfuel-authored root. The rest of `tests/` is upstream-owned: a test dropped next to upstream files conflicts on every sync merge and (being inside an upstream path) gets mislabelled as a `[bf-patch]` carrying a 14-day upstreaming clock. `tests/bf/` sidesteps both — it's a bf-additive path in `.bf-paths` ([ADR-0003](./adr/0003-change-classification-and-patch-discipline.md)), so a sync never touches it and no decay clock applies. Default here; only put a `bf_*` test beside an upstream file when it genuinely must import an upstream test fixture (rare).

CPU-safe BF tests (no GPU, no model download) run on **every PR** as part of Tier-0 via `bf-cpu-smoke.yml`, which collects the whole `tests/bf/` tree — the same `python -m pytest … tests/bf/` invocation a developer runs locally ([ADR-0007](./adr/0007-functional-test-strategy.md) Tier-0). A BF test that needs a GPU or a model download still lives in `tests/bf/` but **must** guard itself (the `requires_gpu` / `requires_model_download` markers in `tests/bf/conftest.py`) so the offline Tier-0 run skips it cleanly; it executes later under Tier-1. `tests/bf/test_error_codes.py` (the structured OpenAI error-code surface) is the worked reference.

## Local setup

After cloning the repo for the first time:

```bash
make install
```

That target lives in the root [`Makefile`](../Makefile) and does three things:

1. Creates a project venv at `.venv/` using `uv venv --python python3.14`. uv will auto-download Python 3.14 if it's not already on the system (managed-Python feature). Override with `make install PYTHON=python3.13` if needed.
2. Installs `pre-commit` (and any other lint deps upstream pins via `requirements/lint.txt`) into the venv with `uv pip install`.
3. Registers all three git hooks (`pre-commit`, `commit-msg`, and `pre-push`) so every `git commit` and every `git push` runs the same lint suite CI runs, and DCO/signoff checks fire on each commit message.

Verify with `make lint` — should produce identical output to the `bf-precommit` CI workflow.

**Tearing down**: `make clean` removes the venv. Re-run `make install` for a fresh setup.

**Alternative for users who don't want a project venv**: `uv tool install pre-commit` followed by `pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push` gives the same hooks but installs `pre-commit` per-user in `~/.local/bin`. `pipx install pre-commit` is equivalent. The Makefile path is preferred because it pins to upstream's `requirements/lint.txt` version (currently `pre-commit>=4.5.1`) so everyone in the repo runs the same lint stack.

That registers three git hooks at once:

- **`pre-commit`** — runs on every `git commit`, scoped to staged files. Fast per-commit feedback.
- **`commit-msg`** — runs on every `git commit` against the commit message itself (e.g. DCO `Signed-off-by` enforcement). Fires after `pre-commit` succeeds.
- **`pre-push`** — runs on every `git push`, on every commit in the push range. Catches anything that slipped past the per-commit hook (rebases, amends, fixups, agent-authored commits that bypassed the staged-only hook).

Both share the same `.pre-commit-config.yaml` at the repo root — bumped on every upstream sync, identical to what `vllm-project/vllm` enforces.

**Why both, not just `pre-commit`**: a per-commit hook only sees staged files. A push that rewrites history (amend, rebase, fixup), or a commit authored by an agent that didn't run the hook, can land lint debt that CI only catches after the round-trip. The `pre-push` hook closes that gap — same lint suite, fires once per push, exits non-zero before the network call when something would fail in CI. **Always run pre-commit before pushing**; this is the cheap mechanical way to make sure of it.

## Running CI lint locally

Two equivalent invocations:

```bash
make lint
# or, without make:
pre-commit run --all-files --hook-stage manual
```

The `--hook-stage manual` flag matches the set CI runs in the `bf-precommit` workflow: hooks declared with `stages: [manual]` (the mypy variants `mypy-3.10` through `mypy-3.13`) plus every default-staged hook. The cheaper `mypy-local` hook is `stages: [pre-commit]` (excluded by `--hook-stage manual`) and exists for faster local feedback during normal `git commit`. The `make` target is the canonical entry point and will pick up extensions over time.

## SPDX headers and BF-authored paths

The two SPDX lines upstream vLLM puts on its source files carry different meaning, and only one of them belongs on Blackfuel's own files.

`# SPDX-License-Identifier: Apache-2.0` is the **license**. It applies to every file in the repo — upstream or BF-authored alike, since the whole mirror inherits Apache-2.0. Keep it on every file.

`# SPDX-FileCopyrightText: Copyright contributors to the vLLM project` is **provenance** — it asserts the file originates from the vLLM project. It belongs only on vLLM-origin files. Editing an upstream file as a `[bf-patch]` never strips it (the file is still vLLM's). BF-authored files under `bf-tools/` and `bf-docs/` must **not** carry it: Blackfuel wrote them, so claiming vLLM authorship would be dishonest provenance.

The upstream `check-spdx-header` hook exact-matches that one vLLM copyright string and cannot be told to accept an alternate (a `[bf-patch]` BF copyright line, for instance). So the only way to keep honest provenance on BF files is to exclude BF-authored paths from the hook rather than let it auto-insert the vLLM line. The same reasoning extends to the other upstream-convention hooks that encode vLLM house rules irrelevant to BF code — `ruff-check`/`ruff-format` (vLLM's formatting style), `check-forbidden-imports` (vLLM's `import regex` mandate), and `markdownlint-cli2` (vLLM's Markdown style). Each of these carries a hook-level `exclude: '^(bf-tools|bf-docs)/'` in `.pre-commit-config.yaml`.

That config edit is itself a `[bf-patch]` because `.pre-commit-config.yaml` is an upstream-owned file. When a new top-level `bf-*` directory is added, extend the `(bf-tools|bf-docs)` alternation in each of those excludes (and the matching prefix in the `markdownlint-cli2` exclude) to cover it. Do not add a companion hook that inserts a BF copyright line; BF files deliberately carry the license line only, with no copyright/provenance line at all.

## Agent boundaries ([ADR-0006](./adr/0006-agentic-workflow-boundaries.md))

Highlights — see the ADR for the full rule set:

- **Author-side agents are advisor-only for 90 days.** They propose; humans decide. The six Day-0 skills (`resolve-sync-conflict`, `harmonize-backends`, `upstream-patch`, `track-upstream-models`, `auto-research`, `patch-decay-audit`) all open PRs but never auto-merge.
- **`agent-pilot` label gate, 30 days per skill.** For the first 30 days after a skill ships, every invocation requires the `agent-pilot` label on the triggering PR or issue. No label → the skill dispatch is silently skipped. Graduation is a PR that drops the gate.
- **Reviewer-side agent is a required gate from Day 0.** `bf-pr-review` submits `--approve` or `--request-changes` on every non-draft PR. A `--request-changes` review blocks merge under branch protection.
- **No writes to `vllm-project/vllm`.** Agents may read upstream but never PR, comment, label, or close. Upstream contributions go through a human-clicked link prepared by `bf-vllm:upstream-patch`.
- **No releases.** `bf-release.yml` is human-only — agents can't tag or publish.
- **No CI self-edits.** Agents may not modify `.github/workflows/bf-*` files. Changes to the agent system itself stay human-authored.
- **No GPU spend without an explicit budget envelope** in the task prompt. GPU runners cost real money; this is the hardest financial guardrail in ADR-0006.
- **No force-push to `main`, `upstream-main`, or release branches; no GitHub repo create / archive / rename.** See ADR-0006 for the complete hard-guardrail list.

Every agent-authored action carries an `[agent][<skill-id>]` prefix in PR titles and an `ai.bf-vllm.build.author=agent:<skill-id>` OCI label on any built image.

## Build channels and image labels ([ADR-0004](./adr/0004-build-channels-versioning-labels.md))

Three image channels, each with a distinct consumer:

- **`release`** — `ghcr.io/blackfuel-ai/bf-vllm/vllm-openai-rocm:0.20.2_bf.0.1.0`. Reviewed, blessed, what production pulls.
- **`upstream`** — its own repository `ghcr.io/blackfuel-ai/bf-vllm/upstream/vllm-openai-rocm`, bare vLLM-identical tags (`:latest`, `:vX.Y.Z`, `:sha-<short>`). Built off bare `upstream-main` on every sync. Fast-path for new-model availability — preview deployments use this. Routed by repository path, not by label (ADR-0004 amendment).
- **`dev`** — shared repository `ghcr.io/blackfuel-ai/bf-vllm/vllm-openai-<arch>`, label-routed (`ai.bf-vllm.build.channel=dev`), never auto-deployed. Per-PR builds tag `:dev-pr-N-<sha>`; a manual dev build tags `:<base-version>-dev-<sha7>` (e.g. `:v0.23.1rc0_bf.0.1.0-dev-aebd933`), the tree's clean `git describe` base plus the commit (bare sha7, no git-describe `g` prefix), so the engine — which shows the tag verbatim — displays a version, not a bare sha. `bf-tools/version.py:dev_image_tag()` derives it. See [ADR-0004](./adr/0004-build-channels-versioning-labels.md) 2026-06-22 dev-tag amendment.

Versioning is `v<upstream-semver>+bf.<bf-semver>` (e.g. `v0.20.2+bf.0.1.0`). OCI image tag is the same string with `+` → `_` per OCI naming rules. See [ADR-0004](./adr/0004-build-channels-versioning-labels.md) for the full label schema (`org.opencontainers.image.*` + `ai.vllm.*` + `ai.bf-vllm.*`).

## PR title convention

Every PR title (and the commit subject it lands as on `main`) follows:

```
[Type][Area] short subject in present-tense imperative
```

Two brackets, separated by no space, then a single space, then the subject.

**Type** (first bracket) — one of:

| Type | Meaning |
|---|---|
| `[Feat]` | New feature or capability. |
| `[Fix]` | Bug fix. |
| `[Docs]` | Documentation-only change (ADRs, READMEs, this file). |
| `[Refactor]` | Code change that neither adds a feature nor fixes a bug. |
| `[Chore]` | Housekeeping (deps bump, lockfile regen, license headers). |
| `[Test]` | Tests only — no behavior change. |
| `[CI]` | Pure CI/workflow change with no other component touched. (Use `[Feat][CI]` or `[Fix][CI]` when there's also a feature or fix narrative.) |
| `[bf-patch]` | **Required** for any commit editing an upstream file (ADR-0003). Pair with the touched-file area. |
| `[Sync]` | Upstream sync PR (typically authored by the sync bot). |
| `[Release]` | Release/tag PR. |

**Area** (second bracket) — short, scanable label for the touched surface. No hard dictionary; common values:

`[CI]` · `[bf-tools]` · `[bf-docs]` · `[ADR-NNNN]` · `[CODEOWNERS]` · `[Makefile]` · `[ROCm]` · `[CUDA]` · `[CPU]` · `[XPU]` · `[TPU]` · `[Sync]` · `[Build]` · `[Release]`

Pick the most specific one. If the change spans many areas, pick the dominant one — or split the PR.

**Examples** (representative from current open PRs):

- `[Feat][CI] add .bf-paths manifest + bf-classification-lint workflow`
- `[Feat][bf-tools] add version helper per ADR-0004`
- `[bf-patch][CODEOWNERS] route Blackfuel-owned paths to core owners`
- `[Docs][ADR-0005] switch build cache from Scaleway S3 to Blacksmith-native`

**Enforcement**: not wired today — this is convention, not lint. A small `bf-pr-title-lint` step inside `bf-lint.yml` is a candidate follow-up; it would regex-check `^\[[A-Za-z][A-Za-z0-9-]*\]\[[^\]]+\] \S`, exempt Renovate/Dependabot/sync-bot authors, and start as a soft-fail before being promoted to a required check.

**Why this shape**:
- Two brackets match vLLM upstream's scanable style (`[CI]`, `[Bugfix]`, `[Doc]` are everywhere in the upstream PR feed).
- Type + Area separation matches Conventional Commits semantics, which ai-platform already uses internally.
- `[bf-patch]` survives as the literal ADR-0003 marker — `bf-patches-trailer-lint` looks for it on commit subjects too.

## Code comment discipline

**The code is the source of truth; comments are the exception, not the default.** Readable code — clear names, small steps, obvious structure — carries the *what*. A comment earns its place only by adding what the code cannot say itself: a **surprising why**. Default to no comment; add one when a reader who understands the code would still be surprised, or would undo something load-bearing without the note.

Comment the non-obvious *why*, never the *what*. If a comment restates what the next line plainly does, delete it. Do not narrate the design, re-explain a standard mechanism (what `fail-fast`, a matrix, an `if:` guard does), or justify a choice the reader would already accept — that is prose the code does not need. Prefer making the code clearer (a better name, an intermediate variable) over explaining unclear code with a comment.

Keep each comment to the fewest words that land the idea — usually one line. A multi-paragraph comment is a smell: either the idea belongs in an ADR/`bf-docs` (link to it) or the code should be refactored so the comment isn't needed. Reserve length for the rare genuinely-subtle trap (a footgun a future reader *would* hit — e.g. a working-directory gotcha, an injection surface), and even then state it once, tersely.

A file that is mostly comments is a defect, not thoroughness: it buries the code that is supposed to be read.

Forbidden are change-narration comments: `# changed from X`, `# new approach`, `# previously we did Y`, `# now using Z instead`, `# fixed bug where…`, `# was …`, and the like. A reader of the code at any point in time should see only the present state described; whoever needs the "before" reaches for `git blame` and `git log`.

Two kinds of comment that *look* historical are legitimate and must be preserved, not stripped:

- **Persistent upstream/external constraints** — e.g. `# 1-indexed because the upstream API returns a 1-indexed list`. These explain *why the current code is shaped this way* and stay true for as long as the constraint holds.
- **Deprecation notices marking a deliberate transition state** — e.g. `# Deprecated: use new_auth() instead; remove after the v2 migration`. These follow the phased-migration convention and describe a current, intentional state rather than narrating a past edit.

This applies equally to bf-authored code and to `[bf-patch]` edits of upstream files: do not narrate the patch in a comment. The `[bf-patch]` subject, the `Upstream-status:` trailer, and git history already carry that story ([ADR-0003](./adr/0003-change-classification-and-patch-discipline.md)).

## Documentation index

- [`bf-docs/README.md`](./README.md) — BF doc index.
- [`bf-docs/adr/`](./adr/) — Architecture Decision Records, numbered 0001+.
- [`bf-docs/template.md`](./template.md) — ADR template.

## When in doubt

Open an issue against `blackfuel-ai/bf-vllm` and tag the runtime team. ADRs are normative; this file is descriptive — if they disagree, ADRs win.
