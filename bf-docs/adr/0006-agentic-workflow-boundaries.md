# ADR-0006: Agentic workflow boundaries (Day 0)

**Status**: Accepted
**Date**: 2026-05-17

## Context

Maintaining a vendor distribution of vLLM is a cost that scales with human attention. Upstream ships hundreds of PRs per week. Patches accumulate. Backend drift (CUDA vs ROCm vs TPU) needs constant attention. Each new model architecture is a triage event.

`bf-vllm` is, by design, a playground for autonomous agents to amortize that cost. But agentic autonomy is also a risk:

- Bad PR to `vllm-project/vllm` damages our reputation with maintainers permanently.
- Agent-merged code can ship a regression to production.
- Drive-by automation on external repos is a trust violation, not a feature.

We need explicit boundaries that protect against worst-case agent behavior without throttling the value.

## Decision

**Hard guardrails — agents may NOT, at any phase**:

- Push directly to `main` or `upstream-main`. PRs only.
- Force-push `main`. Sync workflow uses normal merge (ADR-0002).
- Force-push release branches.
- Modify CI workflows under `.github/workflows/bf-*` (changes to the agent system itself stay human-authored).
- Create, archive, or rename GitHub repositories.
- Issue releases (run `bf-release.yml`). Releases are human-decided events.
- Spend GPU runner time without an explicit budget envelope in the task prompt.
- **Interact with `vllm-project/vllm` in any write capacity** — no PRs, no comments, no labels, no closes. Read-only. By extension, no `bf-*` workflow may carry a token with write access to `vllm-project/vllm`.

**Day 0 skill set** (six author-side skills, all advisor-only for the first 90 days):

| Skill | Mode | Purpose |
| --- | --- | --- |
| `bf-vllm:resolve-sync-conflict` | Reactive (sync workflow dispatches on conflict) | Drafts a proposed conflict resolution on the sync PR; human reviews and merges |
| `bf-vllm:harmonize-backends` | Scheduled (weekly) | Scans upstream changes for AMD/Nvidia/TPU drift; files an issue with a proposed fix |
| `bf-vllm:upstream-patch` | Label-triggered (`ready-for-upstream`) | Prepares a branch on `blackfuel-ai/vllm` with a draft upstream PR; **human clicks the link** to open the PR on `vllm-project/vllm` |
| `bf-vllm:track-upstream-models` | Scheduled (daily) | Detects new model architectures in upstream commits; files an issue with deployment recipe |
| `bf-vllm:auto-research` | Issue-triggered (`research-question` label) | Explores a research question with explicit per-phase human approval gates |
| `bf-vllm:patch-decay-audit` | Scheduled (post-release + weekly) | Flags stale `candidate` patches past the 14-day rule; suggests action |

### Reviewer-side agent (exception to advisor-only)

The PR review pipeline (`.github/workflows/bf-pr-review.yml`, ADR-0006-companion) runs `blackfuel-ai/bf-review-action` on every non-draft PR and submits an `--approve` or `--request-changes` review. The `bf-pr-review / verdict` status check is **a required status check on `main` from Day 0** — a `--request-changes` outcome blocks merge.

This is an explicit, narrow exception to the "advisor-only for the first 90 days" rule, scoped only to PR-merge-gating:

| Rule | Author-side skills (six above) | Reviewer-side agent (`bf-pr-review`) |
| --- | --- | --- |
| Can land code on `main`? | No — humans review and merge | No — only acts as a reviewer, never as a committer |
| Status today | Advisory only for 90 days | Required gate from Day 0 |
| Blast radius | One PR per invocation, human-merged | One PR per invocation, but only blocks merge — never lands code |

Why the exception:

- A reviewer agent's worst case is a false-positive `--request-changes`, which is **annoying but reversible** (a human dismisses the review and merges). It cannot ship a regression to production.
- The author-side skills' worst case is shipping bad code; those need the 90-day window.
- Asymmetry of cost ⇒ asymmetry of policy.

If the reviewer agent's false-positive rate becomes intolerable (defined as: ≥1 dismissed `--request-changes` per workday over a rolling 7-day window), the gate is demoted to advisory and an issue is opened to retune the verdict prompt. The dismissal counter is sourced from GitHub's review-event log.

### `agent-pilot` label gate (author-side only)

**`agent-pilot` label gate**: for the first 30 days, every skill invocation requires the `agent-pilot` label on the triggering PR or issue. No label → no agent action. Graduation criterion: 30 days of clean track record per skill, removed via a PR to drop the gate. After 30 days a skill graduates; after 90 days the supervised window ends entirely.

**Observability**: every agent-authored action carries `[agent][<skill-id>]` prefix in PR titles and an OCI label `ai.bf-vllm.build.author=agent:<skill-id>` on any built image. Every invocation logs to `bf-tools/agent-log/` (rotated CSV). Weekly digest posted to `#bf-vllm-agents` Slack.

## Alternatives considered

- **No agents at Day 0** — defer agentic workflow entirely until the manual baseline is stable. Rejected: the value of agents accrues with usage and log data; deferring loses that runway. Advisor-only mode is low-risk.
- **Full agent autonomy from Day 0**. Rejected: no track record yet; no way to measure success rate; bad agent action on `vllm-project/vllm` is irreversible reputationally.
- **Per-skill graduation criterion based on success-rate, not calendar**. Considered — more rigorous, but harder to track and explain. Calendar (30 days then 90 days) is simple and revisitable.
- **Allow agent writes to `vllm-project/vllm` if reviewed by a human first**. Rejected: defeats the boundary; once a token has write access to upstream, a single bug can spam.

- **Reviewer-side agent also advisory for 90 days** (initial draft of this ADR). Rejected after the bf-pr-review workflow PR landed: a reviewer agent cannot ship code to production — its worst case is a noisy `--request-changes` that a human dismisses. Gating from Day 0 is strictly safer than gating later, because it forces the verdict prompt to harden against false positives early while the PR volume is small.

## Consequences

- **Author-side agents are advisors, not actors**, for the first 90 days. They propose; humans decide. This caps the value (advisor-only is slower than full automation), but caps the risk too. The reviewer-side agent (`bf-pr-review`) is an explicit exception — see the "Reviewer-side agent" subsection above.
- **Boundary at `blackfuel-ai/vllm`**: agents push topic branches to our public fork (which they own), but never open PRs from there to `vllm-project/vllm`. The transition to upstream is a human click on a prepared link.
- **The `agent-pilot` label** lives only for 30 days per skill. Removing it is a PR to graduate the skill. Visible, reviewable, reversible.
- **Observability is non-negotiable**. We must be able to distinguish agent work from human work at every layer (PR titles, image labels, logs). Without that, we can't measure value and we lose trust.
- **Day 90 retro**: the supervised window ends after 90 days *if* metrics support it. If a skill is regressing or causing churn, we extend supervision, not graduate it. Honest assessment over deadline.
- **Skill-evaluator (Phase 2 skill, not Day 0)** lands once we have ~30 days of `agent-log` data to reason over. It reads the log, identifies failure patterns per skill, proposes prompt/tooling improvements. Recursion that makes the playground compounding.

## References

- ADR-0001: Private mirror with repurposed public fork (where upstream PRs are launched from)
- ADR-0003: Patch discipline (what `upstream-patch` skill operates on)
- `.claude/skills/bf-vllm-*/` (skill definitions, future PR)
- `bf-tools/agent-log/` (invocation history)
- `.github/workflows/bf-pr-review.yml` (the reviewer-side agent governed by this ADR)
- `blackfuel-ai/bf-review-action` (the action backing `bf-pr-review.yml`)
- Linear: BLA-1445 §A.4 (Day 0 skill set sign-off), BLA-1483 (bf-review-action v1 tag)
