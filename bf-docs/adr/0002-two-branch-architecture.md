# ADR-0002: Two-branch architecture, tag-gated merge sync

**Status**: Accepted
**Date**: 2026-05-17

## Context

`bf-vllm` is a long-lived fork of `vllm-project/vllm`. We need to:

1. Track upstream closely enough to serve new models early, **without putting untrusted upstream onto the branch we release from**. An arbitrary `vllm-project/vllm:main` HEAD commit is not a trusted artifact — it can be mid-refactor, carry a regression upstream has not yet caught, or sit between a breaking change and its follow-up fix. Only the commits upstream itself tags (`v0.23.0`, `v0.23.1rc0`, …) are points upstream has blessed; upstream tags a meaningful version roughly every two weeks.
2. Carry our own patches and additive content (plugins, kernel configs, build infra) without losing them when we pull upstream.
3. Keep release tags stable, reproducible, and *true* — a release named after upstream `v0.23.0` must contain exactly that upstream tree plus our layer, not arbitrary post-tag drift.

The conventional vendor-fork patterns are:

- **Single integration branch, force-pushed on every sync** (Linux kernel / RHEL kernel model). Compact history, automatic self-cleaning of upstream-incorporated patches, but force-pushes invalidate tags on `main` and break local clones.
- **Separate patch-queue branch, rebased onto upstream every sync**. Patches stay clean and upstreamable, integration branch stays merge-only, but adds a second mental model (where do my workflows live? on patches or files?).
- **Long-lived merge-based integration**. Standard GitHub pattern. No force-push, stable SHAs. The remaining question is *what* upstream point gets merged: every daily HEAD (fast but untrusted), or only tagged releases (trusted, the choice below).

## Decision

Two long-lived branches:

| Branch | Role | Push policy |
| --- | --- | --- |
| `upstream-main` | Fast-forward-only mirror of `vllm-project/vllm:main`. Source for nightly side-branches; carries the upstream tags. | Bot-only (sync App). No force-push, no delete. |
| `main` | Integration branch. Every BF change (additive or patch) lands here directly. **Sync merges upstream into `main` only at release-tag commits.** | Protected; normal PR-required flow. No force-push, no delete. |

No separate `bf/patches` branch. No force-push of `main`. Tags on `main` are stable forever.

**Tag-gated merge.** The sync workflow runs daily: it fast-forwards `upstream-main` to `vllm-project/vllm:main` and mirrors any new upstream tags (so tags are always visible). But it merges into `main` **only the commit of a new upstream release tag** — never `upstream-main`'s drifting HEAD. Most days upstream has no new tag, so most runs merge nothing. After every sync merge, `main`'s upstream layer is *exactly* a trusted upstream tag. This makes `main` trustworthy by construction: untrusted upstream cannot reach the branch we release from, and a release named after a tag genuinely contains that tag's tree plus our layer.

What counts as a "release tag" — finals only (`vX.Y.Z`) or also release candidates (`vX.Y.ZrcN`) — is the one tunable in this model; both are upstream-blessed points (see Alternatives).

When the merge happens, the common conflict case — upstream has merged a patch we previously cherry-picked — is auto-resolved with `--theirs` when the BF commit carries an `Upstream-status: pending-upstream-#NNNN` trailer. A second case — upstream independently shipping a feature that supersedes one of our patches — is deliberately left unresolved for an operator decision, never auto-dropped (both cases in ADR-0003).

**Untrusted upstream is opt-in, off `main`.** When BF needs upstream content *before* it is tagged — the Day-0 new-model case — a disposable branch is forked from `main`, the desired upstream content is merged into it (`upstream-main` HEAD, or a specific upstream branch / PR ref), and a **nightly** preview image is built from that branch (the `upstream` channel of ADR-0004: SHA-tagged, preview pool only, no release tag). The branch is discarded; its untrusted commits never reach `main`.

Release branches are **optional**. We tag `main` directly for releases — correct precisely because tag-gated merge leaves no post-tag drift on `main` to contaminate the release. A `release/v<upstream>+bf.<bf>.x` branch is only created when a release line needs hotfix isolation from `main` (e.g., backporting to an older upstream version that `main` has moved past).

## Alternatives considered

- **Merge `upstream-main` HEAD daily (every commit, not just tags).** Rejected: puts untrusted upstream (a mid-flight `vllm-project/vllm:main` commit, typically dozens-to-hundreds of commits past the last tag) onto `main` — the branch we release from. A release would then ship that drift, the release version becomes fiction (setuptools-scm computes a guessed `0.23.0.dev109+g<sha>` for an untagged commit), and the release machinery would have to either ship drift or reconstruct "tag + BF layer" out of band. The one advantage — urgent upstream fixes reach `main` within a day — is recovered without the cost by the `pending-upstream-#NNNN` cherry-pick (ADR-0003, the deliberate fast-path for an urgent fix, self-cleaning at the next tag merge) and by the nightly side-branch for Day-0 serving needs.
- **A git tag per daily sync** (e.g. `v0.23.0.dev{n}+bf.…dev{m}`), promoted to an official tag when upstream tags. Rejected: mints permanent, immutable git tags on untrusted commits — the exact thing the trust principle forbids — and asserts a guessed next-version upstream can contradict. The legitimate "moving dev identifier" need is met by the `upstream`-channel *image* tags (not git tags), which are explicitly preview-only.
- **Finals-only vs rc-and-final release tags** (the tag-gated tunable). Finals-only keeps `main` maximally trusted and advances it ~monthly, leaning on the nightly branch for anything between finals. Rc-and-final advances `main` ~biweekly, matching upstream's tag rhythm, at the cost of treating release candidates as trusted (they are upstream-blessed points, just not finals). Both are coherent; the choice sets how often `main` moves and how often we can release, and rc-and-final additionally requires the version helper to accept a pre-release upstream segment.
- **Force-push on `main` (vendor-kernel pattern).** Rejected: tags on `main` become orphaned commits after each sync; local checkouts need force-pull; PRs in flight have their bases rewritten. Cost > benefit in a GitHub-native team.
- **Separate `bf/patches` branch, rebased on every sync.** Rejected: doubles the branch model for daily contributors; requires our workflow files to be replicated onto the patch branch (CI on patch PRs needs the workflows visible); two CODEOWNERS surfaces. Originally proposed in early design, dropped after iteration.
- **Mandatory release branches for every release.** Rejected: extra ceremony with no win. Since `main` is never force-pushed and its upstream layer is always a trusted tag, tags on `main` are already stable and reproducible. Release branches are kept as an exception for hotfix isolation only.

## Consequences

- **`main` is trustworthy by construction.** Its upstream layer is always a tagged release; untrusted upstream cannot reach it. Releasing is "tag `main`", correctly — there is no post-tag drift to guard against, so the release named after upstream `v0.23.0` genuinely is that tree plus the BF layer.
- **`main` lags upstream by up to one tag interval (~2 weeks) of upstream fixes**, including bug/security fixes that have landed on upstream `main` but are not yet in a tag. The deliberate fast-path for an urgent such fix is an ADR-0003 `pending-upstream-#NNNN` cherry-pick onto `main` (self-cleaning at the next tag merge); the nightly side-branch covers Day-0 serving needs. This is the price of structural trust.
- **The sync runs daily but merges into `main` only on a new tag.** It still mirrors `upstream-main` and pushes tags every day (so tags are visible and nightly branches can fork), and `upstream-main` is the source for those nightly branches — but it is no longer a daily merge target for `main`.
- **`main` accumulates merge-commit topology** from tag merges. `git log --first-parent main` shows the release-tag cadence; `git log main` shows full upstream history. Both work.
- **BF commit SHAs are stable forever.** No need for `bf-tools/find-patch.py` to map "semantic patch identity → current SHA" (which would be required with rebase). Old SHAs in commit messages, PR descriptions, etc., all keep pointing at real commits.
- **One manual merge resolution per upstream-incorporated patch**, roughly 1-2 per month at our scale. This is the cost of merge over rebase — patches now in upstream become merge conflicts when sync pulls them in at the next tag. Auto-resolved when the trailer says so; flagged for human otherwise.
- **Release branches are optional, not standard.** Less ceremony for normal releases. The hotfix-isolated case (backport security fix to an old upstream version that `main` has moved past) is rare but documented.
- **Local development is normal-GitHub**: `git pull main` is always a clean fast-forward; PRs target `main`; no SHA churn.

## References

- ADR-0003: Change classification and patch discipline (defines `Upstream-status: pending-upstream-*`; the cherry-pick is the urgent-fix fast-path under tag-gated merge)
- ADR-0004: Three image channels, versioning, OCI labels (the `upstream` channel is the nightly side-branch output; the `release` channel pins to the tag on `main`)
- ADR-0006: Agentic workflow boundaries (sync App is bot-only on `upstream-main`)
- `.github/workflows/bf-sync-upstream.yml`, `.github/workflows/bf-release.yml`
