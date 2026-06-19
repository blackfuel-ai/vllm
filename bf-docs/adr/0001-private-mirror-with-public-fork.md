# ADR-0001: Private mirror with repurposed public fork

**Status**: Accepted
**Date**: 2026-05-17

## Context

Blackfuel runs vLLM in production (ROCm 6.4.1/7.0.0, vLLM ≥ 0.19) and is building inference infrastructure on top of it. Until now, work happened on `blackfuel-ai/vllm` — a public fork of `vllm-project/vllm`. That setup was acceptable when the work was small and uncontroversial; it broke as soon as we started accumulating:

- Pre-disclosure performance work and scheduling tweaks.
- ROCm-specific patches that may never upstream cleanly.
- Infrastructure-specific build configuration (Blacksmith runners, GHCR paths, lightweight CPU bench images).
- Future kernel work and model-architecture explorations.

A public fork exposes all of this to anyone who looks. GitHub does not allow turning a public fork private.

We also have a real, ongoing need to **contribute back upstream**: many of our improvements are generally useful, and the right home for them is `vllm-project/vllm`, not our private repo.

## Decision

Two repos, two roles:

| Repo | Visibility | Role |
| --- | --- | --- |
| `blackfuel-ai/bf-vllm` | private | Canonical Blackfuel vLLM. All daily work. Builds production images. |
| `blackfuel-ai/vllm` | public | Upstream-PR staging only. `main` tracks `vllm-project/vllm:main` cleanly; in-flight contributions live on short-lived `bf-upstream-<topic>` branches. |

`bf-vllm` is bare-mirrored from `vllm-project/vllm` (selectively: just `main` + tags). All BF additions land there. The existing public fork is reset to a clean upstream mirror and repurposed as the launchpad for contributions back to `vllm-project/vllm`.

## Alternatives considered

- **Stay on the public fork.** Rejected — exposes IP, mixes experimental and production work, no clear authorization boundary.
- **Single private repo, contribute upstream via personal forks.** Rejected — every contributor would maintain their own fork, fragmenting reputation with vLLM maintainers and complicating attribution.
- **Single private repo with `git push` to upstream contributors' forks.** Rejected — same fragmentation, plus operational complexity in tracking which fork a PR is in flight from.
- **Archive `blackfuel-ai/vllm` entirely.** Rejected — we lose the established maintainer reputation built on that repo, and we'd need to create a new public fork for upstream PRs anyway. Repurposing is cheaper.

## Consequences

- **Two repos to maintain**, but they have orthogonal jobs: one is private and active, the other is public and mechanical (cron-synced from upstream, occasionally hosts a topic branch).
- **Upstream-PR identity stays consistent**: contributions continue to come from `blackfuel-ai/vllm`. Maintainers see a familiar name.
- **Existing GHCR image paths change**: from `ghcr.io/blackfuel-ai/vllm/*` to `ghcr.io/blackfuel-ai/bf-vllm/*`. Engine and other consumers need a coordinated cutover (Phase B of the migration).
- **Reset of `blackfuel-ai/vllm:main` is a one-time destructive op**: force-push to upstream HEAD + delete the 12 historical BF branches. The historical BF state is preserved on `bf-vllm` (every commit is migrated); old SHAs on the public fork stop being reachable.
- **No automated writes to `vllm-project/vllm`** at Day 0 (see ADR-0006). Upstream contributions are human-initiated.

## References

- ADR-0006: Agentic workflow boundaries
- `bf-vllm` PR #1 — infrastructure foundation
- Linear: BLA-1445 (pre-flight checklist)
