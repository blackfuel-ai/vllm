# ADR-0005: Build CI optimized for development velocity

**Status**: Accepted
**Date**: 2026-05-18

## Context

**The critical metric for `bf-vllm` is development velocity — wall-clock time from `git push` to a deployable image on a production cluster.** Everything else in the build CI is in service of compressing that loop.

Why velocity, not just correctness or coverage:

- Agents are the primary consumer of this CI (ADR-0006). The `bf-vllm:auto-research` skill iterates on questions like "does FlashAttention-4 give us throughput on H100?" by building a candidate, deploying to staging, measuring, and revising. **At 30 min per build cycle, an agent gets ~16 iterations per workday. At 5 min, it gets ~96.** Compounding effect on exploration throughput.

- Humans benefit too: the dev image lands on the PR, the engineer pulls it, tests against a staging deployment, iterates. Slow CI = context-switching while waiting = lost focus.

- Faster sync-to-production-image cycle = faster reaction to upstream model releases. Goal #1 in the spec is "first to serve a new upstream model." That requires the build CI to keep pace with upstream's release tempo, not lag it.

- vLLM's compile surface is large: C++ extensions, CUDA kernels (~thousands of nvcc invocations per build), HIP kernels for ROCm. A naive build on a 4-core GitHub-hosted runner takes 25-45 minutes cold, every time. **That's the baseline we're competing against.**

So the architectural question is: **what's the minimum-feasible time from PR open → deployable image, and what does each lever cost?**

## Decision

Build CI for ROCm + Nvidia (plus CPU variants), tuned aggressively for velocity. Concrete targets, with status:

| Build event | Target | Current achievable | Lever |
| --- | --- | --- | --- |
| **Sync → upstream `:latest`** rebuilt | ≤ 30 min | ~25 min | Blacksmith cache warm |
| **PR build (Python-only change)** | **≤ 5 min** | Achievable in v1 with wheel-base | Pre-compiled wheel base image |
| **PR build (C++/CUDA change, warm cache)** | **≤ 10 min** | ~6-9 min | Blacksmith cache, arch-list trim |
| **PR build (cold cache or invalidated)** | ≤ 30 min | ~20-30 min | arch-list trim |
| **Release build** | ≤ 15 min | ~7-12 min | All of the above + release tagging overhead |

### Levers, ordered by impact

**1. Blacksmith-native build cache (5-10× speedup on warm)** — the single largest lever. Without it, every PR pays the full compile cost; with it, most layers and most compiler invocations are cache hits. We use `useblacksmith/setup-docker-builder@v1` + `useblacksmith/build-push-action@v2` — the same Blacksmith-actions pattern as the public-fork `build-rocm-image.yml` we are inheriting from. (That fork additionally sets `USE_SCCACHE=1` for a local-disk sccache layer; we drop it because the Blacksmith builder already caches compiler output via [sticky disks](https://docs.blacksmith.sh/blacksmith-caching/dependencies-sticky-disks) — see Consequences for the rationale.) Per Blacksmith's [docker-builds caching docs](https://docs.blacksmith.sh/blacksmith-caching/docker-builds), "Docker layer caching executes within the same runners that process your GitHub Actions workflows" — cache lookups stay on the runner's local disk, no cross-region round-trip on every nvcc/hipcc invocation. Credentials are the standard `github.token` injected by the action itself — no separate bucket, no separate IAM application, no per-runner sccache config. Hit-rate targets: 70% ROCm, 85% Nvidia post-warm-up; build-action cache stats artifacted per build, weekly retrospective in `#bf-vllm-sync`. Cache reset: bust by changing the `cache-key` input on the action, or wipe a runner's sticky disk from the Blacksmith dashboard if a bad layer needs evicting.

**2. Architecture-list trim to production hardware (~25% cold-build speedup)** — every additional CUDA arch is a full nvcc pass. We target only what we actually run:

- **Nvidia**: `torch_cuda_arch_list=9.0;10.0` (Hopper H100/H200 + Blackwell B200/B300). No PTX fallback for older arches. vLLM's CMake auto-routes architecture-specific kernels (Machete, DeepEP, cutlass_scaled_mm) to `*a` variants via `cuda_archs_loose_intersection()`; we don't force `*a` explicitly.
- **ROCm**: `pytorch_rocm_arch=gfx942;gfx950` (MI300X/MI325X/MI355X). No `gfx90a`/MI210.

Override available for one-off builds via `workflow_dispatch` inputs (testing on A100, MI210, L40S kernel re-enablement, etc.).

**3. Blacksmith runners (~2× speedup vs GitHub-hosted)** — `blacksmith-4vcpu-ubuntu-2404` for GPU image builds, `ubuntu-latest` for CPU builds. Already used by the existing public-fork workflow; matches the established team paid plan. Faster CPUs, faster I/O.

**4. Per-PR concurrency cancellation** — `concurrency: { cancel-in-progress: true }` on PR-triggered builds. Pushing commit B over commit A cancels A's still-running build. Stops wasted CI on stale commits and prevents `:dev-pr-N-<sha-A>` images cluttering GHCR.

**5. (v1) Pre-compiled wheel base image** — separate the slow part (C++/CUDA compilation, hours of CMake) from the fast part (Python-only changes). Nightly job builds the wheel once; PR builds layer Python changes on top in ~30 seconds. Adopts the [NeMo-RL pattern](https://docs.nvidia.com/nemo/rl/latest/guides/use-custom-vllm.html) (NeMo-RL ships the wheel as `VLLM_PRECOMPILED_WHEEL_LOCATION`). Lands after v0 ships and we have a stable CI baseline to compare against.

All workflow filenames start with `bf-` per the mandatory prefix convention (ADR-0003). This is what distinguishes our workflows from upstream-inherited ones; the sync workflow **deletes** anything in `.github/workflows/` that doesn't match `bf-*.yml` as part of every merge, so inherited workflows never persist on `main`.

### Workflow inventory

Eight workflow files, all `bf-`-prefixed:

| File | Source Dockerfile | Notes |
| --- | --- | --- |
| `bf-build-rocm-image.yml` | `Dockerfile.rocm` | `release` + `dev` channels; Blacksmith cache on |
| `bf-build-nvidia-image.yml` | `Dockerfile` | Blacksmith cache on |
| `bf-build-cpu-image.yml` | `Dockerfile.cpu` | Blacksmith cache on; mainly for testing without GPUs |
| `bf-build-cpu-bench-image.yml` | `Dockerfile.bf-cpu-bench` | lightweight; for `vllm bench` runs |
| `bf-build-{rocm,nvidia,cpu,cpu-bench}-image-from-upstream.yml` | same Dockerfiles | bare-upstream source (`VLLM_REF=upstream-main` or a tag); produces the `upstream` channel |

The `*-from-upstream.yml` variants are the velocity speedrun for new-model availability: dispatched automatically by every sync, building bare-upstream HEAD and bare-upstream release tags as soon as they exist. That's how Goal #1 ("first to serve a new upstream model") materializes — model lands upstream, sync detects, image builds within minutes, preview engine deployment picks it up by tag pin.

## Alternatives considered

- **Build for all upstream-supported arches** (`7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX`, vLLM upstream default). Rejected — ~25-40% longer cold builds for hardware we don't run. Velocity > coverage of arches no Blackfuel engineer ever touches.

- **Force arch-specific `9.0a;10.0a`** explicitly. Rejected — vLLM's CMake auto-routes performance-critical kernels to `*a` variants via loose intersection. Forcing `*a` everywhere breaks portable kernels that only declare base arches.

- **External S3-backed sccache** (an own-region bucket, e.g. on Scaleway or AWS). Rejected — the bucket region is necessarily different from the runner region, so every `nvcc`/`hipcc` miss costs a cross-region round-trip. Beats no cache, but loses to a builder-local cache for the same hit rate, and adds a second IaC surface (bucket + IAM application + lifecycle policy) to maintain.

- **GitHub Actions cache (`type=gha`) only**. Considered — `type=gha` is good for Docker layer caching but doesn't cache individual compiler invocations across runners. The Blacksmith builder bundles both: layer caching plus compiler-output caching, both co-located with the runner pool. We don't need a second cache plane on top.

- **Standard `ubuntu-latest` runners for GPU image builds.** Rejected — ~2× slower; the team already pays for Blacksmith and uses it on existing public-fork workflows. No reason to regress. Also rules itself out by virtue of lever #1: the Blacksmith-native cache is only available when the build runs on a Blacksmith runner.

- **Per-commit incremental builds without a compiler-output cache** (relying on Docker layer caching alone). Rejected — works for very small Python-only changes but falls apart when any C++ unit changes, which invalidates the build stage. The Blacksmith cache catches what Docker layer caching misses.

- **Self-hosted GPU build runners with persistent ccache on local SSD.** Deferred to v1. Would push warm builds under 3 min, but adds operational burden (runner provisioning, monitoring, SSD management). Not worth the complexity at v0 when Blacksmith-native caching already hits the target.

- **Pre-compiled wheel base image at Day 0.** Deferred to v1. Adds a separate "build wheel" pipeline and image-layering discipline. Worth it once v0 baseline is stable and we have measured how many PRs are Python-only (estimated >60%).

- **Accept slow CI and compensate with longer cycle times.** Rejected on first principles — slow CI is the silent productivity killer of any project compiling at this scale. At our throughput, every minute saved per build compounds to hours of agent + engineer time per week.

## Consequences

- **Velocity is now a first-class metric.** Build time (per channel, per kind of change) is tracked weekly in `#bf-vllm-sync`. If a workflow change makes cold builds slower, the retro catches it. Slow PRs are debugged; fast PRs are not assumed.

- **Cost: Blacksmith CI minutes only.** No separate object-storage line item (the cache lives with the runner pool). Blacksmith bill scales with usage. Net cost is dominated by engineer + GPU time saved, so the financial trade is trivially in our favor.

- **No new infra-as-code commitments outside bf-vllm.** Build CI is fully self-contained inside `blackfuel-ai/bf-vllm` and the `useblacksmith/*` actions. There is no bucket, no IAM application, and no lifecycle policy to maintain in this repo or anywhere else in the org. Blacksmith itself is a paid SaaS configured at the GitHub-org level (subscription, runner pool sizing) — that's a one-time admin setting, not version-controlled infrastructure we own.

- **Architecture override is one click.** If someone needs to test on A100, `workflow_dispatch` accepts `torch_cuda_arch_list=8.0;9.0;10.0` for a one-off build without changing the default. No PR needed for ad-hoc arch experiments.

- **vLLM Dockerfile compatibility**: vLLM's CUDA Dockerfile supports `USE_SCCACHE=1`, but we don't rely on it — the Blacksmith builder caches at the Docker layer/compiler-output level instead, which works uniformly across both CUDA and ROCm Dockerfiles. The vLLM `USE_SCCACHE` plumbing is left at its default (off) on our build args.

- **Known vLLM sccache miss-rate bug** (`vllm-project/vllm#13697`, temp-dir paths leaked into compile hashes) is not relevant to us — we don't run vLLM's built-in sccache layer. Documented here only so future readers don't reach for `USE_SCCACHE=1` and rediscover the same trap.

- **L40S kernel JSONs** (from the public-fork `feat/l40s-fp8-kernel-configs` branch) ship in the image as data files, even though we don't compile for `sm_89`. Re-enable by adding `8.9` to the arch list via `workflow_dispatch` input. No code change needed.

- **The wheel-base v1 follow-up** is the single biggest velocity win remaining. Worth scheduling once v0 ships and we have a baseline to measure against.

- **Agent exploration scales with this CI's speed.** Doubling build speed roughly doubles how many candidate ideas the `bf-vllm:auto-research` skill can run through per week. The investment in Blacksmith caching + arch trimming pays directly into research throughput.

## Amendment (2026-06-15): sccache restored, backed by the Blacksmith-persisted BuildKit cache mount

The original decision (lever #1, and the two Consequences notes on `USE_SCCACHE`) dropped vLLM's sccache layer outright, on the reasoning that "the Blacksmith builder already caches compiler output via sticky disks." That reasoning conflated two distinct things: persistent *storage* (a Blacksmith sticky disk / a persisted BuildKit cache mount) is not the same as a *compiler cache* (sccache). A persisted mount gives you durable bytes across runs; it does not, by itself, turn an `nvcc`/`gcc` invocation into a cache hit. Docker layer caching only helps at the granularity of a whole `RUN` layer — the moment any C++/CUDA source in the build stage changes, that layer is invalidated and every compiler invocation in it re-runs from scratch, persisted storage or not. The thing that makes an *individual* compiler invocation a hit is a compiler cache keyed on the preprocessed source — that is exactly what sccache (or ccache) does, and nothing else in the original design did it. So the two were never substitutes; we removed the compiler cache and kept only the storage, and lost the per-invocation hit rate that motivated lever #1 in the first place.

Revised decision: **run sccache (the compiler cache) in its local-disk backend, with that local directory backed by a Blacksmith-persisted BuildKit `type=cache` mount (no S3).** Concretely, the build stage exports `SCCACHE_DIR=/sccache` and the wheel-build `RUN` mounts `type=cache,target=/sccache,id=vllm-nvidia-sccache,sharing=locked`. The Blacksmith Docker builder persists BuildKit cache mounts across runs automatically, so `/sccache` stays warm between builds — that mount, not a separate sticky-disk action, is the persistence layer. This keeps everything inside the Dockerfile build and the `useblacksmith/*` actions: no bucket, no IAM application, no second IaC surface, which preserves the "no new infra-as-code commitments" consequence intact. The S3 backend remains available for anyone who passes a non-empty `SCCACHE_BUCKET_NAME`; it simply is not what our build CI uses.

This is proven prior art, not a new invention: the public fork shipped exactly this pattern for ROCm in `blackfuel-ai/vllm@dc789f5` ("Enable local-disk sccache for ROCm image build"), which added `SCCACHE_DIR=/sccache` plus the `type=cache` mount on the wheel-build stages of `Dockerfile.rocm`, opting in via `USE_SCCACHE=1` and an empty `SCCACHE_BUCKET_NAME`. The NVIDIA `docker/Dockerfile` carried only the S3 plumbing, so a `[bf-patch]` adds the same local-disk path there; the `bf-build-nvidia-image-from-upstream.yml` workflow opts in the same way (`USE_SCCACHE=1`, empty `SCCACHE_BUCKET_NAME`).

The `vllm-project/vllm#13697` caveat — temp-dir paths leaking into compile hashes and depressing the sccache hit rate — does apply once sccache is actually running, where the original decision had dismissed it as irrelevant. The mitigation falls out of the design: a stable `SCCACHE_DIR` on the persisted mount (a fixed `/sccache`, not a per-build temp path) keeps the cache location constant across runs, so the hash instability that issue describes does not defeat the persisted cache. We watch the `sccache --show-stats` hit rate the build already prints before and after the wheel build; if it degrades, that is the signal to revisit, not a reason to drop the compiler cache again.

Net effect on the original consequences: lever #1 is unchanged in spirit (compiler-output caching co-located with the runner) but now names the correct mechanism; the two `USE_SCCACHE`-is-off consequence notes are superseded by this amendment.

## References

- ADR-0004: Build channels and image labels (where images go and how they're tagged)
- ADR-0006: Agentic workflow boundaries (the agents consuming this fast CI)
- `.github/workflows/bf-build-*.yml` (the eight build workflows)
- [`useblacksmith/build-push-action`](https://github.com/useblacksmith/build-push-action) (the cache-aware builder used by every `bf-build-*` workflow)
- [`useblacksmith/setup-docker-builder`](https://github.com/useblacksmith/setup-docker-builder)
- Public-fork reference workflow: [`blackfuel-ai/vllm:gemma4-mtp-rocm/.github/workflows/build-rocm-image.yml`](https://github.com/blackfuel-ai/vllm/blob/gemma4-mtp-rocm/.github/workflows/build-rocm-image.yml)
- [NeMo-RL custom vLLM (pre-compiled wheel pattern)](https://docs.nvidia.com/nemo/rl/latest/guides/use-custom-vllm.html)
- Blacksmith CI caching architecture: [Docker builds](https://docs.blacksmith.sh/blacksmith-caching/docker-builds), [Sticky disks](https://docs.blacksmith.sh/blacksmith-caching/dependencies-sticky-disks)
