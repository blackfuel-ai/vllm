# ADR-0004: Three image channels, versioning, OCI label conventions

**Status**: Accepted (amended 2026-06-22 — see Amendment: upstream channel registry)
**Date**: 2026-05-17

## Context

`bf-vllm` produces container images for production. We need three different categories:

1. **Releases** that the engine deploys to production — must be reproducible, immutable per version, versioned in a way that conveys "this is upstream X plus our changes."
2. **Fast-path images** that track upstream HEAD — for serving new models the same day they land upstream, before our review cycle catches up.
3. **Dev images per PR** — for testing changes before merge; the deploy pipeline must refuse to auto-deploy them.

Each category has different stability, freshness, and trust properties. Engine routing must distinguish them mechanically (a label, not a tag-name regex).

Versioning is also a constraint: we ship downstream of an upstream version. SemVer 2.0.0 has a slot for that (build metadata, `+`); PEP 440 has the same idea (local version identifier, `+`). OCI image tags don't allow `+`. We need a single human-readable version that round-trips between git tags, image tags, and labels.

## Decision

**Three build channels**, declared via the `ai.bf-vllm.build.channel` OCI label:

| Channel | Repository | Content | Image tag pattern | Engine routing |
| --- | --- | --- | --- | --- |
| `release` | `…/vllm-openai-<arch>` | Upstream + BF patches + additive content, versioned & reviewed | `:0.20.2_bf.0.1.0`, `:0.20.2_bf-latest`, `:bf-stable` | Production pool |
| `upstream` | `…/upstream/vllm-openai-<arch>` | Bare upstream only, no BF layer; our Dockerfile, upstream's vLLM source | `:vX.Y.Z` (at a tag), `:<scm-version>` (between tags), `:latest` | Preview pool only — **distinguished by repository, not label** |
| `dev` | `…/vllm-openai-<arch>` | PR / manual / local | `:dev-pr-<N>-<sha>`, `:<base-version>-dev-<sha7>` | Never auto-deployed |

The `upstream` channel lives in a **dedicated repository** (`upstream/vllm-openai-<arch>`), so the registry path itself is the channel signal (see Amendment below). The other two channels share the main `vllm-openai-<arch>` repository and are distinguished by the `ai.bf-vllm.build.channel` label as originally decided.

**Versioning scheme**: `v<upstream-semver>+bf.<bf-semver>` — e.g., `v0.20.2+bf.0.1.0`.

- SemVer 2.0.0 §10 build-metadata form. Also valid PEP 440 local version (`<public>+<local>`).
- BF SemVer semantics: MAJOR = breaking BF surface change; MINOR = new BF content; PATCH = bugfix.
- OCI image tag form substitutes `+` → `_`: `0.20.2_bf.0.1.0`. Standard mapping (GoReleaser, ko, kustomize).

**OCI label conventions** — three namespaces, no overlap, each owns its semantics:

```text
# OCI standard — describes the artifact actually built
org.opencontainers.image.source        = https://github.com/blackfuel-ai/bf-vllm
org.opencontainers.image.revision      = <bf-vllm main SHA>
org.opencontainers.image.version       = 0.20.2+bf.0.1.0
org.opencontainers.image.url           = <Actions run URL>
org.opencontainers.image.title         = vLLM (Blackfuel, Nvidia|ROCm)
org.opencontainers.image.vendor        = Blackfuel

# ai.vllm.* — vLLM-defined labels, kept with their ORIGINAL semantic (upstream values)
ai.vllm.build.commit                   = <vllm-project/vllm SHA — upstream provenance>
ai.vllm.image.tag                      = 0.20.2 (the upstream tag)
ai.vllm.build.target-arch              = <build config — actual>
ai.vllm.build.cpu-x86                  = <actual>
ai.vllm.build.python-version           = <actual>

# ai.bf-vllm.* — concepts vLLM doesn't have; parallel structure to ai.vllm.*
ai.bf-vllm.build.commit                = <bf-vllm main SHA — same as image.revision>
ai.bf-vllm.image.tag                   = 0.20.2_bf.0.1.0
ai.bf-vllm.build.channel               = release | upstream | dev
ai.bf-vllm.bf.version                  = 0.1.0 (only when channel=release)
ai.bf-vllm.scm.version                 = <git-describe of the tree, +→_> (provenance; see 2026-06-22 dev-tag amendment)
```

The namespace rule is unambiguous: **a label's namespace identifies who defined it, not what value it holds**. `ai.vllm.*` labels keep their original semantic (so vLLM-native tooling works on our images unmodified). `ai.bf-vllm.*` holds BF-specific facts and parallel-structured provenance for our build.

`ai.bf-vllm.build.channel` is still set on every image (it documents what was built), but for the `upstream` channel it is **descriptive provenance, not the routing input** — the repository path is authoritative there (see Amendment below). For `release` and `dev`, which share one repository, the label remains the routing input as originally decided.

## Amendment (2026-06-22): the upstream channel has its own registry

The `upstream` channel is moved out of the shared `vllm-openai-<arch>` repository into a dedicated `upstream/vllm-openai-<arch>` repository, and engine routing for it is **by repository path, not by label**.

The original design used one repository per arch and a single distinguisher — the `ai.bf-vllm.build.channel` label — to keep one source of truth. In practice the upstream channel wanted bare, vLLM-identical tags (`:v0.23.1rc0`, `:latest`) so an image is recognisable as "upstream vLLM X" at a glance and so vLLM-native tooling reads it unmodified. In a shared repository that is impossible without a disambiguating tag prefix (`upstream-…`), which re-encodes the channel a second time — the label says `channel=upstream` *and* the tag carries `upstream-`. That is two parallel encodings of the same fact: the redundancy the ADR set out to avoid, reappearing in the tag namespace.

Giving the upstream channel its own repository collapses that back to a single source of truth, located at the registry: the path `…/upstream/vllm-openai-<arch>` *is* the channel. The engine recognises an upstream image directly from where it pulled it, with no label lookup and no tag-prefix parsing, and the tags can be bare. The production pool simply never points at the `upstream/` repository, so an upstream image cannot reach production structurally — a stronger guarantee than a label the engine must remember to check, and the right risk direction for a preview-only channel.

This is a deliberate, scoped reversal of "route by label, not by location" — **for the upstream channel only**. It holds exactly one invariant in exchange: **push access to each repository is restricted to the workflow that owns it.** Location-as-trust is only sound if nothing untrusted can land at a trusted location; the `upstream/` repository accepts pushes only from the from-upstream build workflows, and the release repository only from the release workflow. With that lock in place the registry path is a trustworthy channel signal.

`release` and `dev` are **not** moved — they share the `vllm-openai-<arch>` repository and stay label-routed. Splitting all three was considered and rejected: release↔dev distinction is low-risk (both are BF-layer builds in the same trust family) and a per-PR `dev` repository multiplies registry credentials and GC policy for no safety gain. Only `upstream` — the one channel that is a different trust family (bare upstream, no review) and wants vLLM-native tags — earns its own repository.

Upstream tag derivation: the version name is taken the way vLLM takes its own (`setuptools-scm`, i.e. `git describe` against upstream tags), `+` → `_` sanitised for OCI. A build exactly at an upstream tag is the clean name (`v0.23.1rc0`); a build between tags carries the scm dev-distance suffix, which honestly marks it as not-a-release. `:latest` moves only when building the canonical `upstream-main` ref.

## Amendment (2026-06-22): dev tags embed the scm-describe version

The `dev` channel tag is `:<base-version>-dev-<sha7>` (e.g. `:v0.23.1rc0_bf.0.1.0-dev-f107520`), not the bare `:dev-<sha>` of the original table.

The engine renders an image tag **verbatim** wherever it shows a version — the model-version row, the deployment detail, benchmark rows — with no label lookup and no parsing (it stores `image_tag` as a plain string and echoes it). A bare `:dev-<sha>` therefore surfaces to an operator as a commit hash with no version, which is exactly the readout the channel split set out to make legible. Putting the version *in the tag* is the only lever that changes that surface, because the tag is the one field the engine displays.

The base version is the bf-vllm tree's own `git describe` (`setuptools-scm` form), `+` → `_` sanitised for OCI — the same scheme the upstream channel already uses, applied to the BF tag line. It is taken **BF-first, upstream-fallback**: `git describe --match 'v*+bf.*'`, and if no BF release is reachable (a feature branch cut before the release, or a fresh upstream line with no BF release yet) it falls back to `--match 'v[0-9]*'`. The fallback always resolves on the mirror, so the derivation never fails a build. git-describe's between-tags distance suffix (`-<commits-since-tag>-g<sha>`) is stripped to keep the base clean; the commit is instead carried as a trailing `-dev-<sha7>` marker — the `-dev-` placed *before* the sha so the version reads cleanly up to the marker. Unlike git-describe, the sha carries no `g` prefix: this is our own tag format, and a bare `<sha7>` is what an operator pastes straight into `git show`. The `<sha7>` suffix keeps the tag unique per commit, so `:dev-<sha>`'s uniqueness and per-commit `cancel-in-progress` are preserved. `bf-tools/version.py` exposes `dev_image_tag()`/`parse_dev_image_tag()` so the derivation lives in one place rather than being hand-assembled in each build workflow. The per-PR `:dev-pr-<N>` form is unchanged and is added by the PR-triggered build wiring.

A new descriptive label `ai.bf-vllm.scm.version` carries the same scm-describe version on every channel, so an agent reading labels (`get_image_labels`) gets the lineage even where it would otherwise have to parse the tag. It is provenance, not a routing input.

## Amendment (2026-06-23): image repos follow the `vllm-openai[-<accel>]` upstream convention

The three OpenAI-server image repos take vLLM's own published-image stem `vllm-openai`, suffixed by accelerator: `vllm-openai` (CUDA, the unsuffixed default), `vllm-openai-rocm`, `vllm-openai-cpu`. Earlier these were `vllm`, `vllm-rocm`, `vllm-cpu` (the `-openai` stem dropped, CUDA bare `vllm`). The bench repo stays `vllm-cpu-bench`, unchanged.

vLLM publishes its server images as `vllm/vllm-openai` (CUDA), `vllm/vllm-openai-rocm`, and `vllm/vllm-openai-cpu`. Matching that stem makes the BF repos recognisable as the vLLM OpenAI-server image for each accelerator at a glance, and keeps the `-<accel>` suffix the only thing that varies — the same dimension the build workflows already key on. CUDA stays unsuffixed (`vllm-openai`) because it is upstream's default and the most-pulled image.

The bench image keeps `vllm-cpu-bench` and does **not** take the `vllm-openai` stem: it is a `vllm bench` tool image (latency/throughput), not an OpenAI-compatible API server, and has no upstream `vllm-openai-*` analog — naming it `vllm-openai-cpu-bench` would falsely imply a server. It is never engine-routed, so it is exempt from the server-image convention.

This renames the GHCR repository paths production and previews pull from, so consuming charts and deployments (the engine inference chart, any pinned `image:` refs) must move to the new paths in lockstep before the next release re-cut — there is no dual-publish window.

## Alternatives considered

- **Single `:latest` image with embedded version metadata, no channel labels.** Rejected — engine can't route reliably; no way to refuse to deploy a `dev` image without parsing tags.
- **Use `-bf` separator instead of `+bf`** (e.g., `v0.20.2-bf.0.1.0`). Rejected — `-bf.*` is technically a SemVer pre-release identifier, which makes our build version sort *less than* upstream's. The `+` form correctly marks us as a downstream build of the upstream version, per SemVer §10.
- **Custom `bf.patches-revision` and `bf.files-revision` labels.** Considered, then dropped — the linear-main model means `image.revision` (the `main` SHA) captures everything. Recovering specific patches/files is `git log` filtered by trailer, a forensic query, not a label.
- **Overload `ai.vllm.*` labels with bf-vllm values** (e.g., `ai.vllm.build.commit` = bf-vllm SHA). Rejected — silently breaks vLLM-native tooling that expects upstream provenance.
- **Keep the upstream channel in the shared repository with a label-authoritative `upstream-` tag prefix** (the original design). Reconsidered in the 2026-06-22 amendment and narrowed — it forces the channel to be encoded twice (label + tag prefix) and blocks bare vLLM-identical tags. Replaced by a dedicated upstream repository for that channel only.
- **Split all three channels into per-channel repositories.** Rejected — release↔dev are the same trust family and gain nothing from physical separation, while per-PR dev repositories multiply credentials and GC policy. Only `upstream` (different trust family, wants vLLM-native tags) earns its own repository.

## Consequences

- **Single canonical version per release** that works as git tag, GitHub release name, and (with `+` → `_`) OCI image tag. `bf-tools/version.py` exposes mechanical `to_image_tag()`/`from_image_tag()` for release tags and `dev_image_tag()`/`parse_dev_image_tag()` for the `dev` channel's `-dev-<sha7>` form.
- **Engine routing has one source of truth per channel.** `release` and `dev` share a repository and are routed by the `ai.bf-vllm.build.channel` label. `upstream` is routed by its dedicated repository path; its label is descriptive only. No regex on image tags, no fragile string parsing, and no channel encoded twice.
- **The upstream repository carries bare, vLLM-identical tags** (`:v0.23.1rc0`, `:latest`), so an image is recognisable as a specific upstream vLLM and vLLM-native tooling reads it unmodified — at the cost of one operational invariant: push access to each repository is restricted to the workflow that owns it (location-as-trust is only sound if nothing untrusted can land at a trusted location).
- **vLLM-native tooling works unmodified** on our images — `ai.vllm.build.commit` still answers "which upstream is this?" with the upstream SHA.
- **Five image-tag aliases per release** (`:0.20.2_bf.0.1.0`, `:0.20.2_bf-latest`, `:bf-stable`) sounds like a lot, but each has a specific consumer:
    - `:0.20.2_bf.0.1.0` — immutable, what release notes point at, what reproducibility queries land on.
    - `:0.20.2_bf-latest` — newest BF release for this upstream version (useful for engine pinning during an upstream-line lifecycle).
    - `:bf-stable` — manually moved by engine team after staging bake; what "production" pulls.
- **One quirk**: `org.opencontainers.image.revision` and `ai.bf-vllm.build.commit` hold the same value. Same redundancy upstream has between `image.revision` and `ai.vllm.build.commit`. Keep both for namespace consistency; ~80 bytes per manifest.

## References

- ADR-0005: Build CI (how channel labels get set in practice)
- `bf-tools/version.py`
- vLLM upstream `docker/Dockerfile` (label definitions we inherit)
- SemVer 2.0.0 §10, PEP 440 local version identifiers
