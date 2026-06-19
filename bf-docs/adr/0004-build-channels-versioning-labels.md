# ADR-0004: Three image channels, versioning, OCI label conventions

**Status**: Accepted
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

| Channel | Content | Image tag pattern | Engine routing |
| --- | --- | --- | --- |
| `release` | Upstream + BF patches + additive content, versioned & reviewed | `:0.20.2_bf.0.1.0`, `:0.20.2_bf-latest`, `:bf-stable` | Production pool |
| `upstream` | Bare upstream only, no BF layer; our Dockerfile, upstream's vLLM source | `:upstream-<sha>`, `:upstream-latest`, `:upstream-vX.Y.Z`, `:upstream-stable` | Preview pool only |
| `dev` | PR / manual / local | `:dev-pr-<N>-<sha>`, `:dev-<sha>` | Never auto-deployed |

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
```

The namespace rule is unambiguous: **a label's namespace identifies who defined it, not what value it holds**. `ai.vllm.*` labels keep their original semantic (so vLLM-native tooling works on our images unmodified). `ai.bf-vllm.*` holds BF-specific facts and parallel-structured provenance for our build.

## Alternatives considered

- **Single `:latest` image with embedded version metadata, no channel labels.** Rejected — engine can't route reliably; no way to refuse to deploy a `dev` image without parsing tags.
- **Use `-bf` separator instead of `+bf`** (e.g., `v0.20.2-bf.0.1.0`). Rejected — `-bf.*` is technically a SemVer pre-release identifier, which makes our build version sort *less than* upstream's. The `+` form correctly marks us as a downstream build of the upstream version, per SemVer §10.
- **Custom `bf.patches-revision` and `bf.files-revision` labels.** Considered, then dropped — the linear-main model means `image.revision` (the `main` SHA) captures everything. Recovering specific patches/files is `git log` filtered by trailer, a forensic query, not a label.
- **Overload `ai.vllm.*` labels with bf-vllm values** (e.g., `ai.vllm.build.commit` = bf-vllm SHA). Rejected — silently breaks vLLM-native tooling that expects upstream provenance.

## Consequences

- **Single canonical version per release** that works as git tag, GitHub release name, and (with `+` → `_`) OCI image tag. `bf-tools/version.py` exposes mechanical `to_image_tag()`/`from_image_tag()`.
- **Engine has one label to look at** (`ai.bf-vllm.build.channel`) to decide routing. No regex on image tags, no fragile string parsing.
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
