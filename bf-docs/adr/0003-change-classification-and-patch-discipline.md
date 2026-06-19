# ADR-0003: Change classification via `.bf-paths` manifest; patch discipline

**Status**: Accepted
**Date**: 2026-05-17

## Context

Every change on `bf-vllm` is one of two things:

- **Additive content** — files we own that upstream doesn't touch. Workflows, dockerfiles, kernel configs, plugin code, BF docs, build tools.
- **A patch to upstream files** — an edit to a file `vllm-project/vllm` owns. Carries the risk of conflict on every sync; upstream may incorporate or supersede it.

These two categories have very different lifecycle properties and review needs. Additive content lands and stays. Patches need a path to upstream, conflict-resolution discipline, and decay management.

We need a mechanical, lint-checkable rule that classifies any PR into one bucket — and prevents PRs from mixing the two.

## Decision

**A manifest at the repo root, `.bf-paths`, declares which paths are BF-owned.** It's a glob list (gitignore-style). Examples: `.github/workflows/bf-*.yml`, `docker/Dockerfile.bf-*`, `vllm/plugins/blackfuel/**`, MoE kernel JSONs matched by device name.

**Classification rule** (enforced by `bf-classification-lint.yml`, required check on every PR):

- PR diff matches only manifest globs → **additive PR** — direct to `main`, single review.
- PR diff matches only non-manifest paths → **patch PR** — same `main`, additional discipline.
- PR diff mixes both → **fails the lint** — author must split into two PRs.

**Patch discipline** (additional rules for patch PRs):

Every commit touching non-manifest paths must have:

1. Subject prefix `[bf-patch]`.
2. A trailer `Upstream-status: <state>` where state is one of:
   - `candidate` — we think it's upstreamable; default for new BF-authored patches.
   - `submitted-#NNNN` — upstream PR opened.
   - `pending-upstream-#NNNN` — we cherry-picked from an in-flight upstream PR.
   - `rejected` — upstream said no (link required).
   - `bf-only` — never going upstream (requires `Rationale:` trailer).

**14-day rule**: a patch with `Upstream-status: candidate` must transition to `submitted-*`, `bf-only`, or `rejected` within 14 calendar days. The `bf-vllm:patch-decay-audit` skill flags overdue patches with a triage issue.

**Cherry-pick from in-flight upstream PR**: a first-class pattern. When upstream has a PR (e.g., new model support) we want to use *now*, cherry-pick the PR's commits onto `main` with `Upstream-status: pending-upstream-#NNNN` + `Upstream-original-sha: <sha>` + `Cherry-picked-by: <us>`. Authorship preserved via git cherry-pick. When upstream merges, the next sync's `git merge` brings in upstream's canonical version; conflict is auto-resolved with `--theirs` on files touched by `pending-upstream-*` commits.

### Upstream supersedes a `bf-patch`: operator decides, never auto-drop

A distinct conflict case from the ones above. When a sync's `git merge` conflicts on a file because **upstream has independently built a feature that replaces one of our patches** — not merged our patch (that is the `incorporated` case below), but shipped its own implementation of the same capability — the conflict resolver must **not** silently take either side. Three outcomes are possible and only the operator chooses between them: keep our patch (upstream's approach is worse for us), take upstream's and retire our patch (upstream's supersedes ours), or hybridize (keep parts of each). Dropping a `bf-patch` is abandoning a deliberate Blackfuel decision, so it is an operator call, not a resolver default and not an agent default.

Mechanically this is **not** auto-resolvable: unlike `pending-upstream-*` (auto `--theirs`) or modify/delete (auto re-delete), there is no trailer that signals intent, because the patch's author never anticipated upstream replacing it. The sync's deterministic resolver therefore leaves these conflicts in place and the PR carries `needs-manual-rebase` (the same escape hatch used for any conflict the resolver cannot handle). The human or agent resolving the sync PR must, for each superseded patch, surface the three-way choice to the operator rather than picking one — a draft PR comment enumerating "patch X is superseded by upstream Y; keep / take-upstream / hybridize?" is the expected artifact. When the decision is *take upstream and retire ours*, the retiring commit records why in its body so the dropped patch remains auditable (the same rationale discipline `bf-only` patches carry). When the decision is *keep ours*, the patch's `Upstream-status:` is reviewed — a patch upstream has now solved its own way is a strong signal to move `candidate`/`submitted-*` toward `rejected` or `bf-only`.

### `bf-` filename prefix (mandatory)

Every BF-authored file under `.github/workflows/`, `.github/actions/`, `docker/Dockerfile.*`, and `requirements/*.txt` **MUST** have its filename prefixed with `bf-` (or `bf.` for file-extension-prefixed paths like `Dockerfile.bf-cpu-bench`). Concretely:

- `.github/workflows/bf-sync-upstream.yml` ✅
- `.github/workflows/bf-build-rocm-image.yml` ✅
- `.github/actions/bf-claudish-review/action.yml` ✅
- `docker/Dockerfile.bf-cpu-bench` ✅
- `requirements/bf-cpu-bench.txt` ✅

This is not just a stylistic convention — it serves **two enforcement purposes**:

1. **Manifest classification** — `.bf-paths` globs like `.github/workflows/bf-*.yml` rely on the prefix to identify BF-owned files. Without the prefix, the classification lint can't distinguish them from upstream-owned content in the same directory.

2. **Inherited-workflow elimination via delete-on-sync** — `bf-vllm` inherits every workflow file `vllm-project/vllm` has under `.github/workflows/`. Those workflows enforce upstream's contributor policies (require 4 merged PRs, AI-agent self-evaluation, daily scheduled macOS smoke) that are meaningless or wasteful on a private mirror. We **delete them as part of every sync merge**:

   - The sync workflow's prune step inspects `.github/workflows/` after merging `upstream-main`. Any file not matching `bf-*.yml` (or `bf-*.yaml`) is `git rm`'d in the same sync commit, along with supporting subdirs like `.github/workflows/matchers/` and `.github/workflows/scripts/`.
   - For the modify/delete conflict case (upstream modified a file we previously deleted), the sync workflow's conflict-resolution loop auto-resolves by re-deleting. Deterministic; no human required.
   - Sync PRs are exempt from `bf-classification-lint.yml` and `bf-patches-trailer-lint.yml` via the `[sync]` title prefix — they intentionally touch upstream-owned paths.

   Result: `main` never has any non-`bf-*` workflow file. File doesn't exist → workflow can't fire → cannot leak, period. No race window.

   **The `bf-` prefix is the keep-list**, not just a naming convention. Without it, distinguishing our workflows from inherited ones requires path-by-path allowlisting, which doesn't scale as upstream adds new workflow files.

## Alternatives considered

- **Author-intent classification** (let the author choose "additive" or "patch" in the PR description). Rejected — error-prone, no enforcement, easy to mix accidentally.
- **Separate `bf/files` branch for additive content + `bf/patches` for patches**. Rejected (see ADR-0002) — doubles branch model for daily contributors; classification by branch is the same lint check, just expressed structurally.
- **Implicit classification via path prefixes alone** (no manifest file). Rejected — paths drift, new BF-owned subtrees would need lint code changes; a manifest is one source of truth, editable by PR like any other file.
- **Loose trailer discipline (just `[bf-patch]` prefix, no required `Upstream-status:`)**. Rejected — patches accrete without anyone knowing their upstreaming state. The trailer + 14-day rule is the cost we pay for being able to answer "why does this patch still exist?" months later.
- **Per-workflow allowlist by GitHub workflow ID** (instead of filename prefix). Rejected — workflow IDs are generated by GitHub when a workflow is first registered; they're not stable across rename/move; and they're invisible in the repo (you'd have to query the API to know which is which). A filename prefix is visible, grep-able, and survives any operation that preserves filenames.
- **Move all upstream workflows out of `.github/workflows/`** (e.g. into `.github/disabled-workflows/`). Rejected — would be a `[bf-patch]` against every upstream workflow file, with the same modify/delete conflict surface as delete-on-sync. Delete is the simpler and more correct expression of intent.
- **Disable inherited workflows via GitHub's per-workflow API** (`gh api -X PUT .../workflows/{id}/disable`). Considered as the first iteration: less invasive (no git changes), reversible per-workflow. Rejected as the primary mechanism because: (a) GitHub registers workflows lazily — workflows that haven't fired yet aren't in the disable list, so there's a race window between first-trigger and disable; (b) "file exists but is disabled" is a weaker guarantee than "file doesn't exist"; (c) reviewers looking at `.github/workflows/` on `main` see clutter that has nothing to do with our CI. The per-workflow disable API may still be used as a one-time bootstrap before the first sync runs delete-on-sync.

## Consequences

- **`bf-` prefix is enforced by lint.** PRs that add a workflow file without the prefix fail `bf-classification-lint.yml` because the file lands outside the `.bf-paths` globs.
- **Inherited upstream workflows do not exist on `main`.** Every sync's prune step removes them; the modify/delete auto-resolution handles future upstream changes deterministically. Each sync where upstream touches a workflow file produces a small "prune inherited" diff in the sync PR.
- **Sync PRs are exempt from classification + trailer lints**, identified by the `[sync]` title prefix. Their commits intentionally span both manifest and non-manifest paths (the merge brings in everything; the prune deletes non-bf workflow files).
- **Path-based classification is mechanical and reliable.** Lint catches misclassification before review.
- **Adding a new BF-owned subtree requires updating `.bf-paths`** in the same PR. Reviewer sees both the new content and the ownership claim together.
- **Patch upstreaming is incentivized** by the 14-day clock. Forces deliberate decisions: upstream it now, or commit to `bf-only` with rationale.
- **`bf-only` patches stay forever**. That's fine — they're the genuinely BF-specific changes (infrastructure, customer-tuned heuristics, ROCm-specific guards we never want to ship upstream). The `Rationale:` trailer makes them auditable.
- **Cherry-pick mechanic lets us ship new upstream models within hours**, while still self-cleaning when the upstream PR merges. The `pending-upstream-*` trailer is the auto-resolution signal.
- **One quirk**: when upstream merges a patch we authored (`submitted-#NNNN`), the sync's merge conflicts on the same lines. We don't auto-resolve `submitted-*` (only `pending-upstream-*`), because the patch may have been edited during review. A human resolves and updates the trailer to `incorporated`.
- **Upstream superseding a patch is an operator decision, not a resolver default.** When upstream independently ships a replacement for a feature we patched, the sync conflict is left unresolved (`needs-manual-rebase`) and the three-way choice — keep ours, take upstream's, hybridize — is surfaced to the operator. Agents resolving a sync PR must not unilaterally drop a `bf-patch`; doing so abandons a deliberate decision without review. A retired patch records its retirement rationale and a kept patch revisits its `Upstream-status:`.

## References

- ADR-0002: Two-branch architecture (sync merge mechanics)
- `.bf-paths` (repo root)
- `.github/workflows/bf-classification-lint.yml`
- `.github/workflows/bf-patches-trailer-lint.yml`
