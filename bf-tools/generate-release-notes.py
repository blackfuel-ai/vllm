"""Generate the release-notes markdown body for a bf-vllm release.

Per ADR-0004 a Blackfuel release is a single canonical version
``v<upstream>+bf.<bf>`` tagged on ``main`` (ADR-0002: we tag ``main``
directly; release branches are optional, for hotfix isolation only). The
``bf-release`` workflow calls this module to render the GitHub-release body
from the commit range ``<prev-release-tag>..<release-ref>``.

The body has three sections, each derived mechanically from the commit
metadata produced by the change-classification discipline of ADR-0003:

* **Blackfuel changes** — commits authored on ``main`` that are not upstream
  syncs. Split into *patches* (subject ``[bf-patch]``, carrying an
  ``Upstream-status:`` trailer) and *additive* content (everything else).
  Patches are annotated with their upstream status so the release reader can
  see which carried changes are still candidates vs. already submitted.
* **Upstream changes pulled in** — the upstream history merged into ``main``
  over the range, summarised by the ``[sync]`` merge commits (ADR-0002's
  daily ``git merge upstream-main``). Each names the upstream short SHA it
  advanced to.
* **Images** — the GHCR pull strings for the release, one per arch target,
  using the OCI image-tag form from :mod:`version` (``to_image_tag``).

Importable (the workflow imports :func:`generate_release_notes`) and runnable
as a CLI (``python bf-tools/generate-release-notes.py --version v0.20.2+bf.0.1.0
--prev-tag v0.20.2+bf.0.0.1``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field

# bf-tools/ is hyphenated and therefore cannot be a Python package. Sibling
# modules are imported flat; pytest and the workflow both put this directory
# on sys.path (the workflow runs the script from within bf-tools/).
from version import parse_git_tag

# GHCR image repositories per arch target. The release workflow's
# ``image_targets`` input selects which to include; the keys here are the
# stable target names the workflow passes.
IMAGE_REPOS: dict[str, str] = {
    "nvidia": "ghcr.io/blackfuel-ai/bf-vllm/vllm-openai",
    "rocm": "ghcr.io/blackfuel-ai/bf-vllm/vllm-openai-rocm",
    "cpu": "ghcr.io/blackfuel-ai/bf-vllm/vllm-openai-cpu",
    "cpu-bench": "ghcr.io/blackfuel-ai/bf-vllm/vllm-cpu-bench",
}

# ASCII unit/record separators delimit git-log fields and records, so
# subjects/trailers containing common punctuation never split a record. These
# control characters are reserved: a commit message that itself contained one
# would be rejected with a malformed-record error rather than silently
# misparsed (see collect_commits). Real commit messages never contain them.
_FIELD_SEP = "\x1f"
_RECORD_SEP = "\x1e"


@dataclass(frozen=True)
class Commit:
    """One non-merge commit in the release range."""

    sha: str
    short: str
    subject: str
    upstream_status: str  # value of the Upstream-status: trailer, or "".

    @property
    def is_patch(self) -> bool:
        """A patch to upstream-owned files (ADR-0003 ``[bf-patch]`` prefix)."""
        return self.subject.startswith("[bf-patch]")


@dataclass(frozen=True)
class SyncMerge:
    """One ``[sync]`` merge commit (ADR-0002 daily upstream merge)."""

    short: str
    subject: str


@dataclass
class ReleaseRange:
    """The classified contents of a ``<prev>..<ref>`` release range."""

    patches: list[Commit] = field(default_factory=list)
    additive: list[Commit] = field(default_factory=list)
    syncs: list[SyncMerge] = field(default_factory=list)


# Upper bound on any single git invocation, so a wedged git (e.g. a stuck
# credential prompt) fails the release run instead of hanging CI forever.
_GIT_TIMEOUT_SECONDS = 120


def _run_git(args: list[str], *, cwd: str | None = None) -> str:
    """Run a git command and return its stdout.

    Raises ``RuntimeError`` with the captured stderr on a non-zero exit (git's
    own diagnostics — "not a git repository", "unknown revision" — are on
    stderr, which a bare ``CalledProcessError`` would swallow), and on timeout.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {exc.returncode}): "
            f"{(exc.stderr or '').strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_SECONDS}s"
        ) from exc
    return result.stdout


def _git_range(prev_tag: str | None, ref: str) -> str:
    """The git revision range. Open-ended (whole history to ``ref``) when there
    is no previous release tag — the first release."""
    return f"{prev_tag}..{ref}" if prev_tag else ref


def collect_commits(
    prev_tag: str | None, ref: str, *, cwd: str | None = None
) -> ReleaseRange:
    """Classify every commit in ``<prev_tag>..<ref>`` into BF patches, BF
    additive content, and upstream sync merges.

    Sync merges are detected by their ``[sync]`` subject prefix and surfaced
    from the full log (``--merges`` excludes them from the non-merge passes).
    BF commits are the non-merge commits in the range, partitioned by the
    ``[bf-patch]`` prefix.
    """
    rng = _git_range(prev_tag, ref)
    release = ReleaseRange()

    # --- sync merges: merge commits whose subject starts with [sync].
    sync_fmt = f"%h{_FIELD_SEP}%s{_RECORD_SEP}"
    sync_out = _run_git(["log", "--merges", f"--format={sync_fmt}", rng], cwd=cwd)
    for record in sync_out.split(_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        fields = record.split(_FIELD_SEP)
        if len(fields) != 2:
            raise ValueError(
                f"Malformed sync-log record (expected 2 fields, got "
                f"{len(fields)}): {record!r}"
            )
        short, subject = fields
        if subject.startswith("[sync]"):
            release.syncs.append(SyncMerge(short=short, subject=subject))

    # --- BF commits: non-merge commits, with the Upstream-status: trailer.
    commit_fmt = (
        f"%H{_FIELD_SEP}%h{_FIELD_SEP}%s{_FIELD_SEP}"
        f"%(trailers:key=Upstream-status,valueonly){_RECORD_SEP}"
    )
    commit_out = _run_git(
        ["log", "--no-merges", f"--format={commit_fmt}", rng], cwd=cwd
    )
    for record in commit_out.split(_RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        # The Upstream-status trailer is the last field and is the only one
        # that could legitimately be empty; the leading three are git-generated
        # (SHAs, subject) and never empty. A field count other than 4 means a
        # field carried the reserved unit separator — surface the record.
        fields = record.split(_FIELD_SEP)
        if len(fields) != 4:
            raise ValueError(
                f"Malformed commit-log record (expected 4 fields, got "
                f"{len(fields)}): {record!r}"
            )
        sha, short, subject, status = fields
        commit = Commit(
            sha=sha,
            short=short,
            subject=subject,
            upstream_status=status.strip(),
        )
        if commit.is_patch:
            release.patches.append(commit)
        else:
            release.additive.append(commit)

    return release


def _render_bf_section(release: ReleaseRange) -> list[str]:
    lines = ["## Blackfuel changes", ""]
    if not release.patches and not release.additive:
        lines.append("_No Blackfuel changes in this range._")
        return lines

    if release.additive:
        lines.append("### Additive")
        lines.append("")
        for c in release.additive:
            lines.append(f"- {c.subject} ({c.short})")
        lines.append("")

    if release.patches:
        lines.append("### Patches to upstream")
        lines.append("")
        for c in release.patches:
            status = c.upstream_status or "unspecified"
            lines.append(f"- {c.subject} ({c.short}) — `Upstream-status: {status}`")
        lines.append("")

    # Trailing blank line from the last sub-block is fine; collapse here.
    return lines


def _render_upstream_section(release: ReleaseRange) -> list[str]:
    lines = ["## Upstream changes pulled in", ""]
    if not release.syncs:
        lines.append("_No upstream syncs merged in this range._")
        return lines
    lines.append(
        "Upstream history merged into `main` via the daily sync "
        "(ADR-0002), most recent first:"
    )
    lines.append("")
    for s in release.syncs:
        lines.append(f"- {s.subject} ({s.short})")
    return lines


def _render_images_section(git_tag: str, image_targets: list[str]) -> list[str]:
    image_tag = parse_git_tag(git_tag).image_tag
    lines = ["## Images", ""]
    lines.append(
        f"Release channel images (`ai.bf-vllm.build.channel=release`), "
        f"immutable tag `{image_tag}`:"
    )
    lines.append("")
    lines.append("```")
    for target in image_targets:
        repo = IMAGE_REPOS.get(target)
        if repo is None:
            # An unknown target is surfaced rather than silently dropped, so a
            # typo'd image_targets input is visible in the release body.
            lines.append(f"# unknown image target: {target}")
            continue
        lines.append(f"docker pull {repo}:{image_tag}")
    lines.append("```")
    return lines


def generate_release_notes(
    git_tag: str,
    prev_tag: str | None,
    image_targets: list[str],
    *,
    ref: str = "HEAD",
    cwd: str | None = None,
) -> str:
    """Render the full release-notes markdown body.

    Args:
        git_tag: The release version being tagged, canonical git-tag form
            (``v<upstream>+bf.<bf>``). Validated via :func:`parse_git_tag`.
        prev_tag: The previous release tag, or ``None`` for the first release
            (the range is then open-ended to ``ref``).
        image_targets: Arch targets to render pull strings for (keys of
            :data:`IMAGE_REPOS`, e.g. ``["nvidia", "rocm", "cpu"]``).
        ref: The release ref the range ends at (``main`` HEAD by default).
        cwd: Repository working directory for the git calls.

    Returns:
        The markdown body for the GitHub release.
    """
    # Validate the version up-front so a malformed tag fails loudly before any
    # git work, rather than producing a body with a broken image tag.
    parse_git_tag(git_tag)

    release = collect_commits(prev_tag, ref, cwd=cwd)

    parts: list[str] = []
    parts.append(f"# bf-vllm {git_tag}")
    parts.append("")
    if prev_tag:
        parts.append(f"Changes since `{prev_tag}`.")
    else:
        parts.append("First Blackfuel release for this upstream line.")
    parts.append("")
    parts.extend(_render_bf_section(release))
    parts.append("")
    parts.extend(_render_upstream_section(release))
    parts.append("")
    parts.extend(_render_images_section(git_tag, image_targets))
    parts.append("")

    # Normalise: no more than one consecutive blank line.
    body_lines: list[str] = []
    for line in parts:
        if line == "" and body_lines and body_lines[-1] == "":
            continue
        body_lines.append(line)
    return "\n".join(body_lines).rstrip("\n") + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the bf-vllm release-notes markdown body."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version, canonical git-tag form (v<upstream>+bf.<bf>).",
    )
    parser.add_argument(
        "--prev-tag",
        default=None,
        help="Previous release tag. Omit for the first release.",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Release ref the range ends at (default: HEAD).",
    )
    parser.add_argument(
        "--image-targets",
        default="nvidia,rocm,cpu,cpu-bench",
        help="Comma-separated arch targets for the pull strings.",
    )
    parser.add_argument(
        "--repo-dir",
        default=None,
        help="Repository working directory for the git calls (default: cwd).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    targets = [t.strip() for t in args.image_targets.split(",") if t.strip()]
    body = generate_release_notes(
        git_tag=args.version,
        prev_tag=args.prev_tag,
        image_targets=targets,
        ref=args.ref,
        cwd=args.repo_dir,
    )
    sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
