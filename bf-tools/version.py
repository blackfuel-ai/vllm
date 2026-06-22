# SPDX-License-Identifier: Apache-2.0
"""bf-vllm version helper.

Per ADR-0004, the canonical Blackfuel release identifier has the shape:

    v<upstream-semver>+bf.<bf-semver>

Where:
- ``upstream-semver`` is whatever upstream ``vllm-project/vllm`` released
  (e.g. ``0.20.2``).
- ``bf-semver`` is our own SemVer of the BF layer on top, where each
  component is an integer (e.g. ``0.1.0``).

The ``+`` separator is illegal in OCI image tags (the distribution-spec
restricts tags to ``[A-Za-z0-9_.-]``), so image tags use ``_`` instead:

    Git tag:    v0.20.2+bf.0.1.0
    Image tag:    0.20.2_bf.0.1.0

This module defines the round-trip between the two forms and validates
the structure of both. Consumers (release workflow, sync workflow, engine
deployment scripts) should use :func:`to_image_tag` / :func:`from_image_tag`
rather than hand-rolled string munging.

The dev channel (ADR-0004 dev-tag amendment) uses a separate tag form derived
from the tree's ``git describe`` self-name plus the commit sha:

    Dev image tag:    v0.23.1rc0_bf.0.1.0-dev-aebd933

:func:`dev_image_tag` / :func:`parse_dev_image_tag` centralise that derivation
so the build workflows don't hand-assemble git-describe output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Strict regex: matches v<upstream>+bf.<bf>, both halves dot-numeric SemVer.
#
# The upstream half is ``\d+.\d+.\d+`` with an OPTIONAL ``rcN`` pre-release
# suffix, so a Blackfuel release may be cut against either a final
# (``0.23.1``) or a release candidate (``0.23.1rc0``). The two release paths
# differ deliberately: the ``bf-sync-upstream`` automation stays FINALS-ONLY
# (its ``RELEASE_TAG_RE`` rejects rc, so it never auto-tracks a candidate),
# while the operator-gated ``bf-release`` path MAY release the temporary rc
# that ``main`` is sitting on between finals. The BF half is dot-numeric (a BF
# release is always a finished increment — no rc on our own layer).
_UPSTREAM = r"\d+\.\d+\.\d+(?:rc\d+)?"
_BF = r"\d+\.\d+\.\d+"

# The f-string braces interpolate the pre-assembled regex fragments above
# (compile-time literals, not runtime input — no injection surface).
_GIT_TAG_RE = re.compile(rf"^v(?P<upstream>{_UPSTREAM})\+bf\.(?P<bf>{_BF})$")

_IMAGE_TAG_RE = re.compile(rf"^(?P<upstream>{_UPSTREAM})_bf\.(?P<bf>{_BF})$")

# git-describe appends ``-<commits-since-tag>-g<short-sha>`` once HEAD has moved
# past the nearest matching tag (the ``g`` is git's own "git-hash" prefix). The
# dev image tag carries the commit itself (``-dev-<sha7>``), so this whole
# distance suffix is stripped to recover the base version. ``[0-9a-f]+`` matches
# the abbreviated sha describe emits (length varies with repo ambiguity, hence
# not fixed-width). The trailing ``$`` anchor means ``sub`` only ever removes a
# suffix — never a mid-string match — so a base whose own text contains
# ``-N-gHEX`` earlier is untouched.
_DESCRIBE_DISTANCE_RE = re.compile(r"-\d+-g[0-9a-f]+$")

# A dev image tag: ``<base-version>-dev-<sha7>`` (e.g.
# ``v0.23.1rc0_bf.0.1.0-dev-aebd933``). The base keeps the leading ``v`` and the
# ``_``-sanitised scm form — it is the tree's git-describe self-name (BF release
# line or upstream fallback), NOT the ADR-0004 release image tag, so it is not
# re-validated against the BfVersion grammar here. Unlike git-describe, the sha
# carries no ``g`` prefix: this is our own tag format, and a bare sha7 is what an
# operator pastes straight into ``git show``.
_DEV_IMAGE_TAG_RE = re.compile(r"^(?P<base>.+)-dev-(?P<sha7>[0-9a-f]{7})$")

# A git commit sha is lowercase hex; the dev tag's ``<sha7>`` is the first 7
# chars, so a non-hex sha would emit a tag the parser above rejects. Validate at
# the source instead of producing an unparsable tag.
_SHA_RE = re.compile(r"^[0-9a-f]+$")


@dataclass(frozen=True)
class BfVersion:
    """Parsed Blackfuel release identifier.

    Attributes:
        upstream: Upstream vLLM SemVer (e.g. ``"0.20.2"``).
        bf: Blackfuel layer SemVer (e.g. ``"0.1.0"``).
    """

    upstream: str
    bf: str

    @property
    def git_tag(self) -> str:
        """Canonical git tag form, e.g. ``v0.20.2+bf.0.1.0``."""
        return f"v{self.upstream}+bf.{self.bf}"

    @property
    def image_tag(self) -> str:
        """OCI-compatible image tag, e.g. ``0.20.2_bf.0.1.0``."""
        return f"{self.upstream}_bf.{self.bf}"


def parse_git_tag(tag: str) -> BfVersion:
    """Parse a canonical git tag.

    Args:
        tag: A string like ``v0.20.2+bf.0.1.0``.

    Returns:
        Parsed :class:`BfVersion`.

    Raises:
        ValueError: If the tag doesn't match the BfVersion grammar.
    """
    m = _GIT_TAG_RE.match(tag)
    if not m:
        raise ValueError(
            f"Not a valid bf-vllm git tag: {tag!r}. "
            "Expected shape: v<upstream-semver>+bf.<bf-semver>, "
            "e.g. v0.20.2+bf.0.1.0"
        )
    return BfVersion(upstream=m["upstream"], bf=m["bf"])


def parse_image_tag(tag: str) -> BfVersion:
    """Parse an OCI image tag.

    Args:
        tag: A string like ``0.20.2_bf.0.1.0``.

    Returns:
        Parsed :class:`BfVersion`.

    Raises:
        ValueError: If the tag doesn't match the image-tag grammar.
    """
    m = _IMAGE_TAG_RE.match(tag)
    if not m:
        raise ValueError(
            f"Not a valid bf-vllm image tag: {tag!r}. "
            "Expected shape: <upstream-semver>_bf.<bf-semver>, "
            "e.g. 0.20.2_bf.0.1.0"
        )
    return BfVersion(upstream=m["upstream"], bf=m["bf"])


def to_image_tag(git_tag: str) -> str:
    """Convert a git tag to its OCI-compatible image-tag form.

    Args:
        git_tag: e.g. ``v0.20.2+bf.0.1.0``.

    Returns:
        e.g. ``0.20.2_bf.0.1.0``.
    """
    return parse_git_tag(git_tag).image_tag


def from_image_tag(image_tag: str) -> str:
    """Convert an OCI image tag back to its canonical git-tag form.

    Args:
        image_tag: e.g. ``0.20.2_bf.0.1.0``.

    Returns:
        e.g. ``v0.20.2+bf.0.1.0``.
    """
    return parse_image_tag(image_tag).git_tag


# -----------------------------------------------------------------------------
# Dev-channel image tags
#
# A dev build pushes ``<base-version>-dev-<sha7>`` to the shared
# ``vllm-<arch>`` repository (ADR-0004 dev-tag amendment). The base version is
# the bf-vllm tree's git-describe self-name (BF release line ``v…+bf.…``, or the
# upstream tag line ``v…`` when no BF release is reachable) with ``+`` -> ``_``
# for OCI; the distance suffix git-describe appends between tags is stripped, and
# the literal commit is carried as ``<sha7>`` instead. Marker placement is
# ``-dev-`` BEFORE the sha so the version reads cleanly up to the marker. The sha
# has no ``g`` prefix — unlike git-describe, this is our own tag format and a
# bare sha7 is what an operator pastes straight into ``git show``.
# -----------------------------------------------------------------------------


def dev_image_tag(scm_describe: str, sha: str) -> str:
    """Build the dev-channel image tag from a git-describe name and commit sha.

    Args:
        scm_describe: The tree's ``git describe`` output, e.g.
            ``v0.23.1rc0+bf.0.1.0-2-gaebd933d0`` (the ``-N-gSHA`` distance
            suffix is optional — a build exactly at a tag has none). Either the
            ``+`` git form or the ``_`` OCI form is accepted.
        sha: The full commit sha (hex); its first 7 chars become the
            ``<sha7>``.

    Returns:
        e.g. ``v0.23.1rc0_bf.0.1.0-dev-aebd933``.

    Raises:
        ValueError: If ``sha`` is shorter than 7 chars or not lowercase hex —
            either would yield a tag :func:`parse_dev_image_tag` rejects.
    """
    if len(sha) < 7 or not _SHA_RE.match(sha):
        raise ValueError(f"sha must be >=7 lowercase-hex chars, got {sha!r}")
    base = _DESCRIBE_DISTANCE_RE.sub("", scm_describe).replace("+", "_")
    return f"{base}-dev-{sha[:7]}"


def parse_dev_image_tag(tag: str) -> tuple[str, str]:
    """Split a dev image tag into its base version and 7-char sha.

    Args:
        tag: e.g. ``v0.23.1rc0_bf.0.1.0-dev-aebd933``.

    Returns:
        ``(base_version, sha7)``, e.g.
        ``("v0.23.1rc0_bf.0.1.0", "aebd933")``.

    Raises:
        ValueError: If the tag doesn't match the dev image-tag grammar.
    """
    m = _DEV_IMAGE_TAG_RE.match(tag)
    if not m:
        raise ValueError(
            f"Not a valid bf-vllm dev image tag: {tag!r}. "
            "Expected shape: <base-version>-dev-<sha7>, "
            "e.g. v0.23.1rc0_bf.0.1.0-dev-aebd933"
        )
    return m["base"], m["sha7"]
