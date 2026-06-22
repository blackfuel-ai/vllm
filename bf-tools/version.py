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
