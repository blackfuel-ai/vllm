# SPDX-License-Identifier: Apache-2.0
"""Tests for bf-tools/version.py — git-tag <-> image-tag round-trip per ADR-0004."""

from __future__ import annotations

import pytest

# bf-tools/ is hyphenated and therefore cannot be a Python package. Tests
# live alongside the modules they cover; pytest adds the test file's
# directory to sys.path so a flat import resolves.
from version import (
    BfVersion,
    from_image_tag,
    parse_git_tag,
    parse_image_tag,
    to_image_tag,
)


# -----------------------------------------------------------------------------
# Round-trip
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "git_tag, image_tag",
    [
        # The canonical example from ADR-0004.
        ("v0.20.2+bf.0.1.0", "0.20.2_bf.0.1.0"),
        # Larger numbers in every field.
        ("v10.20.30+bf.4.5.6", "10.20.30_bf.4.5.6"),
        # Zero-padded edges: leading zeros are NOT SemVer-legal, so we
        # intentionally don't test them (would surface as ValueError).
        # Single-digit minor/patch on both halves.
        ("v1.0.0+bf.0.0.1", "1.0.0_bf.0.0.1"),
    ],
)
def test_round_trip(git_tag: str, image_tag: str) -> None:
    """to_image_tag and from_image_tag are mutual inverses on valid input."""
    assert to_image_tag(git_tag) == image_tag
    assert from_image_tag(image_tag) == git_tag


def test_parsed_attributes() -> None:
    """parse_git_tag exposes upstream + bf separately."""
    v = parse_git_tag("v0.20.2+bf.0.1.0")
    assert v.upstream == "0.20.2"
    assert v.bf == "0.1.0"
    assert v.git_tag == "v0.20.2+bf.0.1.0"
    assert v.image_tag == "0.20.2_bf.0.1.0"


def test_bfversion_is_frozen() -> None:
    """BfVersion is immutable so it can be safely shared across pipelines."""
    v = BfVersion(upstream="0.20.2", bf="0.1.0")
    with pytest.raises(AttributeError):
        v.upstream = "9.9.9"  # type: ignore[misc]


# -----------------------------------------------------------------------------
# Rejection of malformed input
# -----------------------------------------------------------------------------

_INVALID_GIT_TAGS = [
    # No leading v
    "0.20.2+bf.0.1.0",
    # Missing +bf. separator
    "v0.20.2bf.0.1.0",
    # Image-tag form (uses _)
    "v0.20.2_bf.0.1.0",
    # Single-component upstream
    "v0.20+bf.0.1.0",
    # Four-component upstream
    "v0.20.2.1+bf.0.1.0",
    # Non-numeric in upstream
    "v0.20.x+bf.0.1.0",
    # Non-numeric in bf
    "v0.20.2+bf.alpha.0.0",
    # Empty string
    "",
    # Pre-release upstream tags are NOT releasable (finals-only policy,
    # ADR-0002): rcN, hyphenated rc, alpha/beta/dev all rejected.
    "v0.23.1rc0+bf.0.1.0",
    "v0.20.2-rc1+bf.0.1.0",
    "v0.20.2a1+bf.0.1.0",
    "v0.20.2b2+bf.0.1.0",
    "v0.20.2.dev5+bf.0.1.0",
    # Pre-release suffix on the BF half: a BF release is always final.
    "v0.20.2+bf.0.1.0rc1",
    # Build metadata other than bf.*
    "v0.20.2+build.42",
]


@pytest.mark.parametrize("bad", _INVALID_GIT_TAGS)
def test_parse_git_tag_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="Not a valid bf-vllm git tag"):
        parse_git_tag(bad)


_INVALID_IMAGE_TAGS = [
    # Has leading v (image tags don't)
    "v0.20.2_bf.0.1.0",
    # Has +bf. instead of _bf.
    "0.20.2+bf.0.1.0",
    # Single-component upstream
    "0.20_bf.0.1.0",
    # Non-numeric component
    "0.20.x_bf.0.1.0",
    # Empty string
    "",
    # Missing _bf. separator
    "0.20.2.0.1.0",
    # Pre-release upstream (finals-only): rc image tag is not valid.
    "0.23.1rc0_bf.0.1.0",
]


@pytest.mark.parametrize("bad", _INVALID_IMAGE_TAGS)
def test_parse_image_tag_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="Not a valid bf-vllm image tag"):
        parse_image_tag(bad)


def test_to_image_tag_propagates_validation_error() -> None:
    """to_image_tag refuses obviously malformed input rather than silently
    producing garbage output."""
    with pytest.raises(ValueError):
        to_image_tag("not-a-tag")


def test_from_image_tag_propagates_validation_error() -> None:
    with pytest.raises(ValueError):
        from_image_tag("not-a-tag")
