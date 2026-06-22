# SPDX-License-Identifier: Apache-2.0
"""Tests for bf-tools/version.py — git-tag <-> image-tag round-trip per ADR-0004."""

from __future__ import annotations

import pytest

# bf-tools/ is hyphenated and therefore cannot be a Python package. Tests
# live alongside the modules they cover; pytest adds the test file's
# directory to sys.path so a flat import resolves.
from version import (
    BfVersion,
    dev_image_tag,
    from_image_tag,
    parse_dev_image_tag,
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
        # Release-candidate upstream half: the operator-gated bf-release path
        # may release the temporary rc main sits on between finals. The `rcN`
        # suffix survives the +bf -> _bf image-tag mapping unchanged (rc has no
        # `+`, so it is OCI-tag-legal).
        ("v0.23.1rc0+bf.0.1.0", "0.23.1rc0_bf.0.1.0"),
        ("v0.23.1rc2+bf.1.0.0", "0.23.1rc2_bf.1.0.0"),
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
    # Upstream `rcN` (no separator) IS accepted — see the round-trip cases.
    # Every OTHER pre-release form is still rejected: only vLLM's own `rcN`
    # spelling is a real upstream tag; hyphenated rc, alpha/beta, and PEP 440
    # `.devN` are not shapes upstream releases under.
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


# -----------------------------------------------------------------------------
# Dev-channel image tags
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "scm_describe, sha, expected",
    [
        # The canonical example: a BF release line, between-tags git-describe
        # output. The -2-gaebd933d0 distance suffix is stripped; the bare 7-char
        # sha is re-appended as -dev-<sha7> (no `g` prefix); `+` becomes `_`.
        (
            "v0.23.1rc0+bf.0.1.0-2-gaebd933d0",
            "aebd933d0cafef00dba5eba11",
            "v0.23.1rc0_bf.0.1.0-dev-aebd933",
        ),
        # Build exactly at the BF tag: git-describe emits no distance suffix.
        (
            "v0.23.1rc0+bf.0.1.0",
            "aebd933d0cafef00dba5eba11",
            "v0.23.1rc0_bf.0.1.0-dev-aebd933",
        ),
        # Already `_`-sanitised input round-trips unchanged through the helper.
        (
            "v0.23.1rc0_bf.0.1.0-2-gaebd933d0",
            "aebd933d0cafef00dba5eba11",
            "v0.23.1rc0_bf.0.1.0-dev-aebd933",
        ),
        # Upstream-fallback line (no BF release reachable): no `+`, leading `v`
        # preserved, distance suffix stripped just the same.
        (
            "v0.23.1rc0-5-gdeadbeef0",
            "deadbeef0cafef00dba5eba11",
            "v0.23.1rc0-dev-deadbee",
        ),
        # Final (non-rc) upstream half.
        (
            "v0.20.2+bf.0.1.0-11-gd30cceb56",
            "d30cceb561234567890abcdef",
            "v0.20.2_bf.0.1.0-dev-d30cceb",
        ),
    ],
)
def test_dev_image_tag(scm_describe: str, sha: str, expected: str) -> None:
    """dev_image_tag strips git-describe distance, sanitises `+`, and appends
    the bare 7-char commit marker."""
    assert dev_image_tag(scm_describe, sha) == expected


def test_dev_image_tag_uses_underscore_not_plus() -> None:
    """The `+` of the BF release separator never survives into an image tag."""
    tag = dev_image_tag("v0.23.1rc0+bf.0.1.1-2-gaebd933d0", "aebd933d0cafef00d")
    assert "+" not in tag
    assert tag == "v0.23.1rc0_bf.0.1.1-dev-aebd933"


def test_dev_image_tag_takes_exactly_seven_sha_chars() -> None:
    """The marker is the first 7 sha chars (bare, no `g`) regardless of input
    length."""
    tag = dev_image_tag("v0.20.2+bf.0.1.0", "0123456789abcdef")
    assert tag.endswith("-dev-0123456")


def test_dev_image_tag_rejects_short_sha() -> None:
    with pytest.raises(ValueError, match="lowercase-hex"):
        dev_image_tag("v0.20.2+bf.0.1.0", "abc123")


@pytest.mark.parametrize(
    "bad_sha",
    [
        "ZZZZZZZ",  # non-hex
        "AEBD933",  # uppercase hex (git emits lowercase)
        "aebd-933",  # punctuation
        "aebd 933",  # whitespace
    ],
)
def test_dev_image_tag_rejects_non_hex_sha(bad_sha: str) -> None:
    """A non-hex sha would emit a tag parse_dev_image_tag rejects, so reject it
    at the source."""
    with pytest.raises(ValueError, match="lowercase-hex"):
        dev_image_tag("v0.20.2+bf.0.1.0", bad_sha)


@pytest.mark.parametrize(
    "tag, base, sha7",
    [
        ("v0.23.1rc0_bf.0.1.0-dev-aebd933", "v0.23.1rc0_bf.0.1.0", "aebd933"),
        ("v0.20.2_bf.0.1.0-dev-d30cceb", "v0.20.2_bf.0.1.0", "d30cceb"),
        ("v0.23.1rc0-dev-deadbee", "v0.23.1rc0", "deadbee"),
    ],
)
def test_parse_dev_image_tag(tag: str, base: str, sha7: str) -> None:
    """parse_dev_image_tag splits a dev tag into its base version and sha7."""
    assert parse_dev_image_tag(tag) == (base, sha7)


def test_dev_image_tag_round_trip() -> None:
    """dev_image_tag's output parses back to the same base + sha7."""
    sha = "aebd933d0cafef00dba5eba11"
    tag = dev_image_tag("v0.23.1rc0+bf.0.1.0-2-gaebd933d0", sha)
    base, sha7 = parse_dev_image_tag(tag)
    assert base == "v0.23.1rc0_bf.0.1.0"
    assert sha7 == sha[:7]


_INVALID_DEV_IMAGE_TAGS = [
    # No -dev- marker at all.
    "v0.23.1rc0_bf.0.1.0-aebd933",
    # Trailing -dev (the OLD format this PR replaces) is no longer valid.
    "v0.23.1rc0_bf.0.1.0-2-gaebd933d0-dev",
    # sha7 too short (6 chars).
    "v0.23.1rc0_bf.0.1.0-dev-aebd93",
    # sha7 too long (8 chars).
    "v0.23.1rc0_bf.0.1.0-dev-aebd9333",
    # A `g` prefix is now disallowed: `gaebd93` is 7 chars but `g` isn't hex.
    "v0.23.1rc0_bf.0.1.0-dev-gaebd93",
    # Non-hex in the sha.
    "v0.23.1rc0_bf.0.1.0-dev-zzzzzzz",
    # Uppercase hex (git emits lowercase).
    "v0.23.1rc0_bf.0.1.0-dev-AEBD933",
    # Empty string.
    "",
]


@pytest.mark.parametrize("bad", _INVALID_DEV_IMAGE_TAGS)
def test_parse_dev_image_tag_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="Not a valid bf-vllm dev image tag"):
        parse_dev_image_tag(bad)
