"""Tests for bf-tools/generate-release-notes.py.

The module under test is hyphenated (matching the script name the release
workflow invokes) and so cannot be imported by name; it is loaded from its
file path via importlib. Commit classification is exercised against a real
throwaway git repository built per-test, since the generator's whole job is
to read git history.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("generate-release-notes.py")
_spec = importlib.util.spec_from_file_location("generate_release_notes", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
grn = importlib.util.module_from_spec(_spec)
# Register before exec so @dataclass can resolve the module via sys.modules
# (dataclasses looks up cls.__module__ there during class processing).
sys.modules["generate_release_notes"] = grn
_spec.loader.exec_module(grn)


# -----------------------------------------------------------------------------
# git fixture
# -----------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _commit(repo: Path, filename: str, message: str) -> None:
    path = repo / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x\n")
    _git(repo, "add", filename)
    _git(repo, "commit", "--no-verify", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repo with a prev-release tag, one additive commit, one
    [bf-patch] commit carrying an Upstream-status trailer, and one [sync]
    merge commit bringing in 'upstream' history."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@bf.ai")
    _git(r, "config", "user.name", "Test")

    _commit(r, "base.txt", "base")
    _git(r, "tag", "v0.20.2+bf.0.0.1")

    # A divergent 'upstream-main' lineage that main merges via a [sync] commit.
    _git(r, "checkout", "-q", "-b", "upstream-main")
    _commit(r, "upstream_feature.py", "Add a new model")
    _git(r, "checkout", "-q", "main")
    _git(
        r,
        "merge",
        "--no-ff",
        "--no-verify",
        "-m",
        "[sync] merge upstream-main (abc1234) into main",
        "upstream-main",
    )

    _commit(r, "bf-tools/new_tool.py", "[Feat][bf-tools] add a helper")
    _commit(
        r,
        "vllm/patched.py",
        "[bf-patch] tweak upstream\n\nUpstream-status: candidate",
    )
    return r


# -----------------------------------------------------------------------------
# collect_commits classification
# -----------------------------------------------------------------------------


def test_collect_classifies_each_bucket(repo: Path) -> None:
    rng = grn.collect_commits("v0.20.2+bf.0.0.1", "HEAD", cwd=str(repo))

    additive_subjects = [c.subject for c in rng.additive]
    patch_subjects = [c.subject for c in rng.patches]
    sync_subjects = [s.subject for s in rng.syncs]

    assert "[Feat][bf-tools] add a helper" in additive_subjects
    assert any(s.startswith("[bf-patch]") for s in patch_subjects)
    assert any(s.startswith("[sync]") for s in sync_subjects)
    # The patch carries its Upstream-status trailer.
    assert rng.patches[0].upstream_status == "candidate"
    # The additive bucket must NOT contain the patch.
    assert not any(s.startswith("[bf-patch]") for s in additive_subjects)


def test_collect_open_range_is_first_release(repo: Path) -> None:
    """With no prev tag the range is open-ended and includes the base commit."""
    rng = grn.collect_commits(None, "HEAD", cwd=str(repo))
    # Base commit (pre-tag) is additive and present in the open range.
    assert any(c.subject == "base" for c in rng.additive)


# -----------------------------------------------------------------------------
# generate_release_notes rendering
# -----------------------------------------------------------------------------


def test_body_has_all_sections_and_image_strings(repo: Path) -> None:
    body = grn.generate_release_notes(
        git_tag="v0.20.2+bf.0.1.0",
        prev_tag="v0.20.2+bf.0.0.1",
        image_targets=["nvidia", "cpu"],
        ref="HEAD",
        cwd=str(repo),
    )

    assert "# bf-vllm v0.20.2+bf.0.1.0" in body
    assert "Changes since `v0.20.2+bf.0.0.1`." in body
    assert "## Blackfuel changes" in body
    assert "## Upstream changes pulled in" in body
    assert "## Images" in body
    # Pull string is `<repo>:<image_tag>` — the `:` separates repo from tag,
    # and the tag itself is the OCI (_) form from version.to_image_tag.
    assert "ghcr.io/blackfuel-ai/bf-vllm/vllm:0.20.2_bf.0.1.0" in body
    assert "ghcr.io/blackfuel-ai/bf-vllm/vllm-cpu:0.20.2_bf.0.1.0" in body
    # Only the requested targets are rendered.
    assert "vllm-rocm" not in body
    # The patch's upstream status surfaces in the body.
    assert "Upstream-status: candidate" in body


def test_first_release_body_wording(repo: Path) -> None:
    body = grn.generate_release_notes(
        git_tag="v0.20.2+bf.0.1.0",
        prev_tag=None,
        image_targets=["nvidia"],
        cwd=str(repo),
    )
    assert "First Blackfuel release for this upstream line." in body
    # The open-ended range must actually carry commits — assert a known one
    # is present so a silently-empty range can't pass this test.
    assert "[Feat][bf-tools] add a helper" in body


def test_empty_image_targets_renders_no_pull_strings(repo: Path) -> None:
    """An empty image_targets list yields an Images block with no pull lines
    (and never a stray pull string)."""
    body = grn.generate_release_notes(
        git_tag="v0.20.2+bf.0.1.0",
        prev_tag="v0.20.2+bf.0.0.1",
        image_targets=[],
        cwd=str(repo),
    )
    assert "## Images" in body
    assert "docker pull" not in body


def test_unknown_image_target_is_surfaced(repo: Path) -> None:
    body = grn.generate_release_notes(
        git_tag="v0.20.2+bf.0.1.0",
        prev_tag="v0.20.2+bf.0.0.1",
        image_targets=["nvidia", "bogus"],
        cwd=str(repo),
    )
    assert "# unknown image target: bogus" in body


def test_invalid_version_rejected(repo: Path) -> None:
    with pytest.raises(ValueError, match="Not a valid bf-vllm git tag"):
        grn.generate_release_notes(
            git_tag="0.20.2_bf.0.1.0",  # image-tag form, not a git tag
            prev_tag=None,
            image_targets=["nvidia"],
            cwd=str(repo),
        )


def test_no_double_blank_lines(repo: Path) -> None:
    body = grn.generate_release_notes(
        git_tag="v0.20.2+bf.0.1.0",
        prev_tag="v0.20.2+bf.0.0.1",
        image_targets=["nvidia"],
        cwd=str(repo),
    )
    assert "\n\n\n" not in body


# -----------------------------------------------------------------------------
# error surfacing
# -----------------------------------------------------------------------------


def test_git_failure_surfaces_stderr(tmp_path: Path) -> None:
    """A git failure is raised as a RuntimeError carrying git's stderr, not a
    bare CalledProcessError with an empty stdout."""
    not_a_repo = tmp_path / "empty"
    not_a_repo.mkdir()
    with pytest.raises(RuntimeError, match="git log"):
        grn.collect_commits(None, "HEAD", cwd=str(not_a_repo))


def test_malformed_commit_record_is_surfaced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A log record with the wrong field count raises with the offending
    record, rather than an opaque unpack ValueError."""

    def fake_run_git(args: list[str], *, cwd: str | None = None) -> str:
        # A commit record with only 2 fields (a field carried the separator).
        if "--no-merges" in args:
            return f"sha{grn._FIELD_SEP}short{grn._RECORD_SEP}"
        return ""

    monkeypatch.setattr(grn, "_run_git", fake_run_git)
    with pytest.raises(ValueError, match="Malformed commit-log record"):
        grn.collect_commits(None, "HEAD")
