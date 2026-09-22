"""Shared plumbing for the pipeline steps.

Renku has no job orchestration: every job gets its own ephemeral PVC, so a
step cannot leave anything on disk for the next one. Steps therefore hand off
through the container registry -- the one durable, credential-free-to-read
store a job can reach. Ordering is enforced outside Renku, by whatever drives
the jobs.

Each step is runnable two ways so it can be tested without a cluster:
  --in/--out       local directories
  --pull/--push    OCI references
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import oci  # noqa: E402

DEFAULT_TOKEN_FILE = Path("/secrets/ghcr_token")


def read_token(token_file: Path = DEFAULT_TOKEN_FILE) -> str:
    """The GitHub token, from the Renku secret mount or the environment."""
    if token_file.is_file():
        token = token_file.read_text().strip()
        if token:
            return token
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            f"no token: {token_file} is absent and GITHUB_TOKEN is unset. "
            "Attach the Renku secret to this launcher."
        )
    return token


def optional_token(token_file: Path = DEFAULT_TOKEN_FILE) -> str | None:
    """A token if one is available, else None for an anonymous pull."""
    try:
        return read_token(token_file)
    except SystemExit:
        return None


def resolve_input(pull: str | None, into: Path, local: Path | None) -> Path:
    """Get a step's input, from a local directory or from the registry."""
    if local is not None:
        if not local.is_dir():
            raise SystemExit(f"--in {local} does not exist")
        print(f"Input: {local}")
        return local
    if not pull:
        raise SystemExit("one of --in or --pull is required")
    # Authenticate the pull: a package a PAT has just created is private
    # until someone makes it public, so anonymous would 401 here.
    digest = oci.pull(pull, into, token=optional_token())
    print(f"Input: {pull} ({digest[:19]})")
    return into


def publish(push: str | None, directory: Path, files: list[str], source: str) -> None:
    """Publish a step's output, if it was asked to."""
    if not push:
        print(f"Output: {directory} (not published)")
        return
    digest = oci.push(
        push,
        [directory / name for name in files],
        token=read_token(),
        annotations={
            "org.opencontainers.image.source": source,
            "org.opencontainers.image.description": f"Pipeline output: {push}",
        },
    )
    print(f"Published {push} ({digest})")
