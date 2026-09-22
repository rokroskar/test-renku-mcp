"""Push and pull a small file bundle as an OCI artifact, using only stdlib.

The precompute job publishes its results to the GitHub container registry
next to the app image, and the app pulls them at startup. Both sides run in
images we do not control the dependencies of, so this speaks the registry's
HTTP API directly rather than depending on oras or docker being present.

    push("ghcr.io/owner/repo/model:latest", [Path("results.npz")], token=...)
    directory = pull("ghcr.io/owner/repo/model:latest")

Anonymous pull needs no token. Push needs a GitHub token with write:packages.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path

ARTIFACT_TYPE = "application/vnd.renku.mnist-results.v1+json"
EMPTY_CONFIG = b"{}"
EMPTY_CONFIG_TYPE = "application/vnd.oci.empty.v1+json"
MANIFEST_TYPE = "application/vnd.oci.image.manifest.v1+json"

ACCEPT_MANIFESTS = ",".join(
    [
        MANIFEST_TYPE,
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)

TIMEOUT = 60


class RegistryError(RuntimeError):
    """The registry refused, or the artifact is not usable."""


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _split(reference: str) -> tuple[str, str, str]:
    registry, _, remainder = reference.partition("/")
    repository, separator, tag = remainder.rpartition(":")
    if not separator or "/" in tag:  # no tag, just a path
        return registry, remainder, "latest"
    return registry, repository, tag


def _request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        url, data=body, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers or {}), error.read()
    except urllib.error.URLError as error:
        raise RegistryError(f"{method} {url}: {error.reason}") from error


def _token(registry: str, repository: str, scope: str, secret: str | None) -> str:
    url = (
        f"https://{registry}/token"
        f"?scope=repository:{repository}:{scope}&service={registry}"
    )
    headers = {}
    if secret:
        # GHCR accepts any username alongside a token as the password.
        basic = base64.b64encode(f"token:{secret}".encode()).decode()
        headers["Authorization"] = f"Basic {basic}"

    status, _, payload = _request("GET", url, headers)
    if status != 200:
        raise RegistryError(
            f"token request for {repository}:{scope} returned {status}"
        )
    body = json.loads(payload)
    token = body.get("token") or body.get("access_token")
    if not token:
        raise RegistryError(f"no token in registry response for {repository}")
    return token


# --------------------------------------------------------------------------
# Push
# --------------------------------------------------------------------------


def _upload_blob(
    registry: str, repository: str, auth: dict[str, str], payload: bytes
) -> str:
    digest = _digest(payload)
    base = f"https://{registry}/v2/{repository}/blobs"

    status, _, _ = _request("HEAD", f"{base}/{digest}", auth)
    if status == 200:  # already there
        return digest

    status, headers, body = _request("POST", f"{base}/uploads/", auth)
    if status not in (200, 202):
        raise RegistryError(f"blob upload not started ({status}): {body[:200]!r}")

    location = headers.get("Location") or headers.get("location")
    if not location:
        raise RegistryError("registry gave no upload location")
    if location.startswith("/"):
        location = f"https://{registry}{location}"
    joiner = "&" if "?" in location else "?"

    status, _, body = _request(
        "PUT",
        f"{location}{joiner}digest={digest}",
        {**auth, "Content-Type": "application/octet-stream"},
        payload,
    )
    if status != 201:
        raise RegistryError(f"blob PUT failed ({status}): {body[:200]!r}")
    return digest


def push(
    reference: str,
    files: list[Path],
    token: str,
    annotations: dict[str, str] | None = None,
) -> str:
    """Upload files as one OCI artifact. Returns the manifest digest."""
    registry, repository, tag = _split(reference)
    bearer = _token(registry, repository, "pull,push", token)
    auth = {"Authorization": f"Bearer {bearer}"}

    layers = []
    for path in files:
        payload = path.read_bytes()
        media_type = (
            mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        digest = _upload_blob(registry, repository, auth, payload)
        layers.append(
            {
                "mediaType": media_type,
                "digest": digest,
                "size": len(payload),
                "annotations": {"org.opencontainers.image.title": path.name},
            }
        )

    config_digest = _upload_blob(registry, repository, auth, EMPTY_CONFIG)
    manifest = {
        "schemaVersion": 2,
        "mediaType": MANIFEST_TYPE,
        "artifactType": ARTIFACT_TYPE,
        "config": {
            "mediaType": EMPTY_CONFIG_TYPE,
            "digest": config_digest,
            "size": len(EMPTY_CONFIG),
        },
        "layers": layers,
        "annotations": annotations or {},
    }
    payload = json.dumps(manifest, separators=(",", ":")).encode()

    status, _, body = _request(
        "PUT",
        f"https://{registry}/v2/{repository}/manifests/{tag}",
        {**auth, "Content-Type": MANIFEST_TYPE},
        payload,
    )
    if status not in (200, 201):
        raise RegistryError(f"manifest PUT failed ({status}): {body[:300]!r}")
    return _digest(payload)


# --------------------------------------------------------------------------
# Pull
# --------------------------------------------------------------------------


def pull(reference: str, into: Path, token: str | None = None) -> Path:
    """Download an artifact's named layers into `into`. Returns `into`."""
    registry, repository, tag = _split(reference)
    bearer = _token(registry, repository, "pull", token)
    auth = {"Authorization": f"Bearer {bearer}"}

    status, _, payload = _request(
        "GET",
        f"https://{registry}/v2/{repository}/manifests/{tag}",
        {**auth, "Accept": ACCEPT_MANIFESTS},
    )
    if status != 200:
        raise RegistryError(f"manifest GET failed ({status}) for {reference}")

    manifest = json.loads(payload)
    if "manifests" in manifest:  # an index; follow its first entry
        digest = manifest["manifests"][0]["digest"]
        status, _, payload = _request(
            "GET",
            f"https://{registry}/v2/{repository}/manifests/{digest}",
            {**auth, "Accept": ACCEPT_MANIFESTS},
        )
        if status != 200:
            raise RegistryError(f"nested manifest GET failed ({status})")
        manifest = json.loads(payload)

    into.mkdir(parents=True, exist_ok=True)
    written = 0
    for layer in manifest.get("layers", []):
        name = layer.get("annotations", {}).get("org.opencontainers.image.title")
        if not name:
            continue
        status, _, blob = _request(
            "GET",
            f"https://{registry}/v2/{repository}/blobs/{layer['digest']}",
            auth,
        )
        if status != 200:
            raise RegistryError(f"blob GET failed ({status}) for {name}")
        if _digest(blob) != layer["digest"]:
            raise RegistryError(f"digest mismatch for {name}")
        (into / Path(name).name).write_bytes(blob)
        written += 1

    if not written:
        raise RegistryError(f"{reference} carried no named layers")
    return into
