"""DanteDojo ecosystem — DanteForge HTTP client.

Communicates with the DanteForge organ over HTTP.
Fail-closed: any error raises ForgeClientError — never returns a false GREEN.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from dojo.core.exceptions import ForgeClientError


class ForgeClient:
    """HTTP client for the DanteForge organ.

    Args:
        base_url: Base URL of the Forge API. Defaults to DANTE_FORGE_URL env var
                  or "http://127.0.0.1:8002".
    """

    def __init__(self, base_url: str = "") -> None:
        if not base_url:
            base_url = os.environ.get("DANTE_FORGE_URL", "http://127.0.0.1:8002")
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_policy(
        self,
        candidate_manifest: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """POST /v1/policy/verify.

        Returns {"passed": bool, "reasons": list[str], "receipt": str}.
        Fail-closed: any error raises ForgeClientError.
        """
        url = f"{self._base_url}/v1/policy/verify"
        try:
            response = httpx.post(url, json=candidate_manifest, timeout=timeout)
            response.raise_for_status()
            return dict(response.json())
        except httpx.ConnectError as exc:
            raise ForgeClientError(
                f"Cannot connect to DanteForge at {self._base_url}. "
                f"Check DANTE_FORGE_URL or ensure the Forge service is running. "
                f"Original error: {exc}",
                reason="connection_refused",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ForgeClientError(
                f"DanteForge policy verification timed out after {timeout}s. "
                f"The Forge service may be overloaded. Original error: {exc}",
                reason="timeout",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ForgeClientError(
                f"DanteForge returned HTTP {exc.response.status_code} for policy/verify: "
                f"{exc.response.text}",
                reason="http_error",
            ) from exc
        except Exception as exc:
            raise ForgeClientError(
                f"Unexpected error during policy verification with DanteForge: {exc}",
                reason="unexpected",
            ) from exc

    def check_passport_revocation(self, passport_id: str) -> bool:
        """GET /v1/passport/{passport_id}/revoke.

        Returns True if the passport has been revoked.
        Raises ForgeClientError on any failure.
        """
        url = f"{self._base_url}/v1/passport/{passport_id}/revoke"
        try:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            return bool(data.get("revoked", False))
        except httpx.ConnectError as exc:
            raise ForgeClientError(
                f"Cannot connect to DanteForge at {self._base_url}. "
                f"Check DANTE_FORGE_URL or ensure the Forge service is running. "
                f"Original error: {exc}",
                reason="connection_refused",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ForgeClientError(
                f"DanteForge returned HTTP {exc.response.status_code} for passport '{passport_id}': "
                f"{exc.response.text}",
                reason="http_error",
            ) from exc
        except Exception as exc:
            raise ForgeClientError(
                f"Unexpected error checking passport revocation from DanteForge: {exc}",
                reason="unexpected",
            ) from exc

    def get_forge_public_key(self) -> bytes:
        """GET /v1/keys/public.

        Returns raw public key bytes.
        Raises ForgeClientError on any failure.
        """
        url = f"{self._base_url}/v1/keys/public"
        try:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            return response.content
        except httpx.ConnectError as exc:
            raise ForgeClientError(
                f"Cannot connect to DanteForge at {self._base_url}. "
                f"Check DANTE_FORGE_URL or ensure the Forge service is running. "
                f"Original error: {exc}",
                reason="connection_refused",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ForgeClientError(
                f"DanteForge returned HTTP {exc.response.status_code} fetching public key: "
                f"{exc.response.text}",
                reason="http_error",
            ) from exc
        except Exception as exc:
            raise ForgeClientError(
                f"Unexpected error fetching Forge public key: {exc}",
                reason="unexpected",
            ) from exc
