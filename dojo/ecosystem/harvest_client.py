"""DanteDojo ecosystem — DanteHarvest HTTP client.

Communicates with the DanteHarvest organ over HTTP.
On connection error → raises HarvestClientError with actionable message.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from dojo.core.exceptions import HarvestClientError


class HarvestClient:
    """HTTP client for the DanteHarvest organ.

    Args:
        base_url: Base URL of the Harvest API. Defaults to DANTE_HARVEST_URL env var
                  or "http://127.0.0.1:8001".
    """

    def __init__(self, base_url: str = "") -> None:
        if not base_url:
            base_url = os.environ.get("DANTE_HARVEST_URL", "http://127.0.0.1:8001")
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_pack(self, pack_uri: str) -> dict[str, Any]:
        """GET /v1/packs/{pack_uri}.

        Returns pack manifest dict.
        Raises HarvestClientError on failure.
        """
        url = f"{self._base_url}/v1/packs/{pack_uri}"
        try:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            return dict(response.json())
        except httpx.ConnectError as exc:
            raise HarvestClientError(
                f"Cannot connect to DanteHarvest at {self._base_url}. "
                f"Check DANTE_HARVEST_URL or ensure the Harvest service is running. "
                f"Original error: {exc}",
                reason="connection_refused",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise HarvestClientError(
                f"DanteHarvest returned HTTP {exc.response.status_code} for pack '{pack_uri}': "
                f"{exc.response.text}",
                reason="http_error",
            ) from exc
        except Exception as exc:
            raise HarvestClientError(
                f"Unexpected error fetching pack '{pack_uri}' from DanteHarvest: {exc}",
                reason="unexpected",
            ) from exc

    def verify_receipt(self, receipt_id: str) -> dict[str, Any]:
        """GET /v1/receipts/{receipt_id}.

        Returns {"valid": bool, "revoked": bool}.
        Raises HarvestClientError on failure.
        """
        url = f"{self._base_url}/v1/receipts/{receipt_id}"
        try:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            return dict(response.json())
        except httpx.ConnectError as exc:
            raise HarvestClientError(
                f"Cannot connect to DanteHarvest at {self._base_url}. "
                f"Check DANTE_HARVEST_URL or ensure the Harvest service is running. "
                f"Original error: {exc}",
                reason="connection_refused",
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise HarvestClientError(
                f"DanteHarvest returned HTTP {exc.response.status_code} for receipt '{receipt_id}': "
                f"{exc.response.text}",
                reason="http_error",
            ) from exc
        except Exception as exc:
            raise HarvestClientError(
                f"Unexpected error verifying receipt '{receipt_id}' from DanteHarvest: {exc}",
                reason="unexpected",
            ) from exc

    def check_revocation(self, receipt_id: str) -> bool:
        """Returns True if the receipt has been revoked.

        Raises HarvestClientError on connection failure.
        """
        data = self.verify_receipt(receipt_id)
        return bool(data.get("revoked", False))
