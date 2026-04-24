"""EvidenceSovereign: operation seal/halt within the stream organ."""

import hashlib
import json
from typing import Any, Dict


class EvidenceSovereign:
    def __init__(self) -> None:
        self.state = "Verified"

    def seal_operation(self, op_details: Dict[str, Any]) -> Dict[str, Any]:
        op_str = json.dumps(op_details, sort_keys=True)
        op_hash = hashlib.sha256(op_str.encode()).hexdigest()
        return {"operation_hash": op_hash, "bundle_ref": f"ULID-{op_hash[:8]}", "status": "ok"}

    def trigger_halt(self, reason: str) -> Dict[str, Any]:
        self.state = "Halting"
        return {"status": "Halted", "reason": reason}
