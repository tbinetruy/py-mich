from dataclasses import dataclass
from typing import Dict
from stubs import *


def require(condition: bool, message: str) -> int:
    if not condition:
        raise Exception(message)

    return 0


@dataclass
class Contract:
    admin: address
    manifest_url: str
    manifest_hash: str
    _open: str
    _close: str
    artifacts_url: str
    artifacts_hash: str

    def open(self, _open: str, manifest_url: str, manifest_hash: str):
        require(SENDER == self.admin, "Only admin can call this entrypoint")
        self._open = _open
        self.manifest_url = manifest_url
        self.manifest_hash = manifest_hash

    def close(self, _close: str):
        require(SENDER == self.admin, "Only admin can call this entrypoint")
        self._close = _close

    def artifacts(self, artifacts_url: str, artifacts_hash: str):
        require(SENDER == self.admin, "Only admin can call this entrypoint")
        self.artifacts_url = artifacts_url
        self.artifacts_hash = artifacts_hash

