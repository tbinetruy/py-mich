import json

from compiler import Compiler
from vm import VM
from compiler_backend import CompilerBackend


admin =  "tzaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
source = f"""
@dataclass
class Storage:
    admin: address
    manifest_url: str
    manifest_hash: str
    open: str
    close: str
    artifacts_url: str
    artifacts_hash: str

@dataclass
class OpenArg:
    open: str
    manifest_url: str
    manifest_hash: str

@dataclass
class ArtifactsArg:
    artifacts_url: str
    artifacts_hash: str

class Contract:
    def deploy():
        return Storage("{admin}", '', '', '', '', '', '')

    def open(params: OpenArg) -> Storage:
        self.storage.open = params.open
        self.storage.manifest_url = params.manifest_url
        self.storage.manifest_hash = params.manifest_hash

        return self.storage

    def close(params: str) -> Storage:
        self.storage.close = params

        return self.storage

    def artifacts(params: ArtifactsArg) -> Storage:
        self.storage.artifacts_url = params.artifacts_url
        self.storage.artifacts_hash = params.artifacts_hash

        return self.storage
"""
vm = VM(isDebug=False)
c = Compiler(source, isDebug=False)
instructions = c._compile(c.ast)

micheline = CompilerBackend().compile_contract(c.contract)

with open("my_contract.json", "w+") as f:
    f.write(json.dumps(micheline))
