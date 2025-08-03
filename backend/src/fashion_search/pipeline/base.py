from typing import Protocol, Dict, Any

class PipelineStep(Protocol):
    def run(self) -> Dict[str, Any]:
        ...
