"""Operator registry for managing registered operator families."""

from pathlib import Path
from typing import Optional

from triton_agent.core.contract import OpContract


class RegistryError(Exception):
    """Raised on registry lookup failures."""


class OpRegistry:
    """Manages registered operators and their contracts."""

    def __init__(self) -> None:
        self._ops: dict[str, OpContract] = {}

    def register(self, contract: OpContract) -> None:
        """Register an operator from a contract."""
        self._ops[contract.op] = contract

    def get(self, op_name: str) -> OpContract:
        """Get a contract by operator name."""
        if op_name not in self._ops:
            raise RegistryError(f"Operator '{op_name}' not registered")
        return self._ops[op_name]

    def list(self) -> list[str]:
        """Return sorted list of registered operator names."""
        return sorted(self._ops.keys())

    def has(self, op_name: str) -> bool:
        """Check if an operator is registered."""
        return op_name in self._ops

    def load_from_dir(self, ops_dir: str | Path) -> int:
        """Load all operators from a directory structure.

        Scans for `contract.yaml` in subdirectories and registers them.
        Returns the number of operators loaded.
        """
        ops_path = Path(ops_dir)
        count = 0
        for contract_path in ops_path.rglob("contract.yaml"):
            try:
                contract = OpContract.from_yaml(contract_path)
                self.register(contract)
                count += 1
            except Exception:
                pass
        return count


_registry: Optional[OpRegistry] = None


def get_registry() -> OpRegistry:
    """Return the global registry singleton."""
    global _registry
    if _registry is None:
        _registry = OpRegistry()
    return _registry
