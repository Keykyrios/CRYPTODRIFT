"""
MAC_THEN_ENCRYPT detection rule.

Detects MAC-then-Encrypt ordering, which is vulnerable to
padding oracle attacks. The correct order is Encrypt-then-MAC.

This is a heuristic-based detection that looks for:
1. HMAC computation followed by encryption
2. encrypt() called on data that includes a MAC/tag
"""

from __future__ import annotations

import ast

from . import BaseRule, Severity


class MACOrderRule(BaseRule):

    @property
    def rule_id(self) -> str:
        return "MAC_THEN_ENCRYPT"

    @property
    def description(self) -> str:
        return "MAC-then-Encrypt ordering (vulnerable to padding oracle)"

    def __init__(self):
        super().__init__()
        self._mac_vars: set[str] = set()
        self._mac_lines: dict[str, int] = {}
        self._encrypt_calls: list[tuple[int, ast.Call]] = []

    def reset(self) -> None:
        super().reset()
        self._mac_vars.clear()
        self._mac_lines.clear()
        self._encrypt_calls.clear()

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track HMAC/MAC computation results."""
        if isinstance(node.value, ast.Call):
            call_name = self._get_call_name(node.value)

            # Track HMAC computations
            if call_name in ("hmac.new", "HMAC.new", "hmac.digest",
                             "CMAC.new"):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self._mac_vars.add(target.id)
                        self._mac_lines[target.id] = node.lineno

            # Track .digest() / .hexdigest() results
            if call_name.endswith((".digest", ".hexdigest")):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name_lower = target.id.lower()
                        if any(kw in name_lower for kw in
                               ("mac", "tag", "hmac", "digest")):
                            self._mac_vars.add(target.id)
                            self._mac_lines[target.id] = node.lineno

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track encrypt calls and check for MAC-then-Encrypt."""
        call_name = self._get_call_name(node)

        if call_name.endswith(".encrypt") or call_name in (
            "encrypt", "cipher.encrypt"
        ):
            # Check if any argument references a MAC variable
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in self._mac_vars:
                    mac_line = self._mac_lines.get(arg.id, 0)
                    if mac_line < node.lineno:
                        self._add_finding(
                            severity=Severity.HIGH,
                            line=node.lineno,
                            col=node.col_offset,
                            description=(
                                f"MAC-then-Encrypt pattern: MAC '{arg.id}' "
                                f"(line {mac_line}) is encrypted at line "
                                f"{node.lineno}. Use Encrypt-then-MAC "
                                "ordering to prevent padding oracle attacks."
                            ),
                            confidence="MEDIUM",
                        )

                # Check for concatenation with MAC var (e.g., mac + data)
                if isinstance(arg, ast.BinOp) and isinstance(
                    arg.op, ast.Add
                ):
                    for operand in (arg.left, arg.right):
                        if (isinstance(operand, ast.Name)
                                and operand.id in self._mac_vars):
                            self._add_finding(
                                severity=Severity.HIGH,
                                line=node.lineno,
                                col=node.col_offset,
                                description=(
                                    "Possible MAC-then-Encrypt: MAC value "
                                    "concatenated into encryption input."
                                ),
                                confidence="MEDIUM",
                            )

        self.generic_visit(node)
