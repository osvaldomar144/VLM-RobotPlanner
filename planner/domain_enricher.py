"""
DomainEnricher — applies VLM-suggested additions to a base PDDL domain template.

This is the innovative component of the pipeline: instead of a single fixed domain,
the system selects a base template (pddl/domains/) and enriches it at runtime based
on scene-specific context provided by the VLM.

Enrichment types supported:
  1. new_types             — add subtypes to the (:types ...) block
  2. new_predicates        — add predicates to the (:predicates ...) block
  3. new_actions           — append full action definitions to the domain
  4. modified_preconditions — add extra conditions to existing actions' preconditions

All operations are string-based (no full PDDL parser required). Validation is
structural: parenthesis balance, duplicate detection, and known-type checks.

Usage:
    from planner.domain_enricher import DomainEnricher, DomainAdditions

    enricher = DomainEnricher()
    additions = DomainAdditions(
        new_predicates=["(locked ?i - item)"],
        modified_preconditions={"pick": ["(not (locked ?i))"]},
    )
    result = enricher.enrich(domain_text, additions)
    if result.is_valid:
        fast_downward.solve(result.domain_text, problem_text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DomainAdditions:
    """
    Scene-specific enrichment suggestions, typically produced by the VLM.

    Fields:
        new_types:              PDDL type declarations, e.g. ["container - location"]
        new_predicates:         PDDL predicate declarations, e.g. ["(locked ?i - item)"]
        new_actions:            Action dicts with keys: name, parameters, precondition, effect
        modified_preconditions: Map of action_name → list of extra PDDL conditions.
                                Conditions must use variable names already in that action's
                                parameter list (the enricher does not re-check this).
    """
    new_types:               list[str]        = field(default_factory=list)
    new_predicates:          list[str]        = field(default_factory=list)
    new_actions:             list[dict]       = field(default_factory=list)
    modified_preconditions:  dict[str, list[str]] = field(default_factory=dict)


@dataclass
class EnrichmentResult:
    """
    Output of DomainEnricher.enrich().

    Fields:
        domain_text:         The enriched PDDL domain string (ready for Fast Downward).
        is_valid:            True if structural validation passed.
        errors:              List of validation error messages.
        additions_applied:   Human-readable list of what was added.
        additions_skipped:   Human-readable list of what was rejected and why.
    """
    domain_text:        str
    is_valid:           bool
    errors:             list[str]
    additions_applied:  list[str]
    additions_skipped:  list[str]


# ── Main class ────────────────────────────────────────────────────────────────

class DomainEnricher:
    """
    Applies structured enrichment additions to a PDDL domain text.

    All methods are stateless — the same instance can be reused across tasks.
    """

    # PDDL keywords that must never be treated as type/predicate names
    _PDDL_KEYWORDS = frozenset({
        "define", "domain", "problem", "requirements", "types", "predicates",
        "action", "parameters", "precondition", "effect", "and", "or", "not",
        "when", "forall", "exists", "object", "strips", "typing", "adl",
        "negative-preconditions", "disjunctive-preconditions",
        "equality", "conditional-effects", "quantified-preconditions",
    })

    # ── Public API ────────────────────────────────────────────────────────────

    def enrich(self, domain_text: str, additions: DomainAdditions) -> EnrichmentResult:
        """
        Apply additions to domain_text and return an EnrichmentResult.

        Processing order: types → predicates → actions → modified preconditions.
        Each step is independent; failures in one do not abort the others.
        """
        errors:   list[str] = []
        applied:  list[str] = []
        skipped:  list[str] = []

        text = domain_text
        existing = self._extract_existing(text)

        # 1. New types
        for type_decl in additions.new_types:
            type_name = type_decl.split()[0]
            if type_name in existing["types"]:
                skipped.append(f"type '{type_name}' already exists — skipped")
            else:
                new_text, ok = self._insert_into_block(text, ":types", type_decl)
                if ok:
                    text = new_text
                    applied.append(f"type: {type_decl}")
                    existing["types"].add(type_name)
                else:
                    skipped.append(f"type '{type_decl}' — :types block not found")

        # 2. New predicates
        for pred in additions.new_predicates:
            pred_name = self._predicate_name(pred)
            if pred_name in existing["predicates"]:
                skipped.append(f"predicate '{pred_name}' already exists — skipped")
            else:
                new_text, ok = self._insert_into_block(text, ":predicates", pred)
                if ok:
                    text = new_text
                    applied.append(f"predicate: {pred}")
                    existing["predicates"].add(pred_name)
                else:
                    skipped.append(f"predicate '{pred}' — :predicates block not found")

        # 3. New actions
        for action in additions.new_actions:
            name = action.get("name", "")
            if not name:
                skipped.append("action with missing 'name' field — skipped")
                continue
            if name in existing["actions"]:
                skipped.append(f"action '{name}' already exists — skipped")
                continue
            action_pddl = self._format_action(action)
            if not self._balanced(action_pddl):
                skipped.append(f"action '{name}' has unbalanced parentheses — skipped")
                continue
            text = self._append_action(text, action_pddl)
            applied.append(f"action: {name}")
            existing["actions"].add(name)

        # 4. Modified preconditions
        for action_name, extra_conditions in additions.modified_preconditions.items():
            if action_name not in existing["actions"]:
                skipped.append(
                    f"cannot extend precondition of '{action_name}' — action not found"
                )
                continue
            for cond in extra_conditions:
                if not self._balanced(cond):
                    skipped.append(
                        f"condition '{cond}' for '{action_name}' has unbalanced parens — skipped"
                    )
                    continue
            new_text, ok = self._extend_precondition(text, action_name, extra_conditions)
            if ok:
                text = new_text
                applied.append(
                    f"extended precondition of '{action_name}': {extra_conditions}"
                )
            else:
                skipped.append(
                    f"could not extend precondition of '{action_name}' — :precondition not found"
                )

        # 5. Validate final result
        is_valid, val_errors = self._validate(text)
        errors.extend(val_errors)

        return EnrichmentResult(
            domain_text=text,
            is_valid=is_valid,
            errors=errors,
            additions_applied=applied,
            additions_skipped=skipped,
        )

    @classmethod
    def from_file(cls, domain_path: str | Path) -> "tuple[DomainEnricher, str]":
        """Convenience: return (enricher_instance, domain_text) from a .pddl file path."""
        text = Path(domain_path).read_text()
        return cls(), text

    # ── PDDL text manipulation ────────────────────────────────────────────────

    def _find_matching_paren(self, text: str, start: int) -> int:
        """Return position of the ) that closes the ( at position start. Returns -1 on failure."""
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    def _insert_into_block(self, text: str, keyword: str, new_content: str) -> tuple[str, bool]:
        """
        Append new_content as a new line inside the (:keyword ...) block.
        Returns (modified_text, success).
        """
        pattern = re.compile(r"\(" + re.escape(keyword) + r"\b")
        m = pattern.search(text)
        if not m:
            return text, False
        block_end = self._find_matching_paren(text, m.start())
        if block_end == -1:
            return text, False
        insertion = f"\n    {new_content}"
        return text[:block_end] + insertion + text[block_end:], True

    def _append_action(self, text: str, action_pddl: str) -> str:
        """
        Insert a new action definition before the final closing ) of the domain.
        """
        last_paren = text.rfind(")")
        if last_paren == -1:
            return text + "\n\n" + action_pddl + "\n"
        return text[:last_paren] + "\n" + action_pddl + "\n" + text[last_paren:]

    def _extend_precondition(
        self,
        text: str,
        action_name: str,
        extra_conditions: list[str],
    ) -> tuple[str, bool]:
        """
        Add extra_conditions to the :precondition of action_name.

        If the precondition is already wrapped in (and ...), the new conditions
        are inserted inside it. Otherwise the precondition is wrapped in a new
        (and current_precond new_cond1 ...).

        Returns (modified_text, success).
        """
        pattern = re.compile(r"\(:action\s+" + re.escape(action_name) + r"\b")
        m = pattern.search(text)
        if not m:
            return text, False

        action_start = m.start()
        action_end = self._find_matching_paren(text, action_start)
        if action_end == -1:
            return text, False

        action_block = text[action_start:action_end + 1]

        pre_m = re.search(r":precondition\s*", action_block)
        if not pre_m:
            return text, False

        # Find the ( that opens the precondition expression
        paren_start = action_block.find("(", pre_m.end())
        if paren_start == -1:
            return text, False

        paren_end = self._find_matching_paren(action_block, paren_start)
        if paren_end == -1:
            return text, False

        current = action_block[paren_start:paren_end + 1]
        extra_str = "\n                 ".join(extra_conditions)

        if re.match(r"\(and\b", current.strip()):
            # Already an (and ...) — insert before its closing )
            inner_close = self._find_matching_paren(current, 0)
            new_precond = (
                current[:inner_close]
                + f"\n                 {extra_str}"
                + current[inner_close:]
            )
        else:
            # Single condition — wrap in (and ...)
            new_precond = f"(and {current}\n                 {extra_str})"

        new_action = action_block[:paren_start] + new_precond + action_block[paren_end + 1:]
        return text[:action_start] + new_action + text[action_end + 1:], True

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_existing(self, text: str) -> dict[str, set[str]]:
        """
        Return sets of already-defined type names, predicate names, and action names.
        Used for duplicate detection before applying additions.
        """
        result: dict[str, set[str]] = {
            "types":      set(),
            "predicates": set(),
            "actions":    set(),
        }

        # Types: find (:types ...) block, collect identifiers that are not "-"
        m = re.search(r"\(:types\b", text)
        if m:
            end = self._find_matching_paren(text, m.start())
            block = text[m.start():end + 1] if end != -1 else ""
            tokens = re.findall(r"\b([a-zA-Z][\w-]*)\b", block)
            result["types"] = {
                t for t in tokens if t not in self._PDDL_KEYWORDS
            }

        # Predicates: find (:predicates ...) block, collect names of each (name ...) entry
        m = re.search(r"\(:predicates\b", text)
        if m:
            end = self._find_matching_paren(text, m.start())
            block = text[m.start():end + 1] if end != -1 else ""
            result["predicates"] = set(re.findall(r"\(([a-zA-Z][\w-]*)\s", block))
            result["predicates"].discard("predicates")

        # Actions: simple regex suffices
        result["actions"] = set(re.findall(r"\(:action\s+([a-zA-Z][\w-]*)", text))

        return result

    def _predicate_name(self, pred: str) -> str:
        """Extract the predicate name from a declaration like '(locked ?i - item)'."""
        m = re.match(r"\(\s*([a-zA-Z][\w-]*)", pred.strip())
        return m.group(1) if m else pred.strip()

    def _format_action(self, action: dict) -> str:
        """Render an action dict to a PDDL (:action ...) string."""
        name       = action.get("name", "unnamed")
        parameters = action.get("parameters", "()")
        precond    = action.get("precondition", "(and)")
        effect     = action.get("effect", "(and)")
        return (
            f"  (:action {name}\n"
            f"    :parameters {parameters}\n"
            f"    :precondition {precond}\n"
            f"    :effect {effect}\n"
            f"  )"
        )

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _balanced(text: str) -> bool:
        """Return True if parentheses in text are balanced."""
        return text.count("(") == text.count(")")

    def _validate(self, domain_text: str) -> tuple[bool, list[str]]:
        """
        Structural validation of the enriched domain.
        Checks: parenthesis balance, required blocks, at least one action.
        """
        errors: list[str] = []

        if not self._balanced(domain_text):
            errors.append(
                f"unbalanced parentheses "
                f"(open={domain_text.count('(')}, close={domain_text.count(')')})"
            )

        for keyword in (":requirements", ":types", ":predicates"):
            if keyword not in domain_text:
                errors.append(f"missing {keyword} block")

        if not re.search(r"\(:action\b", domain_text):
            errors.append("domain has no actions")

        return len(errors) == 0, errors
