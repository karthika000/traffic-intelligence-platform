"""Does a report say an incident is over? Rule-based ("cleared", "reopened", "back to normal",
"moving again", ...), with negations and forward-looking phrasing excluded ("not cleared yet",
"will reopen"). fusion_service uses it to resolve incidents sooner."""

import re

ACTIVE, CLEARED = "active", "cleared"

_CLEARED = re.compile(
    r"\b(?:"
    r"(?:has|have|now|been|finally|is|are)\s+(?:now\s+)?(?:been\s+)?(?:cleared|resolved|reopened"
    r"|removed|dealt with|recovered|towed(?: away)?)"
    r"|cleared\b(?!\s+(?:to|for|by)\b)|resolved|reopened|all lanes (?:have )?(?:re)?opened"
    r"|(?:is|are) gone|gone and|sorted(?: at last)?|moving (?:again|freely)|back to normal"
    r"|returning to normal|normal service|road is clear(?: again)?|delays are easing"
    r"|recovery is complete|stand down|incident \S+ closed|left the scene|all good now"
    r"|no longer (?:blocked|closed|an issue)"
    r")",
    re.IGNORECASE,
)
_NOT_YET = re.compile(
    r"\b(?:not|isn't|hasn't|haven't|aren't|no sign of|yet to be|still|until|expected to|will"
    r"|should|once|while|whilst|wile|being|to be)\b[^.!?]{0,25}$",
    re.IGNORECASE,
)


def report_stage(text: str) -> str:
    """CLEARED if the report says the incident is over, else ACTIVE."""
    for m in _CLEARED.finditer(text):
        before = text[max(0, m.start() - 40) : m.start()]
        if not _NOT_YET.search(before):
            return CLEARED
    return ACTIVE
