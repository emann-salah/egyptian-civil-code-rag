import re

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
DIGIT_TRANSLATION = str.maketrans(ARABIC_DIGITS, WESTERN_DIGITS)

# Parentheses (and brackets, for safety) flip to their mirror image when a
# line gets reversed - "(" needs to become ")" and vice versa, not just move
# position like the digits do.
MIRROR_TRANSLATION = str.maketrans("()[]{}", ")(][}{")

DIGIT_PATTERN = re.compile(r'[٠-٩0-9]+')


SUBSECTION_MARKER_RE = re.compile(r'[()]\s*([٠-٩0-9]+)\s*[()]')


def fix_pdfplumber_arabic_line(line: str) -> str:
    """Fix a single line of Arabic text extracted by pdfplumber.

    pdfplumber (via pdfminer) returns Arabic text in visual character
    order instead of logical reading order for this PDF's font. Reversing
    the whole line fixes word/letter order, but that reversal:
      1. also incorrectly reverses embedded digit runs (digits are always
         written left-to-right, even inside Arabic text), so we reverse
         those back afterwards.
      2. leaves parentheses/brackets pointing the wrong way (mirrored
         characters), so we swap "(" <-> ")" and "[" <-> "]" afterwards too.
    """
    reversed_line = line[::-1]
    reversed_line = DIGIT_PATTERN.sub(lambda m: m.group()[::-1], reversed_line)
    reversed_line = reversed_line.translate(MIRROR_TRANSLATION)
    return reversed_line


def fix_subsection_markers(text: str) -> str:
    """Normalize any parenthesis-digit-parenthesis marker to the canonical
    '(N)' form, regardless of which paren glyph the source used on each
    side (see SUBSECTION_MARKER_RE above for why this is necessary)."""
    return SUBSECTION_MARKER_RE.sub(r'(\1)', text)


def fix_pdfplumber_arabic_cell(cell_text: str) -> str:
    """Apply the line-level fix to every line in a multi-line table cell,
    then normalize subsection markers across the whole cell."""
    lines = cell_text.split("\n")
    fixed = "\n".join(fix_pdfplumber_arabic_line(line) for line in lines)
    return fix_subsection_markers(fixed)


def arabic_digits_to_western(text: str) -> str:
    """Convert Arabic-Indic digits (١٢٣) to Western digits (123)."""
    return text.translate(DIGIT_TRANSLATION)


def extract_article_number(heading_text: str) -> int | None:
    """Pull the article number out of a heading like 'مادة (13)'.

    Returns None if no digits found, so the caller can flag it instead
    of silently producing a wrong number.
    """
    western_text = arabic_digits_to_western(heading_text)
    match = re.search(r'\d+', western_text)
    if match is None:
        return None
    return int(match.group())