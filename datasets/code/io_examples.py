"""Author-created search corpus. These functions are never executed by the lessons."""


def read_json(path):
    """Read a UTF-8 file and parse its JSON object."""
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    """Serialize an object as indented JSON and write it to a UTF-8 file."""
    import json
    from pathlib import Path
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def parse_iso_date(text):
    """Parse a calendar date formatted as YYYY-MM-DD."""
    from datetime import date
    return date.fromisoformat(text)


def exponential_delay(attempt, maximum=30):
    """Compute a capped exponential retry delay without sleeping or making a request."""
    return min(2 ** attempt, maximum)
