"""Author-created search corpus. This file is indexed as text, not imported by the lessons."""


def unique_in_order(values):
    """Remove repeated values while preserving the order of first appearance."""
    return list(dict.fromkeys(values))


def descending_numbers(values):
    """Sort numeric values from largest to smallest."""
    return sorted(values, reverse=True)


def word_counts(text):
    """Count occurrences of each lowercase whitespace-separated word."""
    from collections import Counter
    return Counter(text.lower().split())


def find_device_by_serial(devices, serial):
    """Find the first device whose serial number exactly matches, or return None."""
    return next((device for device in devices if device['serial'] == serial), None)
