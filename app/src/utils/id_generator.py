"""Deterministic ID generation utilities for knowledge graph entities."""

import hashlib


def generate_file_id(file_path: str) -> str:
    """
    Generate a deterministic, stable ID for a file based on its contents.
    Returns a 32-character lowercase hex string (16 bytes, SHA-256 truncated).
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:32]


def generate_class_id(class_iri: str, label: str, source: str) -> str:
    """
    Generate a deterministic, stable ID for a class instance.
    The ID is derived from (class_iri, label, source) — the source text
    disambiguates same-label concepts extracted from different passages.
    Returns a 32-character lowercase hex string (16 bytes, SHA-256 truncated).
    """
    raw = class_iri + "\x00" + label + "\x00" + source
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def generate_property_id(src_id: str, property_iri: str, dest_id: str) -> str:
    """
    Generate a deterministic, stable ID for a property (relationship) instance.
    Returns a 32-character lowercase hex string (16 bytes, SHA-256 truncated).
    """
    raw = src_id + "\x00" + property_iri + "\x00" + dest_id
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
