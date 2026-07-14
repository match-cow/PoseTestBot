"""Validated object registry shared by scene, rendering, and BOP export."""

from posetestbot.objects.registry import (
    ObjectRegistry,
    ObjectRegistryEntry,
    load_object_registry,
)

__all__ = ["ObjectRegistry", "ObjectRegistryEntry", "load_object_registry"]
