#!/usr/bin/env python3
"""Texture size changes never reuse a live DearPyGui alias."""

from contextlib import contextmanager

from viewer import object_transport_segmentation_app as app


def test_reallocation_uses_a_fresh_texture_alias(monkeypatch) -> None:
    existing = set()
    added = []
    deleted = []
    configured = []

    @contextmanager
    def registry():
        yield

    def add_raw_texture(width, height, value, *, tag, format):
        assert tag not in existing
        existing.add(tag)
        added.append(tag)

    monkeypatch.setattr(app.dpg, "texture_registry", registry)
    monkeypatch.setattr(app.dpg, "add_raw_texture", add_raw_texture)
    monkeypatch.setattr(
        app.dpg,
        "does_item_exist",
        lambda tag: tag in existing or tag == "object_result_image",
    )
    monkeypatch.setattr(
        app.dpg,
        "configure_item",
        lambda tag, **kwargs: configured.append((tag, kwargs)),
    )

    def delete_item(tag):
        deleted.append(tag)
        existing.discard(tag)

    monkeypatch.setattr(app.dpg, "delete_item", delete_item)
    app.S.buffers.clear()
    app.S.texture_shapes.clear()
    app.S.texture_items.clear()
    app.S.texture_generation.clear()

    app.alloc_texture(app.RESULT, 8, 8)
    app.alloc_texture(app.RESULT, 31, 47)

    assert added == [app.RESULT, f"{app.RESULT}__1"]
    assert app.S.texture_items[app.RESULT] == f"{app.RESULT}__1"
    assert deleted == [app.RESULT]
    assert configured[-1][1]["texture_tag"] == f"{app.RESULT}__1"

