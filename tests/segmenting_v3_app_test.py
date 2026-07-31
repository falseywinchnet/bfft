import numpy as np

from viewer import segmenting_v3_app as app


def test_display_sampling_toggle_updates_only_result_panel(monkeypatch):
    source = np.zeros((8, 9, 3), dtype=np.float64)
    reconstruction = np.ones_like(source)
    result = {"reconstruction_rgb": reconstruction}
    calls = []

    monkeypatch.setattr(
        app,
        "_push_texture",
        lambda tag, image, result=None, **kwargs: calls.append(
            (tag, image, result, kwargs)
        ),
    )
    monkeypatch.setattr(app.dpg, "configure_item", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        app.dpg,
        "get_value",
        lambda tag: "Reconstruction",
    )

    state = {
        "busy": app.S.busy,
        "rgb": app.S.rgb,
        "result": app.S.result,
        "display_key": app.S.display_key,
        "source_display_key": app.S.source_display_key,
        "resampled_display": app.S.resampled_display,
    }
    try:
        app.S.busy = False
        app.S.rgb = source
        app.S.result = result
        app.S.display_key = (id(result), "Reconstruction", False)
        app.S.source_display_key = id(source)
        app.S.resampled_display = False

        app.toggle_display_sampling()

        assert app.S.resampled_display is True
        assert len(calls) == 1
        assert calls[0][0] == app.RESULT
        assert calls[0][1] is reconstruction
        assert calls[0][2] is result
    finally:
        for name, value in state.items():
            setattr(app.S, name, value)
