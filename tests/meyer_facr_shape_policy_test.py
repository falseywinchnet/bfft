from bfft._core import _meyer_facr_work_shape


def test_facr_shape_policy_keeps_small_minimum_area_column_path():
    assert _meyer_facr_work_shape(360, 640) == (512, 640)
    assert _meyer_facr_work_shape(480, 854) == (512, 854)


def test_facr_shape_policy_prefers_video_scale_contiguous_rows():
    assert _meyer_facr_work_shape(720, 1280) == (720, 2048)
    assert _meyer_facr_work_shape(1280, 720) == (1280, 1024)
    assert _meyer_facr_work_shape(1000, 1536) == (1000, 2048)
    assert _meyer_facr_work_shape(1440, 2560) == (1440, 4096)


def test_facr_shape_policy_retains_column_path_for_large_area_saving():
    assert _meyer_facr_work_shape(1920, 1080) == (2048, 1080)


def test_facr_shape_policy_does_not_pad_when_one_axis_is_transformable():
    assert _meyer_facr_work_shape(1024, 1366) == (1024, 1366)
    assert _meyer_facr_work_shape(1080, 2048) == (1080, 2048)
