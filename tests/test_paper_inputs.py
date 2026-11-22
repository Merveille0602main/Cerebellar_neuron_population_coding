import json

from cerebellar_reproduction.paper_inputs import build_paper_mossy_input_schedule


def test_bsb3103_noise_is_reproducible_poisson_window_and_sd_is_ignored(tmp_path):
    config = {
        "simulations": {
            "stim_on_MFs": {
                "duration": 1250,
                "devices": {
                    "initial_mf": {
                        "cell_types": ["mossy_fibers"],
                        "targetting": "by_id",
                        "targets": [10],
                        "parameters": {
                            "noise": True,
                            "start": 200,
                            "interval": 30,
                            "number": 2,
                            "sd": 2,
                        },
                    },
                    "background": {
                        "cell_types": ["mossy_fibers"],
                        "targetting": "cell_type",
                        "parameters": {
                            "noise": True,
                            "start": 500,
                            "interval": 250,
                            "number": 3,
                        },
                    },
                }
            }
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    first = build_paper_mossy_input_schedule(path, [10, 11], seed=123)
    second = build_paper_mossy_input_schedule(path, [10, 11], seed=123)
    different = build_paper_mossy_input_schedule(path, [10, 11], seed=124)

    assert first.spike_times_ms == second.spike_times_ms
    assert first.spike_times_ms != different.spike_times_ms
    assert first.ignored_sd_devices == ("initial_mf",)
    initial = tuple(value for value in first.spike_times_ms[10] if value < 500.0)
    assert all(200.0 < value <= 260.0 for value in initial)
    assert all(value > 500.0 for value in first.spike_times_ms[11])


def test_explicit_device_targets_can_be_supplied_for_empty_paper_target_list(tmp_path):
    config = {
        "simulations": {
            "stim_on_MFs": {
                "duration": 1200,
                "devices": {
                    "mossy_fiber_sensoryBurst": {
                        "cell_types": ["mossy_fibers"],
                        "targetting": "by_id",
                        "targets": [],
                        "spike_times": [1100.0, 1103.0, 1106.0, 1131.0, 1151.0],
                    }
                }
            }
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))

    schedule = build_paper_mossy_input_schedule(
        path,
        [213, 214],
        device_names=("mossy_fiber_sensoryBurst",),
        target_overrides={"mossy_fiber_sensoryBurst": (214,)},
    )

    assert schedule.spike_times_ms[213] == ()
    assert schedule.spike_times_ms[214] == (1100.0, 1103.0, 1106.0, 1131.0, 1151.0)
