# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.utils.isaaclab_utils.simulation_app import (
    LIVESTREAM_DYNAMIC_RESIZE_SETTING,
    _configure_livestream_dynamic_resize,
)


def test_public_livestream_enables_dynamic_resize():
    args = argparse.Namespace(livestream=1, kit_args="--ext-folder=/tmp/extensions")

    _configure_livestream_dynamic_resize(args)

    assert args.kit_args == f"--ext-folder=/tmp/extensions --{LIVESTREAM_DYNAMIC_RESIZE_SETTING}=true"


def test_livestream_environment_enables_dynamic_resize(monkeypatch):
    monkeypatch.setenv("LIVESTREAM", "1")
    args = argparse.Namespace(livestream=-1, kit_args="")

    _configure_livestream_dynamic_resize(args)

    assert args.kit_args == f"--{LIVESTREAM_DYNAMIC_RESIZE_SETTING}=true"


def test_explicit_dynamic_resize_setting_is_preserved():
    explicit_setting = f"--{LIVESTREAM_DYNAMIC_RESIZE_SETTING}=false"
    args = argparse.Namespace(livestream=1, kit_args=explicit_setting)

    _configure_livestream_dynamic_resize(args)

    assert args.kit_args == explicit_setting


def test_disabled_livestream_does_not_change_kit_args():
    args = argparse.Namespace(livestream=0, kit_args="--ext-folder=/tmp/extensions")

    _configure_livestream_dynamic_resize(args)

    assert args.kit_args == "--ext-folder=/tmp/extensions"
