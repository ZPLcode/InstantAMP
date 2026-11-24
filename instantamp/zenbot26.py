"""Zenbot 26-DoF articulation profile for the shared AMP tasks."""

from functools import partial

from . import zenbot
from .zenbot import (
    ACTION_SCALE as ACTION_SCALE,
    DEFAULT_HEIGHT as DEFAULT_HEIGHT,
    FEET as FEET,
    HANDS as HANDS,
    HIST as HIST,
    KEY_BODIES as KEY_BODIES,
    KNEES as KNEES,
    RESET_LIFT as RESET_LIFT,
    RISE_HEIGHT as RISE_HEIGHT,
    SUPPORT_BODIES as SUPPORT_BODIES,
    TORSO as TORSO,
    UP_HEIGHT as UP_HEIGHT,
)
from .zenbot_models import ZENBOT26 as MODEL

MIXED_TASK = MODEL.mixed_task
MASS = MODEL.mass
DOF = MODEL.dof
KP, KD = MODEL.kp, MODEL.kd
DEFAULT_Q, GROUP = MODEL.default_q, MODEL.group
EFFORT, SPEED = MODEL.effort, MODEL.speed
WRIST_IDX = MODEL.wrist_indices
TERMINATION_BODIES = tuple(
    MODEL.head_body if body == zenbot.MODEL.head_body else body
    for body in zenbot.TERMINATION_BODIES
)
# This MJCF already contains the physical capsules.
CAPSULES = {}
actuator_ctrl = partial(
    zenbot.actuator_ctrl, kp=KP, kd=KD, effort=EFFORT, speed=SPEED
)
