"""Text prompts.

ActionRoPE never puts the action into the text: the prompt only describes the scene,

    "In a 2.5D top down view, the player is in a high rocky mountain pass."

and ``NULL_PROMPT`` (no scene) is used for scene dropout during training and for all evaluation clips.
"""
from __future__ import annotations

import re

NULL_PROMPT = "In a 2.5D top down view."
SCENE_TEMPLATE = "In a 2.5D top down view, the player is {scene}."

_WITH_ACTION = re.compile(r"^In a 2\.5D top down view, (?P<scene>.+), the player is (?P<action>.+)\.$")
_ACTION_ONLY = re.compile(r"^In a 2\.5D top down view, the player is (?P<action>.+)\.$")


def scene_prompt(scene: str | None) -> str:
    """Scene description (e.g. "in a dark cave lit by blue crystal glow") -> prompt; None -> NULL_PROMPT."""
    return NULL_PROMPT if not scene else SCENE_TEMPLATE.format(scene=scene)


def strip_action(prompt: str) -> str:
    """"In a 2.5D top down view, {scene}, the player is {action}." -> scene_prompt(scene).

    Captions that carry an action but no scene collapse to NULL_PROMPT."""
    m = _WITH_ACTION.match(prompt)
    if m:
        return scene_prompt(m.group("scene"))
    if _ACTION_ONLY.match(prompt):
        return NULL_PROMPT
    raise ValueError(f"prompt does not follow the caption template: {prompt!r}")
