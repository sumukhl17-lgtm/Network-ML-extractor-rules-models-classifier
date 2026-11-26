from __future__ import annotations

from types import MappingProxyType

TAXONOMY_VERSION = "audio-sound-groups-v2"

_GROUP_LABELS = {
    "animals": (
        "dog",
        "rooster",
        "pig",
        "cow",
        "frog",
        "cat",
        "hen",
        "insects",
        "sheep",
        "crow",
    ),
    "nature": (
        "rain",
        "sea_waves",
        "crackling_fire",
        "crickets",
        "chirping_birds",
        "water_drops",
        "wind",
        "pouring_water",
        "thunderstorm",
    ),
    "human_sounds": (
        "crying_baby",
        "sneezing",
        "clapping",
        "breathing",
        "coughing",
        "footsteps",
        "laughing",
        "brushing_teeth",
        "snoring",
        "drinking_sipping",
    ),
    "household": (
        "door_wood_knock",
        "mouse_click",
        "keyboard_typing",
        "door_wood_creaks",
        "can_opening",
        "washing_machine",
        "vacuum_cleaner",
        "clock_alarm",
        "clock_tick",
        "glass_breaking",
        "toilet_flush",
    ),
    "transport": (
        "helicopter",
        "car_horn",
        "engine",
        "train",
        "airplane",
    ),
    "tools_and_machinery": (
        "chainsaw",
        "hand_saw",
    ),
    "alarms_and_events": (
        "siren",
        "church_bells",
        "fireworks",
    ),
}

ESC50_TO_SOUND_GROUP = MappingProxyType(
    {
        label: group
        for group, labels in _GROUP_LABELS.items()
        for label in labels
    }
)

ESC50_CLASSES = tuple(sorted(ESC50_TO_SOUND_GROUP))
