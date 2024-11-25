from __future__ import annotations

from collections.abc import Generator
from enum import StrEnum
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from content.dynamic_content import DynamicContent
from main import LIBRARY

if TYPE_CHECKING:
    from content.setting import Setting

REPO_PATH = Path(__file__).parents[2]

HA_REPO = REPO_PATH.parent / "home-assistant"
MQTT_DIR = HA_REPO / "entities/mqtt"


def get_all_settings() -> Generator[Setting[Any], None, None]:
    for content in LIBRARY:
        if isinstance(content, DynamicContent):
            yield from content.settings.values()


def titleify(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").title()


def mqtt_number(setting: Setting[int] | Setting[float]) -> None:
    unit = setting.unit_of_measurement

    if not unit:
        unit = '" "'
    elif unit == "%":
        unit = '"%"'

    step = (
        1
        if setting.type_ is int
        else max(0.001, getattr(setting, "transition_rate", 0.01))
    )

    yaml = (
        dedent(
            f"""
    ---
    name: "MtrxPi | {titleify(setting.instance.id)}: {titleify(setting.slug)}"

    unique_id: mtrxpi_{setting.instance.id.replace("-", "_")}_{setting.slug}

    command_topic: {setting.mqtt_topic}

    icon: {setting.icon}

    min: {setting.minimum}

    max: {setting.maximum}

    mode: {setting.display_mode}

    retain: true

    step: {step}

    unit_of_measurement: {unit}

    state_topic: {setting.mqtt_topic}
    """,
        ).strip()
        + "\n"
    )

    file_path = MQTT_DIR.joinpath(
        "number",
        "mtrxpi",
        setting.instance.id.replace("-", "_"),
        setting.slug,
    ).with_suffix(".yaml")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(yaml)


def mqtt_select(setting: Setting[Any]) -> None:
    options = "\n      ".join(f"- {titleify(option)}" for option in setting.type_)

    yaml = (
        dedent(
            f"""
    ---
    name: "MtrxPi | {titleify(setting.instance.id)}: {titleify(setting.slug)}"

    unique_id: mtrxpi_{setting.instance.id.replace("-", "_")}_{setting.slug}

    command_topic: {setting.mqtt_topic}

    command_template: "{{{{ slugify(value) | tojson }}}}" # hacv disable: InvalidTemplateVar:value,slugify

    icon: {setting.icon}

    retain: true

    options:
      {options}

    state_topic: {setting.mqtt_topic}

    value_template: "{{{{ value_json.replace('_', ' ').title() }}}}"
    """,
        ).strip()
        + "\n"
    )

    file_path = MQTT_DIR.joinpath(
        "select",
        "mtrxpi",
        setting.instance.id.replace("-", "_"),
        setting.slug,
    ).with_suffix(".yaml")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(yaml)


def mqtt_switch(setting: Setting[bool]) -> None:
    yaml = (
        dedent(
            f"""
    ---
    name: "MtrxPi | {titleify(setting.instance.id)}: {titleify(setting.slug)}"

    unique_id: mtrxpi_{setting.instance.id.replace("-", "_")}_{setting.slug}

    command_topic: {setting.mqtt_topic}

    icon: {setting.icon}

    payload_on: "true"

    payload_off: "false"

    retain: true

    state_topic: {setting.mqtt_topic}

    state_on: "true"

    state_off: "false"
    """,
        ).strip()
        + "\n"
    )

    file_path = MQTT_DIR.joinpath(
        "switch",
        "mtrxpi",
        setting.instance.id.replace("-", "_"),
        setting.slug,
    ).with_suffix(".yaml")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(yaml)


def main() -> None:
    for setting in get_all_settings():
        if setting.type_ in {int, float}:
            mqtt_number(setting)
        elif issubclass(setting.type_, StrEnum):
            mqtt_select(setting)
        elif setting.type_ is bool:
            mqtt_switch(setting)
        else:
            print(f"Unsupported type for {setting.slug!r}: {setting.type_}")


if __name__ == "__main__":
    main()
