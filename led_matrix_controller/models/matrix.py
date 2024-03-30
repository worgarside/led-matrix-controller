"""Constants and class for managing the RGB LED Matrix."""

from __future__ import annotations

from json import dumps
from logging import DEBUG, getLogger
from math import ceil
from threading import Thread
from time import sleep, time
from typing import ClassVar, Literal, TypedDict

from paho.mqtt.publish import multiple
from utils import const
from wg_utilities.loggers import add_stream_handler

from ._rgbmatrix import DrawText, RGBMatrix, RGBMatrixOptions
from .artwork_image import NULL_IMAGE, ArtworkImage
from .text_label import FONT, Text

LOGGER = getLogger(__name__)
LOGGER.setLevel(DEBUG)
add_stream_handler(LOGGER)


class LedMatrixOptions(TypedDict, total=False):
    """Typing info for the matrix options."""

    cols: int
    """Columns in the LED matrix, the 'width'."""

    rows: int
    """Rows in the LED matrix, the 'height'."""

    brightness: int

    gpio_slowdown: int
    """Reduce the speed of writing to the GPIO pins.

    The Raspberry Pi starting with Pi2 are putting out data too fast for almost all LED panels I have seen. In
    this case, you want to slow down writing to GPIO. Zero for this parameter means 'no slowdown'.

    The default 1 (one) typically works fine, but often you have to even go further by setting it to 2 (two). If
    you have a Raspberry Pi with a slower processor (Model A, A+, B+, Zero), then a value of 0 (zero) might work
    and is desirable.

    A Raspberry Pi 3 or Pi4 might even need higher values for the panels to be happy.

    https://github.com/hzeller/rpi-rgb-led-matrix#gpio-speed
    """

    hardware_mapping: Literal[
        "regular",
        "adafruit-hat",
        "adafruit-hat-pwm",
        "regular-pi1",
        "classic",
        "classic-pi1",
        "compute-module",
    ]
    """The hardware mapping to use.

    https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/wiring.md#alternative-hardware-mappings
    """

    inverse_colors: bool
    """Switch if your matrix has inverse colors on."""

    chain_length: int
    """The chain_length is the number of displays daisy-chained together. Defaults to 1.

    i.e output of one connected to input of next

    https://github.com/hzeller/rpi-rgb-led-matrix#panel-connection
    """

    parallel: int
    """The newer Raspberry Pis allow to connect multiple chains in parallel.

    https://github.com/hzeller/rpi-rgb-led-matrix#panel-connection
    """

    pwm_bits: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    """The number of bits to use for PWM.

    The LEDs can only be switched on or off, so the shaded brightness perception is achieved via PWM.
    In order to get a good 8 Bit per color resolution (24Bit RGB), the 11 bits default per color are good
    (why ? Because our eyes are actually perceiving brightness logarithmically, so we need a lot more physical
    resolution to get 24Bit sRGB).

    With this flag, you can change how many bits it should use for this; lowering it means the lower bits (=more
    subtle color nuances) are omitted. Typically you might be mostly interested in the extremes: 1 Bit for
    situations that only require 8 colors (e.g. for high contrast text displays) or 11 Bit for everything else
    (e.g. showing images or videos). Why would you bother at all ? Lower number of bits use slightly less CPU and
    result in a higher refresh rate.
    """

    pwm_lsb_nanoseconds: int
    """This allows to change the base time-unit for the on-time in the lowest significant bit in nanoseconds.

    Lower values will allow higher frame-rate, but will also negatively impact qualty in some panels (less
    accurate color or more ghosting).

    Good values for full-color display (PWM=11) are somewhere between 100 and 300.

    If you you use reduced bit color (e.g. PWM=1) and have sharp contrast applications, then higher values
    might be good to minimize ghosting.

    https://github.com/hzeller/rpi-rgb-led-matrix#misc-options
    """

    scan_mode: Literal[0, 1]
    """This switches from progressive scan and interlaced scan.

    0 = progressive; 1 = interlaced (Default: 0)

    The latter might look be a little nicer when you have a very low refresh rate, but typically it is more
    annoying because of the comb-effect (remember 80ies TV ?).
    """

    multiplexing: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    """The multiplexing type of the panel.

    The outdoor panels have different multiplexing which allows them to be faster and brighter, but by default
    their output looks jumbled up. They require some pixel-mapping of which there are a few types you can try
    and hopefully one of them works for your panel.
    """

    row_address_type: Literal[0, 1, 2, 3, 4]
    """This option is useful for certain 64x64 or 32x16 panels.

    For 64x64 panels, that only have an A and B address line, you'd use --led-row-addr-type=1. This is only
    tested with one panel so far, so if it doesn't work for you, please send a pull request.

    For 32x16 outdoor panels, that have have 4 address line (A, B, C, D), it is necessary to use
    --led-row-addr-type=2.
    """

    disable_hardware_pulsing: bool
    """This disables the hardware-pulsing of the GPIO pins.

    This library uses a hardware subsystem that also is used by the sound. You can't use them together. If your
    panel does not work, this might be a good start to debug if it has something to do with the sound subsystem.
    This is really only recommended for debugging; typically you actually want the hardware pulses as it results
    in a much more stable picture.
    """

    show_refresh_rate: bool
    """This shows the current refresh rate of the LED panel, the time to refresh a full picture.

    Typically, you want this number to be pretty high, because the human eye is pretty sensitive to flicker.
    Depending on the settings, the refresh rate with this library are typically in the hundreds of Hertz but
    can drop low with very long chains. Humans have different levels of perceiving flicker - some are fine
    with 100Hz refresh, others need 250Hz. So if you are curious, this gives you the number (shown on the
    terminal).

    The refresh rate depends on a lot of factors, from --led-rows and --led-chain to --led-pwm-bits,
    --led-pwm-lsb-nanoseconds and --led-pwm-dither-bits. If you are tweaking these parameters, showing the
    refresh rate can be a useful tool.
    """

    led_rgb_sequence: Literal["RGB", "BGR", "GBR", "RBG", "BRG", "GRB"]
    """Switch if your matrix has led colors swapped."""

    pixel_mapper_config: str
    """Mapping the logical layout of your boards to your physical arrangement.

    https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/examples-api-use#remapping-coordinates
    """

    panel_type: str
    """Chipset of the panel.

    In particular if it doesn't light up at all, you might need to play with this option because it indicates that
    the panel requires a particular initialization sequence.
    """

    pwm_dither_bits: int
    """Time dithering of lower bits (Default: 0)

    The lower bits can be time dithered, i.e. their brightness contribution is achieved by only showing them some
    frames (this is possible, because the PWM is implemented as binary code modulation). This will allow higher
    refresh rate (or same refresh rate with increased --led-pwm-lsb-nanoseconds). The disadvantage could be
    slightly lower brightness, in particular for longer chains, and higher CPU use. CPU use is not of concern for
    Raspberry Pi 2 or 3 (as we run on a dedicated core anyway) but probably for Raspberry Pi 1 or Pi Zero.

    Default: no dithering; if you have a Pi 3 and struggle with low frame-rate due to high multiplexing panels
    (1:16 or 1:32) or long chains, it might be worthwhile to try.
    """

    limit_refresh_rate_hz: int
    """This allows to limit the refresh rate to a particular frequency to approach a fixed refresh rate.

    This can be used to mitigate some situations in which you have a faint flicker, which can happen due
    to hardware events (network access) or other situations such as other IO or heavy memory access by other
    processes. Also when you see wildly changing refresh frequencies with --led-show-refresh.

    You trade a slightly slower refresh rate and display brightness for less visible flicker situations.

    For this to calibrate, run your program for a while with --led-show-refresh and watch the line that shows
    the current refresh rate and minimum refresh rate observed. So wait a while until that value doesn't change
    anymore (e.g. a minute, so that you catch tasks that happen once a minute, such as ntp updated). Use this
    as a guidance what value to choose with --led-limit-refresh.

    The refresh rate will now be adapted to always reach this value between frames, so faster refreshes will be
    slowed down, but the occasional delayed frame will fit into the time-window as well, thus reducing visible
    brightness fluctuations.

    You can play with value a little and reduce until you find a good balance between refresh rate and flicker
    suppression.
    """

    daemon: bool
    """Make the process run in the background as daemon.

    If this is set, the program puts itself into the background (running as 'daemon'). You might want this if
    started from an init script at boot-time.
    """

    drop_privileges: bool
    """Don't drop privileges from 'root' after initializing the hardware.

    You need to start programs as root as it needs to access some low-level hardware at initialization time. After
    that, it is typically not desirable to stay in this role, so the library then drops the privileges.

    This flag allows to switch off this behavior, so that you stay root. Not recommended unless you have a specific
    reason for it (e.g. you need root to access other hardware or you do the privilege dropping yourself).
    """


class HAPendingUpdatesInfo(TypedDict):
    """Typing info for the record of pending attribute updates."""

    artist: bool
    entity_picture: bool
    media_title: bool


class HAPayloadInfo(TypedDict):
    """Typing info for the payload of the MQTT message to Home Assistant."""

    state: bool
    media_title: str | None
    artist: str | None
    album: str | None
    album_artwork_url: str | None


class Matrix:
    """Class for displaying track information on an RGB LED Matrix."""

    OPTIONS: ClassVar[LedMatrixOptions] = {
        "cols": 64,
        "rows": 64,
        "brightness": 80,
        "gpio_slowdown": 4,
        "hardware_mapping": "adafruit-hat-pwm",
        "inverse_colors": False,
        "led_rgb_sequence": "RGB",
        "show_refresh_rate": False,
    }

    def __init__(self, brightness: int | None = None) -> None:
        options = RGBMatrixOptions()
        for k, v in self.OPTIONS.items():
            if k == "brightness" and brightness:
                options.brightness = brightness
                continue
            setattr(options, k, v)

        self.matrix = RGBMatrix(options=options)
        self.canvas = self.matrix.CreateFrameCanvas()

        artist_y_pos = self.matrix.height - 2
        media_title_y_pos = artist_y_pos - (const.FONT_HEIGHT + 1)

        self.image_size = media_title_y_pos - (const.FONT_HEIGHT + 3)
        self.image_x_pos: int = int((self.matrix.width - self.image_size) / 2)
        self.image_y_pos: int = int(
            (self.matrix.height - (const.FONT_HEIGHT * 2 + 2) - self.image_size) / 2
        )

        self._media_title: Text = Text(
            "-", media_title_y_pos, matrix_width=self.matrix.width
        )
        self._artist: Text = Text("-", artist_y_pos, matrix_width=self.matrix.width)
        self._artwork_image: ArtworkImage = NULL_IMAGE

        self.scroll_thread = Thread(target=self._scroll_worker)
        self.ha_update_thread = Thread(target=self._update_ha_worker)

        self._pending_ha_updates: HAPendingUpdatesInfo = {
            "media_title": False,
            "artist": False,
            "entity_picture": False,
        }
        self._ha_last_updated = time()

    def _clear_text(self, text: Text, *, update_canvas: bool = False) -> None:
        """Clear a line on the canvas by writing a line of black "█" characters.

        Args:
            text (str): the text instance to clear
            update_canvas (bool, optional): whether to update the canvas after clearing
                the text. Defaults to False.
        """
        DrawText(
            self.canvas,
            FONT,
            0,
            text.y_pos,
            text.CLEAR_TEXT_COLOR,
            "█" * ceil(self.matrix.width / const.FONT_WIDTH),
        )
        if update_canvas:
            self.matrix.SwapOnVSync(self.canvas)

    def _scroll_worker(self) -> None:
        """Actively scrolls the media title and artist text when required."""

        while self.scrollable_content:
            if self.artist.scrollable:
                self.write_artist(clear_first=True)

            if self.media_title.scrollable:
                self.write_media_title(clear_first=True)

            self.matrix.SwapOnVSync(self.canvas)
            sleep(0.5)

        LOGGER.debug("Scroll worker exiting")

    def _start_update_ha_worker(self) -> None:
        """Start the HA update worker thread if it is not already running."""
        try:
            if self.ha_update_thread.is_alive():
                LOGGER.warning("HA update thread is already running")
            else:
                self.ha_update_thread.start()
                LOGGER.debug("HA update thread is dead, restarted")
        except (RuntimeError, AttributeError) as exc:
            LOGGER.debug("Recreating HA update thread: %s", repr(exc))
            self.ha_update_thread = Thread(target=self._update_ha_worker)
            self.ha_update_thread.start()

    def _start_scroll_worker(self) -> None:
        """Start the scroll worker thread if it is not already running."""
        try:
            if self.scroll_thread.is_alive():
                LOGGER.warning("Scroll thread is already running")
            else:
                self.scroll_thread.start()
                LOGGER.debug("Scroll thread is dead, restarted")
        except (RuntimeError, AttributeError) as exc:
            LOGGER.debug("Recreating scroll thread: %s", repr(exc))
            self.scroll_thread = Thread(target=self._scroll_worker)
            self.scroll_thread.start()

    def _update_ha_worker(self) -> None:
        start_time = time()

        # Wait up to 2.5 seconds
        while time() < start_time + 2.5 and not all(self.pending_ha_updates.values()):
            sleep(0.1)

        if not all(self.pending_ha_updates.values()):
            LOGGER.warning(
                "Timed out waiting for pending Home Assistant updates, sending current"
                " values"
            )

        multiple(
            msgs=[
                {
                    "topic": const.HA_LED_MATRIX_STATE_TOPIC,
                    "payload": (
                        "ON"
                        if any(
                            [
                                self.artwork_image != NULL_IMAGE,
                                self.artist.display_content != "",
                                self.media_title.display_content != "",
                            ]
                        )
                        else "OFF"
                    ),
                },
                {
                    "topic": const.HA_MTRXPI_CONTENT_TOPIC,
                    "payload": dumps(self.home_assistant_payload),
                },
            ],
            auth={"username": const.MQTT_USERNAME, "password": const.MQTT_PASSWORD},
            hostname=const.MQTT_HOST,
        )

        self._pending_ha_updates = {
            "entity_picture": False,
            "media_title": False,
            "artist": False,
        }

        LOGGER.debug(
            "Sent all pending updates to HA: %s", dumps(self.home_assistant_payload)
        )

    def clear_artist(self, *, update_canvas: bool = False) -> None:
        """Clear the artist text.

        Args:
            update_canvas (bool, optional): whether to update the canvas after clearing
                the text. Defaults to False.
        """
        self._clear_text(self.artist, update_canvas=update_canvas)

    def clear_media_title(self, *, update_canvas: bool = False) -> None:
        """Clear the media title text.

        Args:
            update_canvas (bool, optional): whether to update the canvas after clearing
                the text. Defaults to False.
        """
        self._clear_text(self.media_title, update_canvas=update_canvas)

    def write_artist(
        self, *, clear_first: bool = False, swap_on_vsync: bool = False
    ) -> None:
        """Force the artist to be written to the canvas.

        Args:
            clear_first (bool, optional): whether to clear the artist text before
                writing
            swap_on_vsync (bool, optional): update the canvas after writing the text
        """

        if clear_first:
            self.clear_artist()

        DrawText(
            self.canvas,
            FONT,
            self.artist.get_next_x_pos(),
            self.artist.y_pos,
            self.artist.color,
            self.artist.display_content,
        )

        if swap_on_vsync:
            self.matrix.SwapOnVSync(self.canvas)

    def write_artwork_image(self, *, swap_on_vsync: bool = False) -> None:
        """Write the artwork image to the canvas.

        Args:
            swap_on_vsync (bool, optional): whether to swap the canvas on vsync.
                Defaults to False.
        """
        self.canvas.SetImage(
            self.artwork_image.get_image(
                self.image_size,
            ).convert("RGB"),
            offset_x=self.image_x_pos,
            offset_y=self.image_y_pos,
        )

        if swap_on_vsync:
            self.matrix.SwapOnVSync(self.canvas)

    def write_media_title(
        self, *, clear_first: bool = False, swap_on_vsync: bool = False
    ) -> None:
        """Force the media title to be written to the canvas.

        Args:
            clear_first (bool, optional): whether to clear the media title text before
                writing
            swap_on_vsync (bool, optional): update the canvas after writing the text
        """
        if clear_first:
            self.clear_media_title()

        DrawText(
            self.canvas,
            FONT,
            self.media_title.get_next_x_pos(),
            self.media_title.y_pos,
            self.media_title.color,
            self.media_title.display_content,
        )

        if swap_on_vsync:
            self.matrix.SwapOnVSync(self.canvas)

    @property
    def artist(self) -> Text:
        """Returns the media title content."""
        return self._artist

    @artist.setter
    def artist(self, value: str) -> None:
        """Set the artist text content."""
        if not isinstance(value, str):
            raise TypeError(f"Value for `artist` must be a string: {value!r}")

        if value == self.artist.original_content:
            return

        self._artist.display_content = value
        self.write_artist(clear_first=True, swap_on_vsync=True)

        self.pending_ha_updates = {
            "artist": True,
            "entity_picture": self.pending_ha_updates["entity_picture"],
            "media_title": self.pending_ha_updates["media_title"],
        }

        if self.artist.scrollable:
            LOGGER.debug("Sending request to start scroll thread from artist setter")
            self._start_scroll_worker()

    @property
    def artwork_image(self) -> ArtworkImage:
        """Returns the current artwork image.

        Returns:
            Image: the current artwork image
        """
        return self._artwork_image

    @artwork_image.setter
    def artwork_image(self, image: ArtworkImage) -> None:
        """Set the current artwork image.

        Args:
            image (ArtworkImage): the new artwork image
        """
        if image == self._artwork_image:
            return

        self._artwork_image = image

        self.write_artwork_image()

        self.pending_ha_updates = {
            "artist": self.pending_ha_updates["artist"],
            "entity_picture": True,
            "media_title": self.pending_ha_updates["media_title"],
        }

    @property
    def brightness(self) -> float:
        """Gets the brightness of the display.

        Returns:
            float: the brightness of the display
        """
        return float(self.matrix.brightness)

    @brightness.setter
    def brightness(self, value: int) -> None:
        """Set the brightness of the display.

        Force updates all canvas content to apply the brightness.

        Args:
            value (int): the brightness of the display
        """
        self.matrix.brightness = value

        self.write_artwork_image()

        if not self.artist.scrollable:
            # It'll get written in <=0.5s anyway, so no need to write it again. This
            # also causes a brief overlap glitch on the matrix with scrolling text
            self.write_artist()

        if not self.media_title.scrollable:
            self.write_media_title()

        self.matrix.SwapOnVSync(self.canvas)

    @property
    def home_assistant_payload(self) -> HAPayloadInfo:
        """Creates the payload to send to Home Assistant for sensor updates.

        Returns:
            HAPayloadInfo: the payload to send to Home Assistant for sensor updates
        """

        if self.artwork_image == NULL_IMAGE:
            album = None
            album_artwork_url = None
        else:
            album = self.artwork_image.album
            album_artwork_url = self.artwork_image.url

        return {
            "state": any(
                [
                    self.artwork_image != NULL_IMAGE,
                    self.artist.display_content != "",
                    self.media_title.display_content != "",
                ]
            ),
            "media_title": self.media_title.original_content or None,
            "artist": self.artist.original_content or None,
            "album": album,
            "album_artwork_url": album_artwork_url,
        }

    @property
    def media_title(self) -> Text:
        """Returns the media title content."""
        return self._media_title

    @media_title.setter
    def media_title(self, value: str) -> None:
        """Set the media title content."""
        if not isinstance(value, str):
            raise TypeError(f"Value for `media_title` must be a string: {value!r}")

        if value == self.media_title.display_content:
            return

        LOGGER.debug("Setting media title to: %s", value)

        self.media_title.display_content = value
        self.write_media_title(clear_first=True, swap_on_vsync=True)

        self.pending_ha_updates = {
            "artist": self.pending_ha_updates["artist"],
            "entity_picture": self.pending_ha_updates["entity_picture"],
            "media_title": True,
        }

        if self.media_title.scrollable:
            LOGGER.debug("Sending request to start scroll thread from media_title setter")
            self._start_scroll_worker()

    @property
    def pending_ha_updates(self) -> HAPendingUpdatesInfo:
        """Returns a record of any pending attribute updates.

        Returns:
            HAPendingUpdatesInfo: a record of any pending attribute updates
        """
        return self._pending_ha_updates

    @pending_ha_updates.setter
    def pending_ha_updates(self, value: HAPendingUpdatesInfo) -> None:
        """Update the HA pending updates record and then send updates to HA.

        Args:
            value (dict): this must be the entire dict: updating a single value will not
                trigger this setter method and the updates won't be sent to HA
        """
        self._pending_ha_updates = value

        if any(self._pending_ha_updates.values()):
            self._start_update_ha_worker()

            if (
                not self.home_assistant_payload.get("media_title")
                and not self.home_assistant_payload.get("artist")
                and self.artwork_image == NULL_IMAGE
            ):
                LOGGER.info("No content found, clearing matrix")
                self.matrix.Clear()

    @property
    def ha_updates_available(self) -> bool:
        """Checks whether there are any pending HA updates.

        Returns:
            bool: True if any of the pending HA updates are True
        """
        return any(self.pending_ha_updates.values())

    @property
    def scrollable_content(self) -> bool:
        """Returns whether the display has any scrollable content.

        Returns:
            bool: True if there is scrollable content, False otherwise
        """
        return bool(self.media_title.scrollable or self.artist.scrollable)
