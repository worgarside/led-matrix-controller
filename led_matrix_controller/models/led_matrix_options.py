"""Typing info for the matrix options."""

from __future__ import annotations

from typing import Literal, TypedDict


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
