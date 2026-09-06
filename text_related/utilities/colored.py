"""
colored.py -- a small, dependency-free toolkit for styling terminal text with ANSI escape codes.

Overview
========
This module provides a rich, composable way to add color and text effects
(bold, underline, italic, blink, reverse video, strikethrough, ...) to text
printed in an ANSI-capable terminal. Everything is built on top of standard
:class:`enum.Enum` classes, so colors and effects are discoverable,
type-checkable and self-documenting.

Building blocks
===============
Color enums
    A series of small enums group the raw escape sequences by purpose:

    * :class:`ForegroundStandardANSI` / :class:`BackgroundStandardANSI` --
      the 8 standard foreground/background colors.
    * :class:`ForegroundBrightStandardANSI` /
      :class:`BackgroundBrightStandardANSI` -- the bright variants.
    * ``*RgbANSI`` / ``*HexANSI`` / ``*HslANSI`` -- truecolor (24-bit) samples.

    These are merged into :data:`StandardANSI` and finally into the single
    combined :data:`Palette` enum used throughout the module.

:class:`Effects`
    ANSI SGR style codes (bold, dim, italic, underline, blink, reverse,
    hidden, strikethrough, double underline, framed, encircled, overline) and
    their matching ``*_OFF`` reset codes. Note that terminal support for some
    of these varies.

:data:`Palette`
    The one-stop combined enum containing every color and effect, plus
    ``ENDC``/``RESET`` (``\\033[0m``). It is extended at runtime with two helper
    methods:

    * ``Palette.<MEMBER>.colorize(text)`` -- wrap ``text`` in the member's
      escape code and a trailing reset.
    * ``Palette.register(name, code)`` -- register a brand new color/effect on
      the enum at runtime (see :func:`palette_register`).

:class:`Styleit`
    A small helper implementing the ``style >> text`` operator syntax for
    one-shot styling.

:class:`cprint`
    A drop-in replacement for the built-in :func:`print` that supports a fluent
    ``.style(...)`` call for applying colors and effects.

Quick start
===========
>>> Palette.GREEN.colorize("hi") == "\\033[32mhi\\033[0m"
True
>>> (Styleit(Palette.RED, B=True) >> "x") == "\\033[1m\\033[31mx\\033[0m"
True

Running this file
=================
Execute the module directly to either run the doctests or print a set of
styled demonstration lines::

    python colored.py                  # demo (default)
    python colored.py --mode doctest   # run the embedded doctests
    python colored.py --mode demo      # print styled sample output (default)
"""

import enum
from typing import Optional


class ForegroundStandardANSI(enum.Enum):
    """
    Foreground standard ANSI color codes for terminal text formatting.
    """
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"


class BackgroundStandardANSI(enum.Enum):
    """
    Background standard ANSI color codes for terminal text formatting.
    """
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class ForegroundBrightStandardANSI(enum.Enum):
    """
    Foreground bright standard ANSI color codes for terminal text formatting.
    """
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class BackgroundBrightStandardANSI(enum.Enum):
    """
    Background bright standard ANSI color codes for terminal text formatting.
    """
    BG_BRIGHT_BLACK = "\033[100m"
    BG_BRIGHT_RED = "\033[101m"
    BG_BRIGHT_GREEN = "\033[102m"
    BG_BRIGHT_YELLOW = "\033[103m"
    BG_BRIGHT_BLUE = "\033[104m"
    BG_BRIGHT_MAGENTA = "\033[105m"
    BG_BRIGHT_CYAN = "\033[106m"
    BG_BRIGHT_WHITE = "\033[107m"

combined_standard_members = [(m.name, m.value) for m in ForegroundStandardANSI] + [(m.name, m.value) for m in BackgroundStandardANSI] + [(m.name, m.value) for m in ForegroundBrightStandardANSI] + [(m.name, m.value) for m in BackgroundBrightStandardANSI]

StandardANSI = enum.Enum('StandardANSI', combined_standard_members)
StandardANSI.__doc__ = "Standard ANSI color codes for terminal text formatting."


class ForegroundRgbANSI(enum.Enum):
    RED_RGB = "\033[38;2;255;0;0m"
    ORANGE_RGB = "\033[38;2;255;165;0m"


class BackgroundRgbANSI(enum.Enum):
    BG_RED_RGB = "\033[48;2;255;0;0m"
    BG_ORANGE_RGB = "\033[48;2;255;165;0m"


class ForegroundHexANSI(enum.Enum):
    RED_HEX = "\033[38;2;255;0;0m"
    ORANGE_HEX = "\033[38;2;255;165;0m"


class BackgroundHexANSI(enum.Enum):
    BG_RED_HEX = "\033[48;2;255;0;0m"
    BG_ORANGE_HEX = "\033[48;2;255;165;0m"


class ForegroundHslANSI(enum.Enum):
    RED_HSL = "\033[38;2;255;0;0m"
    ORANGE_HSL = "\033[38;2;255;165;0m"


class BackgroundHslANSI(enum.Enum):
    BG_RED_HSL = "\033[48;2;255;0;0m"
    BG_ORANGE_HSL = "\033[48;2;255;165;0m"


class Effects(enum.Enum):
    """
    ANSI text style/effect SGR codes.

    Note: support varies by terminal. ``BLINK``, ``RAPID_BLINK``,
    ``STRIKETHROUGH`` and ``FRAMED``/``ENCIRCLED``/``OVERLINE`` are not
    honoured everywhere. Each effect has a matching ``*_OFF`` reset code so an
    effect can be turned off without resetting everything.
    """
    # Enabling codes
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    RAPID_BLINK = "\033[6m"
    REVERSE = "\033[7m"          # swap fg/bg (inverse video)
    HIDDEN = "\033[8m"           # concealed
    STRIKETHROUGH = "\033[9m"
    DOUBLE_UNDERLINE = "\033[21m"
    FRAMED = "\033[51m"
    ENCIRCLED = "\033[52m"
    OVERLINE = "\033[53m"

    # Disabling / reset codes
    BOLD_OFF = "\033[22m"        # also turns off DIM
    DIM_OFF = "\033[22m"
    ITALIC_OFF = "\033[23m"
    UNDERLINE_OFF = "\033[24m"
    BLINK_OFF = "\033[25m"
    REVERSE_OFF = "\033[27m"
    HIDDEN_OFF = "\033[28m"
    STRIKETHROUGH_OFF = "\033[29m"
    FRAMED_OFF = "\033[54m"      # also turns off ENCIRCLED
    OVERLINE_OFF = "\033[55m"


combined_palette_members = [(m.name, m.value) for m in StandardANSI] + [(m.name, m.value) for m in ForegroundRgbANSI] + [(m.name, m.value) for m in BackgroundRgbANSI] + [(m.name, m.value) for m in ForegroundHexANSI] + [(m.name, m.value) for m in BackgroundHexANSI] + [(m.name, m.value) for m in ForegroundHslANSI] + [(m.name, m.value) for m in BackgroundHslANSI] + [(m.name, m.value) for m in Effects]
combined_palette_members.append(("ENDC", "\033[0m"))
combined_palette_members.append(("RESET", "\033[0m"))
Palette = enum.Enum('Palette', combined_palette_members)
Palette.__doc__ = "Combined ANSI color codes for terminal text formatting."
def palette_colorize(self, text: str,) -> str:
    """
    Colorize the given text with the color represented by this enum member.

    The returned string wraps ``text`` in this member's escape code and a
    trailing reset (``ENDC``), so styles can be safely nested.

    >>> Palette.GREEN.colorize("This text will be green") == "\\033[32mThis text will be green\\033[0m"
    True

    >>> Palette.RED.colorize("oops") == "\\033[31moops\\033[0m"
    True

    >>> Palette.GREEN.colorize(Palette.BOLD.colorize("nested"))
    '\\x1b[32m\\x1b[1mnested\\x1b[0m\\x1b[0m'
    """
    return f"{self.value}{text}{Palette.ENDC.value}"
def palette_register(cls, name: str, color_code: str):
    """
    Register a new color or style with the given name and color code.

    A genuine enum member is created (or an alias is added when the code
    already exists), so the new name behaves exactly like the built-in ones.

    >>> Palette.register("PINK_DEMO", "\\033[38;2;255;105;180m")  # doctest: +ELLIPSIS
    <Palette.PINK_DEMO: ...>
    >>> Palette.PINK_DEMO.colorize("pink") == "\\033[38;2;255;105;180mpink\\033[0m"
    True
    >>> Palette.register("PINK_DEMO", "\\033[0m")
    Traceback (most recent call last):
        ...
    ValueError: 'PINK_DEMO' is already registered in Palette
    """
    if name in cls._member_map_:
        raise ValueError(f"{name!r} is already registered in {cls.__name__}")
    if color_code in cls._value2member_map_:
        # Reuse the existing member for this value, just add an alias name.
        member = cls._value2member_map_[color_code]
    else:
        # Build a genuine enum member so it behaves like the others.
        member = object.__new__(cls)
        member._name_ = name
        member._value_ = color_code
        cls._value2member_map_[color_code] = member

    # Bypass EnumType.__setattr__, which forbids adding/altering members.
    type.__setattr__(cls, name, member)
    cls._member_map_[name] = member
    return member

setattr(Palette, "register", classmethod(palette_register))
setattr(Palette, "colorize", palette_colorize)


class Styleit:
    """
    A class for styling text with ANSI escape codes using the ``>>`` operator.

    Construct with a foreground color/effect (a :class:`Palette` or
    :class:`Effects` member, or a raw escape string) plus optional background
    and effect flags, then apply it to text with ``style >> text``.

    >>> (Styleit(Palette.GREEN, B=True, U=True) >> "ankit") == "\\033[1m\\033[4m\\033[32mankit\\033[0m"
    True
    """

    def __init__(self, style_code: str | Palette, bg_color: Optional[str | Palette] = None, B: bool = False, U: bool = False, I: bool = False, dim: bool = False, blink: bool = False, reverse: bool = False, hidden: bool = False, strikethrough: bool = False, effects: Optional[list] = None):
        if isinstance(style_code, (Palette, Effects)):
            self.value = style_code.value
        else:
            self.value = style_code
        
        if bg_color is None:
            self.bg_color = ""
        else:
            if isinstance(bg_color, (Palette, Effects)):
                self.bg_color = bg_color.value
            else:
                self.bg_color = bg_color
        
        flag_map = [
            (B, Effects.BOLD),
            (dim, Effects.DIM),
            (I, Effects.ITALIC),
            (U, Effects.UNDERLINE),
            (blink, Effects.BLINK),
            (reverse, Effects.REVERSE),
            (hidden, Effects.HIDDEN),
            (strikethrough, Effects.STRIKETHROUGH),
        ]
        self.effects = "".join(effect.value for enabled, effect in flag_map if enabled)
        for extra in effects or []:
            if isinstance(extra, (Palette, Effects)):
                self.effects += extra.value
            elif isinstance(extra, str) and extra in Effects._member_map_:
                self.effects += Effects[extra].value
            else:
                self.effects += str(extra)

    def __rshift__(self, text: str) -> str:
        """
        Apply the style to the given text and return the styled string.

        >>> (Styleit(Palette.BOLD) >> "This text will be bold") == "\\033[1mThis text will be bold\\033[0m"
        True

        >>> (Styleit(Palette.UNDERLINE) >> "u") == "\\033[4mu\\033[0m"
        True
        """
        return f"{self.effects}{self.value}{self.bg_color}{text}{Palette.ENDC.value}"


class cprint:
    """
    A drop-in replacement for ``print`` that supports fluent ANSI styling.

    Call it exactly like ``print`` to output plain text::

        cprint("hello", "world")

    Or chain ``.style(...)`` to apply ANSI codes before printing::

        cprint("text").style(B=True, U=True, I=False, bg_color=Palette.BG_RED)
        cprint("warn").style(color=Palette.YELLOW, B=True)

    ``.style`` accepts:
        color     -- foreground color (str or Palette member)
        bg_color  -- background color (str or Palette member)
        B         -- bold
        U         -- underline
        I         -- italic

    All the usual ``print`` keyword arguments (``sep``, ``end``, ``file``,
    ``flush``) are supported and remembered when styling.

    >>> import io
    >>> buf = io.StringIO()
    >>> cprint("hi", file=buf).style(B=True, color=Palette.RED)  # doctest: +ELLIPSIS
    <...cprint object at ...>
    >>> buf.getvalue() == "\\033[1m\\033[31mhi\\033[0m\\n"
    True
    """

    def __init__(self, *values, sep: str = " ", end: str = "\n", file=None, flush: bool = False):
        self._sep = sep
        self._end = end
        self._file = file
        self._flush = flush
        self._text = sep.join(str(v) for v in values)
        # Printing is deferred: if ``.style(...)`` is chained the styled text is
        # printed instead. Otherwise the plain text is flushed when this
        # temporary object is discarded (see ``__del__``), so ``cprint`` still
        # behaves like ``print`` when used on its own.
        self._printed = False

    def __del__(self):
        if not self._printed:
            self._emit(self._text)

    def _emit(self, text: str) -> None:
        print(text, end=self._end, file=self._file, flush=self._flush)
        self._printed = True

    @staticmethod
    def _code(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (Palette, Effects)):
            return value.value
        if isinstance(value, str) and value in Palette._member_map_:
            return Palette[value].value
        if isinstance(value, str) and value in Effects._member_map_:
            return Effects[value].value
        return str(value)

    def style(
        self,
        color: Optional[str | Palette] = None,
        bg_color: Optional[str | Palette] = None,
        B: bool = False,
        U: bool = False,
        I: bool = False,
        dim: bool = False,
        blink: bool = False,
        rapid_blink: bool = False,
        reverse: bool = False,
        hidden: bool = False,
        strikethrough: bool = False,
        double_underline: bool = False,
        overline: bool = False,
        framed: bool = False,
        encircled: bool = False,
        effects: Optional[list] = None,
    ) -> "cprint":
        """
        Apply ANSI styling to the previously supplied text and (re)print it.

        Boolean flags toggle common effects; ``effects`` accepts any extra
        ``Effects`` members (or their names/codes) for full control.

        >>> import io
        >>> buf = io.StringIO()
        >>> _ = cprint("x", end="", file=buf).style(bg_color=Palette.BG_RED, blink=True)
        >>> buf.getvalue() == "\\033[5m\\033[41mx\\033[0m"
        True
        """
        flag_map = [
            (B, Effects.BOLD),
            (dim, Effects.DIM),
            (I, Effects.ITALIC),
            (U, Effects.UNDERLINE),
            (double_underline, Effects.DOUBLE_UNDERLINE),
            (blink, Effects.BLINK),
            (rapid_blink, Effects.RAPID_BLINK),
            (reverse, Effects.REVERSE),
            (hidden, Effects.HIDDEN),
            (strikethrough, Effects.STRIKETHROUGH),
            (overline, Effects.OVERLINE),
            (framed, Effects.FRAMED),
            (encircled, Effects.ENCIRCLED),
        ]
        codes = "".join(effect.value for enabled, effect in flag_map if enabled)

        for extra in effects or []:
            codes += self._code(extra)

        fg = self._code(color)
        bg = self._code(bg_color)

        styled = f"{codes}{fg}{bg}{self._text}{Palette.ENDC.value}"
        self._emit(styled)
        return self

# Entry point for testing and demonstration
def _run_demo() -> None:
    """Print a set of styled sample lines demonstrating the module's features."""
    # Basic colorize helpers
    print(Palette.GREEN.colorize("This text will be green"))
    print(Palette.BG_YELLOW.colorize(Palette.RED.colorize("This text will have a yellow background")))

    # Using the Styleit class
    print(Styleit(Palette.RED, B=True, U=True) >> "This text will be bold, underlined, and red")
    print(Styleit(Palette.BLUE, I=True) >> "This text will be italic and blue")
    print(Styleit(Palette.BRIGHT_GREEN, bg_color=Palette.BG_BRIGHT_YELLOW) >> "This text will be bright green with a bright yellow background")

    # Registering a brand new color at runtime
    Palette.register("CUSTOM_COLOR", "\033[38;2;123;45;67m")
    print(Palette.CUSTOM_COLOR)
    print(Palette.CUSTOM_COLOR.colorize("This text will be in a custom RGB color"))

    # Using cprint as a styled replacement for print
    cprint("Plain text, printed just like print()")
    cprint("Bold + underlined on red background").style(B=True, U=True, I=False, bg_color=Palette.BG_RED)
    cprint("Italic yellow foreground").style(color=Palette.YELLOW, I=True)
    cprint("Custom color via name").style(color="CUSTOM_COLOR", B=True)

    # Text effects (support varies by terminal)
    cprint("Blinking red text").style(color=Palette.RED, blink=True)
    cprint("Reverse video").style(color=Palette.GREEN, reverse=True)
    cprint("Struck through").style(strikethrough=True)
    cprint("Dim + italic").style(dim=True, I=True)
    cprint("Overline + double underline").style(overline=True, double_underline=True)
    cprint("Mixed effects").style(color=Palette.CYAN, effects=[Effects.BOLD, Effects.UNDERLINE, Effects.BLINK])


def _run_doctests(verbose: bool = False) -> int:
    """Run the module's doctests and return the number of failures."""
    import doctest
    results = doctest.testmod(verbose=verbose)
    print(f"doctests: {results.attempted} run, {results.failed} failed")
    return results.failed


def main(argv: Optional[list] = None) -> int:
    """Parse command line arguments and dispatch to demo or doctest mode."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate or test the colored.py ANSI styling toolkit.",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=("demo", "doctest"),
        default="demo",
        help="'demo' prints styled sample output (default); 'doctest' runs the embedded doctests.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (only affects doctest mode).",
    )
    args = parser.parse_args(argv)

    if args.mode == "doctest":
        return _run_doctests(verbose=args.verbose)

    _run_demo()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())