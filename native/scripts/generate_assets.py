#!/usr/bin/env python3
"""Generate UNIT iOS app icon and splash screen assets."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "ios/App/App/Assets.xcassets/AppIcon.appiconset"
SPLASH_DIR = ROOT / "ios/App/App/Assets.xcassets/Splash.imageset"

BG = "#0a0a0a"
WHITE = "#ffffff"
BLUE = "#0088ff"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/SFNSDisplay-Bold.otf",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_logo(draw: ImageDraw.ImageDraw, cx: int, cy: int, unit_size: int) -> None:
    font = _font(unit_size)
    unit = "UNIT"
    dot = "."
    unit_w = draw.textlength(unit, font=font)
    dot_w = draw.textlength(dot, font=font)
    total_w = unit_w + dot_w
    x = cx - total_w / 2
    y = cy - unit_size * 0.55
    draw.text((x, y), unit, fill=WHITE, font=font)
    draw.text((x + unit_w, y), dot, fill=BLUE, font=font)


def make_icon(size: int = 1024) -> Image.Image:
    img = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(img)
    draw_logo(draw, size // 2, size // 2, int(size * 0.22))
    return img


def make_splash(size: int = 2732) -> Image.Image:
    img = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(img)
    draw_logo(draw, size // 2, size // 2, int(size * 0.12))
    return img


def main() -> None:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    SPLASH_DIR.mkdir(parents=True, exist_ok=True)

    icon = make_icon(1024)
    icon_path = ICON_DIR / "AppIcon-512@2x.png"
    icon.save(icon_path, "PNG")
    print(f"Wrote {icon_path}")

    splash = make_splash(2732)
    for name in (
        "splash-2732x2732.png",
        "splash-2732x2732-1.png",
        "splash-2732x2732-2.png",
    ):
        path = SPLASH_DIR / name
        splash.save(path, "PNG")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
