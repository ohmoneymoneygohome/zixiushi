from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W, H = 1200, 630
NAVY = (17, 38, 79)
CREAM = (242, 234, 217)
PINK = (245, 143, 152)
CYAN = (0, 154, 214)

img = Image.new("RGB", (W, H), NAVY)
d = ImageDraw.Draw(img)

# faint diagonal wash, like a night ink field
for i in range(-H, W, 26):
    d.line([(i, H), (i + H, 0)], fill=(20, 44, 90), width=10)

han = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 110)
han_s = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 30)
latin = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 30)
latin_s = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
mono = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 20)

rabbit = Image.open(ROOT / "assets" / "rabbit.png").convert("RGBA")
rabbit.thumbnail((260, 260), Image.LANCZOS)
img.paste(rabbit, (78, 200), rabbit)

# pink rule
d.rectangle([400, 168, 470, 173], fill=PINK)

d.text((400, 190), "十友齋", font=han, fill=CREAM)
d.text((404, 320), "ZIXIU 自修室", font=han_s, fill=(242, 234, 217))

def track(draw, xy, text, font, fill, gap=6):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + gap

track(d, (404, 378), "EAST / WEST INK", latin, CYAN, 5)
d.text((404, 424), "21 ORIGINAL WORKS · 2026", font=latin_s, fill=(242, 234, 217))
d.text((404, 462), "ORIGINAL $1,000   ·   DIGITAL $9.90", font=mono, fill=PINK)

d.line([(78, 548), (400, 548)], fill=(242, 234, 217, 60), width=1)
d.text((78, 560), "zixiu.homes", font=mono, fill=(242, 234, 217))

out = ROOT / "assets" / "og.jpg"
img.save(out, quality=92)
img.save(ROOT / "preview" / "assets" / "og.jpg", quality=92)
print("saved", out, img.size)
