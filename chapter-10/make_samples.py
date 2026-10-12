"""Regenerates the sample inputs in samples/ (already included in the repository).

    python make_samples.py

belt.jpg      a drawn illustration of a worn AHU fan belt, standing in for a technician's photo
job_card.jpg  a filled-in job card in handwriting style, with one smudged field
voice_note.wav  a short English voice note made with Coqui TTS

Replace them with your own photographs and recordings as soon as you can: synthetic samples
are fine for checking the plumbing and poor for judging accuracy (Section 10.5).
The job card uses the Kalam handwriting font (SIL Open Font Licence) if you download
Kalam-Regular.ttf from Google Fonts and save it here as kalam.ttf; otherwise a plain font.
"""
import os, random
from PIL import Image, ImageDraw, ImageFilter, ImageFont
random.seed(4)
os.makedirs("samples", exist_ok=True)

def font(size, hand=False):
    for f in (["kalam.ttf"] if hand else []) + ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()

# ---- belt.jpg: two pulleys and a cracked, glazed V-belt on a dim machine-room background
W, H = 1024, 768
img = Image.new("RGB", (W, H), (58, 60, 62))
d = ImageDraw.Draw(img)
for _ in range(6000):                                     # concrete-like speckle
    x, y = random.randrange(W), random.randrange(H)
    g = random.randint(45, 80); d.point((x, y), fill=(g, g, g + 2))
d.rectangle([0, 600, W, H], fill=(70, 72, 70))            # plinth
big, small = (330, 380, 175), (760, 330, 95)              # pulleys: centre x, y, radius
for cx, cy, r in (big, small):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(105, 108, 112), outline=(30, 30, 30), width=6)
    d.ellipse([cx - r * 0.35, cy - r * 0.35, cx + r * 0.35, cy + r * 0.35], fill=(80, 82, 85))
    d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(40, 40, 40))
belt = (24, 24, 26)
for off in (0, 26):                                       # two belts of a matched set
    d.line([(big[0], big[1] - big[2] + off), (small[0], small[1] - small[2] + off)], fill=belt, width=22)
    d.line([(big[0], big[1] + big[2] - off), (small[0], small[1] + small[2] - off)], fill=belt, width=22)
    d.arc([big[0] - big[2] + off, big[1] - big[2] + off, big[0] + big[2] - off, big[1] + big[2] - off], 90, 270, fill=belt, width=22)
    d.arc([small[0] - small[2] + off, small[1] - small[2] + off, small[0] + small[2] - off, small[1] + small[2] - off], 270, 90, fill=belt, width=22)
for i in range(14):                                       # cracks across the lower belt run
    t = 0.15 + i * 0.05
    x = big[0] + t * (small[0] - big[0]); y = big[1] + big[2] + t * (small[1] + small[2] - big[1] - big[2])
    d.line([(x, y - 9), (x + random.randint(-3, 3), y + 9)], fill=(150, 140, 125), width=2)
d.line([(big[0] + 40, big[1] - big[2] - 6), (big[0] + 260, big[1] - big[2] + 8)], fill=(110, 110, 115), width=4)   # glazing shine
for i in range(40):                                       # rubber dust under the drive
    x, y = random.randint(380, 700), random.randint(610, 650)
    d.ellipse([x, y, x + 4, y + 3], fill=(20, 20, 20))
d.text((40, 30), "AHU-3  BLOCK B ROOF", fill=(200, 190, 60), font=font(28))
img = img.filter(ImageFilter.GaussianBlur(1.2))
img.save("samples/belt.jpg", quality=85)

# ---- job_card.jpg: printed form, handwritten entries, one field smudged
W, H = 1100, 1400
card = Image.new("RGB", (W, H), (246, 242, 230))
d = ImageDraw.Draw(card)
d.rectangle([40, 40, W - 40, 150], outline=(30, 50, 100), width=3)
d.text((60, 60), "FIELD SERVICE JOB CARD", fill=(30, 50, 100), font=font(44))
d.text((60, 112), "Bengaluru Operations    No. 04417", fill=(30, 50, 100), font=font(26))
fields = [("Date", "11/10/2026"), ("Site", "Prestige Tech Park, Block B"), ("Equipment", "AHU-3 (roof)"),
          ("Fault reported", "Belt squeal at start-up"), ("Work done", "Replaced both belts, set tension"),
          ("Parts used", "2 x B52 V-belt (AB-B52-MS)"), ("Time in / out", "10:40 / 12:15"),
          ("Technician", "Mahesh R."), ("Customer signature", "")]
y = 190
for label, value in fields:
    d.text((60, y), label + ":", fill=(30, 50, 100), font=font(26))
    d.line([(330, y + 40), (W - 60, y + 40)], fill=(120, 130, 160), width=2)
    if value:
        d.text((340, y - 6), value, fill=(25, 35, 120), font=font(40, hand=True))
    y += 120
smudge = Image.new("L", (W, H), 0)                        # a smudge across 'Time in / out'
ImageDraw.Draw(smudge).ellipse([300, 190 + 6 * 120 - 25, 820, 190 + 6 * 120 + 70], fill=250)
card = Image.composite(Image.new("RGB", (W, H), (150, 140, 120)), card, smudge.filter(ImageFilter.GaussianBlur(25)))
d = ImageDraw.Draw(card)
d.line([(620, 190 + 8 * 120 + 30), (700, 190 + 8 * 120 - 5), (760, 190 + 8 * 120 + 35), (840, 190 + 8 * 120)], fill=(25, 35, 120), width=3)
card = card.rotate(1.5, expand=False, fillcolor=(40, 40, 40)).filter(ImageFilter.GaussianBlur(0.8))
card.save("samples/job_card.jpg", quality=80)

# ---- voice_note.wav
try:
    from TTS.api import TTS
    tts = TTS("tts_models/en/vctk/vits")
    speaker = next(s for s in tts.speakers if s.strip() == "p251")   # a male Indian English voice
    tts.tts_to_file(text="Hi, Mahesh here, on the roof of Block B at Prestige Tech Park. AHU three is "
                         "squealing every time it starts, and the belts look cracked and shiny. "
                         "I've sent a photo. Can I replace just the worst belt, and what tension should I set?",
                    speaker=speaker, file_path="samples/voice_note.wav")
except Exception as e:
    print("Voice note skipped:", e)
print("Wrote", sorted(os.listdir("samples")))
