from PIL import Image, ImageDraw, ImageFont
# A red disc on white: "a colour".
i = Image.new("RGB", (320, 240), "white"); d = ImageDraw.Draw(i)
d.ellipse((90, 50, 230, 190), fill=(220, 20, 20)); i.save("red.png")
# Three blue squares: "a count".
i = Image.new("RGB", (420, 200), "white"); d = ImageDraw.Draw(i)
for x in (40, 170, 300):
    d.rectangle((x, 60, x + 80, 140), fill=(30, 60, 220))
i.save("three.png")
# A large printed word.
i = Image.new("RGB", (440, 160), "white"); d = ImageDraw.Draw(i)
f = ImageFont.load_default(size=96)
d.text((30, 20), "TIGER", fill="black", font=f); i.save("word.png")
# 1000 x 600: 3 x 2 tiles and the overview; a green field with a yellow sun.
i = Image.new("RGB", (1000, 600), (120, 180, 250)); d = ImageDraw.Draw(i)
d.rectangle((0, 380, 1000, 600), fill=(40, 160, 60)); d.ellipse((760, 60, 900, 200), fill=(250, 220, 30))
i.save("wide.png")
# Transparency: a black cat-free square with a transparent hole (alpha dropped, as stb does).
i = Image.new("RGBA", (200, 200), (0, 128, 0, 255)); d = ImageDraw.Draw(i)
d.rectangle((50, 50, 150, 150), fill=(255, 0, 0, 0)); i.save("alpha.png")

# prompts-chatml.jsonl: the same conversations rendered by a plain ChatML template with no default
# system prompt (InternVL3.5's GGUF template), from prompts.jsonl (InternVL3's, which adds its system
# turn): the system turn removed, everything else byte-identical.
import json
SYSTEM = ("<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位"
          "联合开发的多模态大语言模型。<|im_end|>\n")
with open("prompts.jsonl", encoding="utf-8") as src, open("prompts-chatml.jsonl", "w", encoding="utf-8") as out:
    for line in src:
        j = json.loads(line)
        assert j["prompt"].startswith(SYSTEM)
        j["prompt"] = j["prompt"][len(SYSTEM):]
        out.write(json.dumps(j, ensure_ascii=False) + "\n")

# Decoders past PNG, for the m12 gate only: 10 prompts after the 100 above, in prompts-chatml.jsonl
# (m10's prompts.jsonl stays as it was calibrated). stb, llama.cpp's decoder, reads JPEG itself, so the
# gate measures larch's JPEG decoder against it. stb ignores EXIF orientation, reads no WebP and
# leaves a GIF's transparent pixels black, where larch shows the picture upright, reads WebP and
# keeps the palette colour (the documented differences): those have an `<image>.oracle.png` twin
# holding the pixels larch means to show, which llama.cpp reads while larch reads the image.
import random
from PIL import ImageOps
random.seed(11)
def noisy(img, amount):
    p = img.load()
    for y in range(img.height):
        for x in range(img.width):
            p[x, y] = tuple(max(0, min(255, c + random.randint(-amount, amount))) for c in p[x, y])
    return img
# 1000 x 600 photo-like, baseline 4:2:0: a red barn in a field (3 x 2 tiles and the overview), with a
# camera's soft grain. Grain of +-12 a pixel, unblurred, made llama.cpp's CPU and CUDA encoders disagree
# with each other (row cosine 0.972), which would loosen the embedding bound for every image.
from PIL import ImageFilter
i = Image.new("RGB", (1000, 600), (110, 170, 240)); d = ImageDraw.Draw(i)
d.rectangle((0, 360, 1000, 600), fill=(60, 150, 50))
d.rectangle((380, 240, 620, 420), fill=(190, 30, 25)); d.polygon([(360, 240), (500, 150), (640, 240)], fill=(150, 25, 20))
noisy(i, 6).filter(ImageFilter.GaussianBlur(1.2)).save("photo.jpg", quality=85)
# Progressive, a printed word.
i = Image.new("RGB", (640, 480), "white"); d = ImageDraw.Draw(i)
d.text((60, 160), "ZEBRA", fill="black", font=ImageFont.load_default(size=120)); i.save("progressive.jpg", quality=90, progressive=True)
# Stored on its side with EXIF orientation 6: shown upright, the word reads "UP".
up = Image.new("RGB", (360, 480), "white"); d = ImageDraw.Draw(up)
d.text((70, 60), "UP", fill=(20, 40, 200), font=ImageFont.load_default(size=150)); d.rectangle((40, 380, 320, 420), fill=(20, 40, 200))
exif = Image.Exif(); exif[0x0112] = 6
up.transpose(Image.Transpose.ROTATE_90).save("exif6.jpg", quality=90, exif=exif)
ImageOps.exif_transpose(Image.open("exif6.jpg")).convert("RGB").save("exif6.jpg.oracle.png")
# Lossless WebP: three green triangles.
i = Image.new("RGB", (400, 300), "white"); d = ImageDraw.Draw(i)
for x in (40, 160, 280):
    d.polygon([(x, 220), (x + 40, 100), (x + 80, 220)], fill=(30, 160, 60))
i.save("lossless.webp", lossless=True); Image.open("lossless.webp").convert("RGB").save("lossless.webp.oracle.png")
# A GIF whose transparent index is light blue under a yellow disc: stb would show black.
i = Image.new("P", (300, 300), 0); i.putpalette([200, 200, 255, 240, 200, 20] + [0, 0, 0] * 254)
ImageDraw.Draw(i).ellipse((70, 70, 230, 230), fill=1)
i.save("transparent.gif", transparency=0); Image.open("transparent.gif").convert("RGB").save("transparent.gif.oracle.png")
QA = [("photo.jpg", "What colour is the barn?", "The barn is red."),
      ("photo.jpg", "Describe the picture in one sentence.", "A red barn stands in a green field under a blue sky."),
      ("progressive.jpg", "What word is written?", "ZEBRA."),
      ("progressive.jpg", "How many letters does the word have?", "Five."),
      ("exif6.jpg.oracle.png", "What word is written?", "UP."),
      ("exif6.jpg.oracle.png", "What colour is the bar under the word?", "Blue."),
      ("lossless.webp.oracle.png", "How many triangles are there?", "There are three triangles."),
      ("lossless.webp.oracle.png", "What colour are the triangles?", "Green."),
      ("transparent.gif.oracle.png", "What colour is the disc?", "The disc is yellow."),
      ("transparent.gif.oracle.png", "What shape is shown?", "A disc.")]
with open("prompts-chatml.jsonl", "a", encoding="utf-8") as out:
    for f, q, a in QA:
        p = "<|im_start|>user\n<__media__>\n" + q + "<|im_end|>\n<|im_start|>assistant\n" + a + "<|im_end|>"
        out.write(json.dumps({"prompt": p, "images": [f]}, ensure_ascii=False) + "\n")
