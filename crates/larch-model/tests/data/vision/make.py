# Makes the decoding fixtures of tests/vision.rs: 4 x 2 images, left half red, right half blue.
#   uv run --with pillow python make.py   (in this folder)
from PIL import Image

def img(mode="RGB", red=(255, 0, 0), blue=(0, 0, 255)):
    i = Image.new(mode, (4, 2), blue)
    for x in range(2):
        for y in range(2):
            i.putpixel((x, y), red)
    return i

rgba = img("RGBA", (255, 0, 0, 255), (0, 0, 255, 128))
rgba.save("rgba.png")
img().quantize(4).save("palette.png")
Image.new("I;16", (4, 2), 0x8000).save("grey16.png")
img().save("progressive.jpg", quality=95, subsampling=0, progressive=True)
img("CMYK", (0, 255, 255, 0), (255, 255, 0, 0)).save("cmyk.jpg", quality=95)
img().save("lossy.webp", quality=95)
img().save("lossless.webp", lossless=True)
other = Image.new("RGB", (4, 2), (0, 255, 0))
img().save("animated.gif", save_all=True, append_images=[other], duration=100, loop=0)
exif = Image.Exif()
exif[0x0112] = 6  # rotate 90 clockwise to show
img().save("exif6.jpg", quality=95, subsampling=0, exif=exif)
