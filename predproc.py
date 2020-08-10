from PIL import Image
import os

data = "photo"
dst = "resized1"

size=128
os.mkdir(dst)

for photo in os.listdir(data):

    img = Image.open(os.path.join(data, photo))
    img=img.resize((size,size))
    img.thumbnail((size,size))


    if img.mode == 'RGBA':
        img.load()
        background = Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        background.save(os.path.join(dst, photo.split('.')[0] + '.jpeg'), 'JPEG')
    elif img.mode=='RGB':
        img.convert('RGB')
        img.save(os.path.join(dst, photo.split('.')[0] + '.jpeg'), 'JPEG')

