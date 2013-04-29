import sys
import svg
import Image, ImageDraw

p = svg.Path(sys.argv[1])
p.parse()

im = Image.new("RGB", (1000,1000), "white")
draw = ImageDraw.Draw(im)

red = (255,0,0)

for l in p.segments():
    draw.line(l, fill=red)

im.show()
