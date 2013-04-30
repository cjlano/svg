import sys, os
import svg
import Image, ImageDraw

p = svg.Path(sys.argv[1])
p.parse()

im = Image.new("RGB", (800,800), "white")
draw = ImageDraw.Draw(im)

red = (255,0,0)

for l in p.segments():
    draw.line(l, fill=red)

im.save(os.path.expanduser("~/public_html/bezier.png"))
#im.show()
