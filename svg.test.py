import sys, os
import svg
import Image, ImageDraw

p = svg.Path(sys.argv[1])
p.parse()

im = Image.new("RGB", (800,800), "white")
draw = ImageDraw.Draw(im)

red = (255,0,0)
green = (0,255,0)

for l in p.segments():
    draw.line([(10*x).coord() for x in l], fill=red)
for l in p.simplify(0.5):
    draw.line([(10*x).coord() for x in l], fill=green)

#im.save(os.path.expanduser("~/public_html/bezier.png"))
im.show()

