import sys, os
import svg
import Image, ImageDraw

f = open(sys.argv[1])
line = f.readline()

p = svg.Path()
p.parse(line)

im = Image.new("RGB", (800,800), "white")
draw = ImageDraw.Draw(im)

red = (255,0,0)
green = (0,255,0)

for l in p.segments(1):
    draw.line([(1*x).coord() for x in l], fill=red)
#for l in p.simplify(5):
#    draw.point([(1*x).coord() for x in l], fill=green)
draw.rectangle([pt.coord() for pt in p.bbox()], outline='blue')
im.save(os.path.expanduser("~/public_html/bezier.png"))
#im.show()

