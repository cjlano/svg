import sys, os
import svg
import Image, ImageDraw

f = svg.Svg(sys.argv[1])

im = Image.new("RGB", (800,800), "white")
draw = ImageDraw.Draw(im)

red = (255,0,0)
green = (0,255,0)

for l in f.segments(1):
    draw.line([(1*x).coord() for x in l], fill=red)
#for l in p.simplify(5):
#    draw.point([(1*x).coord() for x in l], fill=green)
draw.rectangle([pt.coord() for pt in f.bbox()], outline='blue')
#im.save(os.path.expanduser("~/public_html/bezier.png"))
im.show()

