import os
import Image, ImageDraw

def bezier1(p0, p1, t):
    x = p0[0] + t * (p1[0] - p0[0])
    y = p0[1] + t * (p1[1] - p0[1])
    return (x,y)

def bezierN(pts, t):
    res = list(pts)
    for n in range(len(pts), 1, -1):
        for i in range(0,n-1):
            res[i] = bezier1(res[i], res[i+1], t)
    return res[0]

im = Image.new("RGB", (100,100), "white")
draw = ImageDraw.Draw(im)

red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

pts = [(10,90),(5,40),(60,40),(90,90)]
for pt in pts:
    im.putpixel(pt, red)
draw.line(pts,fill=red)

bezier = []
for t in range(0,10):
    xy = bezierN(pts, t*0.1)
    bezier.append(xy)
    im.putpixel((int(round(xy[0])), int(round(xy[1]))), green)

draw.line(bezier, fill=blue)
#im.save(os.path.expanduser("~/public_html/bezier.png"))
im.show()
