import Image

# im = Image.new("1", (100,100), "white")
# im.putpixel((10,10),1)
# im.putpixel((20,10),1)
# im.putpixel((30,30),1)
# 
# im.show()

def bezier1(p0, p1, t):
	x = p0[0] + t * (p1[0] - p0[0])
	y = p0[1] + t * (p1[1] - p0[1])
	return (x,y)

im = Image.new("1", (100,100), "white")

for t in range(0,10):
	xy = bezier1((0,0), (10,10), t*0.1)
	im.putpixel((int(round(xy[0])), int(round(xy[1]))), 1)
#	im.putpixel(xy,1)

im.show()
