import sys, os, math
import svg
import cairo

def draw_with_cairo(cr, drawing):
    for d in drawing:
        if isinstance(d, svg.Path):
            for elt in d.items:
                if isinstance(elt, svg.MoveTo):
                    x,y = elt.dest.coord()
                    cr.move_to(x,y)
                elif isinstance(elt, svg.Line):
                    x,y = elt.end.coord()
                    cr.line_to(x,y)
                elif isinstance(elt, svg.Bezier):
                    if elt.dimension == 3:
                        a,c = elt.pts[1:]
                        b = c
                    else:
                        a,b,c = elt.pts[1:]
                    cr.curve_to(a.x, a.y, b.x, b.y, c.x, c.y)
        if isinstance(d, svg.Circle):
            cx, cy = d.center.coord()
            cr.move_to(cx+d.radius, cy)
            cr.arc(cx, cy, d.radius, 0, 2*math.pi)


def draw_with_segments(cr, drawing):
    for d in drawing:
        if isinstance(d, svg.Path):
            for l in d.segments(1):
                x,y = l[0].coord()
                cr.move_to(x,y)
                for pt in l[1:]:
                    x,y = pt.coord()
                    cr.line_to(x,y)
    #    elif isinstance(d, svg.Circle):
    #        a,b = d.bbox()
    #        draw.arc([int(x) for x in a.coord()+b.coord()],0,360,green)
        else:
            print("Unsupported SVG element")

f = svg.Svg(sys.argv[1])

a,b = f.bbox()

width, height = (a+b).coord()
surface = cairo.SVGSurface("test.svg", width, height)
cr = cairo.Context(surface)

cr.set_source_rgb(0,0,0)
cr.set_line_width(1)

draw_with_cairo(cr, f.flatten())

cr.stroke()

surface.write_to_png('test.png')
cr.show_page()
surface.finish()
