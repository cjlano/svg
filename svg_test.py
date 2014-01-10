import sys, os, math
import cairo

import svg

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
        if isinstance(d, svg.Rect):
            x1,y1 = d.P1.coord()
            x2,y2 = d.P2.coord()
            width = x2 - x1
            height = y2 - y1
            cr.rectangle(x1, y1, width, height)


def draw_with_segments(cr, drawing):
    for d in drawing:
        if hasattr(d, 'segments'):
            for l in d.segments(1):
                x,y = l[0].coord()
                cr.move_to(x,y)
                for pt in l[1:]:
                    x,y = pt.coord()
                    cr.line_to(x,y)
        else:
            print("Unsupported SVG element")

f = svg.parse(sys.argv[1])

a,b = f.bbox()

width, height = (a+b).coord()
surface = cairo.SVGSurface("test.svg", width, height)
cr = cairo.Context(surface)

cr.set_source_rgb(0,0,0)
cr.set_line_width(1)

#draw_with_cairo(cr, f.flatten())
draw_with_segments(cr, f.flatten())

cr.stroke()

surface.write_to_png('test.png')
cr.show_page()
surface.finish()
