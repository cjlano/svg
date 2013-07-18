# SVG parser in Python

# Copyright (C) 2013 -- CJlano < cjlano @ free.fr >

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import re
import numbers, math
import xml.etree.ElementTree as etree

class Transformable:
    '''Abstract class for objects that can be geometrically drawn & transformed'''
    def __init__(self, elt=None):
        # a 'Transformable' is represented as a list of Transformable items
        self.items = []
        # Unit transformation matrix on init
        self.matrix = Matrix()
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        if elt is not None:
            # Parse transform attibute to update self.matrix
            self.getTransformations(elt)

    def bbox(self):
        '''Bounding box'''
        for x in self.items:
            pmin, pmax = x.bbox()
            if self.xmin == None or pmin.x < self.xmin:
                self.xmin = pmin.x
            if self.ymin == None or pmin.y < self.ymin:
                self.ymin = pmin.y
            if self.xmax == None or pmax.x > self.xmax:
                self.xmax = pmax.x
            if self.ymax == None or pmax.y > self.ymax:
                self.ymax = pmax.y

        return (Point(self.xmin,self.ymin), Point(self.xmax,self.ymax))

    # Parse transform field
    def getTransformations(self, elt):
        t = elt.get('transform')
        if t is None: return

        svg_transforms = [
                'matrix', 'translate', 'scale', 'rotate', 'skewX', 'skewY']

        # match any SVG transformation with its parameter (until final parenthese)
        # [^)]*    == anything but a closing parenthese
        # '|'.join == OR-list of SVG transformations
        transforms = re.findall(
                '|'.join([x + '[^)]*\)' for x in svg_transforms]), t)

        for t in transforms:
            op, arg = t.split('(')
            op = op.strip()
            # Keep only numbers
            arg = [float(x) for x in
                    re.findall(r"([+-]?\ *\d+(?:\.\d*)?|\.\d+)", arg)]
            print('transform: ' + op + ' '+ str(arg))

            if op == 'matrix':
                self.matrix *= Matrix(arg)

            if op == 'translate':
                tx, ty = arg
                self.matrix *= Matrix([1, 0, 0, 1, tx, ty])

            if op == 'scale':
                sx = arg[0]
                if len(arg) == 1: sy = sx
                else: sy = arg[1]
                self.matrix *= Matrix([sx, 0, 0, sy, 0, 0])

            if op == 'rotate':
                cosa = math.cos(math.radians(arg[0]))
                sina = math.sin(math.radians(arg[0]))
                if len(arg) != 1:
                    tx, ty = arg[1:3]
                    self.matrix *= Matrix([1, 0, 0, 1, tx, ty])
                self.matrix *= Matrix([cosa, sina, -sina, cosa, 0, 0])
                if len(arg) != 1:
                    self.matrix *= Matrix([1, 0, 0, 1, -tx, -ty])

            if op == 'skewX':
                tana = math.tan(math.radians(arg[0]))
                self.matrix *= Matrix([1, 0, tana, 1, 0, 0])

            if op == 'skewY':
                tana = math.tan(math.radians(arg[0]))
                self.matrix *= Matrix([1, tana, 0, 1, 0, 0])

    def transform(self, matrix=None):
        for x in self.items:
            x.transform(self.matrix)

    def scale(self, ratio):
        for x in self.items:
            x.scale(ratio)
        return self

    def translate(self, offset):
        for x in self.items:
            x.translate(offset)
        return self

    def rotate(self, angle):
        for x in self.items:
            x.rotate(angle)
        return self

class Svg(Transformable):
    '''SVG class: use parse to parse a file'''
    def __init__(self, filename=None):
        Transformable.__init__(self)
        if filename:
            self.parse(filename)

    def parse(self, filename):
        self.filename = filename
        tree = etree.parse(filename)
        self.root = tree.getroot()
        if self.root.tag[-3:] != 'svg':
            raise TypeError('file %s does not seem to be a valid SVG file', filename)
        self.ns = self.root.tag[:-3]

        # Parse XML elements hierarchically with groups <g>
        self.addGroup(self.items, self.root)

        self.transform()

       # Flatten XML tree into a one dimension list
        self.flatten()

    def addGroup(self, group, element):
        for elt in element:
            if elt.tag == self.ns + 'g':
                g = Group(elt)
                # Append to parent group before looking for child elements
                # because Group.append() applies transformations
                # We need to record transformation to propagate to children
                group.append(g)
                self.addGroup(g, elt)
            elif elt.tag == self.ns + 'path':
                group.append(Path(elt))
            elif elt.tag == self.ns + 'circle':
                group.append(Circle(elt))
            else:
                print('Unsupported element: ' + elt.tag)
                #group.append(elt.tag[len(self.ns):])


    def flatten(self):
        self.drawing = []
        for i in self.items:
            if isinstance(i, Group):
                self.drawing += i.flatten()
            else:
                self.drawing.append(i)

    def title(self):
        t = self.root.find(self.ns + 'title')
        if t is not None:
            return t
        else:
            return self.filename.split('.')[0]


class Group(Transformable):
    '''Handle svg <g> elements'''
    def __init__(self, elt=None):
        Transformable.__init__(self, elt)
        if elt is not None:
            self.ident = elt.get('id')

    def append(self, item):
        item.matrix = self.matrix * item.matrix
        self.items.append(item)

    def __repr__(self):
        return 'Group id ' + self.ident + ':\n' + repr(self.items) + '\n'

    def flatten(self):
        ret = []
        for i in self.items:
            if isinstance(i, Group):
                ret += i.flatten()
            else:
                ret.append(i)
        return ret

class Matrix:
    ''' SVG transformation matrix and its operations
    a SVG matrix is represented as a list of 6 values [a, b, c, d, e, f]
    (named vect hereafter) which represent the 3x3 matrix
    ((a, c, e)
     (b, d, f)
     (0, 0, 1))
    see http://www.w3.org/TR/SVG/coords.html#EstablishingANewUserSpace '''

    def __init__(self, vect=[1, 0, 0, 1, 0, 0]):
        # Unit transformation vect by default
        if len(vect) != 6:
            raise ValueError("Bad vect size %d" % len(vect))
        self.vect = list(vect)

    def __mul__(self, other):
        '''Matrix multiplication'''
        if isinstance(other, Matrix):
            a = self.vect[0] * other.vect[0] + self.vect[2] * other.vect[1]
            b = self.vect[1] * other.vect[0] + self.vect[3] * other.vect[1]
            c = self.vect[0] * other.vect[2] + self.vect[2] * other.vect[3]
            d = self.vect[1] * other.vect[2] + self.vect[3] * other.vect[3]
            e = self.vect[0] * other.vect[4] + self.vect[2] * other.vect[5] \
                    + self.vect[4]
            f = self.vect[1] * other.vect[4] + self.vect[3] * other.vect[5] \
                    + self.vect[5]
            return Matrix([a, b, c, d, e, f])

        elif isinstance(other, Point):
            x = other.x * self.vect[0] + other.y * self.vect[2] + self.vect[4]
            y = other.x * self.vect[1] + other.y * self.vect[3] + self.vect[5]
            return Point(x,y)

        else:
            return NotImplemented

    def __str__(self):
        return str(self.vect)


COMMANDS = 'MmZzLlHhVvCcSsQqTtAa'

class Path(Transformable):
    '''SVG <path>'''

    def __init__(self, elt=None):
        Transformable.__init__(self, elt)
        if elt is not None:
            self.ident = elt.get('id')
            self.style = elt.get('style')
            self.parse(elt.get('d'))

    def parse(self, pathstr):
        """Parse path string and build elements list"""

        # (?:...) : non-capturing version of regular parentheses
        pathlst = re.findall(r"([+-]?\ *\d+(?:\.\d*)?|\.\d+|\ *[%s]\ *)"
                % COMMANDS, pathstr)

        pathlst.reverse()

        command = None
        current_pt = Point(0,0)
        start_pt = None

        while pathlst:
            if pathlst[-1].strip() in COMMANDS:
                last_command = command
                command = pathlst.pop().strip()
                absolute = (command == command.upper())
                command = command.upper()
            else:
                if command is None:
                    raise ValueError("No command found at %d" % len(pathlst))

            if command == 'M':
            # MoveTo
                x = pathlst.pop()
                y = pathlst.pop()
                pt = Point(float(x), float(y))
                if absolute:
                    current_pt = pt
                else:
                    current_pt += pt
                start_pt = current_pt

                self.items.append(MoveTo(current_pt))

                # MoveTo with multiple coordinates means LineTo
                command = 'L'

            elif command == 'Z':
            # Close Path
                l = Line(current_pt, start_pt)
                self.items.append(l)


            elif command in 'LHV':
            # LineTo, Horizontal & Vertical line
                # extra coord for H,V
                if absolute:
                    x,y = current_pt.coord()
                else:
                    x,y = (0,0)

                if command in 'LH':
                    x = pathlst.pop()
                if command in 'LV':
                    y = pathlst.pop()

                pt = Point(float(x), float(y))
                if not absolute:
                    pt += current_pt

                self.items.append(Line(current_pt, pt))
                current_pt = pt

            elif command in 'CQ':
                dimension = {'Q':3, 'C':4}
                bezier_pts = []
                bezier_pts.append(current_pt)
                for i in range(1,dimension[command]):
                    x = pathlst.pop()
                    y = pathlst.pop()
                    pt = Point(float(x), float(y))
                    if not absolute:
                        pt += current_pt
                    bezier_pts.append(pt)

                self.items.append(Bezier(bezier_pts))
                current_pt = pt

            elif command in 'TS':
                # number of points to read
                nbpts = {'T':1, 'S':2}
                # the control point, from previous Bezier to mirror
                ctrlpt = {'T':1, 'S':2}
                # last command control
                last = {'T': 'QT', 'S':'CS'}

                bezier_pts = []
                bezier_pts.append(current_pt)

                if last_command in last[command]:
                    pt0 = self.items[-1].control_point(ctrlpt[command])
                else:
                    pt0 = current_pt
                pt1 = current_pt
                # Symetrical of pt1 against pt0
                bezier_pts.append(pt1 + pt1 - pt0)

                for i in range(0,nbpts[command]):
                    x = pathlst.pop()
                    y = pathlst.pop()
                    pt = Point(float(x), float(y))
                    if not absolute:
                        pt += current_pt
                    bezier_pts.append(pt)

                self.items.append(Bezier(bezier_pts))
                current_pt = pt

            elif command == 'A':
                for i in range(0,7):
                    pathlst.pop()
                # TODO

            else:
                pathlst.pop()

    def __str__(self):
        return '\n'.join(str(x) for x in self.items)

    def __repr__(self):
        return 'Path id ' + self.ident

    def segments(self, precision=0):
        '''Return a list of segments, each segment is ended by a MoveTo.
           A segment is a list of Points'''
        ret = []
        seg = []
        for x in self.items:
            if isinstance(x, MoveTo):
                if seg != []:
                    ret.append(seg)
                    seg = []
            else:
                seg += x.segments(precision)
        ret.append(seg)
        return ret

    def simplify(self, precision):
        '''Simplify segment with precision:
           Remove any point which are ~aligned'''
        ret = []
        for seg in self.segments(precision):
            ret.append(simplify_segment(seg, precision))

        return ret

class Point:
    def __init__(self, x=0, y=0):
        '''A Point is defined either by a tuple/list of length 2 or
           by 2 coordinates'''
        if (isinstance(x, tuple) or isinstance(x, list)) and len(x) == 2:
            self.x = x[0]
            self.y = x[1]
        elif isinstance(x, numbers.Real) and isinstance(y, numbers.Real):
            self.x = x
            self.y = y
        else:
            raise TypeError("A Point is defined by 2 numbers or a tuple")

    def __add__(self, other):
        '''Add 2 points by adding coordinates.
        Try to convert other to Point if necessary'''
        if not isinstance(other, Point):
            try: other = Point(other)
            except: return NotImplemented
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if not isinstance(other, Point):
            try: other = Point(other)
            except: return NotImplemented
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if not isinstance(other, numbers.Real):
            return NotImplemented
        return Point(self.x * other, self.y * other)
    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        if not isinstance(other, Point):
            try: other = Point(other)
            except: return NotImplemented
        return (self.x == other.x) and (self.y == other.y)

    def __repr__(self):
        return '(' + format(self.x,'.3f') + ',' + format( self.y,'.3f') + ')'

    def __str__(self):
        return self.__repr__();

    def coord(self):
        '''Return the point tuple (x,y)'''
        return (self.x, self.y)

    def length(self):
        '''Vector length, Pythagoras theorem'''
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def rot(self, angle):
        '''Rotate vector [Origin,self] '''
        if not isinstance(angle, Angle):
            try: angle = Angle(angle)
            except: return NotImplemented
        x = self.x * angle.cos - self.y * angle.sin
        y = self.x * angle.sin + self.y * angle.cos
        return Point(x,y)


class Angle:
    '''Define a trigonometric angle [of a vector] '''
    def __init__(self, arg):
        if isinstance(arg, numbers.Real):
        # We precompute sin and cos for rotations
            self.angle = arg
            self.cos = math.cos(self.angle)
            self.sin = math.sin(self.angle)
        elif isinstance(arg, Point):
        # Point angle is the trigonometric angle of the vector [origin, Point]
            pt = arg
            try:
                self.cos = pt.x/pt.length()
                self.sin = pt.y/pt.length()
            except ZeroDivisionError:
                self.cos = 1
                self.sin = 0

            self.angle = math.acos(self.cos)
            if self.sin < 0:
                self.angle = -self.angle
        else:
            raise TypeError("Angle is defined by a number or a Point")

    def __neg__(self):
        return Angle(Point(self.cos, -self.sin))

class Line:
    '''A line is an object defined by 2 points'''
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return 'Line from ' + str(self.start) + ' to ' + str(self.end)

    def segments(self, precision=0):
        ''' Line segments is simply the segment start -> end'''
        return [self.start, self.end]

    def length(self):
        '''Line length, Pythagoras theorem'''
        s = self.end - self.start
        return math.sqrt(s.x ** 2 + s.y ** 2)

    def pdistance(self, p):
        '''Perpendicular distance between this Line and a given Point p'''
        if not isinstance(p, Point):
            return NotImplemented

        if self.start == self.end:
        # Distance from a Point to another Point is length of a line
            return Line(self.start, p).length()

        s = self.end - self.start
        if s.x == 0:
        # Vertical Line => pdistance is the difference of abscissa
            return abs(self.start.x - p.x)
        else:
        # That's 2-D perpendicular distance formulae (ref: Wikipedia)
            slope = s.y/s.x
            # intercept: Crossing with ordinate y-axis
            intercept = self.start.y - (slope * self.start.x)
            return abs(slope * p.x - p.y + intercept) / math.sqrt(slope ** 2 + 1)


    def bbox(self):
        if self.start.x < self.end.x:
            xmin = self.start.x
            xmax = self.end.x
        else:
            xmin = self.end.x
            xmax = self.start.x
        if self.start.y < self.end.y:
            ymin = self.start.y
            ymax = self.end.y
        else:
            ymin = self.end.y
            ymax = self.start.y
        return (Point(xmin,ymin),Point(xmax,ymax))

    def transform(self, matrix):
        self.start = matrix * self.start
        self.end = matrix * self.end

    def scale(self, ratio):
        self.start *= ratio
        self.end *= ratio
    def translate(self, offset):
        self.start += offset
        self.end += offset
    def rotate(self, angle):
        self.start = self.start.rot(angle)
        self.end = self.end.rot(angle)

class Bezier:
    '''Bezier curve class
       A Bezier curve is defined by its control points
       Its dimension is equal to the number of control points
       Note that SVG only support dimension 3 and 4 Bezier curve, respectively
       Quadratic and Cubic Bezier curve'''
    def __init__(self, pts):
        self.pts = list(pts)
        self.dimension = len(pts)

    def __str__(self):
        return 'Bezier' + str(self.dimension) + \
                ' : ' + ", ".join([str(x) for x in self.pts])

    def control_point(self, n):
        if n >= self.dimension:
            raise LookupError('Index is larger than Bezier curve dimension')
        else:
            return self.pts[n]

    def rlength(self):
        '''Rough Bezier length: length of control point segments'''
        pts = list(self.pts)
        l = 0.0
        p1 = pts.pop()
        while pts:
            p2 = pts.pop()
            l += Line(p1, p2).length()
            p1 = p2
        return l

    def bbox(self):
        return self.rbbox()

    def rbbox(self):
        '''Rough bounding box: return the bounding box (P1,P2) of the Bezier
        _control_ points'''
        xmin = None
        xmax = None
        ymin = None
        ymax = None
        for pt in self.pts:
            if xmin == None or pt.x < xmin:
                xmin = pt.x
            if ymin == None or pt.y < ymin:
                ymin = pt.y
            if xmax == None or pt.x > xmax:
                xmax = pt.x
            if ymax == None or pt.y > ymax:
                ymax = pt.y

        return (Point(xmin,ymin), Point(xmax,ymax))

    def segments(self, precision=0):
        '''Return a polyline approximation ("segments") of the Bezier curve
           precision is the minimum significative length of a segment'''
        segments = []
        # n is the number of Bezier points to draw according to precision
        if precision != 0:
            n = int(self.rlength() / precision) + 1
        else:
            n = 1000
        if n < 10: n = 10
        if n > 1000 : n = 1000

        for t in range(0, n):
            segments.append(self._bezierN(float(t)/n))
        return segments

    def _bezier1(self, p0, p1, t):
        '''Bezier curve, one dimension
        Compute the Point corresponding to a linear Bezier curve between
        p0 and p1 at "time" t '''
        pt = p0 + t * (p1 - p0)
        return pt

    def _bezierN(self, t):
        '''Bezier curve, Nth dimension
        Compute the point of the Nth dimension Bezier curve at "time" t'''
        # We reduce the N Bezier control points by computing the linear Bezier
        # point of each control point segment, creating N-1 control points
        # until we reach one single point
        res = list(self.pts)
        # We store the resulting Bezier points in res[], recursively
        for n in range(self.dimension, 1, -1):
            # For each control point of nth dimension,
            # compute linear Bezier point a t
            for i in range(0,n-1):
                res[i] = self._bezier1(res[i], res[i+1], t)
        return res[0]

    def transform(self, matrix):
        self.pts = [matrix * x for x in self.pts]

    def scale(self, ratio):
        self.pts = [x * ratio for x in self.pts]
    def translate(self, offset):
        self.pts = [x + offset for x in self.pts]
    def rotate(self, angle):
        self.pts = [x.rot(angle) for x in self.pts]

class MoveTo:
    def __init__(self, dest):
        self.dest = dest

    def bbox(self):
        return (self.dest, self.dest)

    def transform(self, matrix):
        self.dest = matrix * self.dest

    def scale(self, ratio):
        self.dest *= ratio
    def translate(self, offset):
        self.dest += offset
    def rotate(self, angle):
        self.dest = self.dest.rot(angle)

class Circle(Transformable):
    '''SVG <circle>'''
    def __init__(self, elt=None):
        Transformable.__init__(self, elt)
        if elt is not None:
            self.center = Point(float(elt.get('cx')), float(elt.get('cy')))
            self.radius = float(elt.get('r'))
            self.style = elt.get('style')
            self.ident = elt.get('id')

    def __repr__(self):
        return 'circle id ' + self.ident

    def bbox(self):
        '''Bounding box'''
        pmin = self.center - Point(self.radius, self.radius)
        pmax = self.center + Point(self.radius, self.radius)
        return (pmin, pmax)

    def transform(self, matrix):
        self.center = self.matrix * self.center

    def scale(self, ratio):
        self.center *= ratio
        self.radius *= ratio
    def translate(self, offset):
        self.center += offset
    def rotate(self, angle):
        self.center = self.center.rot(angle)

    def segments(self, precision=0):
        return self

    def simplify(self, precision):
        return self

def simplify_segment(segment, epsilon):
    '''Ramer-Douglas-Peucker algorithm'''
    if len(segment) < 3 or epsilon <= 0:
        return segment[:]

    l = Line(segment[0], segment[-1]) # Longest line

    # Find the furthest point from the line
    maxDist = 0
    index = None
    for i,p in enumerate(segment[1:]):
        dist = l.pdistance(p)
        if (dist > maxDist):
            maxDist = dist
            index = i+1 # enumerate starts at segment[1]

    if maxDist > epsilon:
        # Recursively call with segment splited in 2 on its furthest point
        r1 = simplify_segment(segment[:index+1], epsilon)
        r2 = simplify_segment(segment[index:], epsilon)
        # Remove redundant 'middle' Point
        return r1[:-1] + r2
    else:
        return [segment[0], segment[-1]]


