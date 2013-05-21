import re
import numbers, math

COMMANDS = 'MmZzLlHhVvCcSsQqTtAa'

class Path:
    """A SVG Path"""

    def __init__(self):
        # The 'path' list contains drawable elements such as Line, Bezier, ...
        self.path = []

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

                self.path.append(MoveTo(current_pt))

                # MoveTo with multiple coordinates means LineTo
                command = 'L'

            elif command == 'Z':
            # Close Path
                l = Line(current_pt, start_pt)
                self.path.append(l)


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

                self.path.append(Line(current_pt, pt))
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

                self.path.append(Bezier(bezier_pts))
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
                    pt0 = self.path[-1].control_point(ctrlpt[command])
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

                self.path.append(Bezier(bezier_pts))
                current_pt = pt

            elif command == 'A':
                for i in range(0,7):
                    pathlst.pop()
                # TODO

            else:
                pathlst.pop()

    def __str__(self):
        return '\n'.join(str(x) for x in self.path)

    def segments(self, precision=0):
        '''Return a list of segments, each segment is ended by a MoveTo.
           A segment is a list of Points'''
        ret = []
        seg = []
        for x in self.path:
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
           Remove any point which is in ~aligned with the current line'''
        ret = []
        for seg in self.segments(precision):
            s = []
            seg.reverse()
            p1 = seg.pop()
            s.append(p1)
            p2 = seg.pop()
            s.append(p2)
            while seg:
                p3 = seg.pop()
                # a is the reference vector
                a = p2 - p1
                if a.length() == 0:
                    s.pop()
                    p2 = p3
                    s.append(p2)
                    continue
                # b is the tested vector
                b = p3 - p1
                # Skip if vector b is null
                if b.length() == 0:
                    continue
                # To ease computation, we make vector a the abscissa
                theta = Angle(a)
                c = b.rot(-theta)
                # We check that vectors a and b are more or less aligned,
                # ie rotated(b) ordinate is below precision
                # We skip the current point is it is ~ aligned
                if abs(c.y) > precision:
                    s.append(p3)
                    p1 = p2
                    p2 = p3
#                else:
#                    print "=== SKIP ==="
#                    c.y = 0
#                    b = c.rot(theta)
#                    p3 = b + p1
#                    s.append(p3)
            s.append(p3)
            ret.append(s)

        return ret

    def bbox(self):
        xmin = None
        xmax = None
        ymin = None
        ymax = None
        for x in self.path:
            pmin, pmax = x.bbox()
            if xmin == None or pmin.x < xmin:
                xmin = pmin.x
            if ymin == None or pmin.y < ymin:
                ymin = pmin.y
            if xmax == None or pmax.x > xmax:
                xmax = pmax.x
            if ymax == None or pmax.y > ymax:
                ymax = pmax.y

        return (Point(xmin,ymin), Point(xmax,ymax))

class Point:
    def __init__(self, x=0, y=0):
        '''A Point is defined either by a tuple of length 2 or by 2 coordinates'''
        if isinstance(x, tuple) and len(x) == 2:
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

class MoveTo:
    def __init__(self, dest):
        self.dest = dest

    def bbox(self):
        return (self.dest, self.dest)

