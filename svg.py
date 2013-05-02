import re
import numbers, math

COMMANDS = 'MmZzLlHhVvCcSsQqTtAa'

class Path:
    """A SVG Path"""

    def __init__(self, pathstr=""):
        self.pathstr = pathstr
        self.pathlst = []
        # The 'path' list contains drawable elements such as Line, Bezier, ...
        self.path = []

    def parse(self, pathstr=""):
        """Parse path string and build elements list"""
        if pathstr != "":
            self.pathstr = pathstr

        # (?:...) : non-capturing version of regular parentheses
        pathlst = re.findall(r"([+-]?\ *\d+(?:\.\d*)?|\.\d+|\ *[%s]\ *)"
                % COMMANDS, self.pathstr)

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

                self.path.append(MoveTo(pt))

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

    def segments(self):
        '''Return a list of segments, each segment is ended by a MoveTo.
           A segment is a list of points coordinates: Points.coord()'''
        ret = []
        seg = []
        for x in self.path:
            if isinstance(x, MoveTo):
                if seg != []:
                    ret.append(seg)
                    seg = []
            else:
                seg += x.segments()
        ret.append(seg)
        return ret

    def simplify(self, precision):
        '''Simplify segment with precision:
           Remove any point which is in the area of the current line'''
        ret = []
        for seg in self.segments():
            s = []
            seg.reverse()
            p1 = seg.pop()
            s.append(p1)
            p2 = seg.pop()
            s.append(p2)
            while seg:
                p3 = seg.pop()
                a = p2 - p1
                b = p3 - p1
                if b.length() == 0:
                    continue
                c = b.rot(a)
                if abs(c.y) > precision:
                    s.append(p3)
                    p1 = p2
                    p2 = p3
            s.append(p3)
            ret.append(s)
        return ret

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
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

    def rot(self, vect):
        '''self rotation vs. vect direction'''
        l = vect.length()
        x = self.x * vect.x/l + self.y * vect.y/l
        y = -self.x * vect.y/l + self.y * vect.x/l
        return Point(x,y)

class Line:
    '''A line is an object defined by 2 points'''
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return 'Line from ' + str(self.start) + ' to ' + str(self.end)

    def segments(self):
        ''' Line segments is simply the segment start -> end'''
        return [self.start, self.end]

    def length(self):
        '''Line length, Pythagoras theorem'''
        s = self.end - self.start
        return math.sqrt(s.x ** 2 + s.y ** 2)

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

    def segments(self):
        segments = []
        for t in range(0,100):
            segments.append(self._bezierN(t*0.01))
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

