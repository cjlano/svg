import re
import numbers

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

                self.path.append(MoveTo())

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
        ret = []
        seg = []
        for x in self.path:
            if isinstance(x, MoveTo):
                ret.append(seg)
                seg = []
                print("MoveTo")
            else:
                seg += x.segments()
        ret.append(seg)
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
        return (self.x, self.y)

class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return 'Line from ' + str(self.start) + ' to ' + str(self.end)

    def segments(self):
        return [self.start.coord(), self.end.coord()]

class Bezier:
    def __init__(self, pts):
        self.pts = list(pts)
        self.dimension = len(pts)

    def __str__(self):
        return 'Bezier' + str(self.dimension) + \
                ' : ' + ", ".join([str(x) for x in self.pts])

    def control_point(self, n):
        if n > self.dimension:
            print("Error")
        else:
            return self.pts[n]

    def segments(self):
        segments = []
        for t in range(0,10):
            segments.append(self._bezierN(t*0.1).coord())
        return segments

    def _bezier1(self, p0, p1, t):
        'Bezier curve, one dimension'
        pt = p0 + t * (p1 - p0)
        return pt

    def _bezierN(self, t):
        'Bezier curve, N dimensions'
        res = list(self.pts)
        for n in range(self.dimension, 1, -1):
            for i in range(0,n-1):
                res[i] = self._bezier1(res[i], res[i+1], t)
        return res[0]

class MoveTo:
    def __init__(self):
        pass
