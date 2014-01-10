__all__ = ['geometry', 'svg']

def parse(filename):
    from . import svg
    f = svg.Svg(filename)
    return f

