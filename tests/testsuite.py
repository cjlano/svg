import os, sys
import nose
sys.path.append('..')
import svg

path = 'W3C_SVG_11_TestSuite/svg/'

def test_files():
    for filename in os.listdir(path):
        if filename[-3:] == 'svg':
            yield svg.Svg, path+filename

if __name__=="__main__":
    import nose
    nose.main()
