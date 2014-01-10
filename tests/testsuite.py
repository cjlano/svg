import os
import svg

path = os.path.abspath(os.path.dirname(__file__)) + '/W3C_SVG_11_TestSuite/svg/'

def test_files():
    for filename in os.listdir(path):
        if filename[-3:] == 'svg':
            yield svg.parse, path+filename

if __name__=="__main__":
    import nose
    nose.main()
