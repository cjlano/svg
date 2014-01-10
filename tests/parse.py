# Parse all W3C SVG testsuite files
# use it with 
## python parse.py | grep ^"No handler" | sort | uniq -c | sort -n
# to get all unhandled elements sorted by occurence.

import os
import sys
sys.path.append('..') #FIXME
import svg

path = 'W3C_SVG_11_TestSuite/svg/'

for f in os.listdir(path):
    if os.path.splitext(f)[1] == '.svg':
        svg.parse(path + f)

