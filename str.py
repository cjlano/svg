#!/usr/bin/python2
import sys
import re

if len(sys.argv) < 2:
    print "Usage: %s \"path string\"" % sys.argv[0]
    sys.exit()

path_str = sys.argv[1]

#print path

path = re.split(r"([+-]?\ *\d+(?:\.\d*)?|\.\d+)", path_str)
# (?:...)non-capturing version of regular parentheses
# because re.split()  If capturing parentheses are used in pattern, then the text of all groups in the pattern are also returned as part of the resulting list
print len(path)

for i,elt in enumerate(path):
    elt = elt.replace(' ','')
    if elt == '':
        path.pop(i)
    if elt.isalpha():
        print elt

print len(path)

#print path
