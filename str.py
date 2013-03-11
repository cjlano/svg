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

# Number of points per commands
cmd = {'M':1, 'C':3, 'Z':0, 'L':1}

for i,elt in enumerate(path):
# Remove whitespaces
    elt = elt.replace(' ','')
# Remove empty elements
    if elt == '':
        path.pop(i)
# Separate multiple elements (e.g.: 'ZM')
    if elt.isalpha() and len(elt) > 1:
        print elt[0]
# Search in cmd
#    if elt.upper() in cmd.keys():
#        print cmd[elt.upper()]

# Change lower case command to upper case (relative to absolute)

print len(path)

#print path
