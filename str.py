#!/usr/bin/python2
import sys
import re

if len(sys.argv) < 2:
    print "Usage: %s \"path string\"" % sys.argv[0]
    sys.exit()

path = re.split(r"([+-]?\ *\d+(?:\.\d*)?|\.\d+)", sys.argv[1])
# (?:...)non-capturing version of regular parentheses
# because re.split()  If capturing parentheses are used in pattern, then the text of all groups in the pattern are also returned as part of the resulting list

# Number of expected values per commands
cmd = {'M':2, 'L':2, 'Z':0, 'H':1, 'V':1, 'A':7, 'Q':4, 'T':2, 'C':6, 'S':4}

# clean-up path in p[]
p = []
for elt in path:
# remove all spaces
    elt = elt.strip()
    elt = elt.replace(' ','')
    if (elt == ''):
        continue;
# remove all commas (not strictly necessary in SVG)
    if (elt == ','):
        continue;
# split commands into single one (e.g.: 'ZM' -> 'Z','M')
# check command validity
    if elt.isalpha():
        for i in list(elt):
            if (i.upper() in cmd.keys()): p.append(i)
            else: print i, " is not a valid command"
# not a command? should be a numeric
    else:
        try: p.append(float(elt))
        except: print elt, " should be numeric"


# Split series of identical commands into individual blocks
i = 1
elt = p[0]   # current element
c = 'M'      # Current command
while i <= len(p):
# Expect a command, remember it
    if str(elt).upper() in cmd.keys():
        c = elt
        l = [c]
        for j in range(0, cmd[c.upper()]):
             l.append(p[i+j])
        i += cmd[c.upper()]
# Get next element
        try: elt = p[i]
        except: elt = '' # End of list?
        i += 1
        print l
    else:
# We expect a new command but did not get one: use previous one and realign
         elt = c
         i -= 1


# Change lower case command to upper case (relative to absolute)

