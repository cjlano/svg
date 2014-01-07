SVG Tests
=========

W3C SVG test suite
------------------
Download the [W3C SVG 1.1 Test suite (14MB)]( http://www.w3.org/Graphics/SVG/Test/20110816/archives/W3C_SVG_11_TestSuite.tar.gz).

Extract it:

    tar -xvzf W3C_SVG_11_TestSuite.tar.gz -C W3C_SVG_11_TestSuite

Test is using [Python nose](http://nose.readthedocs.org/en/latest/index.html). Install it for your environment to run the test.

    nosetests testsuite.py

