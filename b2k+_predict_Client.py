import httplib
import sys


import datetime

if len(sys.argv) != 3:
  print 'ERROR - Input argument error: please define input and output filenames'
  sys.exit(0)

# the input-document to execute b2k
b2k_input_doc = sys.argv[1]
b2k_output    = sys.argv[2]

conn = httplib.HTTPConnection('localhost', 2311)
conn.request('POST', '/b2k_process/?inputs='+str(b2k_input_doc+','+b2k_output), '{}')
print conn.getresponse().read()
conn.close()
