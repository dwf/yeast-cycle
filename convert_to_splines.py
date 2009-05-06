#/opt/sw/Python-2.5.4/bin/python
import tools
import sys
f = open(sys.argv[1])
locs, ids = tools.cells_in_database_per_imagefile([f], headerfile=op'/home/dwf/headers.csv')
features, newids = tools.load_and_process('/home/dwf/binary', locs, ids)
print feat
np.savez('/home/dwf' + sys.argv[2], features=features, ids=ids)