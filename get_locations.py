import numpy as np
import csv


def cells_in_database_per_imagefile(datafiles, headerfile=None):
    """
    Reads in CSV files that contain object records.
    
    Returns a dictionary where the keys are the image filename (with 
    / replaced by _) and the values are NumPy arrays containing coordinate
    lists, as well as another dictionary giving ImageNumber/ObjectNumbers.
    """
    if hasattr(datafiles, 'read'):
        datafiles = [datafiles]
    cells = {}
    if headerfile:
        headers = [h[1:-1] for h in headerfile.read().split(',')]
    else:
        headers = None
    for datafile in datafiles:
        reader = csv.DictReader(datafile, fieldnames=headers,
            delimiter=',',quotechar='"')
        for row in reader:
    	    key = row['Image_PathName_GFP'] + '/' + \
    	        row['Image_FileName_GFP']
    	    imgnum = row['ImageNumber']
    	    objnum = row['ObjectNumber']
    	    location = [np.float64(row['cells_Location_Center_X']),
    	        np.float64(row['cells_Location_Center_Y'])]
    	    cells.setdefault(key, []).append(((imgnum,objnum),location))
    	print "Finished with %s" % datafile.name
    allids = {}
    for image in cells.keys():
        ids, locations = zip(*cells[image])
        ids = np.array(ids,dtype=int)
        locations = np.array(locations)
        del cells[image]
        image = image.replace('/','_')
        cells[image] = locations
        allids[image] = ids
    return cells, allids

