import csv, sys, re
import numpy as np
import pdb

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print >> sys.stderr, "%s <geneDBfile> <wellDBfile> ids.npz all_cell_data.npz" % sys.argv[0]
        sys.exit(1)
    
    orfs_to_labels = {}
    locs_to_orfs = {}
    
    dr = csv.DictReader(open(sys.argv[1]))
    
    # Get each gene record, save ORF as key, entire record as value
    for line in dr:                                         
        orfs_to_labels[line['cgeneorf']] = line
        print line['cgeneorf'] + '...'
    
    dr = csv.DictReader(open(sys.argv[2]))
        
    # Get each well record, save (plate,row,col) as key, orf as value.
    for line in dr:
        if line['cgeneorf'] != "BLANK":
            loc = tuple(map(int, (line['nplate'],line['nrow'],line['ncol'])))
            locs_to_orfs[loc] = line['cgeneorf']
    
    locs_to_labels = {}
    for loc in locs_to_orfs:
        orf = locs_to_orfs[loc]
        try:
            rec = orfs_to_labels[orf]
        except KeyError:
            print >>sys.stderr, "[w] Gene '%s' not in label lookup." % orf
            continue
        localization = None
        for key, val in rec.iteritems():
            if key[:5] == "bloc_":
                if localization == None and val == "1":
                    localization = key
                elif val == "1":
                    print >> sys.stderr, "[w] '%s': multiple loc'ns" % orf
                    continue
        if localization:
            locs_to_labels[loc] = localization
        else:
            print >> sys.stderr, "[w] '%s': no localization" % orf
    f = np.load(sys.argv[3])
    platerowcol2imageno = {}
    for fn in f.files:
        tokens = fn.split("_")
        print tokens
        plateno = re.match(r'plate(\d\d)',tokens[-3]).group(1)
        row, col = re.match(r'(\d{3})(\d{3})(\d{3})',tokens[-2]).groups()[:2]
        imageno = f[fn][0,0]
        platerowcol2imageno[(int(plateno),int(row),int(col))] = imageno
    #print platerowcol2imageno
    g = np.load(sys.argv[4])
    
    labels = np.empty(g['order'].shape,dtype=object)
    imageno2label = {}
    for loc,label in locs_to_labels.iteritems():
        try:
            imageno2label[platerowcol2imageno[loc]] = label
        except KeyError, e:
            pass
    
    labeled_images = np.unique1d(np.array(imageno2label.keys()))
    for ii in labeled_images:
        labels[g['ids'][:,0] == ii] = imageno2label[ii]d
    print "Saving..."
    np.save("/Users/dwf/Data/labels.npz", labels)