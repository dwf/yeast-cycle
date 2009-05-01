import csv, sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, "%s <geneDBfile> <wellDBfile>" % sys.argv[0]
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
    
    # If a gene has more than one associated label.
    

