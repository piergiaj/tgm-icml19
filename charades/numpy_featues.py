import os
import numpy as np

base = '/ssd2/charades/Charades_v1_features_flow'
with open('/ssd2/charades/charades_frames.txt', 'r') as f:
    vids = [x.split(' ')[0] for x in f.readlines() if len(x) > 3]
for vid in vids:
    print vid
    feats = []
    for i in xrange(1,10000000000,4):
        fl = os.path.join(base, vid, vid+'-'+str(i).zfill(6)+'.txt')
        if not os.path.exists(fl):
            break
        feats.append(np.loadtxt(fl))
    feats = np.asarray(feats)
    print vid, feats.shape
    np.save('/ssd2/charades/ts-flow/'+vid, feats)
