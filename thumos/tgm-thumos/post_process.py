import numpy as np
import os
import json

with open('thumos_mapping.txt') as f:
    cls = f.readlines()

cls_map = {c.strip():(a.strip(),b.strip()) for a,c,b in [x.split(' ') for x in cls]}
cls2_map = {a.strip():c.strip() for a,c,b in [x.split(' ') for x in cls]}
to_th_id = {a.strip():b.strip() for a,_,b in [x.split(' ') for x in cls]}


lens = {}
for cls in cls_map.keys():
    with open('annotations/'+cls+'.txt') as f:
        d = f.readlines()
    d = [x for x in d if 'test' not in d]
    ls = [x.split(' ') for x in d]
    ls = [float(x[2]) - float(x[1]) for x in ls]
    lens[cls] = np.asarray(ls).mean()

print lens


with open('preds.json') as f:
    data = json.load(f)


#thrs =[0.1,0.4,0.2,0.3,0.6,0.3,0.2,0.6,0.4,0.2,0.7,0.7,0.2,0.2,0.8,0.2,0.5,0.4,0.3,0.5]
thrs =[0.1,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.2,0.1,0.2,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.2]

intervals = []
for vid in data.keys():
    p, fps = data[vid]
    p = np.asarray(p)
    for c in range(20):
        k = lens[cls2_map[str(c+1)]]
        k = int(k*fps)
        prd = p[:,c] >thrs[c]+0.2
        #np.convolve(p[:,c], np.ones((k,)), mode='same')/k
        in_interval = 0
        start = -1
        for i in range(prd.shape[0]):
            if in_interval == 0 and prd[i] == 1:
                start = i
                in_interval = 1
            if in_interval == 1 and prd[i] == 0:
                if (i-start)/fps < 0.5*lens[cls2_map[str(c+1)]]:
                    #print (i-start)/fps, lens[cls2_map[str(c+1)]]
                    in_interval = 0
                    start = -1
                    continue
                if (i-start)/fps > 2*lens[cls2_map[str(c+1)]]:
                    ti = i - int((lens[cls2_map[str(c+1)]]*fps)/2)
                    intervals.append(vid+' '+str(start/fps)+' '+str(ti/fps)+' '+str(to_th_id[str(c+1)])+' '+str(p[(start+i)//2,c]))
                    start = ti+1
                    #print (i-start)/fps, lens[cls2_map[str(c+1)]]
                #else:
                intervals.append(vid+' '+str(start/fps)+' '+str(i/fps)+' '+str(to_th_id[str(c+1)])+' '+str(p[(start+i)//2,c]))
                #intervals.append((to_th_id[str(c+1)], start, i))
                in_interval = 0
                start = -1
        if in_interval == 1:
            #intervals.append((to_th_id[c], start, i))
            intervals.append(vid+' '+str(start/fps)+' '+str(i/fps)+' '+str(to_th_id[str(c+1)])+' '+str(p[start,c]))

with open('THUMOS14_evalkit_20150930/res.txt', 'w') as out:
    out.write('\n'.join(intervals))

