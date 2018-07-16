#Ricardo Erazo
#load data files
import numpy as np
label1 = 'gsyn='
label2 = str(gsyn)
label3 = ',Vrev='
label4 = str(Vrev)
fname1 = np.core.defchararray.add(label1,label2)
fname2 = np.core.defchararray.add(label3,label4)
fname = str(np.core.defchararray.add(fname1,fname2))
fname = str(np.core.defchararray.add(fname,'.npy'))

[c1,a1]= np.load(fname)
