import pstats, cProfile
import cymlda

import pyximport
pyximport.install()

command = '''
import numpy as np
model = cymlda.mylda()
model.readCorpus()
model.train_corpus(1)
prev_het = np.zeros(np.shape(model._n_het))
np.copyto(prev_het, model._n_het)
for i in range(10):
    print 'iter %d' % (i+1)
    model.train_corpus(1)
    model._abs_delta_het = np.abs(model._n_het - prev_het)
    np.copyto(prev_het, model._n_het)
    if i%5 == 0:
        model.output_topic(i)
'''

cProfile.runctx('cymlda.main()', globals(), locals(), 'Profile.prof')

s = pstats.Stats('Profile.prof')
s.strip_dirs().sort_stats('time').print_stats()
