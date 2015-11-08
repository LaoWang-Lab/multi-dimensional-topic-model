import pstats, cProfile
import mlda

cProfile.runctx('mlda.main()', globals(), locals(), 'mlda')

s = pstats.Stats('mlda')
s.strip_dirs().sort_stats('time').print_stats()
