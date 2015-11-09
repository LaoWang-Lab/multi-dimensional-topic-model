__author__ = 'Linwei'
from settings import run_num
from genSamples import gen_corpus
from cymlda import run_once

def main():
    for run_id in range(run_num):
        gen_corpus()
        run_once(run_id)

if __name__ == '__main__':
    main()
