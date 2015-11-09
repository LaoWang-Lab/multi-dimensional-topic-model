import glob
import subprocess
import argparse
import os

from plot import plot_fig_in_dir

def dance(png_dir, gif):
    os.makedirs(os.path.pathname(git))
    os.chdir(png_dir)
    if gif in os.listdir('.'):
        os.remove(gif)
    subprocess.Popen('convert -delay 10 -loop 0 %s/*.png %s' % (png_dir, gif), shell=True, stdout=subprocess.PIPE).stdout.read()

def dance_all():
    curdir = os.getcwd()
    if not curdir.endswith('multi-dimensional-topic-model'):
        raise Exception('NOT in correct directory!')
    dirs = [curdir, 'harvest']
    for hewotm in os.listdir(os.path.sep.join(dirs)):
        dirs.append(hewotm)
        for run in os.listdir(os.path.sep.join(dirs)):
            dirs.append(run)
            dir_name = os.path.sep.join(dirs)
            print(dir_name)
            if 'done' not in os.listdir(dir_name):
                plot_fig_in_dir(dir_name)
                dance(dir_name, os.path.sep.join(dirs[0] + ['gif'] + '_'.join(dirs[2:])))
                with open('done','w') as fd:
                    pass
            dirs.pop()
        dirs.pop()

if __name__ == '__main__':
    dance_all()
