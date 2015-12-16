#
# Author: Henrique Pereira Coutada Miranda
# Run a GW calculation using yambo
#
from __future__ import print_function
from yambopy.inputfile import *
from yambopy.outputfile import *
from yambopy.analyse import *
from pwpy.inputfile import *
from pwpy.outputxml import *
import subprocess

if not os.path.isdir('database'):
    os.mkdir('database')

#check if the nscf data is present
if os.path.isdir('nscf/si.save'):
    print('nscf calculation found!')
else:
    print('nscf calculation not found!')
    exit()

#check if the SAVE folder is present
if not os.path.isdir('database/SAVE'):
    print('preparing yambo database')
    os.system('cd nscf/si.save; p2y')
    os.system('cd nscf/si.save; yambo')
    os.system('mv nscf/si.save/SAVE database'.split())

#if bse folder is not present, create it
if not os.path.isdir('bse_conv'):
    os.mkdir('bse_conv')
    os.system('cp -r database/SAVE bse_conv')

#create the yambo input file
y = YamboIn('yambo -b -o b -k sex -y d -V all',folder='bse_conv')

#list of variables to optimize and the values they might take
conv = { 'FFTGvecs': [[10,15,20],'Ry'],
         'NGsBlkXs': [[5,10,20], 'Ry'],
         'BndsRnXs': [[1,10],[1,20],[1,30]] }

def run(filename):
    """ Function to be called by the optimize function """
    folder = filename.split('.')[0]
    print(filename, folder)
    os.system('cd bse_conv; yambo -F %s -J %s -C %s 2> %s.log'%(filename,folder,folder,folder))

y.optimize(conv,run=run)

#pack the files in .json files
for folder in subprocess.check_output('ls bse_conv/*/o-*',shell=True).splitlines():
    folder = '/'.join(folder.split('/')[:-1])
    y = YamboOut(folder)
    if not y.locked():
        y.pack()
        y.put_lock()

#plot the results using yambmo analyser
y = YamboAnalyser('bse_conv')
print(y)
y.plot_bse('eps')
print('done!')