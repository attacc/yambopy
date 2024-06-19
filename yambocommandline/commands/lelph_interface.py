from yambopy import LetzElphElectronPhononDB,ConvertElectronPhononDB
import os
import shutil
import subprocess
"""
Calculate gauge-invariant electron-phonon matrix elements with LetzElPhC and convert them into Yambo format

- Usage:

>> yambopy l2y -ph phinp -b b1 b2 -par nq nk [--lelphc lelphc] [--debug]

- Input parameters:
	-ph           : path to ph.x input file, e.g. dvscf/ph.in
	-b            : initial and final band indices (counting from 1)
    -par [OPT]    : MPI pools for q and k (needs mpirun)
	--lelphc [OPT]: path to lelphc executable, default 'lelphc', code will prompt
	--debug [OPT] : won't remove LetzElPhC input and outputs
 
- Prerequisites:

* ph.x phonon calculation must be complete, e.g. the phinp folder should contain:
    - ph.x input file
    - pw.x (scf) save directory
    - [prefix].dyn files
    - _ph* directories
* Yambo SAVE directory must be present. We run in the directory where the SAVE is.
* LetzElPhC must be installed
* mpirun must be linked for parallel runs
"""

def get_input(bands,pools,ph_path):
	"""
	Get LetzElPhC input
	"""
	input_file = []; app = input_file.append
	app("# LetzElPhC input for yambo generated by yambopy")
	app(f"nqpool      = {pools[0]}") # first pool is qpoints
	app(f"nkpool      = {pools[1]}") # second pool is kpoints
	app(f"start_bnd   = {bands[0]}")
	app(f"end_bnd     = {bands[1]}")
	app(f"save_dir    = ./SAVE")
	app(f"kernel      = dfpt")
	app(f"ph_save_dir = {ph_path}/ph_save")
	app(f"convention = yambo")
	return "\n".join(input_file)

def checks(phinp,lelphc,bands,pools):

	## ph.x input file and directory
	if '/' in phinp:
		path_ph = phinp.rsplit('/',1)[0]
		inp_ph  = phinp.rsplit('/',1)[1]
	else:
		path_ph='./'
		inp_ph = phinp

	## check for SAVE
	if not os.path.isdir("SAVE"):
		raise FileNotFoundError("SAVE directory not found. Make sure to run in the directory where the SAVE is.")

	## check for lelphc executable
	lelphc = "lelphc"
	is_lelphc = shutil.which(lelphc)
	if is_lelphc is None: lelphc=input("lelphc executable not found. Please provide absolute path here: \n")

	## check for mpirun
	if int(pools[0])*int(pools[1])>1:
		is_mpirun = shutil.which('mpirun')
		if is_mpirun is None: 
			print("[WARNING] mpirun not found, running in serial") 
			pools = [1,1]

    ## check band indices
    try: assert(int(bands[0])<int(bands[1]))
    except: raise ValueError("[ERROR] band indices must be integers with b1<b2")

	## lelphc input file
	inp_lelphc = get_input(bands,pools,path_ph)
	inp_name = 'lelphc.in'

	return lelphc,path_ph,inp_ph,inp_lelphc,inp_name

def run_preprocessing(lelphc,path_ph,inp_ph):

	print(":: LetzElPhC pre-processing ::")
	pp_run_str = f"{lelphc} -pp --code=qe -F {inp_ph}"
	try:
		pp=subprocess.run(f'cd {path_ph} ; {pp_run_str} ; cd -',shell=True,check=True,capture_output=True,text=True)
	except subprocess.CalledProcessError as e:
		print("PP Error:", e)
		print("Description", e.stderr)

	if not os.path.isdir(f"{path_ph}/ph_save"):
		print("[ERROR] Something wrong with preprocessing step:")
		print(pp.stderr)
		exit()

def run_elph(lelphc,inp_lelphc,inp_name):

	def get_ntasks(inp):
		ntasks=1
		for line in inp.split('\n'):
			if 'pool' in line: ntasks *= int(line.split()[-1])
		return ntasks			

	print(":: LetzElPhC el-ph calculation ::")

	# write input file
	f = open(inp_name, "w")
	f.write(inp_lelphc)
	f.close()

	# manage parallel run
	ntasks = get_ntasks(inp_lelphc)
	if ntasks > 1 : elph_run_str = f"mpirun -np {ntasks} {lelphc} -F {inp_name}"
	elif ntasks==1: elph_run_str = f"{lelphc} -F {inp_name}"
	else: raise ValueError(f'Wrong MPI task selection: {ntasks} tasks?')

	# run lelphc
	try:
		elph_run=subprocess.run(elph_run_str,shell=True,check=True,capture_output=True,text=True)
	except subprocess.CalledProcessError as e:
		print("Error:", e)
		print("Description:", e.stderr)


def letzelph_to_yambo():

	print(":: Load el-ph database ::")
	lelph_obj = LetzElphElectronPhononDB('ndb.elph',div_by_energies=False)

	print(":: Convert el-ph data to yambo format ::")
	l2y       = ConvertElectronPhononDB(lelph_obj,'lelphc','SAVE','SAVE')

def clean_lelphc(debug,inp_name,ph_path):
	if debug:
		print(":: Debug flag ::")
		print("     - lelphc.in input written")
		print("     - ph_save | ndb.elph | ndb.Dmats not removed")
	else:
		[os.remove(C) for C in [inp_name,'ndb.elph','ndb.Dmats'] if os.path.isfile(C)]
		if os.path.isdir(f"{ph_path}/ph_save"): shutil.rmtree(f"{ph_path}/ph_save")

if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Generate electron-phonon coupling databases via LetzElPhC')
	parser.add_argument('-ph','--ph_inp_path', type=str, help='<Required> Path to ph.x (dvscf) input file',required=True)
	parser.add_argument('-b','--bands',nargs='2',type=str,help="<Required> First and last band (counting from 1), e.g. 'b_i b_f'",required=True)
	parser.add_argument('-par','--pools',nargs='2',type=str,default=[1,1],help="<Optional> MPI tasks as 'nqpools nkpools' (default serial)")
	parser.add_argument('-lelphc','--lelphc',type=str,default='lelphc',help="<Optional> Path to lelphc executable (default assumed in Path, otherwise prompted)")
	parser.add_argument('-D','--debug', action="store_true", help="Debug mode")
	args = parser.parse_args()

	phinp  = args.ph_inp_path
	bands  = args.bands
	pools  = args.pools
	lelphc = args.lelphc
	debug  = args.debug

	# Check inputs
	lelphc,ph_path,inp_ph,inp_lelphc,inp_name = checks(phinp,lelphc,bands,pools)

	# run preprocessing
	run_preprocessing(lelphc,ph_path,inp_ph)

	# run el-ph calculation and rotation
	run_elph(lelphc,inp_lelphc,inp_name,pools)

	# load database and convert to yambo format
	letzelph_to_yambo()

	# clean
	clean_lelphc(debug,inp_name,ph_path)	
