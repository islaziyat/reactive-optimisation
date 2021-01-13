## Copyright:  Isla Ziyat, Simon Fraser University 
## Date:2020-10-28

import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import os
import csv
import time
from math import sqrt
import copy
import time


def main():
	start_time = time.time()
	global gridname, griddata, L1, L2, L3, P1, P2, P3, S1, S2, S3, power_loss, VDI
	do_optimise = True

	# Choose grid
	gridname = "bc_grid"
	griddata = load_griddata(gridname,optimised=False)


	if gridname == "bc_grid":
		L1 = 2
		L2 = 15
		L3 = 19
		S1 = 4000
		S2 = 4000
		S3 = 5000
		inputs='bc_summer.csv'

	# SET DG real power based on time of day
	P1 = 0
	P2 = 2000
	P3 = 4000

	# SET load demand based on demand curve
	loading_percent = 1

	# RUN optimisation
	if do_optimise==True:
		model = optimise()
		Q1 = model.output_dict['variable'][0]
		Q2 = model.output_dict['variable'][1]
		Q3 = model.output_dict['variable'][2]
		objective([Q1,Q2,Q3])
		output = [ L1, L2, L3, P1, P2, P3, Q1, Q2, Q3,power_loss, VDI, model.output_dict['function'] ]
	else:
		Q1, Q2, Q3 = 0,0,0
		objective([Q1,Q2,Q3])
		output = [ L1, L2, L3, P1, P2, P3, Q1, Q2, Q3,power_loss, VDI]

	
	with open(r'results.csv', 'a', newline='') as f:
	    writer = csv.writer(f)
	    writer.writerow(output)
	f.close()

	print("--- %s seconds ---" % (time.time() - start_time))

	power_loss_opt = copy.deepcopy(power_loss)
	VDI_opt = copy.deepcopy(VDI)

	# RUN without reactive power for comparison
	objective([0,0,0])
	power_loss_reduction = 100*(power_loss-power_loss_opt)/power_loss
	VDI_reduction = 100*(VDI-VDI_opt)/VDI

	print("Optimal reactive power at DG1 is found to be " + str(Q1) + " Var")
	print("Optimal reactive power at DG2 is found to be " + str(Q2)+ " Var")
	print("Optimal reactive power at DG3 is found to be " + str(Q3)+ " Var")

	
	print("Power loss in the lines goes from " + str(power_loss) + " W to " + str(power_loss_opt) + " W" )
	print("Total power loss reduction of " + str(power_loss_reduction) + " % ")

	print("VDI goes from " + str(VDI) + " to " + str(VDI_opt) )
	print("Total VDIs reduction of " + str(VDI_reduction) + " % ")



	return


def optimise():
	# OR SET VA limit
	Q1_limit1 = sqrt(S1**2-P1**2);
	Q2_limit1 = sqrt(S2**2-P2**2);
	Q3_limit1 = sqrt(S3**2-P3**2);
	varbound=np.array([[-Q1_limit1,Q1_limit1],[-Q2_limit1,Q2_limit1],[-Q3_limit1,Q3_limit1]])

	# OR SET power factor limit
	Q1_limit2 = 0.484*P1
	Q2_limit2 = 0.484*P2
	Q3_limit2 = 0.484*P3

	Q1_limit = min(Q1_limit1,Q1_limit2)
	Q2_limit = min(Q2_limit1,Q2_limit2)
	Q3_limit = min(Q3_limit1,Q3_limit2)
	varbound=np.array([[-Q1_limit,Q1_limit],[-Q2_limit,Q2_limit],[-Q3_limit,Q3_limit]])

	algorithm_param = {'max_num_iteration': 200,\
	                   'population_size':10,\
	                   'mutation_probability':0.1,\
	                   'elit_ratio': 0.01,\
	                   'crossover_probability': 0.5,\
	                   'parents_portion': 0.3,\
	                   'crossover_type':'uniform',\
	                   'max_iteration_without_improv':None}


	model=ga(function=objective,dimension=3,variable_type='real',variable_boundaries=varbound,algorithm_parameters=algorithm_param)

	model.run()

	return model

def load_griddata(gridname, optimised=False, reactive=False):
	base_voltage=1
	reg_exists =0

	with open('gridlabd_results/'+gridname+'/IEEEdata/Grid Info.csv') as csvDataFile:
	    csvReader = csv.reader(csvDataFile)
	    for row in csvReader:
	          if row[0] == "base_voltage":
	                base_voltage = np.float64(row[1])
	          if row[0] == "regulator_location":
	                reg_exists = 1
	                x = np.float64(row[1])
	                y = np.float64(row[2])

	if optimised==False:
	    griddata = dict(  gridlabd_file = 'gridlabd_results/' + gridname +'/',\
	                      current_file = 'gridlabd_results/'+ gridname + '/current.csv',\
	                      voltage_file = 'gridlabd_results/'+ gridname +'/voltage.csv',\
	                      max_current_file = 'gridlabd_results/'+gridname+'/max_current.csv',\
	                      base_voltage = base_voltage,\
	                      regulator = [],\
	                      DG_buses = [],\
	                      loads = [],\
	                      power_loss_reduction=[],\
	                      voltage_deviation_reduction=[],\
	                      )

	return griddata

def loadflow(X):
	sign1 = '+'
	if X[0] < 0:
		sign1 = '-'
		X[0] = abs(X[0])

	sign2 = '+'
	if X[1] < 0:
		sign2 = '-'
		X[1] = abs(X[1])

	sign3 = '+'
	if X[2] < 0:
		sign3 = '-'
		X[2] = abs(X[2])
	 
	variables_DG1 = ' --define DG1_location='+str(L1)+\
	    ' --define DG1_PhaseA='+ str(1000*P1/3)+sign1+ str(1000*X[0]/3) +\
	    ' --define DG1_PhaseB='+ str(1000*P1/3)+sign1+ str(1000*X[0]/3)+\
	    ' --define DG1_PhaseC='+ str(1000*P1/3)+sign1+ str(1000*X[0]/3);

	variables_DG2 = ' --define DG2_location='+str(L2)+\
	    ' --define DG2_PhaseA='+ str(1000*P2/3)+sign2+ str(1000*X[1]/3)\
	    +' --define DG2_PhaseB='+ str(1000*P2/3)+sign2+ str(1000*X[1]/3)\
	    +' --define DG2_PhaseC='+ str(1000*P2/3)+sign2+ str(1000*X[1]/3);

	variables_DG3 = ' --define DG3_location='+str(L3)\
	    +' --define DG3_PhaseA='+ str(1000*P3/3)+sign3+ str(1000*X[2]/3)\
	    +' --define DG3_PhaseB='+ str(1000*P3/3)+sign3+ str(1000*X[2]/3)\
	    +' --define DG3_PhaseC='+ str(1000*P3/3)+sign3+ str(1000*X[2]/3)+' ';

	gridlabd_command= 'gridlabd' + variables_DG1+variables_DG2+variables_DG3+ griddata['gridlabd_file'] + 'gridlabd_optimised.glm';
	os.system(gridlabd_command) 
	return 

def objective(X):
	global power_loss, VDI
	loadflow(X)
	power_loss = calculate_power_loss()
	VDI = calculate_VDI()

	objective = 1000*(VDI/5.4 + 10*power_loss/160000)

	return objective

def save_output_voltage():
	nrow=0
	VA_pu,VB_pu,VC_pu = 0,0,0
	with open('gridlabd_results/'+gridname+'/voltage_optimised.csv') as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			if nrow >= 2 and "node" in row[0]:
				VA_pu = sqrt(float(row[1])**2 + float(row[2])**2)/griddata['base_voltage'] 
				VB_pu = sqrt(float(row[3])**2 + float(row[4])**2)/griddata['base_voltage'] 
				VC_pu = sqrt(float(row[5])**2 + float(row[6])**2)/griddata['base_voltage'] 
				output = [VA_pu, VB_pu, VC_pu]

				with open(r'voltage_results.csv','a', newline='') as f:
					writer = csv.writer(f)
					writer.writerow(output)
				f.close()

			nrow = nrow + 1
	csvDataFile.close()

def calculate_power_loss():
	power_loss = 0
	start = 0
	if os.path.isfile('gridlabd_results/'+gridname+'/underground_line_losses.csv'):
		with open('gridlabd_results/'+gridname+'/underground_line_losses.csv') as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			for row in csvReader:
				if start == 1:
					power_loss = float(row[1]) + float(row[2]) + float(row[3])  
					break
				if row[0] == '# property.. timestamp':
					start = 1
		csvDataFile.close()

	start = 0
	if os.path.isfile('gridlabd_results/'+gridname+'/overhead_line_losses.csv'):
		with open('gridlabd_results/'+gridname+'/overhead_line_losses.csv') as csvDataFile:
			csvReader = csv.reader(csvDataFile)
			for row in csvReader:
				if start == 1:
					power_loss = power_loss + float(row[1]) + float(row[2]) + float(row[3]) 
					break
				if row[0] == '# property.. timestamp':
					start = 1
			csvDataFile.close()
	return power_loss

def calculate_VDI():
	nrow = 0
	VDI = 0
	with open('gridlabd_results/'+gridname+'/voltage_optimised.csv') as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			if nrow >= 2 and "node" in row[0]:
				VA_pu = sqrt(float(row[1])**2 + float(row[2])**2)/griddata['base_voltage'] 
				VB_pu = sqrt(float(row[3])**2 + float(row[4])**2)/griddata['base_voltage'] 
				VC_pu = sqrt(float(row[5])**2 + float(row[6])**2)/griddata['base_voltage'] 
				VDI = VDI + abs(VA_pu-1) + abs(VB_pu-1) + abs(VC_pu-1)
			nrow = nrow + 1
	csvDataFile.close()
	return VDI 

def calculate_apparent_power():
	S = 0
	nrow = 0
	with open('gridlabd_results/'+gridname+'/voltage_optimised.csv') as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			if nrow == 2:
				VA = complex(float(row[1]),float(row[2]))
				VB = complex(float(row[3]),float(row[4]))
				VC = complex(float(row[5]),float(row[6]))
				break
			nrow = nrow + 1
	csvDataFile.close()

	nrow=0
	with open('gridlabd_results/'+gridname+'/current_optimised.csv') as csvDataFile:
		csvReader = csv.reader(csvDataFile)
		for row in csvReader:
			if nrow == 2:
				IA = complex(float(row[1]),float(row[2]))
				IB = complex(float(row[3]), float(row[4]))
				IC = complex(float(row[5]), float(row[6]))		
				break
			nrow = nrow + 1
		csvDataFile.close()

	S = VA*np.conj(IA) + VB*np.conj(IB) + VC*np.conj(IC)

	return S 



if __name__ == "__main__":
    # execute only if run as a script
    main() 


