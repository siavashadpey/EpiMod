#!/usr/bin/env Python3

import argparse
import os
import epimod.data as tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_region_data(folder, region):

	# read population
	file_pop = folder + os.path.sep + "population.json"
	n_pop = int(tools.get_population(file_pop, region))

	# read region's COVID-19 data
	file_data = folder + os.path.sep + "today" + os.path.sep + str(region).lower() + ".txt"
	df = pd.read_csv(file_data, na_values=" nan", index_col ="date", header=0, names = ["date", "day_count", "aggregate_cases", "suspected", "cured", "deaths"])

	# drop rows with insufficient data
	n_min = max(20., 2.E-6*n_pop) # min = 20 or 2/million
	df = df[(df["aggregate_cases"] >= n_min).idxmax():]
	
	# compute daily cases
	x = np.diff(df["aggregate_cases"].values, prepend=0)
	df.insert(loc=3, column="daily_cases", value=x)
	x_ave = moving_average(x,n=7)
	df.insert(loc=3, column="daily_averaged_cases", value=x_ave)

	# read and format shutdown date
	file_sd = folder + os.path.sep + "shutdown_dates.json"
	shutdown_date = tools.get_shutdown_date(file_sd, region)
	if shutdown_date is not None:
		shutdown_day = df.loc[shutdown_date]["day_count"]
	else:
		shutdown_day = None
	
	# Initial condition for SEIR
	S0 = n_pop - df.loc[df.index[0],"aggregate_cases"]
	E0 = 0
	R0 = df.loc[df.index[0],"cured"] - df.loc[df.index[0],"deaths"]
	I0 = df.loc[df.index[0],"aggregate_cases"] - R0
	IC = np.array([S0, E0, I0, R0])
	# output necessary data
	return (df["day_count"].values, df.index.values, df["daily_averaged_cases"].values, n_pop, int(shutdown_day), IC, df["daily_cases"].values)
	
def moving_average(x, n=5):
	# x_ave[i] = 1/n sum_{j=0}^{n-1} x[i-j] and j <= i
	M = len(x)
	x_ave = np.zeros_like(x)
	for i in range(M):
		summ = 0
		count = 0
		for k in range(max(0, i-n+1),i+1):
			summ += x[k]
			count += 1
		x_ave[i] = summ/count

	return x_ave

def main():
	parser = argparse.ArgumentParser(description="Obtains specified region's coronavirus-related data.")
	parser.add_argument('--folder', '-f', default='./data/', help='path of data folder')
	parser.add_argument('--region', '-r', default='canada', help='region of interest')

	args = parser.parse_args()

	folder = args.folder
	region = args.region.lower()
	(t, dates, daily_smooth, n_pop, shutdown_day, IC, daily) = read_region_data(folder, region)
	
	#print(n_pop)
	#print(shutdown_day)
	#for i in range(len(t)):
		#print(t[i], daily_smooth[i])


	plt.plot(t, daily_smooth, color='b', lw=2)
	plt.vlines(t, 0, daily, colors='b')
	plt.show()


if __name__ == '__main__':
	main()

	# TODO: convert the following to a unit test
	#x = np.array([1, 2, 3, 6, 9, 12, 20, 28, 30, 25, 22, 20, 15, 12, 10])
	#x_ave = moving_average(x, 3)
	#assert (np.around(x_ave, decimals = 8) == np.array([1., 1.5, 2., 3.66666667, 6., 9.,
 	#													  13.66666667, 20., 26., 27.66666667, 25.66666667, 22.33333333,
 	#													  19., 15.66666667, 12.33333333])).all(), "moving average is incorrect"
