#!/usr/bin/env Python3

import yaml
import argparse

def get_country_population(file, country):
	with open(file) as json_file:
	    for line in yaml.safe_load(json_file):
	    	if line['country'].lower() == country:
	        	return line['population']

def main():

	parser = argparse.ArgumentParser(description="Obtains specified country's population from json file in specified folder (source: https://github.com/samayo/country-json).")
	parser.add_argument('--file', '-f', default='./data/country_population.json', help='path of json file')
	parser.add_argument('--country', '-c', default='canada', help='country of interest')

	args = parser.parse_args()

	file = args.file
	country = args.country.lower()
	pop = get_country_population(file, country)
	print(pop)

if __name__ == '__main__':
	main()



