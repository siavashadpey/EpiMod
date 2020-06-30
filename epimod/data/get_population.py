#!/usr/bin/env Python3

import yaml
import argparse

def get_population(file, region):
    with open(file) as json_file:
        for line in yaml.safe_load(json_file):
            if line['region'].lower() == region:
                return line['population']

def main():

    parser = argparse.ArgumentParser(description="Obtains specified regions's population from json file in specified folder (source: https://github.com/samayo/country-json).")
    parser.add_argument('--file', '-f', default='./data/population.json', help='path of json file')
    parser.add_argument('--region', '-r', default='canada', help='region of interest')

    args = parser.parse_args()

    file = args.file
    region = args.region.lower()
    pop = get_population(file, region)
    print(pop)

if __name__ == '__main__':
    main()