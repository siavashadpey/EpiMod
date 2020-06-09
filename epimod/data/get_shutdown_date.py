#!/usr/bin/env Python3

import yaml
import argparse

def get_shutdown_date(file, region):
    with open(file) as json_file:
        for line in yaml.safe_load(json_file):
            if line['region'].lower() == region:
                return line['start']    

def main():

    parser = argparse.ArgumentParser(description="Obtains specified region's shutdowndate from json file in specified folder.")
    parser.add_argument('--file', '-f', default='./data/shutdown_dates.json', help='path of json file')
    parser.add_argument('--region', '-c', default='canada', help='region of interest')

    args = parser.parse_args()

    file = args.file
    region = args.region.lower()
    date = get_shutdown_date(file, region)
    print(date)

if __name__ == '__main__':
    main()



