#!/usr/bin/env Python3

import argparse
import requests
import io
import pandas as pd
import os
from datetime import datetime

url = 'https://hgis.uw.edu/virus/assets/virus.csv'
min_data_points = 10
data_types = ['confirmed', 'suspected', 'cured', 'deaths']

def get_csv_data(regions=[], all_regions=False):
    try:
        contents = requests.get(url).content
    except:
        print("Error: couldn't retrieve data from " + url)

    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    df.dropna(axis='columns', thresh=min_data_points, inplace=True) # remove regions with insufficent data

    if all_regions:
        regions = df.columns.values.tolist()[1:]

    regions_data = {}

    print("")
    print("Collecting data of region: ")
    for region in regions:
        print(region + " ...",end=" ")
        try:
            df_filt = df.dropna(subset=[region]) # remove days with no data for 'region'
            regions_data[region] = {}
            regions_data[region]['dates'] = df_filt['datetime'].to_numpy() # store date
            data_points = df_filt[region].to_numpy()

            # store data
            for i in range(len(data_types)):
                regions_data[region][data_types[i]] = []

            for entry in data_points:
                entry_split = [elem.strip() for elem in entry.split('-')]
                for i in range(len(data_types)):
                    if i < len(entry_split) and entry_split[i].isnumeric():
                        regions_data[region][data_types[i]].append(int(entry_split[i]))
                    else:
                        regions_data[region][data_types[i]].append(float('nan'))

            print("done")
        except:
            print("Error: couldn't obtain data of region " + region + " -"*50)

    print("")
    return regions_data


def save_data(folder_dir, regions):
    # create directory if non-existent
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)

    # write to 1 file/region (overwrite existing files if necessary)
    print("")
    print("Writing data of region: ")
    for region_name, region_data in regions.items():
        print(region_name + " ...",end=" ")
        # create file
        file_name = region_name + ".txt"
        f = open(folder_dir + os.path.sep + file_name, "w")

        # write header
        f.write("# date, day count")
        for typ in data_types:
            f.write(", " + str(typ))
        f.write("\n")
        
        # write data
        for day in range(len(region_data['dates'])):
            f.write(region_data['dates'][day] + ", ")
            f.write(str(day))
            for typ in data_types:
                f.write(", " + str(region_data[typ][day]))
            f.write("\n")

        f.close()
        print("done")
    print("")

def main():

    parser = argparse.ArgumentParser(description="Collects coronavirus data from University of Washington's Humanistic GIC Laboratory (https://hgis.uw.edu/virus/).")
    parser.add_argument('--folder', '-f', default='.', help='path of folder where data should be saved')
    parser.add_argument('--all', '-a', action='store_true', help='boolean indicating if all regions should be collected')
    parser.add_argument('--regions', '-r', nargs='+', default=["canada"], help='list of regions')

    args = parser.parse_args()

    regions_data = get_csv_data(args.regions, args.all)

    date_today = datetime.today().strftime('%Y-%m-%d')
    folder_dir = args.folder + os.path.sep + date_today
    save_data(folder_dir, regions_data)

    # make a copy of today's folder
    today_dir = args.folder + os.path.sep + "today"
    cmd = "rm -fr " + today_dir
    os.system(cmd)
    cmd = "cp -r " + folder_dir + " " + today_dir
    os.system(cmd)

if __name__ == '__main__':
    main()
