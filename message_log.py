import time
import csv
import random
import os

output_path = './Logs/'
output_file = None
Task_num = random.randint(1, 1000)

# You should manage your data in the experiment cells, and use the 'output_single_csv' to output.

def output_single_csv(lines, extra_name = ''):
    basic_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + extra_name
    local_table_file = open(output_path + basic_name + ", " + str(Task_num) + '_ind_result.csv', mode='x')
    local_table_writer = csv.writer(local_table_file)
    for line in lines:
        local_table_writer.writerow(line)
    local_table_file.close()

def new_folder(name):
    name_full = name + ', ' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    os.makedirs(output_path + name_full)
    return name_full