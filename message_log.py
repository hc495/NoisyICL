import time
import csv
import random
import os

output_path = './Logs/'
output_file = None
Task_num = random.randint(1, 1000)


# You should manage your data in the experiment cells, and use the 'output_single_csv' to output.

def log(text):
    # Echo and Log a complex string as:
    # ---------------------------------------
    # [TIME STRING]
    # [:text]
    # ---------------------------------------
    global output_file
    timestr = time.asctime(time.localtime(time.time()))
    print('---------------------------------------')
    print(timestr)
    print(text)
    output_file.write('---------------------------------------')
    output_file.write('\n')
    output_file.write(timestr)
    output_file.write('\n')
    output_file.write(text)
    output_file.write('\n')
    
def simple_log(text):
    # Echo and Log a simple string as:
    # [:text]
    print(text)
    global output_file 
    output_file.write(text)
    output_file.write('\n')

def mute_log(text):
    # Log a simple string without echo as:
    # [:text]
    global output_file 
    output_file.write(text)
    output_file.write('\n')
    
def dic_log(dic, dicname):
    global output_file
    timestr = time.asctime(time.localtime(time.time()))
    print(dicname + ':')
    output_file.write(dicname + ':')
    output_file.write('\n')
    print('---------------------------------------')
    output_file.write('---------------------------------------')
    output_file.write('\n')
    for key, value in dic.items():
        print(str(key) + ':' + str(value))
        output_file.write(str(key) + ':' + str(value))
        output_file.write('\n')
    
def experiment_start(experiment_name):
    print('Hello! Welcome to experiment. I wish you with good data!')
    print('よろしくお願いします。よいデータを得られるように。')
    print('Author of Experiment System: Yufeng Zhao')
    basic_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + experiment_name
    global output_file 
    output_file = open(output_path + basic_name + '_experiment_report.txt', mode='x')
    output_file.write('Experiment Start.')
    output_file.write('\n')

def experiment_end():
    global output_file
    output_file.write('Experiment End.')
    output_file.write('\n')
    output_file.close()
    print('The experiment is over, thank you!')
    print('実験終わりました。ゆっくり休んでお願いします。')

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