import re
import pandas as pd


na_list = []
with open("training_100", "r", encoding="utf8") as in_file:
    line = 0
    while line <= 1000000:
        x = in_file.readline()
        if x.startswith("[ID]"):
            y = in_file.readline()
            y = "> " + y
            na_list.append(y.replace("\n", ""))
            z = in_file.readline()
            a = in_file.readline()
            na_list.append(a.replace("\n", ""))
        line += 1

with open('clean_na.txt', 'w') as f:
    for line in na_list:
        f.write(line)
        f.write('\n')

check_list = []
with open("clean_al2.txt", "r", encoding="utf8") as check_file:
    line = 0
    while line <= 5000:
        x = check_file.readline()
        if x.startswith(">"):
            y = check_file.readline()
            check_list.append(y.replace("\n", ""))
            line += 1
        else:
            line += 1
            continue

def collision_check(x):
    for seq in x:
        if seq in na_list:
            print("Collision detected")
            print(seq)
        else:
            continue

collision_check(x=check_list)
