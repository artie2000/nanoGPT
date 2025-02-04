import sys

test_file_name = "config/test_iGSM-med.txt"

with open(test_file_name) as file:
    while line := file.readline():
        sys.argv.append("--start="+line) # set starting prompt
        exec(open('sample.py').read())