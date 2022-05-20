import json
from io import open
import sys
import pickle
import random

random.seed(833434)

in_file = open(sys.argv[1], "r", encoding="utf-8")
out_file = open(sys.argv[2], "w", encoding="utf-8")

in_file_lines = in_file.readlines()
in_file_lines = in_file_lines[1:]  # Remove header
random.shuffle(in_file_lines)

all_labels = {}


for line in in_file_lines:
    line = line.strip().split(",")
    text = line[0]
    label = line[1]
    out_line = "\t".join([label, " ", text, "E"]) + "\n"
    out_file.write(out_line)
    if label not in all_labels:
        all_labels[label] = 0
    all_labels[label] += 1

out_file.close()

if len(sys.argv) >= 4:
    print("Writing Label Space")
    pickle.dump(all_labels, open(sys.argv[3] + "/labelSpace.p", "wb"))
