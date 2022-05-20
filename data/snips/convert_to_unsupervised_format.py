import sys
import io
import pickle
from itertools import chain, combinations
import random
import os


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def generate_complement(input_set, universal_set):
    complement_set = []
    for u in universal_set:
        if u not in input_set:
            complement_set.append(u)
    return complement_set


def write_two_holdout_ood_files(
    in_eval_file, out_eval_file, out_eval_file_clean, acceptable_powerset
):
    allLines = in_eval_file.readlines()
    if verbose:
        print(len(allLines))
    oodLines = 0
    output_lines = []
    output_lines_clean = []
    for line in allLines:
        words = line.strip().split("\t")
        lineLabel = words[0]
        if lineLabel not in acceptable_powerset:
            words[0] = "outOfDomain"
            modified_line = "\t".join(words)
            output_lines.append(modified_line)
            oodLines += 1
        else:
            output_lines.append(line.strip())
            output_lines_clean.append(line.strip())

    for line in output_lines:
        out_eval_file.write(line + "\n")

    for line in output_lines_clean:
        out_eval_file_clean.write(line + "\n")

    if verbose:
        print(oodLines)
    out_eval_file.close()
    out_eval_file_clean.close()


verbose = True

if verbose:
    print(
        "Generating Folders With Unsupervised Splits in the Given Ratio, with ratio being for in-domain domains"
    )
    print("Loading label space")
in_domain_ratio = float(sys.argv[1])
dataset_seed = int(sys.argv[2])
number_of_splits = int(sys.argv[3])
label_space = pickle.load(open("sup/labelSpace.p", "rb"))
if verbose:
    print("Label Space:", label_space)
label_ids = label_space.keys()
train_n = sum(label_space.values())
random.seed(dataset_seed)

if verbose:
    print("Generating powersets")
all_powersets = list(powerset(label_ids))
if verbose:
    print("Total Number Of Powersets:", len(all_powersets))
all_powerset_lengths = [
    sum([label_space[class_name] for class_name in powerset])
    for powerset in all_powersets
]


max_domain_ratio = in_domain_ratio * 1.15
min_domain_ratio = in_domain_ratio * 0.85

acceptable_powersets = []

if verbose:
    print("Finding acceptable powersets")
for i, powerset in enumerate(all_powersets):
    if (
        all_powerset_lengths[i] >= min_domain_ratio * train_n
        and all_powerset_lengths[i] <= max_domain_ratio * train_n
    ):
        acceptable_powersets.append(powerset)
        if verbose:
            print("Accepted Set:", powerset)
            print(
                "Accepted Set Length:",
                all_powerset_lengths[i],
                "Total Length:",
                train_n,
            )
            print("Complement Set:", generate_complement(powerset, label_ids))
print("Number Of Accepted Sets:", len(acceptable_powersets))

random.shuffle(acceptable_powersets)

snips_ood = [
    ["SearchCreativeWork"],
    ["BookRestaurant", "GetWeather"],
    ["PlayMusic", "SearchScreeningEvent"],
    ["BookRestaurant"],
    ["BookRestaurant", "SearchScreeningEvent"],
]

print("acceptable_powersets:", acceptable_powersets)

acceptable_powersets_final = []
for ood in snips_ood:
    for powerset in acceptable_powersets:
        if (len(set(ood) & set(powerset)) == 0) and ((len(ood) + (len(powerset)) == 7)):
            print(powerset, ood)
            acceptable_powersets_final.append(powerset)


acceptable_powersets = (
    acceptable_powersets_final  # acceptable_powersets[:number_of_splits]
)
print("acceptable_powersets:", acceptable_powersets)

for powerset_id, acceptable_powerset in enumerate(acceptable_powersets):
    acceptable_powerset = list(acceptable_powerset)
    if verbose:
        print("Converting for :", acceptable_powerset)

    parent_dir = "unsup" + "_" + str(in_domain_ratio) + "_" + str(powerset_id) + "/"
    os.makedirs(parent_dir)

    # Logging in-domain labels both in a text and pickle file in the directory corresponding to the unsupervised split
    accepted_labels_file = open(
        parent_dir + "accepted_labels.txt", "w", encoding="utf-8"
    )
    accepted_labels_file.write(",".join(acceptable_powerset) + "\n")
    accepted_labels_file.close()
    pickle.dump(acceptable_powerset, open(parent_dir + "accepted_labels.p", "wb"))

    in_train_file = open("sup/train.tsv", "r", encoding="utf-8")
    out_train_file = open(parent_dir + "OODRemovedtrain.tsv", "w", encoding="utf-8")
    for line in in_train_file.readlines():
        words = line.strip().split("\t")
        lineLabel = words[0]
        if lineLabel in acceptable_powerset:
            if len(words) != 4:
                print(line)
                print(len(words))
            out_train_file.write(line)
    out_train_file.close()

    in_eval_file = open("sup/eval.tsv", "r", encoding="utf-8")
    out_eval_file = open(parent_dir + "eval.tsv", "w", encoding="utf-8")
    out_eval_file_clean = open(parent_dir + "OODRemovedeval.tsv", "w", encoding="utf-8")
    write_two_holdout_ood_files(
        in_eval_file, out_eval_file, out_eval_file_clean, acceptable_powerset
    )

    in_test_file = open("sup/test.tsv", "r", encoding="utf-8")
    out_test_file = open(parent_dir + "test.tsv", "w", encoding="utf-8")
    out_test_file_clean = open(parent_dir + "OODRemovedtest.tsv", "w", encoding="utf-8")
    write_two_holdout_ood_files(
        in_test_file, out_test_file, out_test_file_clean, acceptable_powerset
    )
