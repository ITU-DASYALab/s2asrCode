import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument("--validated",  type=str, default="./data/EN/validated.tsv",    help="The path of the tsv file containing all validated clips")
parser.add_argument("--test",       type=str, default="./data/EN/test.tsv",         help="The path of the tsv file containing test clips")
parser.add_argument("--dev",        type=str, default="./data/EN/dev.tsv",          help="The path of the tsv file containing dev clips")
parser.add_argument("--target",     type=str, default="./data/ENCombined/",         help="The target path for the created train file")
args = parser.parse_args()

def construct_voice_dataset(validated, test, dev, target):
    validated_frame = pd.read_csv(validated, sep='\t')
    test_frame = pd.read_csv(test, sep='\t')
    dev_frame = pd.read_csv(dev,sep='\t')

    print("Test frame:\n", test_frame.count())
    print("Dev frame:\n", dev_frame.count())
    print("Validated frame:\n", validated_frame.count())

    validated_frame_without_test = validated_frame[validated_frame.path.isin(test_frame.path) == False]

    print("Validated frame without test:\n", validated_frame_without_test.count())
    training_frame = validated_frame_without_test[validated_frame_without_test.path.isin(dev_frame.path) == False]

    print("Train frame:\n", training_frame.count())
    print("Duplicates:\n", validated_frame.duplicated(subset="path").value_counts())

    save_tsv(training_frame, target, "train")
    save_tsv(test_frame, target, "test")
    save_tsv(dev_frame, target, "dev")

def save_tsv(frame, target, dataset):
    target_path = os.path.join(target, dataset + ".tsv")
    print("Saving to ", target_path)
    frame.to_csv(target_path, sep="\t", index=False)
    print("Done saving to ", target_path)

if __name__ == '__main__':
    construct_voice_dataset(args.validated, args.test, args.dev, args.target)