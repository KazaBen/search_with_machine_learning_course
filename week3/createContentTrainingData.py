import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd

stemmer = nltk.stem.SnowballStemmer("english")
nltk.download('punkt')

def transform_name(product_name):
    tokens = word_tokenize(product_name)
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Directory for product data
directory = r'/workspace/search_with_machine_learning_course/data/pruned_products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")

# Consuming all of the product data will take over an hour! But we still want to be able to obtain a representative sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate

print("Writing results to %s" % output_file)
with open(output_file, 'w') as output:
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            print("Processing %s" % filename)
            f = os.path.join(directory, filename)
            tree = ET.parse(f)
            root = tree.getroot()
            for child in root:
                if random.random() > sample_rate:
                    continue
                # Check to make sure category name is valid
                if (child.find('name') is not None and child.find('name').text is not None and
                    child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
                    child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None):
                      # Choose last element in categoryPath as the leaf categoryId
                      cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
                      # Replace newline chars with spaces so fastText doesn't complain
                      name = child.find('name').text.replace('\n', ' ')
                      output.write("__label__%s %s\n" % (cat, transform_name(name)))

NUM2KEEP = [50, 100, 200]
for num in NUM2KEEP:
    output_df = pd.read_csv(output_file)
    output_df[['y', 'x']] = output_df.iloc[:, 0].str.split(" ", n=1, expand=True)
    output_counts = output_df.groupby(by='y').agg("count")['x']
    labels2keep = output_counts[output_counts > num].index.values
    output_df = output_df[output_df['y'].isin(labels2keep)]
    output = output_df["y"] + " " + output_df["x"]
    output.to_csv("output_filtered{}.fasttext".format(num), index=False)