import sys, fileinput
from sacremoses import *

tokenizer = MosesTokenizer('en')

if __name__ == "__main__":
    for line in fileinput.input():
        if line.strip() != "":
            tokens = tokenizer.tokenize(line.strip())

            sys.stdout.write(" ".join(tokens) + "\n")

        else:
            sys.stdout.write('\n')