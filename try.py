import jsonlines

with open("BoolQ/train.jsonl", "r+") as f:
    for item in jsonlines.Reader(f):
        print(item)
        break