import pandas as pd
import numpy as np
from pathlib import Path

kategorien = ["HUM:ind", "LOC:other", "NUM:count", "NUM:date", "ENTY:other", "ENTY:cremat", "HUM:gr", "LOC:country", "LOC:city", "ENTY:animal", "ENTY:food"]
train2data = []
with open("data/train_data/train_2.csv") as file:
    for line in file:
        line = line.split(";")
        if len(line) == 6:
            if line[3] in kategorien:
                line[4] = int(line[4]) # ID zu int
                line[5] = line[5].replace("\n", "") # getting rid of \n
                train2data.append(line)

df = pd.DataFrame(train2data)
df.columns = ["group", "question", "answer", "answer_type", "document_id", "sentence_with_answer"]

train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])#kann man auch sklearn train_test_split
filenames = ["train", "validate", "test"]
dataframes = [train, validate, test]

for i in filenames:
    filepath = Path("./data/dataframes/{}.csv".format(i))
    filepath.parent.mkdir(parents=True, exist_ok=True)  
i = 0
for dataframe in dataframes:
    dataframe.to_csv("./data/dataframes/" + filenames[i] + ".csv", index = False, sep = ";")
    i += 1