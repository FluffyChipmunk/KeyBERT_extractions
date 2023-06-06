import re
import pandas as pd

bad_chars = ["&", "=", "\"", ","]
names = []
with open('asl signbank files.txt') as f:
    lines = f.readlines()
    for line in lines:
        if "\"name\":" in line:
            for i in bad_chars:
                cleanedLine = ''.join((filter(lambda i: i not in bad_chars, line)))
            name = cleanedLine.split()[1].replace('.mp4', '')
            names.append(name)
    f.close()
print(names)
dataframe = pd.DataFrame(names)
dataframe.to_csv('signbanklisting.csv')





