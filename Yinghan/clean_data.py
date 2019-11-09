import csv
import re
import random

old_file = '../new_data.csv'
new_file = '../train.csv'
dev_file = '../dev.csv'

rows = []

with open(old_file, 'r') as r:
    csv_reader = csv.reader(r)
    # get fields
    fields = next(csv_reader)
    
    for row in csv_reader:
        useful = [re.sub('[\n\t\s+]+', ' ', row[2]).strip(), row[1]]
        rows.append(useful)

with open(new_file, 'w') as w:
    with open(dev_file, 'w') as d:
        csv_writer = csv.writer(w, delimiter='\t')
        dev_writer = csv.writer(d, delimiter='\t')

        train = rows[:len(rows) - 3000]
        print(len(train))
        dev = rows[-3000:]
        print(len(dev))
        for row in train:
            csv_writer.writerow(row)
        
        for row in dev:
            dev_writer.writerow(row)