import csv
import re
import random

old_file = '../new_data.csv'
new_file = '../data/train.csv'
dev_file = '../data/dev.csv'
test_file = '../data/test.csv'

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
        with open(test_file, 'w') as t:

            csv_writer = csv.writer(w, delimiter='\t')
            dev_writer = csv.writer(d, delimiter='\t')
            test_writer = csv.writer(t, delimiter='\t')

            train = rows[:len(rows) - 4000]
            print(len(train))
            dev = rows[-4000:-3000]
            print(len(dev))
            test = rows[-3000:]
            print(len(test))
            for row in train:
                csv_writer.writerow(row)
            
            for row in dev:
                dev_writer.writerow(row)

            for row in test:
                test_writer.writerow(row)
