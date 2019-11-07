import csv

old_file = '../new_data.csv'
new_file = '../clean_data.csv'

rows = []

with open(old_file, 'r') as r:
    csv_reader = csv.reader(r)
    # get fields
    fields = next(csv_reader)
    
    for row in csv_reader:
        useful = [row[2], row[1]]
        rows.append(useful)

with open(new_file, 'w') as w:
    csv_writer = csv.writer(w)
    csv_writer.writerow(['comment','target'])
    
    for row in rows:
        csv_writer.writerow(row)