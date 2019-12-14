import csv
import os

path = 'YOUR_INPUT_DIRECTORY'


with open('YOUR_OUTPUT_FILE', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['identifier', 'file', 'description', 'subject[0]', 'title', 'creator', 'date', 'collection'])
    for root, dirs, files in os.walk('D:/Users/arad/ISIC2018T3/data/ISIC2018_Task3_Test_Input/images'):
        for filename in files:
            writer.writerow(['', os.path.join(root, filename), '', '', '', 'opensource', '', ''])
