import os
import re

def extract_label(filename):
    # Use regex to find all characters up to the first digit
    match = re.match(r"([^\d]*)", filename)  # Matches all characters until a digit
    return match.group(0).strip() if match else ""

def main():
    datapath = '/data/ucfdata/UCF_Train_ten_crop_i3d'
    files = os.listdir(datapath)
    
    unique_labels = set()  # Use a set to store unique labels

    for file in files:
        label = extract_label(file)
        unique_labels.add(label)

    # Print all unique labels
    print("Unique labels in the directory:")
    for label in unique_labels:
        print(label)

if __name__ == '__main__':
    main()
