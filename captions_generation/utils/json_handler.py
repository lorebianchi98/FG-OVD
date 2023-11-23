import json

def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)