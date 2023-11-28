def has_missing_value(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            values = line.split("\t")  
            if '1.00000000000000e+99' in values:
                return True
    return False