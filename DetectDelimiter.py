def detect_delimiter(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        if '\t' in first_line:
            return '\t'
        elif ',' in first_line:
            return ','
        else:
            first_space_index = first_line.find(' ')
            space_count = 1
            for i in range(first_space_index + 1, len(first_line)):
                if first_line[i] == ' ':
                    space_count += 1
                else:
                    break
            return ' ' * space_count
