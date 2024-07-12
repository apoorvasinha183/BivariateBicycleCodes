import json

filename = 'Example.json'

json_objects = []

with open(filename, 'r') as file:
    for line in file:
        line = line.strip()
        if line:  # Check if the line is not empty
            try:
                json_object = json.loads(line)  # Parse each line as a JSON object
                json_objects.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(e)

# Print the retrieved JSON objects
for obj in json_objects:
    print(obj)