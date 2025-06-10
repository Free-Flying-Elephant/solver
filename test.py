

import json

x = [0, 1, 2, -0.5, 0.5, 1.5, 2.5, 0, 1, 2]
y = [0, 0, 0, 0.866025, 0.866025, 0.866025, 0.866025, 1.7320508, 1.7320508, 1.7320508]

idx = [[0, 4, 3],
                        [0, 1, 4],
                        [1, 5, 4],
                        [1, 2, 5],
                        [2, 6, 5],
                        [3, 4, 7],
                        [4, 8, 7],
                        [4, 5, 8],
                        [5, 9, 8],
                        [5, 6, 9]]

# which edge is on the boundary
ed: list[int] = [2, 0, -1, 0, 0, 2, 1, -1, 1, 2] # -1 means no bc on the cell boundary
# which BC is on that edge
bc: list[int] = [2, 1, 0, 1, 3, 2, 1, 0, 1, 3] # bc identifier

dct = {"x" : x, "y" : y}

dct2 = {"edge" : ed, "bc" : bc}

with open(r'case\\bc.json', "w") as f:
    json.dump(dct2, f)

with open(r'case\\bc.json', 'r') as file:
    data = json.load(file)

# Print the data
print(data)

