import math
from tsp import TravelingSalesmanProblem
import json


def main():
    # List of 30 US state capitals and corresponding coordinates on the map
    with open('capitals.json', 'r') as capitals_file:
        capitals = json.load(capitals_file)
    capitals_list = [(k, tuple(v)) for k, v in capitals.items()]

    # Create the problem instance and plot the initial state
    num_cities = 30
    shuffle = False

    capitals_tsp = TravelingSalesmanProblem(capitals_list[:num_cities], shuffle=shuffle)
    starting_city = capitals_tsp.path[0]
    print("Initial path value: {:.2f}".format(-capitals_tsp.utility))
    # print(capitals_tsp.path)  # The start/end point is indicated with a yellow star
    # show_path(capitals_tsp.coords, starting_city)

if __name__ == "__main__":
    main()

    