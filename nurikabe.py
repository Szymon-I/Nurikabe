from itertools import combinations
import copy
from PIL import Image
import numpy as np
import os
from boards import *


class Island:
    def __init__(self, y, x, size):
        self.y = y
        self.x = x
        self.size = size


class Nurikabe:
    def __init__(self, board):
        self.rows = len(board)
        self.columns = len(board[0])
        self.islands = self.gen_islands(board)
        self.board = self.generate_matrix()

    def gen_islands(self, board):
        """
        return list of Island's based on matrix
        """
        islands = []
        for y in range(len(board)):
            for x in range(len(board[0])):
                if board[y][x] != 0:
                    islands.append(Island(y, x, board[y][x]))
        return islands

    def generate_matrix(self):
        """
        return blank nurikabe board
        """
        return [[1] * self.columns for _ in range(self.rows)]

    def get_points(self):

        # get island roots tuples
        island_tuples = set()
        for island in self.islands:
            island_tuples.add((island.y, island.x))

        # check for water points (points between roots)
        water_points = set()
        for y in range(1, self.rows - 1):
            for x in range(1, self.columns - 1):
                if (y - 1, x) in island_tuples and (y + 1, x) in island_tuples or (y, x - 1) in island_tuples and (
                        y, x + 1) in island_tuples:
                    water_points.add((y, x))

        # add neighbours of '1' to water points
        for island in self.islands:
            if island.size == 1:
                y, x = island.y, island.x
                if self.is_valid(self.board, y - 1, x):
                    water_points.add((y - 1, x))
                if self.is_valid(self.board, y + 1, x):
                    water_points.add((y + 1, x))
                if self.is_valid(self.board, y, x - 1):
                    water_points.add((y, x - 1))
                if self.is_valid(self.board, y, x + 1):
                    water_points.add((y, x + 1))

        # select all possible water points
        all_points = []
        for y in range(self.rows):
            for x in range(self.columns):
                if (y, x) not in water_points and (y, x) not in island_tuples:
                    all_points.append((y, x))

        return all_points, list(water_points)

    def get_solution(self):
        """
        find solution for given board and islands roots
        """
        all_points, water_points = self.get_points()

        # calculate quantity of points to pick according to islands sizes and water points
        points_to_pick = self.rows * self.columns - (sum([x.size for x in self.islands]) + len(water_points))
        for comb in combinations(all_points, points_to_pick):
            # create board with actual combination
            actual_board = copy.deepcopy(self.board)
            for x, y in comb:
                actual_board[x][y] = 0
            # add water points to board
            for x, y in water_points:
                actual_board[x][y] = 0
            # validate each combination and return board if valid
            if self.check_board(actual_board):
                return actual_board
        return []

    def isSafe(self, board, row, col, visited):
        """
        check for index error and visited along with island
        """
        return ((row >= 0) and (row < self.rows) and
                (col >= 0) and (col < self.columns) and
                (board[row][col] and not visited[row][col]))

    def DFS(self, board, row, col, visited, count):
        # These arrays are used to get row and column
        # numbers of 4 neighbours of a given cell
        rowNbr = [-1, 0, 0, 1]
        colNbr = [0, -1, 1, 0]

        # Mark this cell as visited
        visited[row][col] = True

        # Recur for all connected neighbours
        for k in range(4):
            if (self.isSafe(board, row + rowNbr[k],
                            col + colNbr[k], visited)):
                # increment region length by one
                count[0] += 1
                self.DFS(board, row + rowNbr[k],
                         col + colNbr[k], visited, count)

    def RegionSize(self, board, i, j):

        # Make a bool array to mark visited cells.
        # Initially all cells are unvisited
        visited = [[0] * self.columns for i in range(self.rows)]

        # Initialize result as 0 and travesle
        # through the all cells of given matrix
        result = 0

        # If a cell with value 1 is not
        if board[i][j] and not visited[i][j]:
            # visited yet, then new region found
            count = [1]
            self.DFS(board, i, j, visited, count)
            # maximum region
            result = count[0]
        return result

    def get_number_of_islands(self, board):
        rows = len(board)
        cols = len(board[0])
        # you can use Set if you like
        # or change the content of binaryMatrix as it is visited
        visited = [[0 for col in range(cols)] for r in range(rows)]
        number_of_island = 0
        for row in range(rows):
            for col in range(cols):
                number_of_island += self.get_island(board, row, col, visited)
        return number_of_island

    def is_valid(self, board, row, col):
        """
        check for index error
        """
        rows = len(board)
        cols = len(board[0])
        return 0 <= row < rows and 0 <= col < cols

    def get_island(self, board, row, col, visited):
        """
        flood algorithm for getting whole island
        """
        if not self.is_valid(board, row, col) or visited[row][col] == 1 or board[row][col] == 0:
            return 0

        # mark as visited
        visited[row][col] = 1
        self.get_island(board, row, col + 1, visited)
        self.get_island(board, row, col - 1, visited)
        self.get_island(board, row + 1, col, visited)
        self.get_island(board, row - 1, col, visited)
        island_size = 0
        for x in range(row):
            for y in range(col):
                if visited[x][y] == 1:
                    island_size += 1
        return 1

    def check_square(self, board):
        """
        check for square water region
        """
        for y in range(self.rows - 1):
            for x in range(self.columns - 1):
                if board[y][x] + board[y + 1][x] + board[y][x + 1] + board[y + 1][x + 1] == 0:
                    return False
        return True

    def check_sizes(self, board):
        """
        check if islands sizes are equal to root values
        """
        for island in self.islands:
            if self.RegionSize(board, island.y, island.x) != island.size:
                return False
        return True

    def check_quantity(self, board):
        """
        check for islands quantity
        """
        return self.get_number_of_islands(board) == len(self.islands)

    def check_board(self, board):
        # check for square water
        if not self.check_square(board):
            return False
        # check each island size
        if not self.check_sizes(board):
            return False
        # check for island quantity
        if not self.check_quantity(board):
            return False
        return True

    def save_board(self, board, file_name):
        """
        convert matrix to bitmap and save as png
        """
        w, h = len(board), len(board[0])
        data = np.ones((h, w), dtype=np.uint8) * 255
        np_board = np.array(board)
        data = data * np_board
        img = Image.fromarray(data.astype(np.uint8), mode='L')
        img.save(os.path.join('images', f'{file_name}.png'))
        img.show()

    @staticmethod
    def solve_all():
        for i, board in boards.items():
            n = Nurikabe(board)
            solution = n.get_solution()
            if solution:
                print(f'Solved board_{i}')
                n.save_board(solution, f'board_{i}')
            else:
                print(f'No solution for board_{i}')


if __name__ == '__main__':
    Nurikabe.solve_all()
