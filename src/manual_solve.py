#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### Name: Ramneet Jaggi Id: 21252485
### GitHub Repo: https://github.com/ramneetjaggi/ARC.git


# Solve  functions commonalities or differences among them .
"""
Goal: To fill in the blank spaces without breaking pattern.
The 3 tasks that are solved as a part of this assignment had few common grounds 
    * All 3 could be solved with the use use of numpy and basic python loop knowledge 
    * All had a pattern which although different from each other while 
        carefully studying the matrix solution could be formed. 
    
The tasks did have different patterns in their grids which did differentiate the methods or grid traversing techniques
that could be used to fill the blank spaces
    * solve_484b58aa : Had a same pattern running through column , but each column could have a different pattern
        Complexity arose from the fact that the pattern could have repetitions of elements too.
    * solve_73251a56 : Had a diagonal symmetry , there was also symmetry between rows and column due to same. The 
        complexity to this problem had to be the issue where a good piece of chunk would be missing around diagonal hence
        symmetry concept would not work there to fill in the blank cell so had a carefully choose a color around 0 to fill in.
    * solve_c3f564a4 :For this task there was symmetry in rows and columns but not via the diagonal.The pattern was of same
        length and same fashioned list elements through out the matrix. So it was easy to find a pattern that suits the whole grid
        and use the same to find which color to use to fill in the blank.
        
Goal: To fill in the blank spaces without breaking pattern.
To quote from the On the Measure of Intelligence from Francois Chollet -"the only intelligence at work here is the engineer’s"
The solutions that are mentioned in this might not work for edge cases as we assume the patterns and grid size
will be same. The algorithm is trained to solve few test scenarios and might not work well for a vast variety of samples .
Also the measure of success is  binary either the whole grid matches or it does not and the result is True or False respectively.
While initially developing a solution the algorithm might generate a grid as close as 90% matching to the expected output, but that
is not taken into account.
"""


def solve_484b58aa(x):
    """" Description: For this solution we see there is a varied length pattern found in each column. We get the
    pattern for each and then go on recreating the appending the new created column using the same pattern.
    Real challenge was to find a pattern and its size as each each row had its unique pattern and colors.

    Solution: The essence really lies in understanding the pattern. First intuition was that the pattern is in the
    rows but for few cases where the row starts with  multiple 0's or that pattern is not available at one stretch
    the algorithm did not work. Switched to columns for better pattern matching. Key point to note that the number of
    unique colors in a column/row which was also same across matrix was also equal to the pattern length. Once it was
    realised the pattern length to match was easier to find. A handy small method was created
    create_new_column_484b58aa to recreate columns per the pattern.
    * Find the column to be processed which has a zero
    * Create a pattern by looking into a continuous stream of numbers which are non-zero
    * Re create the column with the pattern found and return the new matrix

    Training and testing grid solved: All solved
    Packages used: Numpy and normal python conditional and range loops
    """
    y = x.copy();
    rows, columns = np.shape(y)
    # This will provide us unique colors in the grid in ascending order
    unique_colours = np.unique(x)
    # The last color is the pattern length for each column
    pattern_length = unique_colours[-1]
    k = 0
    for column in range(columns):
        # Traversing column by column
        col_list = y[:, column]
        process_flag = 0
        if 0 in col_list:
            # Process only the ones with 0 or black color
            process_flag = 1

        if process_flag == 1:
            k = 1
            pattern_list = []
            for i in range(len(col_list)):
                # Looking through column elements to get a complete
                # pattern with breaks or 0
                pattern_list.append(col_list[i])
                if (k % pattern_length == 0):
                    if 0 in pattern_list:
                        pattern_list = []
                    else:
                        break
                k += 1
            new_column = create_new_column_484b58aa(pattern_list, columns)
            # Creating the new matrix column by column
            y[:, column] = new_column
    return y


def solve_73251a56(x):
    """
    Description: The symmetry was a key feature for this problem. The diagonal running from left to right had same
    element on the opposite side.The elements with position matrix[row][column] that are diagonally opposite to
    matrix[column][row] were easy to fill in , if not blank.
    The left over were the ones that did not find a non empty  cell opposite to diagonal element.
    They were usually around the diagonal and by looking around matrix[row][column] the ones which are not blank
    or 0 and if not a diagonal we get the color we want to fill in the blank spot with.

    Solution:
    Steps Taken to fill the matrix:
        * Finding the diagonal element and then using np.diagonal
        * As there is symmetry filling in y[j][i]=y[i][j] and vice versa if not 0
        * Else calculating the diagonal periphery looking at 6 elements near y[i,j]
            which is not diagonal or 0

    Training and testing grid solved: All solved
    Packages used: Numpy and normal python conditional and range loops
    """
    y = x.copy()
    rows, columns = np.shape(y)
    diagonal_elements = np.unique(y.diagonal())
    # As diagonal elements are same finding a unique color except 0
    diagonal = [i for i in diagonal_elements if i > 0]
    # filling the diagonal across with the color found above
    np.fill_diagonal(y, diagonal)
    for i in range(rows):
        for j in range(columns):
            if y[i][j] == 0 and y[j][i] != 0 and i != j:
                y[i][j] = y[j][i]
            elif y[j][i] == 0 and y[i][j] != 0 and i != j:
                y[j][i] = y[i][j]
            elif i != j and y[i][j] == 0 and y[j][i] == 0:
                fill_element = calculate_diagnol_periphery_73251a56(i, j, y, diagonal)
                y[i][j] = fill_element
                y[j][i] = fill_element
    return y


def solve_c3f564a4(x):
    """
    Description: For this solution we see there is a there was a same length of pattern re-occurring in the whole
    matrix .The ways which could be used solve this one- there is symmetry in rows and columns , also pattern repeats
    with the same length over rows and columns .For example if row 1 is equivalent to column1 and so on. Or the
    second way which was implemented below was to search for the a non blank/ not 0 cell either by checking y[row-1,column]
    and match was found for the same in the pattern and we populate the blank cell with one color ahead per the
    pattern.

    Solution:
    Steps Taken to fill the matrix
    * Find a row with a non zero/blank cell and this gives us the pattern for the grid
    * To fill a blank cell ,find a non zero/ not a blank cell by checking grid[row - 1, column]
    * When a color is found / a cell with not a 0 match it to the same in pattern
    * Fill the blank cell with pattern[position found +1]

    Training and testing grid solved: All solved
    Packages used: Numpy and normal python conditional and range loops
    """
    y = x.copy()
    rows, columns = np.shape(y)
    pattern = []
    for row in y:
        process_flag = 0
        for element in row:
            if element == 0:
                process_flag = 1
        if process_flag == 0:
            ## As pattern is same across the matrix
            ## row with a non zero would contain pattern for the whole matrix
            pattern = row

    for row in range(rows):
        for column in range(columns):
            if y[row, column] == 0:
                if y[row - 1, column] != 0:
                    notZeroColor = y[row - 1, column]
                    for i in range(len(pattern) - 1):
                        ## Finding the color in the pattern which wasnt 0(black)
                        if pattern[i] == notZeroColor:
                            # Populating the blank cell by looking into the pattern
                            y[row, column] = pattern[i + 1]
    return y


def calculate_diagnol_periphery_73251a56(i, j, y, diagonol):
    """Looking at 6 directions to the current y[i,j] location
    which is not equal to the diagonal element and append it to the list"""
    near_diagnol_element = []
    if (y[i + 1][j] != 0 and y[i + 1][j] != diagonol):
        near_diagnol_element.append(y[i + 1][j])
    elif (y[i - 1][j] != 0 and y[i - 1][j] != diagonol):
        near_diagnol_element.append(y[i - 1][j])
    elif (y[i][j + 1] != 0 and y[i][j + 1] != diagonol):
        near_diagnol_element.append(y[i][j + 1])
    elif (y[i][j - 1] != 0 and y[i][j - 1] != diagonol):
        near_diagnol_element.append(y[i][j - 1])
    elif (y[i + 2][j - 1] != 0 and y[i + 2][j - 1] != diagonol):
        near_diagnol_element.append(y[i + 2][j - 1])
    elif (y[i - 1][j + 2] != 0 and y[i - 1][j + 2] != diagonol):
        near_diagnol_element.append(y[i - 1][j + 2])
    # Return unique color found
    return max(set(near_diagnol_element), key=near_diagnol_element.count)


def create_new_column_484b58aa(pattern, column_length):
    new_col_length = 0
    new_column = []
    while (new_col_length < column_length):
        new_column.extend(pattern)
        new_col_length = len(new_column)
        # Repeating the pattern through the column length
        # and if there are few spots left substring the pattern
        # Example column_length=29, len(pattern)=6
        # loop repeats to add 6+6+6+6+pattern[:seq_to_extend]
        if (len(pattern) > (column_length - new_col_length)):
            seq_to_extend = column_length - new_col_length
            arr_to_add = pattern[:seq_to_extend]
            new_column.extend(arr_to_add)
            new_col_length = len(new_column)
    return new_column


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})"
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals():
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1)  # just the task ID
            solve_fn = globals()[name]  # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)


def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""

    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)


def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    if y.shape != yhat.shape:
        print(f"False. Incorrect shape: {y.shape} v {yhat.shape}")
    else:
        print(np.all(y == yhat))


if __name__ == "__main__": main()
