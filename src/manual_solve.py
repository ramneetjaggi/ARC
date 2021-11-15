#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

def solve_484b58aa(x):
    """"
    For this solution we see there is a varied length pattern
    found in each column. We get the pattern for each and then go no
    recreating the column using the same pattern.
    """
    y=x.copy();
    rows, columns = np.shape(y)
    unique_colours=np.unique(x)
    #The last color is the pattern length for each column
    pattern_length = unique_colours[-1]

    unique_colours=np.unique(y)
    pattern_length = unique_colours[-1]
    k=0
    for column in range(columns):
        # Traversing column by column
        col_list = y[:, column]
        process_flag = 0
        if 0 in col_list:
           # Process only the ones with 0 or black color
           process_flag = 1

        if process_flag == 1:
            k= 1
            pattern_list=[]
            for i in range(len(col_list)):
                # Looking through column elements to get a complete
                # pattern with breaks or 0
                pattern_list.append(col_list[i])
                if (k%(pattern_length) ==0):
                    if 0 in pattern_list:
                        pattern_list = []
                    else:
                        break
                k +=1
            new_column=create_new_column(pattern_list,columns)
            # Creating the new matrix column by column
            y[:, column]=new_column
    return y

def create_new_column(pattern,column_length):
    new_col_length=0
    new_column=[]
    while (new_col_length<column_length):
        new_column.extend(pattern)
        new_col_length=len(new_column)
        # repeating the pattern through the column length
        # and if there are few spots left substring the pattern
        # Example column_length=29, len(pattern)=6
        # loop repeats to add 6+6+6+6+pattern[:seq_to_extend]
        if( len(pattern)> (column_length - new_col_length)):
            seq_to_extend=column_length - new_col_length
            arr_to_add = pattern[:seq_to_extend]
            new_column.extend(arr_to_add)
            new_col_length=len(new_column)
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
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
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

