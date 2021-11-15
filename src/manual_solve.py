#!/usr/bin/python

import os, sys
import json
import numpy as np
import re
import math

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
# def solve_6a1e5592(x):
#     return x
#
# def solve_b2862040(x):
#     return x
#
# def solve_05269061(x):
#     return x
def solve_484b58aa(x):
    transpose_flag = False
    column_first = [row[0] for row in x]
    check_pattern_exists = np.where(np.array(column_first) == 0)
    if len(check_pattern_exists[0]) > 1:
        y = x.copy()
        y = np.transpose(y)
        transpose_flag = True
    else:
        y = x.copy()
    rows, columns = np.shape(y)
    pattern = []
    k = 0
    for row in y:
        process_flag = 0
        for element in row:
            if element == 0:
                process_flag = 1
                break

        if process_flag == 1:
            pattern_size, pattern = findpatternsandsize(row)
            deg = degree_of_rotation(row, pattern)
            new_row = rotateArray(pattern, pattern_size, pattern, deg, rows)
            y[k, :] = new_row

        k += 1
    if (transpose_flag):
        return np.transpose(y)
    else:
        return y
def findpatternsandsize(row):
    row=getstringtofindapattern(row)
    max_len = math.ceil(len(row) / 2)
    for x in range(2, int(max_len)):
        if row[0:x] == row[x:2*x] :
            return x,row[0:x]
            break

def getstringtofindapattern(seq):
    dict = {}
    temp_list=[]
    seq_length=len(seq)
    length_counter=0
    for i in seq:
        if i !=0:
            temp_list.append(i)
            length_counter+=1
        if i == 0:
            dict[len(temp_list)]=temp_list
            temp_list=[]
            length_counter+=1
        if length_counter==seq_length:
            dict[len(temp_list)]=temp_list
    final_list = dict[max(dict)]
    return final_list

# function to rotate array by d elements using temp array
def rotateArray(arr, length_of_pattern,pattern, degree_rotate, length_of_row):
    temp = []
    i = 0
    while (i < degree_rotate):
        temp.append(arr[i])
        i = i + 1
    i = 0
    while (degree_rotate < length_of_pattern):
        arr[i] = arr[degree_rotate]
        i = i + 1
        degree_rotate = degree_rotate + 1
    arr[:] = arr[: i] + temp
    while (len(arr) < length_of_row):
        arr.extend(arr)
        if( length_of_pattern> (length_of_row - len(arr))):
            seq_to_extend=length_of_row - len(arr)
            arr_to_add = pattern[:seq_to_extend]
            arr.extend(arr_to_add)
    return arr
def degree_of_rotation(row,pattern):

    #get first index of a2
    final_index=0
    value = row[0]
    pos = np.where(np.array(pattern) ==value)
    if len(pos[0])!= 1:
        #if first value is duplicated in pattern and second elemnet is 0
        #try one by one if pattern and updated value owuld be equal
        for i in range(len(pos[0])):
            ppos = pos[0][i]
            position=True
            for j in row:
                if j==0 or j==pattern[ppos]:
                    if ppos+1 == len(pattern):
                        ppos=0
                    else:
                        ppos +=1
                else:
                    position=False
                    break
            if position == True:
                final_index = pos[0][i]
                break
    else:
        final_index = pos[0][0]
    return final_index

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

