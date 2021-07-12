from RTree import RTree
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
from country import Country

action_set = [1, 2, 3, 4, 5, 0, 6, 8]
country = None

def aprox_time(value):
    return int(0.02276723 * value - 38.54355)

def print_menu():
    print ("CHOOSE ACTION:")
    print ("1. Plot tree")
    print ("2. Add node")
    print ("3. Find")
    print ("4. Save current tree")
    print ("5. Create tree from country index")
    print ("6. Multiple insert from file")
    print ("8. Delete tree")
    print ("0. Exit")

def main(tree):
    
    print_menu()

    action = int(input())

    if not action in action_set: 
        print ("WRONG ACTION")
        return tree
    
    if action == 0:
        tree.save_tree('tmp')
        exit(0)

    elif action == 1:
        print("It can take some time")
        if tree.tree_dim == 2:
            tree.plot2d()
            plt.xlim([-180, 180])
            plt.ylim([-90, 90])
        else:
            ax = plt.axes(projection='3d')
            tree.plot3d(ax)        


        plt.show()
        return tree
    
    elif action == 2:
        print("Input node:")
        print("Example: 2 3 2 3")
        print("or")
        print("Example: 2 3 2 3 2 3")

        inp = list(map(int, input().strip().split(' ')))
        block = np.array(inp)

        try:
            if tree.tree_dim == 2:
                block = np.reshape(block, (2, 2))
            else:
                block = np.reshape(block, (3, 2))
        except ValueError:
            print("WRONG SHAPE !!!")
            return tree


        print(block)

        tree.insert(block)
        return tree

    elif action == 3:
        print("Type block:")
        print("Example: 2 3 2 3")
        inp = list(map(float, input().strip().split(' ')))
        block = np.array(inp)

        try:
            if tree.tree_dim == 2:
                block = np.reshape(block, (2, 2))
            else:
                block = np.reshape(block, (3, 2))
        except ValueError:
            print("WRONG SHAPE !!!")
            return tree

        result = tree.range_search(block)
        print("Result found.")
        print("Plotting...")

        if tree.tree_dim == 2:
            tree.plot2d()
            tree.plot_rec(block, color='b')
            for rec in result:
                tree.plot_rec(rec, color='yellow', ls='-.')
            plt.xlim([-180, 180])
            plt.ylim([-90, 90])

        else:
            tree.plot3d(ax)
            ax.scatter(block[0], block[1], block[2])


        plt.show()
        print("\n\nSEARCH RESULTS:")
        [print(i, end='\n\n') for i in result]

        return tree

    elif action == 4:
        inp = input("Type file name (file):").strip()
        tree.save_tree(inp)
        print("Tree saved")
        exit(0)

    elif action == 8:
        tree.delete()
        exit(0)

    elif action == 6:
        inp = input('Type filename(Example: "file.csv"): ').strip()


        if tree.tree_dim == 2:
            df = pd.read_csv(inp, sep=',', usecols=["x1", "x2", "y1", "y2"])
            df = df.to_numpy().reshape((-1, 2, 2))
        else:
            df = pd.read_csv(inp, sep=',', usecols=["x1", "x2", "y1", "y2", "z1", "z2"])
            df = df.to_numpy().reshape((-1, 3, 2))
        
        print ("Creating...")
        start = time.time()
        for index, block in enumerate(df):
            tree.insert(block)
            if index % 250 == 0:
                print ("{}/{}".format(index, len(df)))
        
        end = time.time()
        print ("{} seconds".format(end - start))
        return tree

    elif action == 5:
        global country
        if country is None:
            country = Country()
        cnt = input("Type country [optional]:")
        pop = input("Type population [optional]:")
            
        
        if pop and cnt:
            if int(pop) <= 0:
                print("Wrong value for population")
                return tree
            try:
                data = country.get_country_with_population_and_name(cnt, int(pop))
            except ValueError as e:
                print(e)
                return tree
    
        elif pop or cnt:
            if pop:
                if int(pop) <= 0:
                    print("Wrong value for population")
                    return tree
                try:
                    data = country.get_country_with_population(int(pop))
                except ValueError as e:
                    print(e)
                    return tree
            elif cnt:
                try:
                    data = country.get_country(cnt)
                except ValueError as e:
                    print(e)
                    return tree
        else:
            print("One of the parametre must be typed")
            return tree

        if tree.tree_dim == 2:
            data = data.to_numpy().reshape((-1, 2, 2))
        else:
            data = data.to_numpy().reshape((-1, 3, 2))
    
        print("Country found.")

        if abs(len(data) // 10 - tree.tree_dim) > 800:
            print("Recomendation:")
            print("Maybe use another maximum amount of children for this data size: {}".format(len(data)))
            print("1. Yes")
            print("2. No")
            inp = int(input())
            if inp == 1:
                tree.max_number_of_children = len(data) // 10
            elif inp == 2:
                pass
            else:
                print ("Wrong value")
                return tree

        print("It will take ~{}:{}".format(aprox_time(len(data)) // 60, aprox_time(len(data)) % 60))

        start = time.time()
        for index, block in enumerate(data):
            tree.insert(block)
            if index % 250 == 0:
                print ("{}/{}".format(index, len(data)))
        print(time.time() - start)
        return tree

    os.system('cls')


if __name__=="__main__":
    os.system('cls')
    print("This program creates RTree\nrepresentation of data.")
    print("Supports working with 2 or 3 dimentional data.")
    print("Data is stored in pickle [*.pkl] binary file.")
    print("Program allowes saving created tree or loading\nexisting one.")
    print("\n\n")
    print ("1. Create empty tree")
    print ("2. Load exising tree")
    inp = int(input())
    if inp == 1:
        inp = int(input("2 or 3 dimentions: "))
        if inp != 2 and inp != 3:
            print("WRONG DIM !!!")
            exit(1)
        inp2 = int(input("Type maximum amount of children for a single node: "))
        if inp2 < 3:
            print("WRONG VALUE !!!")
            exit(1)
        tree = RTree(inp2, tree_dim = inp, folder_name = 'nodes')

    elif inp == 2:

        inp = input('Type name of file (file.pkl): ')

        inp2 = int(input("Type tree depth: "))
        if inp2 < 3:
            print("WRONG VALUE !!!")
            exit(1)
        
        try:
            tree = RTree.loadRTree(int(inp2), folder_name=inp)
        except FileNotFoundError:
            print("TREE STRUCTURE DOESN'T EXIST IN THAT FILE: {}".format(inp))
            exit(1)
            
    else:
        print("WRONG ACTION !!!")
        exit(1)
    while True:
        tree = main(tree)
