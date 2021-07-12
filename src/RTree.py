import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import os
from numba import jit


class Node:
    def __init__(self, number_of_children, parent_id, id_=0):
        self.id = id_
        self.parent = parent_id
        self.amount_of_children = 0
        self.children = []
        self.leaf = False
        self.mbr = None


@jit(nopython=True)
def mbr_support_2d(bb, bb_, block_):
    bb[0, 0] = min(bb_[0, 0], block_[0, 0])
    bb[0, 1] = max(bb_[0, 1], block_[0, 1])
    bb[1, 0] = min(bb_[1, 0], block_[1, 0])
    bb[1, 1] = max(bb_[1, 1], block_[1, 1])
    return bb


@jit(nopython=True)
def mbr_support_3d(bb, bb_, block_):
    bb[0, 0] = min(bb_[0, 0], block_[0, 0])
    bb[0, 1] = max(bb_[0, 1], block_[0, 1])
    bb[1, 0] = min(bb_[1, 0], block_[1, 0])
    bb[1, 1] = max(bb_[1, 1], block_[1, 1])
    bb[2, 0] = min(bb_[2, 0], block_[2, 0])
    bb[2, 1] = max(bb_[2, 1], block_[2, 1])
    return bb


@jit(nopython=True)
def find_x_axis_l_sum(tmp_x_1, tmp_x_2, x_axis_l_sum, w):
    for index_i, i in enumerate(tmp_x_1):
        for index_j, j in enumerate(tmp_x_2):
            x_axis_l_sum[index_i, index_j] = i - j
    return x_axis_l_sum / float(w)


@jit(nopython=True)
def find_y_axis_l_sum(tmp_y_1, tmp_y_2, y_axis_l_sum, w):
    for index_i, i in enumerate(tmp_y_1):
        for index_j, j in enumerate(tmp_y_2):
            y_axis_l_sum[index_i, index_j] = i - j
    return y_axis_l_sum / float(w)


@jit(nopython=True)
def find_z_axis_l_sum(tmp_z_1, tmp_z_2, z_axis_l_sum, w):
    for index_i, i in enumerate(tmp_z_1):
        for index_j, j in enumerate(tmp_z_2):
            z_axis_l_sum[index_i, index_j] = i - j
    return z_axis_l_sum / float(w)


@jit(nopython=True)
def area_index(area):
    '''!
    Method finds area for 2d or volume for 3d object
    @param[in] area Is list of data
    @returns area or volume
    '''
    if area.shape[0] == 3:
        return abs((area[0, 1] - area[0, 0]) * (area[1, 1] - area[1, 0]) * (area[2, 1] - area[2, 0]))
    return abs((area[0, 1] - area[0, 0]) * (area[1, 1] - area[1, 0]))


@jit(nopython=True)
def is_intersect_2d_support(mbr1, mbr2):
    if max(mbr1[0, 0], mbr2[0, 0]) >= min(mbr1[0, 1], mbr2[0, 1]) or max(mbr1[1, 0], mbr2[1, 0]) >= min(mbr1[1, 1], mbr2[1, 1]):
        return False
    if mbr1[1, 0] > mbr2[1, 1] or mbr2[1, 0] > mbr1[1, 1] or mbr1[0, 0] > mbr2[0, 1] or mbr2[0, 0] > mbr1[0, 1]:
        return False
    return True


@jit(nopython=True)
def is_intersect_3d_support(mbr1, mbr2):
    if max(mbr1[0, 0], mbr2[0, 0]) >= min(mbr1[0, 1], mbr2[0, 1]) or max(mbr1[1, 0], mbr2[1, 0]) >= min(mbr1[1, 1], mbr2[1, 1]) or max(mbr1[2, 0], mbr2[2, 0]) > min(mbr1[2, 1], mbr2[2, 1]):
        return False
    if mbr1[1, 0] > mbr2[1, 1] or mbr2[1, 0] > mbr1[1, 1] or mbr1[0, 0] > mbr2[0, 1] or mbr2[0, 0] > mbr1[0, 1] or mbr1[2, 0] > mbr2[2, 1] or mbr2[2, 0] > mbr1[2, 1]:
        return False
    return True



class RTree:
    def __init__(self, max_number_of_children = 3, load = False, tree_dim = 2, folder_name = "nodes"):
        self.max_number_of_children = max_number_of_children
        self.folder_name = folder_name
        self.min_number_of_children = 2
        self.last_index = 1

        if not load:
            self.root = Node(max_number_of_children, None, 0)
            self.tree_dim = tree_dim
            c = {
                'id': self.root.id,
                'amount_of_children': self.root.amount_of_children,
                'children': self.root.children if len(self.root.children) > 0 else None,
                'leaf': False,
                'mbr': self.root.mbr,
                'parent': None
            }
            self.tree = pd.DataFrame(data = c, index = [0])
        else:
            self.tree = pd.read_pickle(folder_name)
            self.root = self.read_node_from_df(0)
            self.tree_dim = self.root.mbr.shape[0]
            

    def getParent(self, node):
        '''!
        Method returns parent node fro given node
        @param[in] node Is node
        @returns parent node
        '''
        return self.read_node_from_df(node.parent)


    def getChild(self, node_id):
        '''!
        Method finds node by its id in DataFrame storage
        @param[in] node_id Is id of node
        @returns node
        '''
        return self.read_node_from_df(node_id)

    
    def getChildren(self, node):
        '''!
        Method finds all children for given node
        @param[in] node Is node
        @returns list of children for node
        '''
        return [self.getChild(i) for i in node.children]


    @staticmethod
    def loadRTree(depth, folder_name):
        '''!
        Method loads tree
        @param[in] depth Is maximum amount of children for a single node
        @param[in] folder_name Is file name
        @returns instance of class RTree
        '''
        return RTree(depth, True, folder_name = folder_name)


    @staticmethod
    def plot_rec(rectangle, color='b', ls='-', lw=3.0):
        '''!
        Method adds constructs rectangle from given block
        @param[in] rectange Is block of data
        @param[in] color Is plotting color
        @param[in] ls Is line style for plotting
        @param[in] lw Is line width for plotting
        '''
        rec = patch.Rectangle((rectangle[0][0], rectangle[1][0]), abs(rectangle[0][1]-rectangle[0][0]), abs(rectangle[1][0]-rectangle[1][1]), linewidth=1, facecolor='none', lw=lw, edgecolor=color, linestyle=ls)

        plt.gca().add_patch(rec)
        plt.text(rectangle[0][0], rectangle[1][0], 'o', color="g")


    def insert(self, block, node = None):
        '''!
        Method inserts block in tree and DataFrame storage
        @param[in] block Is block which needed to be inserted
        @param[in] node Is starting node, should be None to start in root
        @returns parent node for given node and True if node didn't have overflow, False otherwise
        '''
        if node == None:
            node = self.root
            if node.amount_of_children == 0: # It means we have no nodes for now

                block = np.array(block)
                if block.shape[0] != self.tree_dim:
                    raise ValueError("WRONG DIMENTION")

                buff = Node(self.max_number_of_children, node.id, self.last_index)

                self.last_index += 1
                buff.leaf = True
                buff.children.append(block)
                buff.amount_of_children += 1
                buff.mbr = block

                node.children.append(buff.id)
                node.amount_of_children += 1
                node.mbr = buff.mbr

                self.add_node_to_df(node)
                self.add_node_to_df(buff)

                return True

        if node.leaf == True:
            # Control overflow
            if node.amount_of_children == self.max_number_of_children:
                node.children.append(block)
                new_nodes = self.split_node(node.children)


                new_nodes[0].parent = node.parent
                new_nodes[1].parent = node.parent

                
                new_nodes[0].leaf = True
                new_nodes[1].leaf = True

                node_parent = self.getParent(node)
                node_parent.children.remove(node.id)
                node_parent.children.append(new_nodes[0].id)
                node_parent.children.append(new_nodes[1].id)
                node_parent.amount_of_children += 1

                node_parent.mbr = self.find_right_mbr_for_inmemory_node(node_parent, [i.mbr for i in new_nodes])

                self.add_node_to_df(new_nodes[0])
                self.add_node_to_df(new_nodes[1])
                self.add_node_to_df(node_parent)
                return [node_parent, False]

            node.children.append(np.array(block))
            node.mbr = self.find_right_mbr_for_inmemory_node(node, node.children)
            node.amount_of_children += 1

            parent_node = self.getParent(node)
            parent_node.mbr = self.find_right_mbr_for_inmemory_node(parent_node, [node.mbr])

            self.add_node_to_df(node)
            return [parent_node, True]


        node_children = self.getChildren(node)
        right_subtree = self.choose_leaf(node, block, node_children)
        node, status = self.insert(block, right_subtree)
        if not status:
            if node.amount_of_children > self.max_number_of_children: # Current node has overflow

                new_nodes = self.split_node(node, False, self.getChildren(node))

                if node.id == 0: # It's root-node
                    new_root = Node(self.max_number_of_children, None, 0)
                    new_root.children.append(new_nodes[0].id)
                    new_root.children.append(new_nodes[1].id)
                    new_root.amount_of_children += 2

                    new_nodes[0].parent = new_root.id
                    new_nodes[1].parent = new_root.id

                    self.add_node_to_df(new_nodes[0])
                    self.add_node_to_df(new_nodes[1])


                    new_root.mbr = self.find_right_mbr_for_inmemory_node(new_root, [i.mbr for i in new_nodes])
                    self.add_node_to_df(new_root)
                    
                    self.root = new_root

                    return True
                else:
                    new_nodes[0].parent = node.parent
                    new_nodes[1].parent = node.parent

                    node_parent = self.getParent(node)
                    node_parent.children.remove(node.id)
                    node_parent.children.append(new_nodes[0].id)
                    node_parent.children.append(new_nodes[1].id)
                    node_parent.amount_of_children += 1

                    [self.add_node_to_df(i) for i in new_nodes]

                    self.add_node_to_df(node_parent)
        
                return [node_parent, False]

        if node.id != 0:
            parent_node = self.getParent(node)
            parent_node.mbr = self.find_right_mbr_for_inmemory_node(parent_node, [node.mbr])

        if node.id == 0:
            self.root = node

        self.add_node_to_df(node)
        return [node, True]


    def choose_leaf(self, node, block, children = None):
        '''!
        Method chooses the most suitable node for block
        @param[in] node Is node
        @param[in] block Is block
        @param[in] children Is list of nodes
        '''
        best_choise = []
        if children == None:
            children = self.getChildren(node)
        

        for i in range(node.amount_of_children):
            best_choise.append(self.find_area_enlargement(children[i].mbr, block))

        return children[best_choise.index(min(best_choise))]


    def find_right_mbr_for_node(self, node):
        '''!
        Method find mbr(minimum bounding rectangle) for node stored in DataFrame storage
        @param[in] node Is node
        @returns mbr(minimum bounding rectangle)
        '''
        if not node.leaf:
            children = self.getChildren(node)
        else:
            children = node.children

        if node.leaf == True:
            final_mbr = node.mbr
            for block in children:
                final_mbr = self.find_right_mbr(final_mbr, block)
        else:
            final_mbr = node.mbr
            for block in children:
                final_mbr = self.find_right_mbr(final_mbr, block.mbr)

        return final_mbr

    
    def find_right_mbr_for_inmemory_node(self, node, children):
        '''!
        Method finds right mbr (minimum bounding rectangle) for given node
        @param[in] node Is node for which we want to find right mbr
        @param[in] children Is block which is added to node
        @returns 
        '''
        final_mbr = node.mbr
        for child in children:
            final_mbr = self.find_right_mbr(final_mbr, child)
        return final_mbr


    def find_bigger_dist(self, values):
        '''!
        Method finds blocks for which distance between them is the biggest.
        @param[in] values Is list of blocks
        @returns two block with biggest distance
        '''
        results = []
        axesZ = False

        tmp = np.array(values)

        w_x = abs(tmp[:, 0, 1].max() - tmp[:, 0, 0].min())
        w_y = abs(tmp[:, 1, 1].max() - tmp[:, 1, 0].min())

        w = max(w_x, w_y)


        if tmp.shape[1] == 3:
            axesZ = True
            w_z = abs(tmp[:, 2, 1].max() - tmp[:, 2, 0].min())
            w = max(w, w_z)


        tmp_x_1 = tmp[:, 0, 0]
        tmp_x_2 = tmp[:, 0, 1]
        x_axis_l_sum = np.empty([len(values), len(values)])
        x_axis_l_sum = find_x_axis_l_sum(tmp_x_1, tmp_x_2, x_axis_l_sum, w)


        tmp_y_1 = tmp[:, 1, 0]
        tmp_y_2 = tmp[:, 1, 1]
        y_axis_l_sum = np.empty([len(values), len(values)])        
        y_axis_l_sum = find_y_axis_l_sum(tmp_y_1, tmp_y_2, y_axis_l_sum, w)


        if axesZ:
            tmp_z_1 = tmp[:, 2, 0]
            tmp_z_2 = tmp[:, 2, 1]

            z_axis_l_sum = np.empty([len(values), len(values)])
            z_axis_l_sum = find_z_axis_l_sum(tmp_z_1, tmp_z_2, z_axis_l_sum, w)

            overall_l = np.sum((x_axis_l_sum, y_axis_l_sum, z_axis_l_sum), axis=0)

        else:
            overall_l = np.sum((x_axis_l_sum, y_axis_l_sum), axis=0)

        index = np.where(overall_l == overall_l.max())
        index = (index[0][0], index[1][0])
        

        return tmp[index[0]], tmp[index[1]]


    def find_better_node(self, node1, node2, value):
        '''!
        Method chooses which node needs smaller inlargement of mbr(minimum bounding rectangle) to absorb block
        @param[in] node1 Is first node
        @param[in] node2 Is second node
        @param[in] value Is block
        @returns better node
        '''
        if self.find_area_enlargement(node1.mbr, value) < self.find_area_enlargement(node2.mbr, value):
            return node1
        else:
            return node2


    def split_node(self, values, leaf_type = True, children = None):
        '''!
        Method splits values between two nodes
        @param[in] values Is list of values we need to split or node
        @param[in] leaf_type Is argument which indicates if values are from leaf node or non-leaf node
        @param[in] children Is list of childrens mbr, is used for non-leaf node
        @returns two new nodes with splited values
        '''

        first = Node(self.max_number_of_children, None, self.last_index)
        second = Node(self.max_number_of_children, None, self.last_index + 1)

        self.last_index += 2

        if not leaf_type:

            values = children

            values_ = []
            for i in values:
                values_.append(np.array(i.mbr))


            block1, block2 = self.find_bigger_dist(values_)

            node_children = {
                'first' : [],
                'second' : []
            }


            for index, i in enumerate(values):
                if self.min_number_of_children - first.amount_of_children == len(values) - index:
                    for j in range(index, len(values)):
                        first.children.append(values[j].id)
                        values[j].parent = first.id
                        first.mbr = self.find_right_mbr_for_inmemory_node(first, [values[j].mbr])
                        first.amount_of_children += 1
                    break

                elif self.min_number_of_children - second.amount_of_children == len(values) - index:
                    for j in range(index, len(values)):
                        second.children.append(values[j].id)
                        values[j].parent = second.id
                        second.mbr = self.find_right_mbr_for_inmemory_node(second, [values[j].mbr])
                        second.amount_of_children += 1
                    break

                right_node = self.find_better_node(first, second, i.mbr)
                if right_node != False:
                    right_node.children.append(i.id)
                    if right_node == first:
                        node_children['first'].append(index)
                        right_node.mbr = self.find_right_mbr_for_inmemory_node(right_node, [values[k].mbr for k in node_children['first']])
                    else:
                        node_children['second'].append(index)
                        right_node.mbr = self.find_right_mbr_for_inmemory_node(right_node, [values[k].mbr for k in node_children['second']])

                    i.parent = right_node.id
                    right_node.amount_of_children += 1

            for i in values:
                self.add_node_to_df(i)
            
        else: # It's leaf
            block1, block2 = self.find_bigger_dist(values)

            first.leaf = True
            first.mbr = block1

            second.leaf = True
            second.mbr = block2

            for index, block in enumerate(values):

                if self.min_number_of_children - len(first.children) == len(values) - index:
                    for j in range(index, len(values)):    
                        first.children.append(values[j])
                        first.amount_of_children += 1
                        first.mbr = self.find_right_mbr_for_inmemory_node(first, [values[j]])
                    break
            
                elif self.min_number_of_children - len(second.children) == len(values) - index:
                    for j in range(index, len(values)):
                        second.children.append(values[j])
                        second.amount_of_children += 1
                        second.mbr = self.find_right_mbr_for_inmemory_node(second, [values[j]])
                    break

                right_node = self.find_better_node(first, second, block)
                right_node.children.append(block)
                right_node.amount_of_children += 1
                right_node.mbr = self.find_right_mbr_for_inmemory_node(right_node, [block])

        first.amount_of_children = len(first.children)
        second.amount_of_children = len(second.children)

        return first, second


    def find_right_mbr(self, bb_, block_):
        '''!
        Method finds rectangle which absorbs all given blocks
        @param[in] bb_ Is list of blocks
        @param[in] block_ Is block
        @returns new block
        '''
        if bb_ is None:
            return block_

        if bb_.shape[0] == 2:
            bb = np.empty([2, 2])
            return mbr_support_2d(bb, bb_, block_)
    
        bb = np.empty([3, 2])
        return mbr_support_3d(bb, bb_, block_)


    def find_delta(self, bb, tmp):
        '''!
        Method finds difference between two areas or volumes
        @param[in] bb  Is beginning block
        @param[in] tmp Is result block
        @returns difference between blocks 
        '''
        if tmp is None:
            return area_index(bb)
        return area_index(bb) - area_index(tmp)


    def find_area_enlargement(self, bb, tmp):
        '''!
        Method finds enlargement of block which is needed to absorb another block
        @param[in] bb Is block which we enlarge
        @param[in] tmp Is block which must be absorbed
        @returns additionas size for bb
        '''
        buff = self.find_right_mbr(bb, tmp)
        return self.find_delta(buff, bb)


    def is_intersect(self, mbr1, mbr2):
        '''!
        Method finds out if two block instersect or not
        @param[in] mbr1 Is first block
        @param[in] mbr2 Is second block
        @returns True if blocks intersect, False otherwise
        '''
        if self.tree_dim == 2:
            return is_intersect_2d_support(mbr1, mbr2)
        elif self.tree_dim == 3:
            return is_intersect_3d_support(mbr1, mbr2)


    def range_search(self, query, node = None):
        '''!
        Method finds all blocks from tree which have intersection with given block
        @param[in] query Is block we're searching for
        @param[in] node Is starting node, should be None
        @returns list of block from tree
        '''
        if node == None:
            node = self.root
        answer = []
        if node.leaf == False:
            '''examine each entry e of RN to find those e.mbr that intersect Q '''
            '''foreach such entry e call RangeSearch(e.ptr,Q)'''
            for child in self.getChildren(node):
                if self.is_intersect(child.mbr, query):
                    answer = answer + self.range_search(query, child)

        else:
            '''examine all entries e and find those for which e.mbr intersects Q '''
            '''add these entries to the answer set A'''
            for child in node.children:
                if self.is_intersect(child, query):
                    answer.append(child)

        return answer


    def add_node_to_df(self, node):
        '''!
        Method adds node to DataFrame storage
        @param[in] node Is node
        '''
        c = {
            'id': node.id,
            'amount_of_children' : node.amount_of_children,
            'children': list() if node.children is None else node.children,
            'mbr': None if node.mbr is None else node.mbr,
            'leaf': node.leaf,
            'parent': node.parent
        }
        buff = self.tree.loc[self.tree['id'] == node.id]
        if buff.empty:
            tmp = pd.DataFrame([c], index=[0])
            self.tree = self.tree.append(tmp, ignore_index=True)
        else:
            self.tree.at[node.id, 'id'] = node.id
            self.tree.at[node.id, 'amount_of_children'] = node.amount_of_children
            self.tree.at[node.id, 'children'] = node.children
            self.tree.at[node.id, 'leaf'] = node.leaf
            self.tree.at[node.id, 'mbr'] = node.mbr
            self.tree.at[node.id, 'parent'] = node.parent


    def read_node_from_df(self, node_id):
        '''!
        Method reads one node from DataFrame storage by nodes id
        @param[in] node_id Is id of node
        '''
        tmp = self.tree.to_numpy()[node_id]
        buff = Node(self.max_number_of_children, tmp[5], tmp[0])
        buff.amount_of_children = tmp[1]
        buff.children = tmp[2]
        buff.leaf = tmp[3]
        buff.mbr = tmp[4]
        return buff


    def plot2d(self, node=None):
        '''!
        Method adds all 2d data stored in leaf nodes to plotting
        @param[in] node Is starting node, should be None
        '''
        if node == None:
            node = self.root
        if node.leaf:
            for child in node.children:
                self.plot_rec(child, color='r', lw=5.0)
            return
        for child in node.children:
            self.plot2d(self.getChild(int(child)))


    def plot3d(self, ax, node=None):
        '''!
        Method adds all 3d data stored in leaf nodes to plotting
        @param[in] ax Is matplotlib.axes
        @param[in] node Is starting node, should be None
        '''
        if node == None:
            node = self.root
        if node.leaf:
            for child in node.children:
                ax.scatter(child[0], child[1], child[2])
            return
        for child in node.children:
            self.plot3d(ax, self.getChild(int(child)))
        if node == self.root:
            plt.show()


    def save_tree(self, folder_name):
        '''!
        Method saves DataFrame storage to pickle file
        @param[in] folder_name Is file name
        '''
        self.tree.to_pickle('{}.pkl'.format(folder_name))


    def delete(self):
        '''!
        Method deletes current tree and DataFrame storage
        '''
        try:
            os.remove(self.folder_name)
        except FileNotFoundError:
            print("Couldn't delete file, because it wasn't found")
