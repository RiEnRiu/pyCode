# a node is a dict with keys={'depth', 'weight', 'sub_trees'}
# it is a leaf node if node['sub_trees'] is a weight value but not a list
# 'depth' means the depth of this tree
# 'weight' means the sum of all leaves in this tree
# 'sub_trees' means all its children


def _find_new_node_index(other_nodes,new_node):
    if len(other_nodes)<=2:
        for i in range(len(other_nodes)-1,-1,-1):
            this_node = other_nodes[i]
            if this_node['weight']>new_node['weight']:
                return i+1
            elif this_node['weight']==new_node['weight'] and this_node['depth']>=new_node['depth']:
                return i+1
        return 0

    half_index = len(other_nodes)//2+1
    left_nodes = other_nodes[:half_index]
    right_nodes = other_nodes[half_index:]
    if left_nodes[-1]['weight']>new_node['weight'] or left_nodes[-1]['weight']==new_node['weight'] and left_nodes[-1]['depth']>=new_node['depth']:
        if new_node['weight']>right_nodes[0]['weight'] or new_node['weight']==right_nodes[0]['weight'] and new_node['depth']<=right_nodes[0]['depth']:
            return half_index
        else:
            found_right = _find_new_node_index(right_nodes,new_node)
            return found_right+half_index
    else:
        found_left = _find_new_node_index(left_nodes,new_node)
        return half_index-found_left

def _insert_new_node(other_nodes,new_node):
    if new_node['weight']>other_nodes[0]['weight'] or new_node['weight']==other_nodes[0]['weight'] and new_node['depth']<=other_nodes[0]['depth']:
        other_nodes.insert(0,new_node)
        return
    if other_nodes[-1]['weight']>new_node['weight'] or other_nodes[-1]['weight']==new_node['weight'] and other_nodes[-1]['depth']>=new_node['depth']:
        other_nodes.insert(-1,new_node)
        return
    
    index = _find_new_node_index(other_nodes,new_node)
    other_nodes.insert(index,new_node)
    return
    
def _make_new_node(k_c):
    weight = 0
    max_depth = -1
    for node in k_c:
        weight+=node['weight']
        max_depth = max_depth if node['depth']<max_depth else node['depth']
    tree = {'depth':max_depth+1,'weight':weight,'sub_trees':k_c}
    return tree

def _combine_last_k_node(sort_c,k):
    if len(sort_c)<=k:
        return _make_new_node(sort_c)
    new_node = _make_new_node(sort_c[-k:])
    other_nodes = sort_c[:-k]
    _insert_new_node(other_nodes,new_node)
    return other_nodes

def _encede_lenth(tree):
    if type(tree['sub_trees'])!=list:
        return 0
    sum_len = tree['weight']
    for sub_tree in tree['sub_trees']:
        sum_len+=_encede_lenth(sub_tree)
    return sum_len

def huffman_encode(n,k,c):
    # nlogn
    c.sort(reverse=True)

    c = [{'depth':0,'weight':v,'sub_trees':v} for v in c]
  
    # n * logn
    while(type(c)!=dict):
        c = _combine_last_k_node(c,k)

    print('balanced tree = {0}'.format(c))

    # logn
    min_len_encoded, min_max_len_s = _encede_lenth(c), c['depth']
        
    return min_len_encoded, min_max_len_s
    
if __name__=='__main__':

    n = 4
    k = 2
    c = [1,1,2,2]
    print('ans = {0}'.format(huffman_encode(n,k,c)))

    
    


