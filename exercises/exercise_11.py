# print a tree in preorder (root, left, right)
def pre_order_print(root):
  if root == None:
    return
  print(root.data, end=" ")
  pre_order_print(root.left)
  pre_order_print(root.right)
# given an inorder array and a postorder array, construct a binary tree
def create_tree(inorder, postorder):
  class TreeNode():
    def __init__(self, data):
      self.data = data
      self.left = None
      self.right = None
  def build_tree_1(in_array, post_array, in_start_index, in_end_index, current_root_index):
    # base case
    if in_start_index > in_end_index:
      return None
    # current_root_index is the last index in post array, use it to create a node
    node = TreeNode(post_array[current_root_index])
    current_root_index -= 1
    # return the node if it has no child nodes
    if in_start_index == in_end_index:
      return node
    # find this node's index in inorder array, so that we can recursive
    in_index = in_array.index(node.data)
    # use this index in inorder array to construct left and right subtree
    node.right = build_tree_1(in_array, post_array, in_index + 1, in_end_index, current_root_index)
    node.left = build_tree_1(in_array, post_array, in_start_index, in_index - 1, in_index - 1)
    return node
  
  def build_tree_2(in_array, post_array):
    # base case
    if len(in_array) <= 0 or len(post_array) <= 0:
      return None
    # current_root_index is the last index in post array, use it to create a node
    current_root = TreeNode(post_array[-1])
    # return the node if it has no child nodes
    if len(in_array) == 1:
      return current_root
    # find this node's index in inorder array, so that we can recursive
    in_index = in_array.index(current_root.data)
    in_array_left_tree = in_array[0:in_index]
    post_array_left_tree = post_array[0:len(in_array_left_tree)]
    in_array_right_tree = in_array[in_index + 1:]
    post_array_right_tree = post_array[-len(in_array_right_tree) - 1:-1]
    
    # use this index in inorder array to construct left and right subtree
    current_root.right = build_tree_2(in_array_right_tree, post_array_right_tree)
    current_root.left = build_tree_2(in_array_left_tree, post_array_left_tree)
    return current_root
  
      
  size = len(inorder)
  cur_root_index = size - 1
  #root = build_tree_1(inorder, postorder, 0, size-1, cur_root_index)
  root = build_tree_2(inorder, postorder)
  return root


  
tree = create_tree([4, 8, 2, 5, 1, 6, 3, 7], [8, 4, 5, 2, 6, 7, 3, 1])
pre_order_print(tree)

def create_tree_2(inorder, preorder):
  class TreeNode():
    def __init__(self, data):
      self.data = data
      self.left = None
      self.right = None
  def build_tree_2(in_array, pre_array):
    # base
    if len(in_array) <= 0 or len(pre_array) <= 0:
      return None
    # current root the first element in preorder array
    current_root = TreeNode(pre_array[0])
    if len(in_array) == 1:
      return current_root
    # find the index of current root in inorder array, so that we can split the inorder array
    in_index = in_array.index(current_root.data)
    in_array_left_tree = in_array[0:in_index]
    pre_array_left_tree = pre_array[1:len(in_array_left_tree) + 1]
    in_array_right_tree = in_array[in_index + 1:]
    pre_array_right_tree = pre_array[-len(in_array_right_tree):]
    print(f"in_array_left_tree={in_array_left_tree}")
    print(f"pre_array_left_tree={pre_array_left_tree}")
    print(f"in_array_right_tree={in_array_right_tree}")
    print(f"pre_array_right_tree={pre_array_right_tree}")
    current_root.right = build_tree_2(in_array_right_tree, pre_array_right_tree)
    current_root.left = build_tree_2(in_array_left_tree, pre_array_left_tree)
    return current_root

  root = build_tree_2(inorder, preorder)
  return root

# tree = create_tree_2(["D", "B", "E", "A", "F", "C"], ["A", "B", "D", "E", "C", "F"])
# pre_order_print(tree)