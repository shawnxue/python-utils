# Given a Binary Tree, check if all leaves are at same level or not. 
class Node:
  def __init__(self, data):
    self.data = data
    self.left = None
    self.right = None
  
def areAllLeavesSamelevel(root: Node):
  def isSameLevel(root, current_leaf_level):
    if root is None:
      return True
    # found a leaf
    if root.left is None and root.right is None:
      # this is the first time to find a leaf
      if first_leaf_level == 0: 
        first_leaf_level = current_leaf_level
        return True
      # this is not the first time to find leaf
      return current_leaf_level == first_leaf_level
    # not leaf yet
    return (isSameLevel(root.left, current_leaf_level + 1) and isSameLevel(root.right, current_leaf_level + 1))
  
  current_leaf_level = 0
  first_leaf_level = 0
  return isSameLevel(root, current_leaf_level)