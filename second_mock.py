# ’‘’
# Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

# The lowest common ancestor is defined between two nodes `p` and `q` as the lowest node in `T` that has both `p` and `q` as descendants (where we allow **a node to be a descendant of itself**).
# ‘’‘

# 1. clarifying questions
# 2. sketch idea pseudocode 

#    1
#   2 3
#  4 5

# input root, p, q
# if root.val < p and root.val < q ->
# elif : ->
# else: -> 

'''
def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
      if root is None:
          return None
      val = root.val
      if val < min(p.val, q.val):
          return self.lowestCommonAncestor(root.right, p, q)
      elif val > max(p.val, q.val):
          return self.lowestCommonAncestor(root.left, p, q)
      else:
          return root

'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# root, p, q : TreeNode -> TreeNode 


'''
You are given a nested list of integers `nestedList`. Each element is either an integer or a list whose elements may also be integers or other lists.

The **depth** of an integer is the number of lists that it is inside of. For example, the nested list `[1,[2,2],[[3],2],1]` has each integer's value set to its **depth**.

Return *the sum of each integer in* `nestedList` *multiplied by its **depth***.
'''

class NestedInteger:
   def __init__(self, value=None):
       """
       If value is not specified, initializes an empty list.
       Otherwise initializes a single integer equal to value.
       """
    
   def isInteger(self):
        pass
   def getInteger(self):
        pass
   def getList(self):
        pass
# NestedList = [[1,1],2,[1,1]]
# Sum =. 10 

# Input: List[NestedInteger]
# Output: Integer
from collections import deque
def sumOfNestedInteger(Nlist):
    queue = deque(Nlist) # 1,1,1,1
    sum = 0 # 2 + 2 + 2 + 2 + 2
    depth = 1 # 2
    while queue: 
        for i in range(len(queue)): # 3, 4
            NInt = queue.popleft() # 2, 1
            if NInt.isInteger():
                sum += NInt.getInteger() * depth
            else:
                queue.extend(NInt.getList()) 
        depth += 1
    return sum

#[[[[[[[]]]]]]]
'''
NestedList = [[1,1],2,[1,1]]
queue = [[1,1],2,[1,1]], sum = 0

depth = 1
n1 = [1, 1] -> queue = [2, [1, 1], 1, 1]
n2 = 2 -> queue, sum = 2
n3 = [1, 1, ]-> queue = [1, 1, 1, 1]

depth = 2
n1 = 1 -> queue = [1. 1. 1.] sum = 2 + 2 = 4
n2 = 1 -> queue = [1. 1.] sum = 4 + 2 = 6
n3 = 1 -> queue = [1.] sum = 6 + 2 = 8
'''
# N = # nested element
# O(N)
# O(N)

# f(n) = n + f(n - 1)
# f(n) = 2 * f(n - 1)

# quicksort
# f(n) = O(n) + 2 * f(n/2)

# pivot = f(n)
# partition O(n)
# sort(left) + sort(right) = 2 * f(n/2)

# quickselect

# f(n) = O(n) + f(n/2)
# O(n)
# pivot -> partition -> recurse find(left) find(right)
        
