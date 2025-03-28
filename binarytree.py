from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def level_order(root):
    if not root:  # Edge case: empty tree
        return []
    
    queue = deque([root])  # Start with root node in queue
    result = []  # Stores the final level-by-level result
    
    while queue:  # While nodes remain to process
        level_size = len(queue)  # Nodes in current level
        current_level = []  # Stores values at current level
        
        for _ in range(level_size):  # Process all nodes in current level
            node = queue.popleft()  # Get the next node
            current_level.append(node.val)  # Add its value
            
            # Add children to queue for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)  # Add current level to result
    
    return result

# Test Data 1: Simple binary tree
#       1
#      / \
#     2   3
#    / \   \
#   4   5   6
root1 = TreeNode(1)
root1.left = TreeNode(2)
root1.right = TreeNode(3)
root1.left.left = TreeNode(4)
root1.left.right = TreeNode(5)
root1.right.right = TreeNode(6)

print("Test 1 Output:", level_order(root1))
# Expected: [[1], [2, 3], [4, 5, 6]]

# Test Data 2: Empty tree
root2 = None
print("Test 2 Output:", level_order(root2))
# Expected: []

# Test Data 3: Single node tree
#       1
root3 = TreeNode(1)
print("Test 3 Output:", level_order(root3))
# Expected: [[1]]

# Test Data 4: Right-skewed tree
#       1
#        \
#         2
#          \
#           3
root4 = TreeNode(1)
root4.right = TreeNode(2)
root4.right.right = TreeNode(3)
print("Test 4 Output:", level_order(root4))
# Expected: [[1], [2], [3]]

# Test Data 5: Full binary tree
#        1
#      /   \
#     2     3
#    / \   / \
#   4   5 6   7
root5 = TreeNode(1)
root5.left = TreeNode(2)
root5.right = TreeNode(3)
root5.left.left = TreeNode(4)
root5.left.right = TreeNode(5)
root5.right.left = TreeNode(6)
root5.right.right = TreeNode(7)
print("Test 5 Output:", level_order(root5))
# Expected: [[1], [2, 3], [4, 5, 6, 7]]
