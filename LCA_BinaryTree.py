# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

#Solution 1:
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        #Edge Scenario:
        if not root:
            return None

        if root == p or root == q:
            return root

        #Divide:
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        #Conquer:
        if left and right:
            return root
        if not left:
            return right
        if not right:
            return left

#Solution 2:
class Solution:
    def lowestCommonAncestor1(self, root, p, q):
        self.ret = None
        self.helper(root, p, q)
        return self.ret
    
    def helper(self, root, p, q):
        if root is None:
            return 0 
        
        number_of_match = 0
        
        if root == p or root == q:
            number_of_match += 1
        
        #Bottom up:
        l = self.helper(root.left, p, q)
        r = self.helper(root.right, p, q)
        
        number_of_match  = number_of_match + l +r
        if number_of_match == 2 and self.ret is None:
            self.ret = root
            return 2
        else:
            return number_of_match