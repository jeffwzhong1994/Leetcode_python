
# Binary Tree: A tree has two children node so it called binary
# Defintion: At most two children node.

class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None

#Let's do pre-order traversal:

def preorder_traversal(self, root):
	res = []
	self.helper(root, res)
	return res

def Pre_order_helper(self, root, res):
	if not root:
		return 
	res.append(root.val)
	self.helper(root.left, res)
	self.helper(root.right, res)

#Let's do In-order traversal:

def In_order_helper(self, root, res):
	if not root:
		return 
	self.helper(root.left, res)
	res.append(root.val)
	self.helper(root.right, res)

#Let's do Post-order traversal:

def Post_order_helper(self, root, res):
	if not root:
		return
	self.helper(root.left, res)
	self.helper(root.right, res)
	res.append(root.value)

#Get the height of the tree:

def get_height(root):
	if not root:
		return 0
	left = get_height(root.left)
	right = get_height(root.right)
	return 1 + max(left, right)

"""
Ok, let's start doing real world problem:
Q: print the binary tree in the level order.
10 
5, 15
2, 7, 12, 20
1
"""

def level(root):
	q = [root] #current level
	next = [] #next level below current
	line = [] # content to print
	while q:
		head = q.pop(0)
		if head.left:
			next.append(head.left)
		if head.right:
			next.append(head.right)
		line.append(head.val)
		if not q:
			print(line)
			if next:
				q = next
				next = []
				line = []

"""
Given a binary tree where all the right nodes are either leaf nodes with a sibling
(a left node that shares the same parent node) or empty, flip it upside down and
turn it into a tree where the original right nodes turned into left leaf nodes.
return the new root.


Think about Reversing the Linked List:
curr.next.next = curr
curr.next = None
"""

def upside_tree(root):
	if not root:
		return root
	if not root.left and not root.right:
		return root
	left_tree = upside_tree(root.left)
	root.left.left = root.right
	root.left.right= root
	root.left = None
	root.right= None
	return left_tree

"""
For each node in a tree, store the number of ndoes in its left child subtree.

"""
def change_subtree(node):
	if node is None:
		return 0
	left_total = change_subtree(node.left)
	right_total =  change_subtree(node.right)
	node.total_left = left_total
	return 1 + left_total + right_total

"""
Find the node with the max difference in the total number of descendents
in its left subtree and right subtree.

"""

global_max = -1
res = None

def node_diff(root):
	if root is None:
		return 0
	left_total = node_diff(root.left)
	right_total = node_diff(root.right)
	global global_max
	global res
	if abs(left_total - right_total) > global_max:
		global_max = abs(left_total - right_total)
		res = root
	return left_total + right_total + 1

def get_max_diff(root):
	global global_max
	global res
	node_diff(root)
	return res

node_diff(root)

class Solution:
	def __init__(self):
		self.x = 0
	def add(y):
		self.x += y
		return self.x

s = Solution()
s.add(3)
s.add(5)

class TreeNode:
	def __init__(self, x):
		self.val = x
		self.left = None
		self.right = None

class ResultWrapper:
	def __init__(self):
		self.global_max = -1
		self.solution = None

	def max_diff_node(root, res):
		if not root:
			return 0
		left_total = max_diff_node(root.left, res)
		right_total = max_diff_node(root.right, res)
		if abs(left_total - right_total) > res.global_max:
			res.global_max = abs(left_total - right_total)
			res.solution = root
		return left_total + right_total + 1

	def max_diff(root):
		res = ResultWrapper()
		max_diff_node(root, res)
		return res.solution

root = TreeNode(0)
root.left = TreeNode(1)
root.left.left = TreeNode(3)
root.left.right = TreeNode(4)
root.right = TreeNode(2)
root.right.right = TreeNode(5)
root.right.right.right = TreeNode(6)

res = max_diff(root)
print(res)


"""
Get Minimum depth of the tree

"""
def getHeight(root):
	if root is None:
		return 0
	if root.left is None and root.right is None: #Base Case, left node
		return 1
	left = getHeight(root.left) if root.left else float('inf')
	right = getHeight(root.right) if root.right else float('inf')
	return min(left, right) + 1

"""
Tree + Recursion:

Binary Search Tree: for every single node in the tree, the values in its 
left subtree are all smaller than its value, and the values in its right 
subtree are all larger than its value.

"""
def isValidBST(root):
	"""
	:type root: TreeNode
	:rtype: bool
	"""

	prev = [None]
	res = [True]
	inorder(root, pre, res)
	return res[0]

def inorder(root, prev, res):
	if not root:
		return
	inorder(root.left, prev, res)
	if prev[0] and prev[0] >= root.val:
		res[0] = False
	prev[0] = root.val
	inorder(root.right, prev, res)

def isValidBST(root):
	return isBSTHelper(root)[0]

def isBSTHelper(root):
	if not root:
		return (True, None, None)
	left_res = isBSTHelper(root.left)
	right_res = isBSTHelper(root.right)

	#Determine if left_res and right_res is BST, if not return False:
	if not left_res[0] or not right_res[0]:
		return(False, None, None)

	#Determine if root.val is larger than left_res max, if not return False:
	if left_res[2] and root.val <= left_res[2]:
		return(False, None, None)

	#Determine if root.val is smaller than right_res min, if not return False:
	if right_res[1] and root.val > right_res[1]:
		return(False, None, None)

	#Return(True, root min, root max):
	return (True, left_res[1] or root.val, right_res[2] or root.val)

#Queue exercise:
class Queue:
	def __init__(self):
		self.s1 = []
		self.s2= []
		self.head = None

	def enqueue(self, x):
		if len(self.s1) == 0:
			self.head = x
		self.s1.append(x)

	def dequeue(self):
		if len(self.s2) == 0:
			while self.s1:
				self.s2.append(self.s1.pop())
		return self.s2.pop()

	def is_empty(self):
		return not self.s1 and not self.s2

	def peek(self):
		if self.s2:
			retrun self.s2[len(s2)-1]
		return self.head

	def size(self):
		return len(self.s1) + len(self.s2)

"""
Non-recursion tree traversal(very important):

"""
def preorder_traversal(root):
	output = []
	if not root:
		return output
	stack = [(root,1)]
	while stack:
		node, count = stack.pop()
		if count == 1:
			output.append(node.val)
			stack.append((node, count + 1))
			if node.left:
				stack.append((node.left, 1))

		if count == 2:
			if node.right:
				stack.append((node.right, 1))

	return output

def inorder_traversal(root):
	output = []
	if not root:
		return output
	stack = [(root, 1)]
	while stack:
		node, count = stack.pop()
		if count == 2:
			output.append(node.val)
			if node.right:
				stack.append((node.right, 1))

		if count == 1:
			stack.append((node, count + 1))
			if node.left:
				stack.append((node.left, 1))

def postorder_traversal(root):
	output = []
	if not root:
		return output
	stack = [(root, 1)]
	while stack:
		node, count = stack.pop()
		if count == 3:
			output.append(node.val)
		if count == 1:
			stack.append((node, count + 1))
			if node.left:
				stack.append((node.left, 1))
		if count == 2:
			stack.append((node, count + 1))
			if node.right:
				stack.append((node.right, 1))
	return output

"""
Implement a stack with MAX API:

Solution 1: Brute-force: max() iterate each element in the 
stack to find the maximum, O(n)

Solution 2: Trade space for time

"""
class Stack:
	def __init__(self):
		self.stack = []

	def is_empty(self):
		return len(self.stack) == 0

	def max(self):
		if not self.is_empty():
			return self.stack[len(self.stack) - 1][1]
		raise Exception('max(): empty stack')

	def push(self, x):
		tmp = x
		if not self.is_empty():
			tmp = max(tmp, self.max())
		self.stack.append((x,tmp))

	def pop(self):
		if self.is_empty():
			raise Exception('pop(): empty stack')
		elem = self.stack.pop()
		return elem[0]

"""
Q4: Given a binary tree, find the length of the longest consecutive
sequence path. The longest consecutive path need to be from 
parent to child (cannot be the reverse).

"""
def longest(curr, parent, currlen):
	if not curr:
		return currlen
	size = 1 #set the new length, at least 1
	if parent and curr.val == parent.val + 1:
		size = currlen + 1
	return max(currlen, longest(curr.left, curr, size),
		longest(curr.right, curr, size))


# Python BST:
"""
A binary search tree is a binary tree where each node has a comparable
key(optionally, an associated value) that has the following property:
for every single node in this tree, all keys contained within its 
left subtree is smaller than the one in this single node, and all
keys contained within its right subtree is larger than the one in this 
single node.

Recall the concept "collection" that supports insert, remove and 
update items, previously, we have implemented this concept with queue,
stack and singly linked list. Now, with binary search tree, we 
essentially can implement an associated collection that supports 
lookup a value by a given key and all keys are ordered nicely.

"""


"""
Q1: Validate whether a given binary tree is a binary search tree
or not?

Solution 1: 
We can directly have a solution based on the fact that the result of 
our inorder traversal should be in an increasing order. Recall the 
recursive way of doing inorder traversal, what we need for that 
recursive function is to be able to refer to a mutable object for 
storing the previous value we encounter during the process.

"""
def impl(root, prev):
	if not root:
		return True
	if not impl(root.left, prev):
		return False
	if prev[0] >= root.val:
		return False
	prev[0] = root.val
	return impl(root.right, prev)

class Solution:
	def isValidBST(self, root):
		'''
		:typeroot: TreeNode
		:rtype: bool

		'''
		prev = [None]
		return impl(root, prev)

"""
Solution 2: From BST's definition, we know that all values in the 
left subtree are smaller than what root has. That means, the largest
value inside the left subtree is smaller than the root's. 
Similarly, the smallest value inside the right subtree is larger than
the root's.

Based on this, if we know these two quantity for a given tree, we can
know whether a given tree is BST or not by using those results coming 
from a left subtree and right subtree to do the checking that follows
our definition. If we define a function impl in the following way:

impl(root):= This function returns a tuple <bool, value, value>
the first component represents whether the given binary tree rooted
at root is BST or not.
The second component represents the smallest value inside this tree.
For empty tree, this is None.
The third component represents the largest value inside this tree.
For empty tree, this is None.

"""
def impl(root):
	if not root:
		return True, None, None
	lr, lmin, lmax = impl(root.left)
	rr, rmin, rmax = impl(root.right)
	return lr and rr and (not lmax or lmax < root.val) and (not rmin
		or root.val < rmin), lmin or root.val, rmax or root.val

class Solution(object):
	def isValidBST(self, root):
		"""
		:type root: TreeNode
		:rtype: bool
		"""
		return impl(root)[0]

"""
Why Binary Search Tree?
It's an implementation that combines the flexibility of inserting 
in a linkedin list and the efficiency of searching in an ordered
array together. It can help you preserve the order of all keys in 
it while you are dynamically doing insertion and deletion.

"""

#Interface:
class BinarySearchTree(object):

	def __init__(self):
		#initializes the internal to become a valid empty binary
		#search tree.

	def insert(self, key, value):
		#if key is already presented in our tree, the associated value
		#will be updated by our input value.
		#Otherwise, a new key value pair will be put into our tree.

	def query(self, key):
		#If key is already presented in our tree, the associated value 
		#will be returned.
		#Otherwise, None will be returned.

	def delete(self,key):
		#If key is already presented in our tree, this key value pair
		#will be removed.
		#Otherwise, this function is a no-op.

"""
How do we store those key values pairs internally?
Ans: We should have a node concept for a single pair and then use it
in our tree construction. Since Binary search tree is a binary tree, 
we could borrow what we have learnt in previous class.

"""
class _Node(object):
	def __init__(self, key, value):
		self.key = key
		self.value = value
		self.left = self.right = None

"""
Underscore precedent is something you will do if you don't want the 
name of your entity to be imported into other people's code when they
do the following.
from binary_search_tree import *

But, one caveat of this technique is that if you use the following way 
to import names from this libray, it will not work:
import binary_search_tree

So, to some degree, defining a name like this is equivalent to having 
a module-private entity.

"""

#Q2: Given a key, how do we find the corresponding value in a bst?

class _Node(object):
	def __init__(self, key, value):
		self.key = key
		self.value = value
		self.left = self.right = None

class BinarySearchTree(object):
	def __init__(self):
		self._root =None

	def __query(self, root, key):
		if not root:
			return None
		if key < root.key:
			return self.__query(root.left, key)
		elif key > root.key:
			return self.__query(root.right, key)
		else:
			return root.value

	def query(self, key):
		return self.__query(self.__root, key)


#Q2.2: Can we implement this iteratively instead of recursively?
class BinarySearchTree(object):
	def __init__(self):
		self.__root = None

	def query(self, key):
		curr = self.__root
		while curr:
			if key < curr.key :
				curr = curr.left
			elif key > curr.key:
				curr = curr.right
			else:
				return curr.value
		return None

"""
Exercise 1: Given the root of a binary search tree, find the node
that contains the minimum value.

Recursion rule:
base case:
if not root.left:
	return root
return FindMin(root.left)

Key observation:
1. Empty tree should return nothing.
2. Otherwise, we shouldnt start to find the answer recursively in 
the left subtree since all nodes within contain values that are 
smaller than the root's one. If we could not find anything,
then root value must be the answer.

"""
class TreeNode(object):
	def __init__(self, value):
		self.value = value
		self.left = self.right = None

def FindMinimum(root):
	if not root:
		return None
	return FindMinimum(root.left) or root

"""
Exercise 2: Given the root of a binary search tree and a target,
find the first node containing a value that is larger tha our
target value.

Key observations:
1. If the value of the root node equals to target value, then our 
answer should be the leftmost node of the right subtree of our root.
2. If target is larger than the value in our root node, then our answer 
should be contained inside the right subtree. We can use recursion 
to get this since right now we want to get the first node containing 
a value that is larger than our target value in the right subtree. 
You can see that this new problem has exactly the same structure 
as the original problem. but with smaller size in terms of output.
3. When target is smaller, one thing that is different from case 2
is that root node is also a valid candidate for our final answer since
its value is larger than target. Thus, we need to look at the other 
potential candidate in the left subtree by recursion. If this yields
nothing, then root is indeed our answer. Otherwise, the answer we get 
from left subtree will be our final decision.

"""
def FindFirstLargerThanTarget(root, target):
	if not root:
		return None
	if root.value == target:
		return FindMinimum(root.right)
	elif root.value < target:
		return FindFirstLargerThanTarget(root.right, target)
	else:
		return FindFirstLargerThanTarget(root.left, target) or root


"""
Exercise 3:
Given the root of a binary search tree and a target, find the last node 
containing a value that is smaller than our target value.

"""

def FindMaximum(root):
	if not root:
		return None
	return FindMaximum(root.right) or root

def FindLastSmallerThanTarget(root, target):
	if not root:
		return None
	if root.value == target:
		return FindMaximum(root.left)
	elif root.value >target:
		return FindLastSmallerThanTarget(root.left, target)
	else:
		return FindLastSmallerThanTarget(root.right, target)

root = TreeNode(7)
root.left = TreeNode(1)
root.left.right = TreeNode(3)
root.left.right.right = TreeNode(5)
root.right = TreeNode(9)

print(FindMinimum(root).value)
print(FindFirstLargerThanTarget(root, 4).value)
print(FindFirstLargerThanTarget(root, 3).value)
print(FindFirstLargerThanTarget(root, 0).value)
print(FindFirstLargerThanTarget(root,10).value)
print(FindLastSmallerThanTarget(root, 4).value)
print(FindLastSmallerThanTarget(root, 3).value)
print(FindLastSmallerThanTarget(root, 0).value)
print(FindLastSmallerThanTarget(root, 10).value)

"""
Q3: Given a key and an associated value, how do we insert them
into the binary search tree?

Recursion rule:

if root.key == key:
	root.value = value
	return root
elif root.key < key:
	new_root_of_right_subtree = insert(root.right, key, value)
	root.right = new_root_of_right_subtree
	return root
else:
	....

If we find the key, then just update; otherwise, all that we need
to do is replace that link with a new node containing the key 
since a search for a key not in the tree ends at a null link. 
Similarly to query, the recursion algo will look like:
1. If the tree is empty, we return a new node containing the key
and value
2. If key is less than the key at the root, we set the left link 
to the result of inserting the key into the left subtree; otherwise,
we set the right link instead.

How will the API look like when we try to implement the above 
recursive algo?

Ans: As for return values, returning a link to a Node is needed since
we can reflect changes to the tree by assigning the result to the 
link of the root node used as argument.

"""
class BinarySearchTree(object):
	def __insert(self, root, key, value):
		if not root:
			return __Node(key, value)
		if key < root.key:
			root.left = self.__insert(root.left, key, value)
		elif key > root.key:
			root.right = self.__insert(root.right, key, value)
		else:
			root.value = value
		return root

	def insert(self, key, value):
		self.__root = self.__insert(self.__root, key, value)

#Do it iteratively:
class BinarySearchTree(object):
	def __init__(self):
		self.__root = None

	def insert(self, key, value):
		if not self.__root:
			self.__root = _Node(key, value)
		curr, prev, is_left = self.__root, None, None
		while curr:
			prev = curr
			if key < curr.key:
				is_left = True
				curr = curr.left
			elif key > curr.key:
				is_left = False
				curr = curr.right
			else:
				curr.value = value
				break
		if not curr:
			node = _Node(key, value)
			if is_left: 
				prev.left = node
			else:
				prev.right = node

"""
Q4: Given a key, how do we delete the corresponding pair from the tree?

Q4.1 : First, start wtih delete the smallest key?

Recursion rule:
if not root.left:
	return root.right
root.left = DeleteMin(root.left)
return root

The easiest way to remove the leftmost node while preserving the binary
search tree's properties to replace this node with its right child. 
Similar to private insert API, we could define the return value to be
the root node of the tree after deleting the node that contains the 
smallest key. With this, we have the following code:

"""

class BinarySearchTree(object):
	def __init__(self):
		self.__root = None

	def __deleteMin(self, root):
		if not root.left:
			return root.right
		root.left = self.__deleteMin(root.left)
		return root

	def __min(self, root):
		if not root.left:
			return root
		return self.__min(root.left)

	def __delete(self, root, key):
		if not root:
			return None
		if  key < root.key:
			root.left = self.__delete(root.left, key)
		elif key > root.key:
			root.right = self.__delete(root.right, kye)
		else:
			if not root.right:
				return root.left
			if not root.left:
				return root.right
			root = self.__min(root.right)
			root.right = self.__deleteMin(root.right)
			root.left = root.left
		return root

	def delete(self, key):
		self.__root = self.__delete(self.__root, key)

#Python Review:
"""
Midterm Q4:
Given a singly linked list and a target value, remove 
all nodes in this list that contain this target.

10 -> 8 -> 5 -> 1 -> 8, target =8, return 10->5->1

There are two approaches targeting this problem:
1. Prev, Current
2. Current (Current.next)

"""

def remove(self, head, target):
	fake_head = ListNode(None)
	fake_head.next = head
	cur = fake_head
	while cur and cur.next:
		if cur.next.val == target:
			cur.next = cur.next.next
		else:
			cur = cur.next
		return fake_head.next

def RemoveNodes(head, target):
	if not head or not target:
		return head
	fakeHead = ListNode(0)
	fakeHead.next = head
	prevNode, currNode = fakeHead, fakeHead.next
	while currNode:
		if currNode.val == target:
			prevNode.next = currNode.next
		else:
			prevNode = currNode
		currNode = currNode.next
	newHead, fakeHead.next = fakeHead.next, None
	return newHead

"""
Check to see if its a balance tree

"""
class Solution(object):
	def isBalanced(self, root):
		"""
		input: TreeNode root
		return: boolean

		"""
		#Write your solution here:
		if not root:
			return True
		left = self.get_height(root.left)
		right = self.get_height(root.right)
		if abs(left - right) > 1:
			return False
		return self.isBalanced(root.left) and self.isBalanced(root.right)

	def get_height(self, root):
		if not root:
			return 0
		left = self.get_height(root.left)
		right = self.get_height(root.right)
		return 1 + max(left, right)

"""
Optimized Implementation: Above implementation can be optimized 
by calculating the height in the same recursion rather than
calling a height() function separately.

"""
class Solution(object):
	def isBalanced(self, root):
		is self.helper(root) == -1:
			return False
		else:
			return True

	def helper(self, root):
		if root is None:
			return 0

		left = self.helper(root.left)
		right = self.helper(root.right)

		if left == -1 or right == -1:
			return -1
		elif abs(left - right) > 1:
			return -1
		else:
			return max(left, right) + 1

""""
Check if a given binary tree is symmetric.

"""
class Solution(object):
	def isSymmetric(self, root):
		"""
		input: TreeNode root
		return: boolean

		"""
		if not root:
			return True
		return self.helper(root.left, root.right)

	def helper(self, root1, root2):
		if not root1 and not root2:
			return True
		if not root1 or not root2:
			return False
		if root1.val != root2.val:
			return False
		return self.helper(root1.left, root2.right) and 
		self.helper(root1.right, root2.left)

"""
Maximum Subarray:

Given an integer array nums, find the contingous subarray
(containing at least one number) which has the largest 
sum and return its sum.

"""
def maxSubarray(A, left, right):
	if left == right:
		return A[left]

	center = (left + right )/2

	maxLeftSum = maxSubarray(A, left, center);
	maxRightSum = maxSubarray(A, center + 1, right);
	maxCenterSum = getMaxCenterSum(A, center, left, right)

	maxSum = max(maxLeftSum, maxRightSum, maxCenterSum);
	return maxSum;

def getMaxCenterSum(A, center, left, right):
	maxLeftBorderSum = -float('inf');
	leftBorderSum = 0;
	for i in xrange(center, left- 1, -1):
		leftBorderSum += A[i];
		maxLeftBorderSum = max(leftBorderSum, maxLeftBorderSum)

	maxRightBorderSum = -float('inf');
	maxRightBorderSum = 0;
	for i in xrange(center + 1, right + 1):
		rightBorderSum += A[i];
		maxRightBorderSum = max(maxRightBorderSum, rightBorderSum)
	return maxLeftBorderSum + maxRightBorderSum

#Top-down approach:
#Q1: Get Height of a binary tree:
class Solution(object):
	def findHeight(self, root):
		"""
		Input: TreeNode root
		return: int

		"""
		if root is None:
			return 0
		self.result = 0
		self.helper(root, 0)
		return self.result

	def helper(self, root, depth):
		if root is None:
			self.result = max(self.result, depth)
			return
		self.helper(root.left, depth + 1)
		self.helper(root.right, depth + 1)
		return

#Q2: Given a binary tree, find its minimum depth.
class Solution(object):
	def minDepth(self, root):
		if root is None:
			return 0
		self.ret = float('inf')
		self.helper(root, 0)
		return self.ret

	def helper(self, root, depth):
		if root is None:
			return 

		if root.left is None and root.right is None:
			self.ret = min(self.ret, depth + 1)
			return 

		self.helper(root.left, depth + 1)
		self.helper(root.right, depth + 1)
		return

#Q3: Given a binary tree, find its max depth:
class Solution：
	def maxDepth(root):
		if root is None:
			return 0
		self.max = 0
		self.helper(root, 0)
		return self.max

	def helper(self, root, depth):
		if root is None:
			return
		if root.left is None and root.right is None:
			depth += 1
			self.max = max(self.max, depth)
			return

		self.helper(root.left, depth + 1)
		self.helper(root.right, depth + 1)
		return 

"""
Lowest Common Ancestor(LCA):
Given a binary tree and the two nodes in the tree, return its LCA

Assumption:
1. All the node's value will be unique
2. p and q are different and both values will exist in binary tree.

"""
class Solution(object):
	def LCA(self, root, p, q):
		#p, q: TreeNode
		self.ret = None
		self.helper(root, p, q)
		return self.ret

	def helper(self, root, p, q):
		#Return value: 
		if root is None:
			return 0

		number_of_match = 0
		if root == p or root == q:
			number_of_match += 1

		#bottom up:
		l = self.helper(root.left, p, q)
		r = self.helper(root.right, p, q)

		number_of_match = number_of_match + l + r
		if number_of_match == 2 and self.ret is None:
			self.ret = root
			return 2
		else:
			return number_of_match

class Solution(object):
	def lowestCommonAncestor(self, root, p, q):
		if root is None:
			return None

		if root == p or root == q:
			return root

		l = self.lowestCommonAncestor(root.left, p, q)
		r = self.lowestCommonAncestor(root.right, p, q)

		if l is not None and r is not None:
			return root
		elif l is None:
			return r
		else:
			return l

"""
Binary Search Tree:

Operations: 
Insert as leaf

1. target == root.key return; // terminate
2. target < root.key :
	a. Root.left == None: root.left = new Node(target) 
	b. else: 			helper(root.left, target)
3. target > root.key:
	a. root.right == None: root.right = new Node(target)
	b. Else: 			helper(root.right, target)

"""

class BST(object):
	def __insert(self, root, key, value):
		if not root:
			return _Node(key, value)
		if key < root.key:
			root.left = self.__insert(root.left, key, value)
		elif key > root.key:
			root.right = self.__insert(root.right, key, value)
		else:
			root.value = value
		return root

	def insert(self, key, value):
		self.__root = self.__insert(self.__root, key, value)

"""
Leetcode 938: Range sum of BST
Given the root node of a binary search tree, return the sum of values 
of all nodes with value between L and R(inclusive)

The bianry search tree is guaranteed to have unique values.

"""

class Solution:
	def rangeSumBST(self, root, L, R):
		if not root:
			return 0
		r = 0
		if root.val >= L and root.val <= R:
			r += root.val
		r += self.rangeSumBST(root.left, L, R)
		r += self.rangeSumBST(root.right, L, R)
		return r

class Solution(object):
	def RSBST(self, root, L, R):
		if root is None:
			return 0
		if root.val < L:
			return self.RSBST(root.right, L, R )
		if root.val > R:
			return self.RSBST(root.left, L, R)
		else:
			return self.RSBST(root.left, L, root.val) + root.val + 
			self.rangeSumBST(root.right, root.val, R)
			
