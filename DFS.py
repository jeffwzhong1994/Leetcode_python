def subset(num):
	cur = []
	collections = []
	backtrack(num, 0, cur, collections)
	return collections

def backtrack(num, i, cur, collections):
	if i == len(num):
		collections.append(cur[:])
		return
	cur.append(num[i])
	backtrack(num, i+1, cur, collections)
	cur.pop()
	backtrack(num, i+1, cur, collections)

"""
All subsequences of a sorted array:
Given a sorted array of chars with duplicated chars, return all possible 
subsequence. The solution set must not contain duplicate subsequence.

"""

input = "ab1b2"

def subset(num):
	cur = []
	collections = []
	backtrack(num, 0, cur, collections)
	return collections

def backtrack(num, i, cur, collections):
	if i == len(num):
		collections.append(cur[:])
		return 
	cur.append(num[i])
	backtrack(num, i + 1, cur, collections)
	cur.pop()
	backtrack(num, i + 1, cur, collections)

def find_sub_set(input, index, solution, res):
	if index == len(input):
		res.append(solution[:])
		return 

	solution.append(input[index])
	find_sub_set(input, index + 1, solution, res)
	solution.pop()
	i = index
	curNum = input[i]
	while i < len(input) and input[i] == curNum:
		i+= 1

	find_sub_set*(input, i, solution, res)

def find_sub_set(input, index, solution, res):
	if index == len(input):
		res.append(solution)
		return 

	solution.append(input[index])
	find_sub_set(input, index+1, solution)
	solution.pop()
	i = index
	while i < len(input) and input[i] == input[index]:
		i+=1

	find_sub_set(input, i , solution)

"""
Print all combinations of coins that can sum up to a total value
of k.
E.g. total value k = 99 cents.
coin value = 25,10,5,1 cent

"""
def allCombination(coins, target):
	solution = []
	combinations(coins, 0, solution, target)

def combinations(coins, index, solution, target):
	if index == len(coins):
		if target == 0:
			print solution
		return
	for i in range(target / coins[index] + 1):
		solution.append(i)
		combinations(coins, index+1, solution, target - coins[index]*solution[-1])
		solution.pop()

def combinations(coins, index, solution, target):
	if index == len(coins) -1:
		if target % coins[-1] == 0:
			print solution + [target / coins[-1]]
		return
	for i in range(target/coins[index] + 1):
		solution.append(i)
		combinations(coins, index + 1, solution, target - coins[index]*solution[-1])
		solutions.pop()

#Solutions2:
def combinations(coins, index, solution, target):
	if target == 0:
		print solution
	if target < 0:
		return
	for i in range(index, len(coins)):
		solution.append(coins[i])
		combinations(coins, i, solution, target- coins[i])
		solution.pop()


"""
Print All valid combination of factors that form an integer.

Two questions to be answered:
1. what does it store on each level?
2. How many different states should we try to put on this level?

"""
#Method 1:
def getFactors(n):
	"""
	:rtype n: int
	:rtype: List[List[int]]
	"""
	if n == 1:
		return []
	res = []
	cur = []
	factors = getValidFactors(n)
	getFactorCombinations(factors, 0, n, cur, res)
	return res

def getFactorCombinations(factors, index, n, cur, res):
	if index == len(factors):
		if n == 1:
			res.append(convert(factors, cur))
		return
	factor = factors[index]
	numOfFactors = 0
	while True:
		divisor = factor ** numOfFactors
		if n % divisor == 0:
			cur.append(numOfFactors)
			getFactorCombinations(factors, index + 1, n /divisor, cur, res)
			cur.pop()
			numOfFactors += 1
		else:
			break

def getValidFactors(n):
	factors = []
	for i in range(2, n):
		if n % i == 0:
			factors.append(i)
	return factors

def convert(factors, cur):
	res = []
	for i in range(len(factors)):
		num = cur[i]
		while num > 0:
			res.append(factors[i])
			num -= 1
	return res

#Methods2:
def getFactors(n):
	"""
	:type n: int
	:rtype: List[List[int]]
	"""
	solution = []
	res = []
	getFactorsHelper(n, 2, solution, res)
	res.pop()
	return res

	def getFactorsHelper(n, index, solution, res):
		if n == 1:
			res.append(solution[:])
			return
		for i in range(index, n+1):
			if n % i == 0:
				solution.append(i)
				getFactorsHelper(n/i, i, solution, res)
				solution.pop()

"""
Generate All Valid Parentheses:
Given n pairs of parentheses, write a function to generate all combinations
of well-formed parentheses.

For example, given n = 3, a solution set is:
[
"((()))",
"(()())",
"(())()",
"()(())",
"()()()"
]

"""
def generateParentheses(n):
	cur = []
	res = []
	helper(0, 0, n, cur, res)
	return res

def helper(left, right, n, cur, res):
	if left == n and right == n:
		res.append(''.join(cur))
		return
	if left < n:
		cur.append('')
		helper(left + 1, right, n, cur, res)
		cur.pop()
	if right < left:
		cur.append(')')
		helper(left, right + 1, n, cur, res)
		cur.pop()

"""
Follow up:
What if we have 3 kinds of paretheses({}, <>, ())

"""
def ValidParetheses(1, m, n):
	"""
	input: int l, int m, int n
	return: string[]
	"""
	remain = [l, l, m, m, n, n]
	parentheses = ['(', ')', '<', '>', '{', '}']
	targetlen = 2 * (l + m + n)
	cur = []
	stack = []
	result = []
	helper(cur, stack, remain, targetlen, result, parentheses)
	return result

def helper(cur, stack, remain, targetlen, result, paretheses):
	if len(cur) == targetlen:
		result.append(''.join(cur))
		return
	for i in range(len(paretheses)):
		if i % 2 == 0:
			if remain[i] > 0:
				cur.append(parentheses[i])
				stack.append(parentheses[i])
				remain[i] -= 1
				helper(cur, stack, remain, targetlen, result, parentheses)
				cur.pop()
				stack.pop()
				remain[i] += 1
		else:
			if stack and stack[-1] == parentheses[i-1]:
				cur.append(parentheses[i])
				stack.pop()
				remain[i] -= 1
				helper(cur, stack, remain, targetlen, result, paretheses)
				cur.pop()
				stack.append(parentheses[i-1])
				remain[i] += 1

"""
Follow up:
What if we have 3 kinds of paretheses, and we want to force priority
{} > <> > ()

"""
def ValidParetheses(1, m, n):
	"""
	input: int l, int m, int n
	return: string[]
	"""
	remain = [l, l, m, m, n, n]
	parentheses = ['(', ')', '<', '>', '{', '}']
	targetlen = 2 * (l + m + n)
	cur = []
	stack = []
	result = []
	helper(cur, stack, remain, targetlen, result, parentheses)
	return result

def helper(cur, stack, remain, targetlen, result, paretheses):
	if len(cur) == targetlen:
		result.append(''.join(cur))
		return
	for i in range(len(remain)):
		if i % 2 == 0:
			if remain[i] > 0:
				if stack and getPriority(stack[-1]) < getPriority(parentheses[i]):
					continue
					cur.append(parentheses[i])
					stack.append(parentheses[i])
					remain[i] -= 1
					helper(cur, stack, remain, targetlen, result, parentheses)
					cur.pop()
					stack.pop()
					remain[i] += 1
			else:
				if stack and stack[-1] == parentheses[i-1]:
					cur.append(parentheses[i])
					stack.pop()
					remain[i] -= 1
					helper(cur, stack, remain, targetlen, result, paretheses)
					cur.pop()
					stack.append(parentheses[i-1])
					remain[i] += 1

def getPriority(leftParent):
	if leftParent == '{':
		return 2
	if leftParent == '<':
		return 1
	return 0

