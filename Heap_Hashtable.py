
#Sift-up Operation: min-heap

def sift_up(array, index):
	parent_idx = (index - 1)/2
	if parent_idx < 0 or array[index] > array[parent_idx]:
		return
	array[index], array[parent_idx] = array[parent_idx], array[index]
	print("index:",index)
	print("parent index:", parent_idx)
	sift_up(array, parent_idx)

arr = [0,1,5,6,8, -1]
sift_up(arr, len(arr)-1)
print(arr)

class Heap(object):
	def __init__(self):
		self.array = []
	
	def sift_up(self, array, index):
		parent_idx = (index - 1)/2

		if parent_idx < 0 or array[index] > array[parent_idx]:
			return 
		array[index], array[parent_idx] = array[parent_idx], array[index]
		self.sift_up(array, parent_idx)

	def push(self, val):
		#Assume self.array has enough capacity
		self.array.append(val)
		self.sift_up(self.array, len(self.array) - 1)

#Sift_down operation:
#Recursion:

def sift_down(array, index):
	left = index * 2 + 1
	right = index * 2 + 2
	small = index
	if left < len(array) and array[small] > array[left]:
		small = left
	if right < len(array) and array[small] > array[right]:
		small = right
	if small != index:
		array[small], array[index] = array[index], array[small]
		sift_down(array, small)

#Iterations:
def sift_down(array, index):
	left = index * 2 + 1
	right = index * 2 + 2
	while left < len(array) or right > len(array):
		smaller = index
		if left < len(array) and array[smaller] > array[left]:
			smaller = left
		if right < len(array) and array[smaller] > array[right]:
			smaller = right
		if smaller == index:
			break
		array[index] , array[smaller] = array[smaller], array[index]
		index = smaller
		left = index * 2 + 1
		right = index * 2 + 2

#Time Complexity: O(logn)

#How to remove the element at the top of the heap:
class Heap(object):
	def __init__(self):
		self.array = []

	def sift_up(self, array, index):
		parent_idx = (index - 1)/2
		if parent_idx == 0 or array[index] > array[parent_idx]:
			return
		array[index], array[parent_idx] = array[parent_idx], array[index]
		self.sift_up(array, parent_idx)
	
	def push(self, val):
		#Assume self.array has enough capacity:
		self.array.append(val)
		self.sift_up(self.array, len(self.array) -1 )

	def sift_down(self, array, index):
		left = index * 2 + 1
		right = index * 2 + 2
		small = index
		if left < len(array) and array[small] > array[left]:
			small =left
		if right < len(array) and array[small] > array[right]:
			small = right
		if small != index:
			array[small], array[index] = array[index], array[small]
			self.sift_down(array, small)

	def pop(self):
		res = self.array[0]
		self.array[0], self.array[-1] = self.array[-1], self.array[0]
		self.array.pop()
		self.sift_down(self.array, 0)
		return res

#How to initialize a heap from a random array?
def build_heap(arr):
	for i in range(len(arr)/2 - 1, -1, -1):
		sift_down(arr, i)

#Python Hashtable:

"""
If looking up a certain element in a collection is frequently used, you should 
consider using hashtable.

hash_table(general term):
 1. One to one mapping
 2. hash_table is a <key, value> pair,
 3. Does NOT allow duplicate key
 4. allow duplicate value
 5. Hash_set is a set {1,3}, it only contains keys

In python, we can use SET and DICTIONARY to represent hash_set and hash_table

1. Values:
	- Any type(immutable and mutable)
	- Can be duplicates
	- Dictionary value can be a list or another dictionary

2. Keys:
	- Must be unique
	- Immutable type

"""

#Question1: Count word occurences
def words_to_frequencies(words):
	myDict = {}
	for word in words:
		if word in myDict:
			myDict[word] += 1
		else:
			myDict[word] = 1
	return myDict

#Question2: Most common word
def most_common_words(freqs):
	values = freqs.values()
	best = max(values)
	words = [
		key
		for key, val in freqs.items()
		if val == best
	]
	return(words, best)

#Question3: Dictionary Comprehension:
mydic = {'a': 10, 'b': 34, 'A': 7, 'Z': 3}
new_mydic = {
	k.lower(): mydic.get(k.lower(),0) + mydic.get(k.upper(),0)
	for k in mydic.keys()
}
print(new_mydic)

#Example2: Exchange keys and values (no duplicates in the values)
mydict = {'a': 10, 'b': 34}
new_mydict = {v:k for k, v in mydict.items()}
print(new_mydict)

#Set Comprehension:

"""
Q5: Given an array of integers, return indices of the two numbers such that they
add up to a specific target. You may assume that each input would have exactly one
solution, and you may not use the same element twice.

"""
def two_sum(nums, target):
	if len(nums) <= 1:
		return False
	my_dic = {}
	for i in range(len(nums)):
		if nums[i] in my_dic:
			return(my_dic[nums[i]], i)
		else:
			my_dic[target - nums[i]] = i

'''
Q6: Given an array of integers and an integer k, you need to find the total number
of continous subarrays whose sum equals to k.

'''
def subarray_sum(nums, k):
	ans = sums = 0
	my_dic = {}
	for num in nums:
		if sums not in my_dic:
			my_dic[sums] = 0
		my_dic[sums] += 1
		sums += num
		if sums - k in my_dic:
			ans += my_dic[sums - k]
	return ans

'''
Q7: For a composition with different kinds of words, try to find the top-k frequent
words from the composition.

'''

def top_k(nums, k):
	freq = {}
	freq_heap = []
	for num in nums:
		if num in freq:
			freq[num] = freq[num] + 1
		else:
			freq[num] = 1

	for key in freq.keys():
		heapq.heappush(freq_heap, (-freq[key], key))
	topk = []
	for i in range(0,k):
		topk.append(heapq.heappop(freq_heap)[1])
	return topk

'''
Solution2:
	Step1: Read the composition, and count the frequency of each word by using 
	the hash_table.
	Step2: Build a MIN-heap for the first k elements of the hash_table.
	Iterate over each element from the k+1 -th to the n-th element, update the 
	MIN_heap as follows:
		if i-th element > MIN_heap.top(), then remove top and insert i-th element
		otherwise, do nothing
	O(k + (n-k)log(k))

'''

def top_k (nums,k):
	freq = {}
	freq_heap = []
	for num in nums:
		if num in freq:
			freq[num] = freq[num] + 1
		else:
			freq[num] = 1

	for key in freq.keys():
		heapq.heappush(freq_heap, (freq[key],key))
		if len(freq_heap) > k:
			heapq.heappop(freq_heap)
	topk = []
	for i in range(0,k):
		topk.append(heapq.heappop(freq_heap)[1])
	topk.reverse()

'''
Test for palindromic permutations:
A palindromic is a word that reads the same forwards and backwards.
Like, 'level','rotator'. Write a code to test if the letters forming
a string can be permutated to form a palindrome. For example, 
'edified' can be permuted to for 'deified'

'''
def is_palindromic(word):
	freq = {}
	for i in range(len(word)):
		if word[i] in freq:
			freq[word[i]] = freq[word[i]]+ 1
		else:
			freq[word[i]] = 1
	odd_cnt = 0
	for key in freq.keys():
		if freq[key] % 2 == 1:
			odd_cnt += 1
			if odd_cnt > 1:
				return False
	return True

s = 'edified'
print(is_palindromic(s))

'''
Q4: Given a nested dictionary as follows,
print the unfolded version of each element.

'''

data = {'one':{'label': 'This is shot 001', 'start': 1, 'end': 10},
		'two': {'label':'This is shot 002', 'start': 11, 'end': 25},
		'three': 'This is shot 003'}

def nest_dic(data):
	if type(data) is not dict:
		return[str(data)] #Why would you put string in a list[]?
	else:
		result = []
		for key in data.keys():
			next_level = nest_dic(data[key])
			for elm in next_level:
				result.append(str(key) + '->' + elm)
		return result

print(nest_dic(data))
'''
Find the nearest repeated entries in an array.
Write code that takes as input an array and finds the distance between
a closest pair of equal entries.

'''
def nearest_repeat(arr):
	word_ind = {}
	dist = sys.maxsize
	for i in range(len(arr)):
		if arr[i] in word_ind:
			dist = min(dist, i - word_ind[arr[i]])
		word_ind[arr[i]] = i
	return dist

''' 
Find the length of a longest contained range.
Given an integer array, find the size of a largest subset of integers in
the array having the property that if two integers are in the subset,
then so are all the integers between them.

'''
def longest_contained_range(arr):
	unprocessed = set(arr)
	maxlen = 0
	while unprocessed:
		elem = unprocessed.pop()
		lower = elem -1
		while lower in unprocessed:
			unprocessed.remove(lower)
			lower = lower -1
		upper = elem + 1
		while upper in unprocessed:
			unprocessed.remove(upper)
			upper = upper + 1
		maxlen = max(maxlen, upper - lower - 1)
	return maxlen

'''
Find the longest subarray with distinct entries.
arr = [f,s,f,e,t,w,e,n,w,e], the longest distinct subarray is [s,f,e,t,w]

'''
def longest_subarray(arr):
	recent_occur = {}
	start = 0
	result = 0
	for i in range(len(arr)):
		if arr[i] in recent_occur:
			if recent_occur[arr[i]] >= start:
				result = max(result, i - start)
				start = recent_occur[arr[i]] + 1
		recent_occur[arr[i]] = i
	result = max(result, len(arr) - start)
	return result

arr1 = ['f','s','f','e','t','w','e','n','w','e']
print(longest_subarray(arr1))

