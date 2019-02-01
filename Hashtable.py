import sys
"""
Hash_table(general term):
1. one to one mapping
2. Hash_table is a <key,value> pair, key map to value
3. does not allow duplicate key
4. allow duplicate value
5. Hash_set is a set of {1,3}, it only contains keys

"""

#Count word frequencies
def words_to_freq(words):
	myDict ={}
	for word in words:
		if word in myDict:
			myDict[word] += 1
		else:
			myDict[word] = 1

	return myDict


def most_common_words(freqs):
	values = freqs.values()
	best = max(values)
	words =[
	key for key, val in freqs.items()
	if val == best
	]
	return(words, best)


mydic = {'a': 10, 'b': 34 }
new_mydic = {v:k for k, v in mydic.items()}
print(new_mydic)

"""
SET:
Sets are mutable unordered collections of unique elements.
Sets do not record element position or order of insertion. Accordingly,
sets do not support indexing, slicing, or other sequence-like behavior.

Sets are implemented using dictionaries. They cannot contain mutable elements 
such as lists or dictionaries. However, they can contain immutable collections.

SET COMPREHENSION:
Similar to list comprehension:
{expr for value in collection if condition}

"""

def two_sums(nums, target):
	if len(nums) <= 1:
		return False
	my_dic = {}
	for i in range(len(nums)):
		if nums[i] in dic:
			return( my_dic[nums[i]],i)
		else:
			my_dic[target- nums[i]] = i

"""
Given an array of integers and an integer k, you need to find the 
total number of continous subarrays whose sum equals to k.

"""
def subarray_sum(nums, k):
	ans = sums = 0
	my_dic = {}
	for num in nums:
		if sums not in my_dic:
			my_dic[sums] = 0
		my_dic[sums] += 1
		sums += num
		if sums - k in my_dic:
			ans += my_dic[sums -k]
	return ans

nums = [1,6,5,2,3,4,0]
k = 7
print(subarray_sum(nums, k))

"""
For a composition with different kinds of words, try to find the 
top -k frequent words from the composition.

"""
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
	for i in range(0, k):
		topk.append(heapq.heappop(freq_heap)[1])
	return topk

"""
Solution 2:
Step1: Read the composition, and count the frequency of each word 
by using the hash_table.
Step2: Build a MIN-heap for the first k elements of the hash-table
Iterate over each element from the k+1 th to the n-th element, 
and update the MIN_heap as follows

"""
def top_k(nums, k):
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
	for i in range(0, k):
		topk.append(heapq.heappop(freq_heap)[1])
	topk.reverse()


"""
Test for palindromic permutations:
A palindrome is a word that reads the same forwards and backwards, 
write code to test if the letters forming a string can be permuted to
form a palindrome.

"""

def is_palindromic(word):
	freq = {}
	for i in range(len(word)):
		if word[i] in freq:
			freq[word[i]] = freq[word[i]] + 1
		else:
			freq[word[i]] = 1

	odd_cnt =0

	for key in freq.keys():
		if freq[key] % 2 ==1:
			odd_cnt += 1
			if odd_cnt >1:
				return False
	return True

"""
Given a nested dictionary as follows:
data = {'one': ('label': 'This is shoot 1', 'start': 1, 'end': 10},
'two': {'label':'This is shot 2', 'start':'11','end': 25},
'three': {'This is shot 3'}

print the unfoleded version of each element

"""
def nest_dic(data):
	if type(data) is not dict:
		return [str(data)]
	else:
		result = []
		for key in data.keys():
			next_level = nest_dic(data[key])
			for elem in next_level:
				result.append(str(key) + '->' + elm)
		return result

"""
Q: Find the nearest repeated entries in an array.
Write code that takes as input an array and finds the distance
between a closest pair of equal entries

"""
arr = ['All', 'work','and','no','play','makes','for','no','work','no','fun','and','no','results']

def nearest_repeat(arr):
	word_ind = {}
	dist = sys.maxsize
	for i in range(len(arr)):
		if arr[i] in word_ind:
			dist = min(dist, i - word_ind[arr[i]])
			print("dist:" ,dist)
			print(arr[i], word_ind[arr[i]])
		word_ind[arr[i]] = i
	return dist,word_ind


print(nearest_repeat(arr))

"""
Find the length of a longest contained range.
Given an integer array, find the size of a largest subset of integers in the array 
having the property that if two integers are in the subset, then so are all integers
between them 

E.g.: array = [3, -2, 7, 9, 8, 1, 2, 0, -1, 5, 8]
The biggest contained range is [-2, -1, 0, 1, 2, 3], return 6

Solution 1: Brute-force, sort the array, then search from each position
Solution 2: Split from the middle

"""
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

array = [3, -2, 7, 9, 8, 1, 2, 0, -1, 5, 8]
print(longest_contained_range(array))


