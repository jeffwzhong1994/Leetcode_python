class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums = sorted(nums)
    
        res = nums[0] + nums[1] + nums[2]
        
        for i in range(len(nums) -2):
            j, k = i+1, (len(nums) -1)
            while j < k:
                summ = nums[i] + nums[j] + nums[k]
                if summ == target:
                    return summ
                
                if abs(summ-target) < abs(res-target):
                    res = summ
                
                if summ < target:
                    j += 1
                
                elif summ > target:
                    k -= 1
        return res
            