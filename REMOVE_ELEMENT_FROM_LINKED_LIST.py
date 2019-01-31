# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        if head == None:
            return head
        
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        while head:
            if head.val == val:
                pre.next = head.next
                head = pre
            pre = head
            head = head.next
        return dummy.next

