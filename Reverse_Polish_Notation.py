class Solution:
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        import operator
        if tokens  == None or len(tokens) == 0:
            return tokens
        
        stack = []
        operators = []
        ops = {'+': operator.add, '-': operator.sub, '*': operator.mul, 
              '/':operator.truediv}

        for char in tokens:
            #Left parentheses, continue
            if char == '(':
                continue

            #Right parenthesis, pop the two value in the stacks and do operations
            elif char  == ')':
                t1 = int(stack.pop())
                t2 = int(stack.pop())
                stack.append(int(ops[operators.pop()](t2, t1)))

            #Append operators
            elif char in ops:
                operators.append(char)

            #Append value
            else:
                stack.append(int(char))

        return stack[0]