stride = 1
num_blocks = [2, 2, 2, 2]
num_block = num_blocks[0]
strides = [stride] + [3] * (num_block - 1)
print(strides)