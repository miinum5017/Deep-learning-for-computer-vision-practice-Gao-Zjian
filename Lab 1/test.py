import torch
from pytorch101 import create_sample_tensor, mutate_tensor, count_tensor_elements, create_tensor_of_pi, multiples_of_ten

print('--- Test 1: create_sample_tensor ---')
x = create_sample_tensor()
print('Shape correct:', x.shape == (3, 2), '| x[0, 1] correct:', x[0, 1].item() == 10, '| x[1, 0] correct:', x[1, 0].item() == 100)

print('\n--- Test 2: mutate_tensor ---')
mutate_tensor(x, [(0, 0), (1, 0), (1, 1)], [4, 5, 6])
print('x[0, 0] correct:', x[0, 0].item() == 4, '| x[1, 0] correct:', x[1, 0].item() == 5)

print('\n--- Test 3: count_tensor_elements ---')
print('Correctly counted:', count_tensor_elements(x) == 6)

print('\n--- Test 4: create_tensor_of_pi ---')
pi_tensor = create_tensor_of_pi(2, 3)
print('Filled with pi:', (pi_tensor == 3.14).all().item() == True)

print('\n--- Test 5: multiples_of_ten ---')
print('Multiples correct:', multiples_of_ten(5, 25).tolist() == [10.0, 20.0])
print('Empty shape correct:', multiples_of_ten(5, 7).shape == (0,))
