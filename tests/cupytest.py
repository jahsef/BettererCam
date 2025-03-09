import cupy

arr = cupy.asarray([[1,2,3],[4,5,6],[7,8,9]])#hw 3x3
print(arr.shape)
print(arr)


img_lambda = lambda img: img[::-1, ::-1]

fart = img_lambda(arr)

print(fart.shape)
print(fart)


