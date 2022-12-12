import numpy as np
import easynn as nn
import easynn_cpp as cpp

def test_MLP(): # test 8 from grade_p4.py
    relu = nn.ReLU()
    flatten = nn.Flatten()
    f1 = nn.Linear("f1", 28*28, 100)
    f2 = nn.Linear("f2", 100, 10)
    x = nn.Input2d("images", 28, 28, 1)
    x = flatten(x)
    x = f2(relu(f1(x)))
    x.resolve(np.load("msimple_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"][:1000]

    infer = x.compile(cpp.Builder())
    label = infer(images = images).argmax(axis = 1)

def test_CNN(): # test 10 from grade_p4.py
    pool = nn.MaxPool2d(2, 2)
    relu = nn.ReLU()
    flatten = nn.Flatten()

    x = nn.Input2d("images", 28, 28, 1)
    c1 = nn.Conv2d("c1", 1, 8, 3) # 28->26
    c2 = nn.Conv2d("c2", 8, 8, 3) # 26->24
    x = pool(relu(c2(relu(c1(x))))) # 24->12
    c3 = nn.Conv2d("c3", 8, 16, 3) # 12->10
    c4 = nn.Conv2d("c4", 16, 16, 3) # 10->8
    x = pool(relu(c4(relu(c3(x))))) # 8->4
    f = nn.Linear("f", 16*4*4, 10)
    x = f(flatten(x))

    x.resolve(np.load("mnist_params.npz"))
    mnist_test = np.load("mnist_test.npz")
    images = mnist_test["images"][:1000]
    
    infer = x.compile(cpp.Builder())
    label = infer(images = images).argmax(axis = 1)

print("===== Testing MLP =====")
test_MLP()

print()
print("===== Testing CNN =====")
test_CNN()