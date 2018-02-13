import numpy as np  # linear algebra library
nrInputs = 300
# sigmoid function
def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)
    return 1 / (1 + (np.exp(-x)))


training = [[] for i in range(nrInputs)]
answers = [[] for i in range(nrInputs)]
with open("training.txt") as f:
    line = f.readline()
    cnt = 0
    while line and cnt < nrInputs:
        if not line.startswith(("Image", "\n", " ", "#")):
            # print(line.replace(" ", "").replace("\n", ""))
            temp = []
            for y in range(0, 20):
                for c in line.replace("\n", "").split(" "):
                    # print("Line {}: {}".format(cnt, line.strip()))
                    temp.append(int(c)/31)
                line = f.readline()
                # print("Line {}: {}".format(cnt, line.strip()))
            # print(len(temp))
            training[cnt] = temp
            cnt += 1
        line = f.readline()
with open("training-facit.txt") as f:
    line = f.readline()
    cnt = 0
    while line and cnt < nrInputs:
        if not line.startswith(("\n", " ", "#")):
            app = [0 for x in range(0, 4)]
            val = line.split(" ")[1]
            #print("Line {}: {}".format(cnt, line.strip()))
            #print("c is: " + c)
            print(val)
            app[int(val)-1] = 1
            answers[cnt] = app
            cnt += 1
        line = f.readline()
print(answers)
print(len(answers))
print(len(training))

# input dataset
# Each row is a single "training example"
# Each column corresponds to one of our input nodes
X = np.array(training)
# old values
# np.array([[0, 0, 1, 1],
#           [0, 1, 1, 0],
#           [1, 0, 1, 0],
#           [1, 1, 1, 0]])
print("shape of X: " + str(X.shape))
# output dataset
y = np.array(answers)  # = np.array([[0, 0, 1, 1]]).T

print("shape of y: " + str(y.shape))
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(55)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((400, 4))-1
print("first syn0: " + str(syn0))
for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    #print("l1: " + str(l1))

    # how much did we miss?
    l1_error = y - l1
    #print("l1_error: " + str(l1_error))
    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True)
    #print("l1_delta:" + str(l1_delta))
    # update weights
    syn0 += np.dot(l0.T, l1_delta)
    #print("syn0: " + str(syn0[0]))
print("Output After Training:")
#print("syn0: " + str(syn0))

for i, guess in enumerate(l1):
    print(guess)
    print(max(guess))
    print(np.log(guess))
    if str(np.log(guess).argmax(axis=0)+1) is not "1":
        print("guess " +
              str(np.unravel_index(np.argmax(np.log(guess), axis=None),
                                   np.log(guess).shape)[0]+1) + " : " + str(y[i]))
    print()