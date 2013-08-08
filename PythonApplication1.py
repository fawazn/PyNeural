
import numpy

class BackProp:
  numLayers = 0
  _weights = []
  shape = None

  def __init__ (self,size):
    self.numLayers = len(size) - 1 #decrement to remove input layer
    self.shape = size
    self._inputLayers = []
    self._outputLayers = []
    self._previousDelta = []

    for(l1,l2) in zip(size[:-1],size[1:]):
      self._weights.append(numpy.random.normal(scale=0.1, size = (l2, l1+1)))
      self._previousDelta.append(numpy.zeros((l2,l1+1)))

  def run(self,input):
    inCases = input.shape[0]
    self._inputLayers = []
    self._outputLayers = []
    #for input layer
    for index in range(self.numLayers):
      if index == 0:
        layerInput = self._weights[0].dot(numpy.vstack([input.T,numpy.ones([1,inCases])]))
      else:
        layerInput = self._weights[index].dot(numpy.vstack([self._outputLayers[-1],numpy.ones([1,inCases])]))
      self._inputLayers.append(layerInput)
      self._outputLayers.append(self.SigmoidTransfer(layerInput))
    return self._outputLayers[-1].T


  def TrainEpoch(self, input, target, rate = 0.2, momentum = 0.5):
     delta = []
     lnCases = input.shape[0]
     self.run(input)
     #compute deltas for backpropagation
     for index in reversed(range(self.numLayers)): #compare output of current layer to outpt values
       if index == self.numLayers - 1:
         outputDelta = self._outputLayers[index] - target.T
         error = numpy.sum(outputDelta**2)
         delta.append(outputDelta * self.SigmoidTransfer(self._inputLayers[index],True))
         #^^append difference between layer output and target
       else: #or compare to succesive layers
         delta_pullback = self._weights[index + 1].T.dot(delta[-1])
         delta.append(delta_pullback[:-1,:] * self.SigmoidTransfer(self._inputLayers[index],True))
     for index in range(self.numLayers):
         delta_index = self.numLayers - 1 - index
         if index == 0:
           iterlayerOutput = numpy.vstack([input.T, numpy.ones([1, lnCases])])
         else:
           iterlayerOutput = numpy.vstack([self._outputLayers[index - 1],numpy.ones([1,self._outputLayers[index - 1].shape[1]])])
         currentDelta = numpy.sum(iterlayerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0), axis = 0)
         actualDelta = rate*currentDelta + momentum * self._previousDelta[index]
         self._weights[index] -= actualDelta
         self._previousDelta[index] = actualDelta
     return error

  def SigmoidTransfer(self,x, derivative = False):
    if not derivative:
        return 1/(1+numpy.exp(-x))
    else:
        out = self.SigmoidTransfer(x)
        return out*(1-out)


backpObj = BackProp((2,2,1))
#Test -- training to behave as a XOR gate
xortraininp = numpy.array([[0,0], [1,1], [0,1], [1,0]])
xortrainout = numpy.array([[0.05], [0.05], [0.95], [0.95]])
maxinp = 100000
inperror = 1e-4
print("errors for 10000n iterations:")
for i in range(maxinp +1):
    err = backpObj.TrainEpoch(xortraininp,xortrainout)
    if i%10000 == 0: print (err)
    if err<inperror: break
    op = backpObj.run(xortraininp)
print ("output: {0}".format(op) )