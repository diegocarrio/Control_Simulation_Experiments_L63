import numpy as np  
import pickle

class obs_da:
    def __init__(self, t = [0], pos = [0], val = [0], err = [0], bias = [0], xt = [0], name = 'uninitialized', mu_init = [], sigma_init = []):
        tdim = 0
        xdim = 0
        odim = 0
        self.name = name
        self.t = t
        self.pos = np.array(pos)
        self.val = np.array(val)
        self.err = np.array(err)
        self.bias = np.array(bias)
        self.hx = np.array(val)
        self.xt = np.array(xt)
        self.dep = []
        self.climvar = []
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        print("Obs_DA Object Created!")
        print("Obs_DA Object Name: ", self.name)
        print("t: ", self.t)
        print("pos: ", self.pos)
        print("val: ", self.val)
        print("err: ", self.err)
        print("bias: ", self.bias)
        print("xt: ", self.xt)
    
    
    def __str__(self):
        print(self.name)
        print('Observation position:')
        print(self.pos)
        print('Time interval:')
        print(self.t)
        print('Observation values')
        print(self.val)
        return self.name


    def setVal(self, val):
        self.val = np.array(val)
        #print("val: ", self.val)
    
    
    def setPos(self, pos):
        self.pos = np.array(pos).astype(int)
        #print("pos: ", self.pos)
    
    
    def setErr(self, err):
        self.err = np.array(err)
        
        
    def getVal(self):
        return self.val    
    
    
    def getErr(self):
        return self.err
    
    
    def getPos(self):
        return self.pos.astype(int) # not round to
    
    
    def save(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
    
    
    def load(self, infile):
        with open(infile,'rb') as input:
            sv = pickle.load(input)
        return sv