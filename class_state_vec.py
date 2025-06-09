import numpy as np
import pickle

class state_vector:
    # Define __init__ method that initializes the object with the state vector, time vector, name of the object, and name of the parameters
    def __init__(self, al = [0], t = [0], name = 'uninitialized', paraname = 'unknown'):
        self.tdim = np.size(t)
        self.al = al
        self.aldim = np.size(al)
        self.statedim = np.size(al[0:3])
        self.pdim = np.size(al[3:3])
        self.x0 = al[0:3]
        self.p0 = al[3:3]
        self.t = t
        self.name = name
        self.trajectory = np.zeros([self.tdim, self.aldim])
        self.Enstrajectory = np.zeros([self.tdim, self.aldim, 7])
        self.Xatrajectory = np.zeros([self.tdim,self.aldim,7])
        self.paraname = paraname
        print("State_Vector Object Created!")
        print("State_Vector Object Name: ", self.name)
        print("al: ", self.al)
        print("t: ", self.t)
        print("x0: ", self.x0)
 
    # Define __str__ method that prints the name of the object and the number of states and parameters when the object is printed (e.g., print(object))    
    def __str__(self): 
        print(self.name)
        print('Number of states and parameters')
        print(self.aldim)
        print('Parameters:')
        print(self.p0)
        print('Number of parameters:')
        print(self.pdim)
        print('Initial condition:')
        print(self.x0)
        print('Number of states:')
        print(self.statedim)
        print('Trajectory:')
        print(self.trajectory)
        print('EnsTrajectory:')
        print(self.Enstrajectory)
        return self.name
    
    
    # Define setTrajectory method that sets the trajectory of the object:
    def setTrajectory(self,states):
        self.trajectory = states
    
        
    # Define getTrajectory method that returns the trajectory of the object:
    def getTrajectory(self):
        return self.trajectory
    
    
    # Define times method that returns the time vector of the object:
    def getTimes(self):
        return self.t
    
    
    # Define save method that saves the object to a file:
    def save(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self, output)
    
    
    # Define load method that loads the object from a file:
    def load(self, infile):
        with open(infile, 'rb') as input:
            sv = pickle.load(input)
        return sv