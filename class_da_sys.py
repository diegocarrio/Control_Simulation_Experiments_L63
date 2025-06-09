import numpy as np
import random
import pickle
import numpy.matlib 
from scipy import linalg


class da_system:
    
    def __init__(self, x0 = [], yo = [], t0 = 0, dt = 0, state_vector = [], obs_data = [], acyc_step = 1):
        self.xdim           = np.size(x0)
        self.ydim           = np.size(yo)
        self.edim           = 0
        self.x0             = x0
        self.t0             = t0
        self.dt             = dt
        self.t              = t0
        self.acyc_step      = acyc_step       # Number of time steps between assimilation cycles
        self.dtau           = dt * acyc_step  # Time interval between assimilation cycles
        self.fcst_step      = acyc_step       # Number of time steps between forecasts
        self.fcst_dt        = dt
        self.maxit          = 0
        self.statedim       = 0
        self.paradim        = 0
        self.B              = np.matrix(np.identity(self.xdim))  # Background error covariance matrix
        self.R              = np.matrix(np.identity(self.ydim))  # Observation error covariance matrix
        self.H              = np.matrix(np.identity(self.xdim))  # Observation operator matrix
        self.SqrtB          = []
        self.state_vector   = state_vector
        self.obs_data       = obs_data
        self.method         = ''
        self.KH             = []
        self.khidx          = []
        self.das_bias_init  = 0
        self.das_sigma_init = 0.1
    
    
    def __str__(self):
        print('DA System Object')
        print('xdim: ', self.xdim)
        print('ydim: ', self.ydim)
        print('x0: ', self.x0)
        print('t0: ', self.t0)
        print('dt: ', self.dt)
        print('t: ', self.t)
        print('acyc_step: ', self.acyc_step)
        print('dtau: ', self.dtau)
        print('fcst_step: ', self.fcst_step)
        print('fcst_dt: ', self.fcst_dt)
        print('B: ')
        print(self.B)
        print('R: ')
        print(self.R)
        print('H: ')
        print(self.H)
        print('state_vector: ')
        print(self.state_vector)
        print('obs_data: ')
        print(self.obs_data)
        print('method: ')
        print(self.method)
        return 'type::da_system'
    
    
    def setMethod(self, method):
        self.method = method
        
    
    def getMethod(self):
        return self.method
    
    
    def setStateVector(self, sv):
        self.state_vector = sv
    
    
    def getStateVector(self):
        return self.state_vector
    
    
    def setObsData(self, obs):
        self.obs_data = obs
    
    
    def getObsData(self):
        return self.obs_data
    
    
    def setB(self, B):
        self.B     = np.matrix(B)           # Background error covariance matrix
        nr, nc     = np.shape(B)            # Get the number of rows and columns of the background error covariance matrix
        self.xdim  = nr                     # Set the state dimension to the number of rows of the background error covariance matrix
        self.SqrtB = linalg.sqrtm(self.B)   # Compute the square root of the background error covariance matrix
    
    
    def setR(self, R):
        self.R = np.matrix(R)               # Observation error covariance matrix
    
    
    def setH(self, H):
        self.H = np.matrix(H)               # Observation operator matrix
    
    
    def getB(self):
        return self.B
    
        
    def getR(self):
        return self.R
    
    
    def getH(self):
        return self.H
    
    # Defining initial ensemble generation:
    def initEns(self, x0, mu = 0, sigma = 0.1, edim = 4, separate = 'undecided'):
        x0 = np.matrix(x0).flatten().T    # Convert the initial state vector to a column vector
        mu = np.matrix(mu).flatten().T    # Convert the mean to a matrix
        np.random.seed(2)                    # Set the seed for the random number generator
        
        if separate == 'no':
            xdim      = len(x0)
            Xrand     = np.random.normal(mu, sigma, (xdim, edim))   # Generate random numbers from a normal distribution
            Xrand     = np.matrix(Xrand)
            rand_mean = np.mean(Xrand, axis = 1) - mu               # Compute the mean across ensemble members for each state variable
            rmat      = np.matlib.repmat(rand_mean, 1, edim)        # Repeat the mean across ensemble members for each state variable
            Xrand     = Xrand - rmat                                # Subtract the mean from the ensemble members, so that the ensemble mean becomes exactly mu
            rmat      = np.matlib.repmat(x0, 1, edim)               # Repeat the initial state vector across ensemble members
            X0        = np.matrix(Xrand + rmat)                     # Adding x0 to the ensemble members yields the initial ensemble
               
                        
        elif separate == 'state':
            # Perturbation only applied to the state variables
            print('Separate state variables')
            
        elif separate == 'parameter':
            # Perturbation only applied to the parameter variables
            print('Separate parameter variables')
            
        else:
            print('Unrecognized separate command.')
            raise SystemExit
            
        return X0
    
    
    # Computing the analysis:
    def compute_analysis(self, Xb, yo):
        method = self.method
        if method == 'skip':
            xa = xb
            KH = np.identity(self.xdim)
        elif method == 'EnKF':
            xa, KH = self.ETKF(Xb, yo)   
        else:
            print('Unrecognized DA method.')
            raise SystemExit
        return xa,KH
    
    
    # Defining the Ensemble Transform Kalman Filter (ETKF) method:
    def ETKF(self, Xb, yo):
        
        #print('Running ETKF')
        
        # Set verbose to False:
        verbose = False                  # Set verbose to True to print out intermediate results
        
        # Make sure inputs are matrices:
        Xb = np.matrix(Xb)              # Convert background ensemble to a matrix (nxK)
        yo = np.matrix(yo).flatten().T  # Turn observation vector into a column vector (px1)
        
        # Get system dimensions:
        nr, nc = np.shape(Xb)             # Get the number of rows (state variables) and columns (ensemble members) of the background ensemble matrix
        xdim   = nr                       # Set the state dimension to the number of rows of the background ensemble matrix
        edim   = nc                       # Set the ensemble dimension to the number of columns of the background ensemble matrix
        ydim   = np.size(yo)              # Set the observation dimension to the number of elements in the observation vector
        Hlin   = self.H                   # Get the observation operator matrix
        
        
        # Apply observation operator to forecast ensemble
        Yb = np.matrix(np.zeros((ydim,edim)))  # Initialize the observation ensemble matrix
        for i in range(edim):
            Yb[:,i] = Hlin * Xb[:,i]           # Apply the observation operator to each ensemble member
        
        # Convert ensemble members to perturbations
        x_mean  = np.mean(Xb, axis = 1)                    # Compute the mean of the background ensemble
        y_mean  = np.mean(Yb, axis = 1)                    # Compute the mean of the observation ensemble
        Xb_pert = Xb - np.matlib.repmat(x_mean, 1, edim)   # Subtract the mean from the background ensemble
        #Xb_pert = 1.04 * Xb_pert                          # Qiwen: Inflate the background ensemble
        Yb_pert = Yb - np.matlib.repmat(y_mean, 1, edim)   # Subtract the mean from the observation ensemble
        Xb = Xb_pert
        Yb = Yb_pert
        
        
        # Compute R^{-1}
        R    = self.R
        Rinv = np.linalg.inv(R)                           # Compute the inverse of the observation error covariance matrix
        
        # Compute the weights: Yb^T * R^{-1}:
        YbT  = np.transpose(Yb)                      # Transpose the observation ensemble matrix
        C    = YbT @ Rinv                                 # Compute the matrix
        
        # Print out intermediate results:
        if verbose:
            print('shape of Xb: ', np.shape(Xb))
            print('Xb: ', Xb)
            print('x_mean: ', x_mean)
            print('Xb_pert: ', Xb_pert)
            print('Xb_pert check: ',  Xb - np.mean(Xb, axis=1))
            print('shape x_mean: ', np.shape(x_mean))
            print('shape of y_mean: ', np.shape(y_mean))
            print('shape of yo: ', np.shape(yo))
            print('Hlin: ', Hlin)
            print('R: ', R)
            print("Es simétrica:", np.allclose(self.R, self.R.T))
            print("Eigenvalores de R:", np.linalg.eigvals(self.R))
            print('YbT: ', YbT)
            print('Rinv: ', Rinv)
            print('C: ', C)
            

        
        # Compute eigenvalue decomposition for Pa_tilde = [(K-1) * I / rho + Yb^T * R^{-1} * Yb]^{-1} = [eigArg]^{-1}:
        rho      = 1.04                                           # Set the inflation factor
        I        = np.identity(edim)                           # Create an identity matrix
        eigArg   = (edim-1)*I/rho + C @ Yb_pert                     # Compute the argument inside parenthesis of Pa_tilde
        #eigArg   = 0.5 * (eigArg + eigArg.T)                   # DCC: Ensure that eigArg is symmetric
        lamda,P  = np.linalg.eigh(eigArg)                      # Compute the eigenvalues and eigenvectors of Pa_tilde. Suppose eigArg is symmetric.
        #lamda,P  = np.linalg.eig(eigArg)                      # Compute the eigenvalues and eigenvectors of Pa_tilde. Suppose eigArg is symmetric.
        Linv     = np.diag(1.0/lamda)                          # Compute the inverse of the eigenvalues
        Pa_tilde = P @ Linv @ P.T                              # Compute the inverse of Pa_tilde
        
        # Print out intermediate results:
        if verbose:
            print('rho: ', rho)
            print('I: ', I)
            print('lamda: ', lamda)
            print('Linv: ', Linv)
            print('eigArg: ', eigArg)
            print('Pa_tilde: ', Pa_tilde)
            print("¿R es simétrica? ", np.allclose(self.R, self.R.T))
            print("Eigenvalores de R:", np.linalg.eigvals(self.R))
            print("¿eigArg es simétrica? ", np.allclose(eigArg, eigArg.T))
            print("Eigenvalores de eigArg:", lamda)
            print("C @ Yb:")
            print(C @ Yb)
        
        # Compute matrix square root of Pa_tilde: Wa = [(K-1) * Pa_tilde]^{1/2}:
        Linvsqrt = np.diag(1/np.sqrt(lamda))                # Compute the square root of the inverse of the eigenvalues
        Wa       = np.sqrt((edim-1)) * P @ Linvsqrt @ P.T       # Compute the square root of Pa_tilde
        
        # Compute the mean update: wa_mean  = Pa_tilde * Yb^T * R^{-1} * (yo - y_mean) = Pa_tilde * C * (yo - y_mean):
        d       = yo - y_mean
        Cd      = C @ d
        wa_mean = Pa_tilde @ Cd
        
        # Print out intermediate results:
        if verbose:
            print('Linvsqrt: ', Linvsqrt)
            print('Wa: ', Wa)
            print('d: ', d)
            print('Cd: ', Cd)
            print('wa_mean: ', wa_mean)
            
        # Compute the perturbation update. Multiply Xb (perturbations) by each wa(i) and add xbbar
        Wa = Wa + np.matlib.repmat(wa_mean, 1, edim)          # Add the mean update to each column of Wa, forming a matrix whose columsn are the analyisis vectors wa(i)
        Xa = Xb @ Wa + np.matlib.repmat(x_mean, 1, edim)      # Compute the analysis ensemble
        
        # Compute KH:
        K = Xb @ Pa_tilde @ YbT @ Rinv
        KH = K @ Hlin
        
        # Print out intermediate results:
        if verbose:
            print ('Wa = ')
            print (Wa)
            print ('Xa = ')
            print (Xa)
            
        return Xa, KH
    
    
    
    # Defining function to save the DA system object:
    def save(self,outfile):
        with open(outfile,'wb') as output:
            pickle.dump(self,output,pickle.HIGHEST_PROTOCOL)
    
    
    # Defining function to load the DA system object:      
    def load(self,infile):
        with open(infile,'rb') as input:
            das = pickle.load(input)
            return das