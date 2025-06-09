import numpy as np
import random
import pickle
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import linalg
from class_da_sys import da_system


class cse:
    
    def __init__(self, x0 = [], yo = [], t0 = 0, dt = 0, state_vector = [], obs_data = [], acyc_step = 1):
        self.xdim           = np.size(x0)
        self.ydim           = np.size(yo)
        self.edim           = 0
        self.yo             = yo
        self.yo_history     = []
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
        self.history        = []
        self.history_final  = []
        self.multiple_number = 4
        self.T_0            = 75.1
        self.Ta             = 8
        self.t_index        = 0
        self.xf4d           = []
    
    
    def __str__(self):
        print('CSE System Object')
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
    

    #
    # Defining the function to detect regime shifts:
    #
    def detect_regime_shift(self, forecast):
        return np.any(forecast < 0)
    
    
    #
    # Defining the function to classify ensemble members:
    #
    def classify_ensemble(self, Xf):
        predicted_changes = []
        safe_members = []
        for k in range(Xf.shape[0]):
            if self.detect_regime_shift(Xf[k, :, 0]):
                predicted_changes.append(k)
            else:
                safe_members.append(k)
        return predicted_changes, safe_members
    
    
    #
    # Defining the function to compute perturbation vector:
    #
    def compute_perturbation(self, xf_safe, xf_shift, D):
        delta = xf_safe - xf_shift
        #print(f' delta: {delta}')
        #print(f' (np.linalg.norm(delta) * D): {np.linalg.norm(delta) * D}')
        amp = 1 / D
        perturbation_vector = delta / (np.linalg.norm(delta, axis=1, keepdims=True) * amp)
        #perturbation_vector = delta / (np.linalg.norm(delta) * D)
        #print(f' perturbation_vector: {perturbation_vector}')
        # Check perturbations are normalized:
        print(f' perturbation_vector shape: {perturbation_vector.shape}')
        #print(f' perturbation_vector: {perturbation_vector[0,:]}')
        print('check perturbations are normalized: ', np.sqrt(perturbation_vector[0:10,0]**2 + perturbation_vector[0:10,1]**2 + perturbation_vector[0:10,2]**2))
        return perturbation_vector
    

    #
    # Defining the function to evolve nature with control:
    #
    def evolve_nature_with_control(self, x_init, perturb_vec, solver, cycle_steps):
        x = x_init.copy()
        history = []
        for q in range(cycle_steps):
            x_perturbed = x + perturb_vec[q, :]
            #print('x_perturbed shape: ', x_perturbed.shape)
            x_next = solver.run(x_perturbed, np.arange(0, 0.015, 0.01), np.arange(0, 0.015, 0.01))[1, :]
            ####history.append(x_perturbed[0])
            #history.append(x_perturbed[:])
            history.append(x_perturbed.tolist())
            #print('x_next shape: ', x_next.shape)
            #print('x_next: ', x_next)
            #print('history_evolve: ', history)
            ##print('history shape: ', np.shape(history))
            #print('---')
            x = x_next
            #print('x shape: ', x.shape)
        return x, history




    #
    # Defining function to perform the analysis step (Step 1 in CSE):
    #
    def CSE_Step1(self, Xf, yo, das):
        """
        STEP 1: PERFORM DA UPDATE:
        """
        print('CSE System Step 1')
        method = das.getMethod()
        print('Method: ', method)
        Xa, KH = das.compute_analysis(Xf,yo)
        return Xa, KH
    
    
    
    #
    # Defining function to perform the forecast step (Step 2 in CSE):
    #
    def CSE_Step2(self, multiple_number, T_0, solver, Xa, das):
        """
        STEP 2: RUN ENSEMBLE FORECAST:
        """
        print('CSE System Step 2')
        position = int(np.ceil(T_0*multiple_number)) #multiple of To
        print(f'Position: {position}')
        xf_4d = np.zeros((das.edim, position, das.xdim))
        print(f'initial condition forecast: {Xa}')
        # Run forecast for each ensemble member, using the initial condition from the posterior ensemble:
        for k in range(das.edim):
            xf_ens = solver.run(np.ravel(Xa[:,k]),np.arange(0,0.01 * position,0.01),np.arange(0,0.01 * position,0.01))
            xf_4d[k,:,:] = xf_ens
        return xf_4d
    
    
    
    #
    # Defining function to detect regime shift (Step 3 in CSE):
    #
    def CSE_Step3(self, Xf):
        """
        STEP 3: DETECT REGIME SHIFT:
        """
        print('CSE System Step 3')
        predicted_changes, safe_members = self.classify_ensemble(Xf)
        return predicted_changes, safe_members
    
    
    
    
    #
    # Defining function to advance Nature Run if none of the members show regime shift (Step 3b in CSE):
    #
    def CSE_Step3b(self, Xf, x_original, solver, Ta, D, das):
        """
        STEP 3B: ADVANCE NATURE RUN IF NONE OF THE MEMBERS SHOW REGIME SHIFT:
        """
        print('CSE System Step 3b: Advance Nature Run to the next time step where observations are available')
        #
        #x_history = [x_original[0]]
        x_history = [x_original[:]]
        #print(f'x_original: {x_original}')
        #x_history = []
        #
        # Evolve NR forward with control
        #x_fcst = solver.run(x_original, np.arange(0, 0.075, 0.01), np.arange(0, 0.075, 0.01))[1, :]
        x_next = solver.run(x_original, np.arange(0, 0.015, 0.01), np.arange(0, 0.015, 0.01))[1, :]
        x_fcst = solver.run(x_next, np.arange(0, 0.075, 0.01), np.arange(0, 0.075, 0.01))
        #print('x_next shape: ', x_next.shape)
        #print('x_fcst[:,0]: ', x_fcst[:,0])
        x_final = x_fcst[-1,:]
        x_history.extend(x_fcst[0:-1,:].tolist())
        ####x_history.append(x_fcst[0:-1,:].tolist())
        #print('x_history shape: ', np.shape(x_history))
        #print('x_history: ', x_history)
        #x_first_component_history = x_history[0,:,:]
        #x_first_component_history = [x[0] for x in x_history]
        return x_final, x_history
    
    
    
    
    
    #
    # Defining function to perform the Control (Step 4 in CSE):
    #
    def CSE_Step4(self, predicted_changes, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 4: CONTROL:
        """
        print('CSE System Step 4')
        #print(f'x_original shape: {x_original.shape}')
        #print(f'x_original: {x_original}')
        x_final_history = []
        #####x_final_history.append(x_original[0])
        x_final_history.append(x_original[:].tolist())
        # Apply control only if some predict shift and others don't
        if 0 < len(predicted_changes) < Xf.shape[0]:
            print(f'At least one ensemble member show regime shift. Activating control.')
            #self.Plot_Ensemble_Fcst_Step4(Xf, self.t_index, self.T_0, self.multiple_number, title = 'Ensemble forecast (T steps) initiated at $t = $'+str(self.t_index)+'\n (At least one ensemble member show regime shift. Activating control)')
            print(f'initial condition shift Xf[:,0,:]: {Xf[:,0,:]}')
            # Select members
            np.random.seed(self.t_index)     # For reproducibility
            shift_idx = np.random.choice(predicted_changes)
            safe_idx  = np.random.choice(safe_members)
            xf_shift  = Xf[shift_idx, :, :]
            xf_safe   = Xf[safe_idx, :, :]
            #
            # Compute perturbation vector
            perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)
            #
            # Evolve NR forward with control
            x_next = solver.run(x_original, np.arange(0, 0.015, 0.01), np.arange(0, 0.015, 0.01))[1, :]
            #print('x_next shape: ', x_next.shape)
            #print('initial conditions: ', x_next)
            x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
            x_final_history.extend(x_history)
            print(f'x_history: {x_history}')
            ####x_final_history.append(x_history)
            #
            #
            #print('history: ', self.history)
            #print('Control applied successfully.')
        else:
            print(f'All ensemble members show regime shift. Activating control using members from previous initial times for an extended forecasting period.')
            #
            # Check if any of the ensemble members show a regime shift starting from the last time step in the history:
            #
            all_shifted    = True
            time_shift     = 1
            time_shift_max = 20
            #self.Plot_Ensemble_Fcst_Step4(Xf, self.t_index, self.T_0, self.multiple_number)
            while all_shifted:
                print(f'self.history shape: {np.shape(self.history_final)}')
                #print(f'self.history: {self.history_final}')
                #print('self.history[-time_shift] shape: ', np.shape(self.history_final[-time_shift]))
                #print('self.history_final[-(time_shift+1): ', self.history_final[-(time_shift+1)][0,:])
                #print('self.history_final[-(time_shift+1)]:', np.array(self.history_final[-(time_shift+1)])[0, :])
                history_arr = np.array(self.history[-(time_shift+1)])[:, :]
                print(f'history_arr: {history_arr}')   
                #shift_ensemble = self.xf4d
                #print(f'shift_ensemble[:,0,:]: {shift_ensemble[:,0,:]}')
                history_current = np.array(self.history[-1])[:, :]
                print(f'history_current: {history_current}')
                #print(f'shift_ensemble shape: {shift_ensemble.shape}')
                print(f'history_arr shape: {history_arr.shape}')

                #
                # STEP 2: Run ensemble forecast with the last time step in the history:
                #
                print(f'  STEP 2: Running ensemble forecast with time shift: {time_shift}')
                #xf_4d = self.CSE_Step2(self.multiple_number, self.T_0, solver, self.history_final[-(time_shift+1)].T, das)
                xf_4d = self.CSE_Step2(self.multiple_number, self.T_0, solver, history_arr.T, das)
                shift_ensemble = xf_4d.copy()
                print(f'  xf_4d shape: {xf_4d.shape}')
                print(f'  initial condition xf_4d[:,8,:].T: {xf_4d[:,8,:].T}')
                print(f'  initial condition xf_4d[:,8,:]: {xf_4d[:,8,:]}')
                t_index_incr = Ta*time_shift
                t_index_shifted = self.t_index-(t_index_incr)
                print(f'  t_index: {self.t_index}')
                print(f'  t_index_shifted: {t_index_shifted}')
                self.Plot_Ensemble_Fcst_Step4(xf_4d, t_index_shifted, self.T_0, self.multiple_number, title = 'All ensemble members show regime shift.')
                #
                # STEP 3: Detect regime shift:
                #
                print('  STEP 3: Detecting regime shift ensemble members...')
                predicted_changes, safe_members = self.CSE_Step3(xf_4d)
                print(f'  Shift predicted by: {predicted_changes}')
                print(f'  Safe ensemble members: {safe_members}')
                #print('len(safe_members): ', len(safe_members))
                # Check if any safe members are found:
                if len(safe_members) > 0:
                    all_shifted = False
                    print(f'  Found safe ensemble members after {time_shift} time steps.')
                    Xf = xf_4d.copy()
                    t_index_incr = Ta*time_shift
                    t_index_shifted = self.t_index-(t_index_incr)
                    print('t_index: ', self.t_index)
                    print('t_index_incr: ', t_index_incr)
                    print('t_index_shifted: ', t_index_shifted)
                    #self.Plot_Ensemble_Fcst_Step4(Xf, t_index_shifted, self.T_0, self.multiple_number)
                    #
                    # Step 4: Control
                    #
                    print('  STEP 4: Applying control...')
                    #print('shape xf_4d: ', xf_4d.shape)
                    print(f'  initial condition shift xf_4d[:,0,:]: {xf_4d[:,0,:]}')
                    #
                    # Select members
                    #
                    shift_idx = np.random.choice(predicted_changes)
                    safe_idx  = np.random.choice(safe_members)
                    #xf_shift  = xf_4d[shift_idx, t_index_incr:Ta+1, :]
                    ttt = xf_4d.shape[1] - t_index_incr
                    print(f' ttt: {ttt}')
                    xf_shift  = shift_ensemble[shift_idx, 0:ttt, :]
                    print(f' safe_idx: {safe_idx}')
                    print(f' t_index_incr: {t_index_incr}')
                    print(f'shape xf_4d:', xf_4d.shape)
                    xf_safe   = xf_4d[safe_idx, t_index_incr:ttt+t_index_incr:1, :]
                    print(f'  xf_shift shape: {xf_shift.shape}')
                    kk = np.arange(t_index_incr,200+t_index_incr,1)
                    print(f'kk shape: ', np.shape(kk))
                    #print(f'  xf_shift: {xf_shift}')
                    #print(f' xf_safe: {xf_safe}')
                    print(f'  xf_safe shape: {xf_safe.shape}')
                    #
                    # Compute perturbation vector
                    print('Computing perturbation vector...')
                    #print(f' D: {D}')
                    perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)
                    #print(f'perturb_vec: {perturb_vec}')
                    #
                    #
                    # Evolve NR forward with control
                    x_next  = solver.run(x_original, np.arange(0, 0.015, 0.01), np.arange(0, 0.015, 0.01))[1, :]
                    x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
                    x_final_history.extend(x_history)
                    print(f'  x_history: {x_history}')
                    print(f'  x_original: {x_original}')
                    print(f'  x_final: {x_final}')
                    print('   x_final_history: ', x_final_history)
                    x_history_only = [x[0] for x in x_final_history]
                    print('   x_history_only: ', x_history_only)
                    print(np.arange(self.t_index,Ta+self.t_index,1))
                    #####self.Plot_All_Regime_Shift(Xf, x_original, x_history_only, t_index_shifted, self.t_index, self.T_0, Ta, self.multiple_number)
                    #####x_final_history.append(x_history)
                    print('  Control applied successfully.')
                else:
                    print(f'  No safe ensemble members found after {time_shift} time steps. Continuing to search.')
                    time_shift += 1
                    if time_shift > time_shift_max:
                        # If no safe members are found after max time steps, exit loop:
                        Xf = xf_4d.copy()
                        print('t_index: ', self.t_index)
                        t_index_incr = Ta*time_shift
                        t_index_shifted = self.t_index-(t_index_incr)
                        self.Plot_Ensemble_Fcst_Step4(Xf, t_index_shifted, self.T_0, self.multiple_number, title = '')
                        print(f'  No safe ensemble members found after {time_shift_max} time steps. Exiting loop.')
                        print('  EXITING CONTROL LOOP!!!!')
                        return
        return x_final, x_final_history
    
    
    
    #
    # Defining function to perform the Control (Step 4 in CSE):
    #
    def CSE_Step4a(self, predicted_changes, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 4a: CONTROL:
        """
        print('CSE System Step 4a')
        x_final_history = []
        x_final_history.append(x_original[:].tolist())
        # Apply control only if some predict shift and others don't
        if 0 < len(predicted_changes) < Xf.shape[0]:
            print(f'At least one ensemble member show regime shift. Activating control.')
            print(f'initial condition shift Xf[:,0,:]: {Xf[:,0,:]}')
            # Select members
            np.random.seed(self.t_index)     # For reproducibility
            shift_idx = np.random.choice(predicted_changes)
            safe_idx  = np.random.choice(safe_members)
            xf_shift  = Xf[shift_idx, :, :]
            xf_safe   = Xf[safe_idx, :, :]
            #
            # Compute perturbation vector
            perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)
            print(f'perturb_vec: {perturb_vec}')
            print(f' x_safe: {xf_safe}')
            print(f' x_shift: {xf_shift}')
            #
            # Evolve NR forward with control
            x_next = solver.run(x_original, np.arange(0, 0.015, 0.01), np.arange(0, 0.015, 0.01))[1, :]
            print('x_next shape: ', x_next.shape)
            print('x_next: ', x_next)
            x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
            x_final_history.extend(x_history)
            print(f'x_history: {x_history}')
        return x_final, x_final_history
    
    
    
    #
    # Defining function to perform the Control (Step 4 in CSE):
    #
    def CSE_Step4b(self, predicted_changes, safe_members, Xf, x_original, solver, Ta, D, das):
        """
        STEP 4b: CONTROL:
        """
        print('CSE System Step 4b')
        x_final_history = []
        x_final_history.append(x_original[:].tolist())
        # Apply control only if some predict shift and others don't
        print(f'All ensemble members show regime shift. Activating control using members from previous initial times for an extended forecasting period.')
        #
        # Check if any of the ensemble members show a regime shift starting from the last time step in the history:
        #
        all_shifted    = True
        time_shift     = 1
        time_shift_max = 200
        while all_shifted:
            print(f'self.history shape: {np.shape(self.history_final)}')
            history_arr = np.array(self.history[-(time_shift+1)])[:, :]
            print(f'history_arr: {history_arr}')   
            history_current = np.array(self.history[-1])[:, :]
            print(f'history_current: {history_current}')
            print(f'history_arr shape: {history_arr.shape}')
            #
            # STEP 2: Run ensemble forecast with the last time step in the history:
            #
            print(f'  Running ensemble forecast with time shift: {time_shift}')
            xf_4d = self.CSE_Step2(self.multiple_number, self.T_0, solver, history_arr.T, das)
            #shift_ensemble = xf_4d.copy()
            print(f'  xf_4d shape: {xf_4d.shape}')
            print(f'  initial condition xf_4d[:,8,:].T: {xf_4d[:,8,:].T}')
            print(f'  initial condition xf_4d[:,8,:]: {xf_4d[:,8,:]}')
            t_index_incr = Ta*time_shift
            t_index_shifted = self.t_index-(t_index_incr)
            print(f'  t_index: {self.t_index}')
            print(f'  t_index_shifted: {t_index_shifted}')
            #
            # STEP 3: Detect regime shift:
            #
            print('  Detecting regime shift ensemble members...')
            predicted_changes, safe_members = self.CSE_Step3(xf_4d)
            print(f'  Shift predicted by: {predicted_changes}')
            print(f'  Safe ensemble members: {safe_members}')
            # Check if any safe members are found:
            if len(safe_members) > 0:
                all_shifted = False
                print(f'  Found safe ensemble members after {time_shift} time steps.')
                #Xf = xf_4d.copy()
                t_index_incr = Ta*time_shift
                t_index_shifted = self.t_index-(t_index_incr)
                print('t_index: ', self.t_index)
                print('t_index_incr: ', t_index_incr)
                print('t_index_shifted: ', t_index_shifted)
                #
                # Step 4: Control
                #
                print('  Applying control...')
                print(f'  initial condition shift xf_4d[:,0,:]: {xf_4d[:,0,:]}')
                #
                # Select members
                #
                shift_idx = np.random.choice(predicted_changes)
                safe_idx  = np.random.choice(safe_members)
                ttt = xf_4d.shape[1] - t_index_incr
                print(f' ttt: {ttt}')
                xf_shift  = Xf[0, 0:ttt, :]
                print(f' safe_idx: {safe_idx}')
                print(f' shift_idx: {shift_idx}')
                print(f' t_index_incr: {t_index_incr}')
                print(f'shape xf_4d:', xf_4d.shape)
                xf_safe   = xf_4d[safe_idx, t_index_incr:ttt+t_index_incr:1, :]
                print(f'  xf_shift shape: {xf_shift.shape}')
                #
                # Compute perturbation vector
                print('Computing perturbation vector...')
                print(f' x_safe: {xf_safe}')
                print(f' x_shift: {xf_shift}')
                perturb_vec = self.compute_perturbation(xf_safe, xf_shift, D)
                #print(f' x_safe - x_shift: {xf_safe - xf_shift}')
                print(f'perturb_vec: {perturb_vec}')
                #
                #
                # Evolve NR forward with control
                x_next  = solver.run(x_original, np.arange(0, 0.015, 0.01), np.arange(0, 0.015, 0.01))[1, :]
                print('x_next shape: ', x_next.shape)
                print('x_next: ', x_next)
                x_final, x_history = self.evolve_nature_with_control(x_next, perturb_vec, solver, Ta - 1)
                x_final_history.extend(x_history)
                print(f'  x_history: {x_history}')
                print(f'  x_original: {x_original}')
                print(f'  x_final: {x_final}')
                print('   x_final_history: ', x_final_history)
                x_history_only = [x[0] for x in x_final_history]
                print('   x_history_only: ', x_history_only)
                print(np.arange(self.t_index,Ta+self.t_index,1))
                print('  Control applied successfully.')
            else:
                print(f'  No safe ensemble members found after {time_shift} time steps. Continuing to search.')
                time_shift += 1
                if time_shift > time_shift_max:
                    # If no safe members are found after max time steps, exit loop:
                    Xf = xf_4d.copy()
                    print('t_index: ', self.t_index)
                    t_index_incr = Ta*time_shift
                    t_index_shifted = self.t_index-(t_index_incr)
                    print(f'  No safe ensemble members found after {time_shift_max} time steps. Exiting loop.')
                    print('  EXITING CONTROL LOOP!!!!')
                    return
        return x_final, x_final_history
    
    
    
    
    #
    # Defining function to create new observations at t + Ta (Step 5 in CSE):
    #
    def CSE_Step5(self, nature, das):
        """
        STEP 5: CREATE NEW OBSERVATIONS AT t + Ta:
        """
        print('CSE System Step 5')
        new_nr = nature.copy()
        yo_new = new_nr + np.random.normal(0, np.sqrt(2), das.xdim)
        return yo_new
        
    
    #
    # Plotting function for the ensemble forecast (Step 2):
    #
    def Plot_Ensemble_Fcst_Step2(self, x_nature, Xf, Xa, t_index, T_0, multiple_number):
        """
        Plot the ensemble forecast:
        """
        print('CSE System Plot Ensemble Forecast Step 2')
        #
        T = int(np.ceil(T_0*multiple_number))                  # multiple of To
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        plt.plot(x_nature[0,105768:105768+3*350], linestyle = 'solid', linewidth=3, marker='', markersize = 3, color = 'black', label="Nature Run")
        for member in range(np.shape(Xf)[0]):
            plt.scatter(t_index, Xa[0, member], marker='o', s=45, color = 'cyan', edgecolors='black', linewidths=0.7)
            if member == 0:
                #plt.plot(np.arange(t_index,T,1), Xf[member,0:T-t_index,0], linestyle = '-', marker='', markersize = 2, color = 'cyan', label = f"Forecast Ensemble")
                plt.plot(np.arange(t_index,T+t_index,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan', label = f"Forecast Ensemble")
            else:
                #plt.plot(np.arange(t_index,T,1), Xf[member,0:T-t_index,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')
                plt.plot(np.arange(t_index,T+t_index,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')    
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'cyan')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel('time step (1 step = 0.01)', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run and Ensemble Forecasts at $t = $'+str(t_index), fontsize = 18)
        #xticks = list(plt.xticks()[0]) + [t_index]
        #xticks = sorted(set(xticks))
        #ax.set_xticks(xticks)
        #labels = [r"Ta" if int(tick) == t_index else str(int(tick)) for tick in xticks]
        #ax.set_xticklabels(labels, fontsize=14)
        #ax.get_xticklabels()[labels.index("Ta")].set_color('cyan')
        ax.set_xticks(np.arange(-1, 21, 1))
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        #plt.xlim(-10,350)
        plt.xlim(-1,20)
        ax.legend(fontsize=14)
        plt.show()
        fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
        
        
    
    #
    # Plotting function for the ensemble forecast (Step 2):
    #
    def Plot_Ensemble_Fcst_Step4(self, Xf, Xa, yo, t_index, T_0, multiple_number, title):
        """
        Plot the ensemble forecast:
        """
        print('CSE System Plot Ensemble Forecast Step 2')
        #
        T = int(np.ceil(T_0*multiple_number))                  # multiple of To
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        for member in range(np.shape(Xf)[0]):
            if member == 0:
                plt.plot(np.arange(t_index,T+t_index,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan', label = f"Ensemble Forecast")
                plt.scatter(t_index, Xa[0, member], marker='x', s=65, color = 'blue', edgecolors='black', linewidths=0.7, label="Ensemble Analysis")
            else:
                plt.plot(np.arange(t_index,T+t_index,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')   
                plt.scatter(t_index, Xa[0, member], marker='x', s=65, color = 'blue', edgecolors='black', linewidths=0.7) 
        plt.scatter(t_index, yo[0], marker='o', s = 30, facecolors='red', edgecolors='red', label="Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'cyan')
        plt.axvline(x = t_index+self.Ta, linestyle='dashed', color = 'cyan')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel('time step (1 step = 0.01)', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        #plt.title('Lorenz-63 Nature Run and Initial Ensemble Forecasts at $t = $'+str(t_index)+'\n ('+title+')', fontsize = 18)
        plt.title(title, fontsize = 18)
#        xticks = list(plt.xticks()[0]) + [t_index]
#        xticks = sorted(set(xticks))
#        ax.set_xticks(xticks)
#        labels = [r"Ta" if int(tick) == t_index else str(int(tick)) for tick in xticks]
#        ax.set_xticklabels(labels, fontsize=14)
#        ax.get_xticklabels()[labels.index("Ta")].set_color('cyan')
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        plt.xlim(-2,self.t_index+350)
        ax.legend(fontsize=10)
        plt.show()
        #fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
    
    
    
    
    
    #
    # Plotting function for the ensemble forecast (Step 2):
    #
    def Plot_All_Regime_Shift(self, Xf, X_orig, X_history, t_index_shift, t_index, T_0, Ta, multiple_number):
        """
        Plot the ensemble forecast:
        """
        print('CSE System Plot Ensemble Forecast Step 2')
        #
        T = int(np.ceil(T_0*multiple_number))                  # multiple of To
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        for member in range(np.shape(Xf)[0]):
            if member == 0:
                plt.plot(np.arange(t_index_shift,T+t_index_shift,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan', label = f"Forecast Ensemble")
            else:
                plt.plot(np.arange(t_index_shift,T+t_index_shift,1), Xf[member,0:T,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')
        plt.plot(np.arange(t_index,Ta+t_index,1), X_history[0:T], linestyle = '-', marker='', markersize = 2, color = 'red', label = f"Forecast Ensemble")  
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index_shift, linestyle='dashed', color = 'cyan')
        plt.axvline(x = t_index, linestyle='dashed', color = 'red')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel('time step (1 step = 0.01)', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run and Initial Ensemble Forecasts at $t = $'+str(t_index), fontsize = 18)
#        xticks = list(plt.xticks()[0]) + [t_index]
#        xticks = sorted(set(xticks))
#        ax.set_xticks(xticks)
#        labels = [r"Ta" if int(tick) == t_index else str(int(tick)) for tick in xticks]
#        ax.set_xticklabels(labels, fontsize=14)
#        ax.get_xticklabels()[labels.index("Ta")].set_color('cyan')
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        #plt.xlim(-10,350)
        ax.legend(fontsize=14)
        plt.show()
        #fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
    
    
    

    #
    # Plotting function for the ensemble forecast (Step 2):
    #
    def Plot_Ensemble_Fcst_Step5(self, x_nature, x_nature_control, Xf, Xa, yo, t_index, T_0, multiple_number):
        """
        Plot the ensemble forecast:
        """
        print('CSE System Plot Ensemble Forecast Step 2')
        #
        T = int(np.ceil(T_0*multiple_number))     # multiple of To
        x_nature_control_3d = np.asarray(x_nature_control)
        #
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        plt.plot(x_nature[0,105768:105768+3*350], linestyle = 'solid', linewidth=3, marker='', markersize = 3, color = 'black', label="Nature Run")
        plt.plot(np.arange(t_index-8,t_index,1), x_nature_control_3d[:,0], linestyle = 'solid', linewidth=2, marker='', markersize = 3, color = 'red', label="Nature Run (Control)")
        print('Xf shape: ', np.shape(Xf))
        print('Xf[:,0, 0]: ', Xf[:, 8, 0])
        #print('Xf[:,t_index-8, 0]: ', Xf[:,t_index-8, 0])
        print('Xa[0, :]: ', Xa[0, :])
        for member in range(np.shape(Xf)[0]):
            if member == 0:
                plt.scatter(t_index, Xa[0, member], marker='o', s=45, color = 'deeppink', edgecolors='black', linewidths=0.7, label="Ensemble Analysis at t")
                plt.scatter(t_index, Xf[member,8, 0], marker='x', s=45, color = 'black', edgecolors='black', linewidths=0.7, label="Ensemble Forecast at t")
                plt.scatter(t_index-8, Xf[member,0, 0], marker='o', s=45, color = 'cyan', edgecolors='black', linewidths=0.7, label="Ensemble Forecast at t-Ta")
                plt.plot(np.arange(t_index-8,t_index+1,1), Xf[member,0:8+1,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')
            else:
                plt.scatter(t_index, Xa[0, member], marker='o', s=45, color = 'deeppink', edgecolors='black', linewidths=0.7)
                plt.scatter(t_index, Xf[member,8, 0], marker='x', s=45, color = 'black', edgecolors='black', linewidths=0.7)
                plt.scatter(t_index-8, Xf[member,0, 0], marker='o', s=45, color = 'cyan', edgecolors='black', linewidths=0.7)
                plt.plot(np.arange(t_index-8,t_index+1,1), Xf[member,0:8+1,0], linestyle = '-', marker='', markersize = 2, color = 'cyan')
        plt.scatter(t_index, yo[0], marker='o', s = 30, facecolors='red', edgecolors='red', label="Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')   
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index-8, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'cyan')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel('time step (1 step = 0.01)', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run and Ensemble Forecasts at $t = $'+str(t_index), fontsize = 18)
        #xticks = list(plt.xticks()[0]) + [t_index]
        #xticks = sorted(set(xticks))
        ax.set_xticks(np.arange(t_index-10, t_index+10, 1))
        #labels = [r"Ta" if int(tick) == t_index else str(int(tick)) for tick in xticks]
        #ax.set_xticklabels(labels, fontsize=14)
        #ax.get_xticklabels()[labels.index("Ta")].set_color('cyan')
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        plt.xlim(t_index-10,t_index+10)
        ax.legend(fontsize=14)
        plt.show()
        fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')
        
        
     
    
    def Plot_New_NR_and_Obs(self, x_init, x_nature, yo_new, yo_old, t_index, Ta, solver, das):
        """
        Plot the new nature run and observations:
        """
        print('Evolve Nature Run 350 - t time steps forward')
        print('t_index: ', t_index)
        print('350 - t_index: ', 350 - t_index)
        print('shape np.arange(t_index,350): ', np.shape(np.arange(t_index,350)))
        print('shape np.arange(t_index,350-t_index): ', np.shape(np.arange(t_index,350-t_index)))   
        print('shape np.arange(0, 0.01 * (350 - t_index), 0.01): ', np.shape(np.arange(0, 0.01 * (350 - t_index), 0.01)))    
        x_nature_evolved = solver.run(x_init, np.arange(0, 0.01 * (350 - t_index - 0.01), 0.01), np.arange(0, 0.01 * (350 - t_index - 0.01), 0.01))[:,0]
        print('x_nature_evolved shape: ', x_nature_evolved.shape)
        print('x_nature_evolved: ', x_nature_evolved)
        print('CSE System Plot New Nature Run and Observations')
        fig = plt.figure(figsize = (12,12))
        ax  = fig.add_subplot(211)
        plt.plot(x_nature[0,105768:105768+350], linestyle = 'solid', linewidth=3, marker='', markersize = 3, color = 'black', label="Nature Run before control")
        plt.plot(np.arange(t_index,350),x_nature_evolved, linestyle = 'solid', linewidth=3, marker='', markersize = 3, color = 'red', label="Nature Run after control")
        plt.scatter(t_index,yo_new[0], marker='o', s = 30, facecolors='deeppink', edgecolors='deeppink', label="Observations")
        plt.scatter(t_index,yo_old[0,t_index], marker='o', s = 30, facecolors='green', edgecolors='green', label="NR Observations")
        plt.axhline(y = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = 0, linestyle='dashed', color = 'grey')
        plt.axvline(x = t_index, linestyle='dashed', color = 'red')
        plt.axvline(x = 300, linestyle='dashed', color = 'grey')
        plt.xlabel('time step (1 step = 0.01)', fontsize = 16)
        plt.ylabel('state x', fontsize = 16)
        plt.title('Lorenz-63 Nature Run after control at time $t = $'+str(t_index), fontsize = 18)
        #xticks = list(plt.xticks()[0]) + [t_index]
        #xticks = sorted(set(xticks))
        #ax.set_xticks(xticks)
        #labels = [r"Ta" if int(tick) == t_index else str(int(tick)) for tick in xticks]
        #ax.set_xticklabels(labels, fontsize=14)
        #ax.get_xticklabels()[labels.index("Ta")].set_color('red')
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.ylim(-18,18)
        plt.xlim(-10,350)
        ax.legend(fontsize=14)
        plt.show()
        #fig.savefig('Forecasts_Step2_Time_Step_'+str(t_index)+'.png', dpi=300, bbox_inches = 'tight')
        plt.close(fig)
        print('Plotting complete')