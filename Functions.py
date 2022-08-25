#!/usr/bin/env python
# coding: utf-8

L = 1
L2 = 40

"""
SVD Methode für die POD Methode
"""
def SVD_Method(X, d, eps):
    Psi,Sigma,PhiT = np.linalg.svd(X)
    r = 0
    Lambda = np.power(Sigma[:d],2)
    for k in range(1,d):
        if(energie_ratio(k, Lambda) >= eps):
            r = k
            break
    lambda_r = Lambda[:r]
    Psi_r = Psi[:,:r]
    return Psi_r, lambda_r


"""
Lumley Methode für die POD Methode
"""
def Lumley_Method(X, d, eps):
    Lambda, Psi = np.linalg.eig(X@X.T)
    indx=np.argsort(Lambda)[::-1]
    Lambda=Lambda[indx]
    Psi=Psi[:,indx]
    r = 0
    for k in range(1,d):
        if(energie_ratio(k, Lambda) >= eps):
            r = k
            break
    lambda_r = Lambda[:r]
    Psi_r = Psi[:,:r]
    return Psi_r, lambda_r


"""
Sirovich Methode (Methode der Snapshots) für die POD Methode
"""
def Sirovich_Method(X, d, eps):
    Lambda, Phi = eigh(X.T@X)
    indx=np.argsort(Lambda)[::-1]
    Lambda=Lambda[indx]
    Phi=Phi[:,indx]
    r = 0
    for k in range(1,d):
        if(energie_ratio(k, Lambda) >= eps):
            r = k
            break
    Lambda_r = Lambda[:r]
    Psi_r = np.zeros((X.shape[0], r), dtype=complex)
    for i in range(r):
        Psi_r[:,i] = X @ ((1/np.sqrt(Lambda[i])) * Phi[:,i])
    return Psi_r, Lambda_r


"""
POD Methode zur Bestimmung der POD Basis.
Benutzt entweder SVD oder Lösung des Eigenwertproblems.

@Input: X (Snapshots als Matrix in n x m), eps (Schwellenwert zwischen [0,1])
@Output: Psi_r (POD Basis als Matrix n x r), lambda_r (Eigenwerte als Array 1 x r)
"""
def POD_Method(X, eps):
    d = np.linalg.matrix_rank(X)
    n, m = X.shape[0], X.shape[1]
    diff = n - m
    if(diff == 0):
        print('Benutze SVD Methode...')
        Psi_r, lambda_r = SVD_Method(X, d, eps)

    elif(diff < 0):
        print('Benutze Lumley Methode...')
        Psi_r, lambda_r = Lumley_Method(X, d, eps)
        
    elif(diff > 0):
        print('Benutze Sirovich Methode...')
        Psi_r, lambda_r = Sirovich_Method(X, d, eps)
        
    print(f'Rang der Approximation: r = {lambda_r.size}')

    return Psi_r, lambda_r


#############################################################################################
# Funktionen zum Simulieren der Wärmeleitungsgleichung sowie Bildung des POD-Galerkin ROMs #

def Simulate_Heat_ROM_with_tolerance(u, u0, x, t, alpha, eps):
    global L
    N = len(u0)
    dx = L/N

    
    ##### Offline-Teil (3): Pre-Computation of linear operator L_op #####
    
    # Compute POD Modes
    Psi_r, lambda_r = POD_Method(u, eps)
    r = Psi_r.shape[1]

    # Approximate second derivative with DFT and iDFT 
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
    dd_Psi_r = np.zeros_like(Psi_r)
    for jj in range(Psi_r.shape[1]):
        dd_Psi_r[:,jj] = fft.ifft(-(np.power(kappa,2)) * fft.fft(Psi_r[:,jj]))
        
    # Precompute linear operator L 
    L_op = np.power(alpha,2)*(Psi_r.T @ dd_Psi_r)

    
    ##### Online-Teil (1): Simulation des ROMs durch Runge-Kutta Time-Stepper #####
        
    # Project initial condition onto POD modes
    q0 = u0@Psi_r
    q0_ri = np.concatenate((np.real(q0),np.imag(q0)))
    
    def rhsHeat_ROM(q_ri, t):
        q = q_ri[:r] + (1j)*q_ri[r:]
        d_q = L_op @ q.T
        d_q_ri = np.concatenate((np.real(d_q),np.imag(d_q)))
        return d_q_ri
    
    q_ri = odeint(rhsHeat_ROM, q0_ri, t)
    q = q_ri[:,:r] + (1j)*q_ri[:,r:]
    
    
    ##### Online-Teil (2): Rekonstruktion des FOMs ######
    u_rom = Psi_r@q.T

    return u_rom


def Simulate_Heat_ROM_with_r(u, u0, x, t, alpha, r):
    global L
    N = len(u0)
    dx = L/N

    
    ##### Offline-Teil (3): Pre-Computation of linear operator L_op #####
    
    # Compute POD Modes 
    U,S,VT = np.linalg.svd(u)
    Psi_r = U[:,:r]
    
    # Approximate second derivative with DFT and iDFT 
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
    dd_Psi_r = np.zeros_like(Psi_r)
    for jj in range(Psi_r.shape[1]):
        dd_Psi_r[:,jj] = fft.ifft(-(np.power(kappa,2)) * fft.fft(Psi_r[:,jj]))
        
    # Precompute linear operator L 
    L_op = np.power(alpha,2)*(Psi_r.T @ dd_Psi_r)
    

    ##### Online-Teil (1): Simulation des ROMs durch Runge-Kutta Time-Stepper #####
    
    # Project initial condition onto POD modes
    q0 = u0@Psi_r
    q0_ri = np.concatenate((np.real(q0),np.imag(q0)))
    
    def rhsHeat_ROM(q_ri, t):
        q = q_ri[:r] + (1j)*q_ri[r:]
        d_q = L_op @ q.T
        d_q_ri = np.concatenate((np.real(d_q),np.imag(d_q)))
        return d_q_ri
    
    q_ri = odeint(rhsHeat_ROM, q0_ri, t)
    q = q_ri[:,:r] + (1j)*q_ri[:,r:]
    
    
    ##### Online-Teil (2): Rekonstruktion des FOMs ######
    u_rom = Psi_r@q.T

    return u_rom

def Simulate_Heat_ROM_with_tolerance_and_timing(u, u0, x, t, alpha, eps):
    global L
    N = len(u0)
    dx = L/N
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
    
    ##### Offline-Teil (3): Pre-Computation of linear operator L_op #####
    start_time = time.time()
    
    # Compute POD Modes 
    Psi_r, lambda_r = POD_Method(u, eps)
    r = Psi_r.shape[1]
    
    # Approximate second derivative with DFT and iDFT 
    dd_Psi_r = np.zeros_like(Psi_r)
    start_time = time.time()
    for jj in range(Psi_r.shape[1]):
        dd_Psi_r[:,jj] = fft.ifft(-(np.power(kappa,2)) * fft.fft(Psi_r[:,jj]))
        
    # Precompute linear operator L 
    L_op = np.power(alpha,2)*(Psi_r.T @ dd_Psi_r)
    linear_operator_time = time.time() - start_time

    
    ##### Online-Teil (1): Simulation des ROMs durch Runge-Kutta Time-Stepper #####
    start_time = time.time()
    
    # Project initial condition onto POD modes
    q0 = u0@Psi_r
    q0_ri = np.concatenate((np.real(q0),np.imag(q0)))

    def rhsHeat_ROM(q_ri, t):
        q = q_ri[:r] + (1j)*q_ri[r:]
        d_q = L_op @ q.T
        d_q_ri = np.concatenate((np.real(d_q),np.imag(d_q)))
        return d_q_ri
    

    q_ri = odeint(rhsHeat_ROM, q0_ri, t)
    simulation_time = time.time() - start_time

    q = q_ri[:,:r] + (1j)*q_ri[:,r:]
    
    
    ##### Online-Teil (2): Rekonstruktion des FOMs ######
    start_time = time.time()
    u_rom = Psi_r@q.T
    reconstruction_time = time.time() - start_time

    return u_rom, [linear_operator_time, simulation_time, reconstruction_time]

def Simulate_Heat_FOM(u0, x, t, alpha):
    global L
    N = len(u0)
    dx = L/N
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
    
    ##### Offline-Teil (1): Simulation im Frequenzbereich #####
    u0hat = fft.fft(u0)
    u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))

    def rhsHeat(uhat_ri,t,kappa,alpha):
        uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
        d_uhat = -alpha**2 * (np.power(kappa,2)) * uhat
        d_uhat_ri = np.concatenate((np.real(d_uhat),np.imag(d_uhat)))
        return d_uhat_ri
    
    uhat_ri = odeint(rhsHeat, u0hat_ri, t, args=(kappa,alpha))
    
    uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]
    uhat = uhat.T  # Since we want to deal with a spatial x temporal (n x m) matrix
    

    ##### Offline-Teil (2): Rekonstruktion im Phasenraum #####
    u = np.zeros_like(uhat)
    for m in range(uhat.shape[1]):
        u[:,m] = fft.ifft(uhat[:,m])

    return u

def Simulate_Heat_FOM_with_timing(u0, x, t, alpha):
    global L
    N = len(u0)
    dx = L/N
    kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)
    
    ##### Offline-Teil (1): Simulation im Frequenzbereich #####
    start_time = time.time()
    u0hat = fft.fft(u0)
    u0hat_ri = np.concatenate((u0hat.real,u0hat.imag))

    def rhsHeat(uhat_ri,t,kappa,alpha):
        uhat = uhat_ri[:N] + (1j) * uhat_ri[N:]
        d_uhat = -alpha**2 * (np.power(kappa,2)) * uhat
        d_uhat_ri = np.concatenate((np.real(d_uhat),np.imag(d_uhat)))
        return d_uhat_ri
    
    uhat_ri = odeint(rhsHeat, u0hat_ri, t, args=(kappa,alpha))
    simulation_time = time.time() - start_time
    
    uhat = uhat_ri[:,:N] + (1j) * uhat_ri[:,N:]
    uhat = uhat.T  # Since we want to deal with a spatial x temporal (n x m) matrix
    

    ##### Offline-Teil (2): Rekonstruktion im Phasenraum #####
    u = np.zeros_like(uhat)
    start_time = time.time()
    for m in range(uhat.shape[1]):
        u[:,m] = fft.ifft(uhat[:,m])
    reconstruction_time = time.time() - start_time

    return u, [simulation_time, reconstruction_time]

####################################################################################
# Funktionen zum Simulieren der NLS Gleichung sowie Bildung des POD-Galerkin  bzw. POD-DEIM Galerkin ROMs #

def Simulate_NLS_FOM(u0, x, t):
    global L2
    
    # Define discrete wavenumbers
    n = len(u0)
    omega = n*(2*np.pi/L2)*np.fft.fftfreq(n)
    
    ##### Simulate in Fourier frequency domain #####
    u0_hat = fft.fft(u0)
    
    def rhsNLS(u_hat,t):
        u = np.fft.ifft(u_hat)
        d_u_hat = -0.5*(1j)*np.power(omega,2)*u_hat + (1j)*np.fft.fft(np.power(np.abs(u),2)*u)
        return d_u_hat

    X_hat = odeintw(rhsNLS, u0_hat, t)
    X_hat = X_hat.T 
    

    ##### Retransform to spatial domain #####
    X = np.zeros_like(X_hat)
    for tt in range(len(t)):
        X[:,tt] = fft.ifft(X_hat[:,tt])

    return X

############### POD ###################

def NLS_POD_Offline_1(u0, x, t):
    ##### (off-1): Simulation des FOMs zur Bestimmung von X #####
    X = Simulate_NLS_FOM(u0, x, t)
    
    return X


def NLS_POD_Offline_2_3(X, r):
    ##### (off-2): Bestimmung der linken Singulärvektoren #####
    Psi,Sigma,PhiT = np.linalg.svd(X,full_matrices=0)
    Psi_r = Psi[:,:r] # Berechnung der POD Basis
    
    ##### (off-3): Vorberechnung der Operatoren #####
    n = len(u0)
    dd_Psi_r = np.zeros_like(Psi_r) # Approximation der zweiten Ableitung der POD Basis
    omega = n*(2*np.pi/L2)*np.fft.fftfreq(n) # Vektor der diskreten Fourier-Frequenzen
    for jj in range(r):
        dd_Psi_r[:,jj] = np.fft.ifft(-np.power(omega,2)*np.fft.fft(Psi_r[:,jj]))

    L_tilde = 0.5 * (1j) * Psi_r.T @ dd_Psi_r # Projektion des linearen Operators
    
    return (Psi_r, L_tilde)



def NLS_POD_Online(u0, t, operators):
    ##### (on-1): Simulation des POD-DEIM ROMs #####
    Psi_r, L_tilde = operators
    q0 = Psi_r.T @ u0 # Projektion der Anfangsbedingung
    
    def NLS_pod_rhs(q,t):
        d_q = L_tilde @ q + (1j) * Psi_r.T @ (np.power(np.abs(Psi_r @ q),2)*(Psi_r @ q))
        return d_q

    q = odeintw(NLS_pod_rhs,q0,t)
    
    ##### (on-2): Rekonstruktion des FOMs #####
    X_rom = Psi_r @ q.T

    return X_rom

##################################

############### DEIM ###################

def DEIM_Methode(XI, p):

    n = XI.shape[0]
    # First DEIM point
    nmax = np.argmax(np.abs(XI[:,0]))
    XI_m = XI[:,0].reshape(n,1)
    z = np.zeros((n,1))
    P = np.copy(z)
    P[nmax] = 1

    # DEIM points 2 to p
    for jj in range(1,p):
        c = np.linalg.solve(P.T @ XI_m, P.T @ XI[:,jj].reshape(n,1))
        res = XI[:,jj].reshape(n,1) - XI_m @ c
        nmax = np.argmax(np.abs(res))
        XI_m = np.concatenate((XI_m,XI[:,jj].reshape(n,1)),axis=1)
        P = np.concatenate((P,z),axis=1)
        P[nmax,jj] = 1
        
    return P

def NLS_POD_DEIM_Offline_1(u0, x, t):
    ##### (off-1): Simulation für Snapshot-Matrix X und Bestimmung der Snapshot-Matrix X_N #####
    X = Simulate_NLS_FOM(u0, x, t)
    X_N = (1j)*np.power(np.abs(X),2)*X
    
    return (X, X_N)

def NLS_POD_DEIM_Offline_2_3_4(X, X_N, p, r):
    global L2
    
    ##### (off-2): Bestimmung der linken Singulärvektoren ######
    Psi,Sigma,PhiT = np.linalg.svd(X)
    Psi_r = Psi[:,:r] # Berechnung der POD Basis
    XI,Sigma_NL,PhiT_NL = np.linalg.svd(X_N)
    XI_p = XI[:,:p]


    ##### (off-3): Bestimmung der Interpolationsindizes mit DEIM Methode #####
    P = DEIM_Methode(XI, p)
    
    
    ##### (off-4): Vorberechnung der Operatoren #####    
    P_Psi_r = P.T @ Psi_r # Interpolation der POD Basis
    dd_Psi_r = np.zeros_like(Psi_r) # Approximation der zweiten Ableitung der POD Basis
    n = len(u0)
    omega = n*(2*np.pi/L2)*np.fft.fftfreq(n) # Vektor der diskreten Fourier-Frequenzen
    for jj in range(r):
        dd_Psi_r[:,jj] = np.fft.ifft(-np.power(omega,2)*np.fft.fft(Psi_r[:,jj]))

    L_tilde = 0.5 * (1j) * Psi_r.T @ dd_Psi_r # Projektion des linearen Operators
    L_N = Psi_r.T @ (XI_p @ np.linalg.inv(P.T @ XI_p)) # Projektion des nicht-linearen Operators
    
    return (Psi_r, P_Psi_r, L_tilde, L_N)
    
    
def NLS_POD_DEIM_Online(u0, t, operators):
    ##### (on-1): Simulation des POD-DEIM ROMs #####
    Psi_r, P_Psi, L_tilde, L_N = operators
    q0 = Psi_r.T @ u0 # Projektion der Anfangsbedingung
    
    def NLS_pod_deim_rhs(q,t):
        N = P_Psi @ q # (p x r) Produkt
        d_q = L_tilde @ q + (1j) * L_N @ (np.power(np.abs(N),2)*N)
        return d_q

    q = odeintw(NLS_pod_deim_rhs,q0,t)
    
    ##### (on-2): Rekonstruktion des FOMs #####
    X_tilde = Psi_r @ q.T

    return X_tilde

##################################

############ Versionen mit Timings ###############

def NLS_POD_Offline_1_timed(u0, x, t):
    ##### (off-1): Simulation des FOMs zur Bestimmung von X #####
    start_time = time.time()
    X = Simulate_NLS_FOM(u0, x, t)
    off_1_time = time.time() - start_time
    
    return X, off_1_time

def NLS_POD_Offline_2_3_timed(X, r):
    ##### (off-2): Bestimmung der linken Singulärvektoren #####
    start_time = time.time()
    Psi,Sigma,PhiT = np.linalg.svd(X,full_matrices=0)
    Psi_r = Psi[:,:r] # Berechnung der POD Basis
    off_2_time = time.time()-start_time
    
    ##### (off-3): Vorberechnung der Operatoren #####
    start_time = time.time()
    n = len(u0)
    dd_Psi_r = np.zeros_like(Psi_r) # Approximation der zweiten Ableitung der POD Basis
    omega = n*(2*np.pi/L2)*np.fft.fftfreq(n) # Vektor der diskreten Fourier-Frequenzen
    for jj in range(r):
        dd_Psi_r[:,jj] = np.fft.ifft(-np.power(omega,2)*np.fft.fft(Psi_r[:,jj]))

    L_tilde = 0.5 * (1j) * Psi_r.T @ dd_Psi_r # Projektion des linearen Operators
    off_3_time = time.time() - start_time
    
    return (Psi_r, L_tilde), (off_2_time, off_3_time)

def NLS_POD_Online_timed(u0, t, operators):
    ##### (on-1): Simulation des POD-DEIM ROMs #####
    start_time = time.time()
    Psi_r, L_tilde = operators
    q0 = Psi_r.T @ u0 # Projektion der Anfangsbedingung
    
    def NLS_pod_rhs(q,t):
        d_q = L_tilde @ q + (1j) * Psi_r.T @ (np.power(np.abs(Psi_r @ q),2)*(Psi_r @ q))
        return d_q

    q = odeintw(NLS_pod_rhs,q0,t)
    on_1_time = time.time() - start_time
    
    ##### (on-2): Rekonstruktion des FOMs #####
    start_time = time.time()
    X_rom = Psi_r @ q.T
    on_2_time = time.time() - start_time

    return X_rom, (on_1_time, on_2_time)

def NLS_POD_DEIM_Offline_1_timed(u0, x, t):
    ##### (off-1): Simulation für Snapshot-Matrix X und Bestimmung der Snapshot-Matrix X_N #####
    start_time = time.time()
    X = Simulate_NLS_FOM(u0, x, t)
    X_N = (1j)*np.power(np.abs(X),2)*X
    off_1_time = time.time() - start_time
    
    return (X, X_N), off_1_time

def NLS_POD_DEIM_Offline_2_3_4_timed(X, X_N, p, r):
    global L2
    
    ##### (off-2): Bestimmung der linken Singulärvektoren ######
    start_time = time.time()
    Psi,Sigma,PhiT = np.linalg.svd(X,full_matrices=0)
    Psi_r = Psi[:,:r] # Berechnung der POD Basis
    XI,Sigma_NL,PhiT_NL = np.linalg.svd(X_N,full_matrices=0)
    XI_p = XI[:,:p]
    off_2_time = time.time() - start_time


    ##### (off-3): Bestimmung der Interpolationsindizes mit DEIM Methode #####
    start_time = time.time()
    P = DEIM_Methode(XI, p)
    off_3_time = time.time() - start_time
    
    
    ##### (off-4): Vorberechnung der Operatoren ##### 
    start_time = time.time()
    P_Psi_r = P.T @ Psi_r # Interpolation der POD Basis
    dd_Psi_r = np.zeros_like(Psi_r) # Approximation der zweiten Ableitung der POD Basis
    n = len(u0)
    omega = n*(2*np.pi/L2)*np.fft.fftfreq(n) # Vektor der diskreten Fourier-Frequenzen
    for jj in range(r):
        dd_Psi_r[:,jj] = np.fft.ifft(-np.power(omega,2)*np.fft.fft(Psi_r[:,jj]))

    L_tilde = 0.5 * (1j) * Psi_r.T @ dd_Psi_r # Projektion des linearen Operators
    L_N = Psi_r.T @ (XI_p @ np.linalg.inv(P.T @ XI_p)) # Projektion des nicht-linearen Operators
    off_4_time = time.time() - start_time
    
    return (Psi_r, P_Psi_r, L_tilde, L_N), (off_2_time, off_3_time, off_4_time)
    
    
def NLS_POD_DEIM_Online_timed(u0, t, operators):
    ##### (on-1): Simulation des POD-DEIM ROMs #####
    start_time = time.time()
    Psi_r, P_Psi, L_tilde, L_N = operators
    q0 = Psi_r.T @ u0 # Projektion der Anfangsbedingung
    
    def NLS_pod_deim_rhs(q,t):
        N = P_Psi @ q # (p x r) Produkt
        d_q = L_tilde @ q + (1j) * L_N @ (np.power(np.abs(N),2)*N)
        return d_q

    q = odeintw(NLS_pod_deim_rhs,q0,t)
    on_1_time = time.time() - start_time
    
    ##### (on-2): Rekonstruktion des FOMs #####
    start_time = time.time()
    X_tilde = Psi_r @ q.T
    on_2_time = time.time() - start_time

    return X_tilde, (on_1_time, on_2_time)

####################################################################################
