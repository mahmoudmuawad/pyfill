# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import pyfill.Filters
    
    # Testing linear kalman filter 
    # Model will be x_pred = x_old + x_dot *dt   --->
    dt = 0.01
    A = np.matrix([[1.0, dt],
                   [1.0, 0.]])
    B = u = 0 # note : there is no control action in the model equation
    x_old = np.matrix([[0.0], 
                       [100.]])
    p_old = np.matrix([[0.5, 0.0], 
                       [0.0, 0.5]])
    R = np.matrix([[0.0, 0.0], 
                   [0.0, 0.0]])
    # prediction step
    x_new, p_new = Linear_KF_predict(x_old, p_old, u, R, A, B)
    print("x_pred is  : \n", x_new)
    print("p_pred is  : \n", p_new)
    
    # update step
    z = np.matrix([0.92])
    Q = np.matrix([0.1])
    C = np.matrix([1.0, 0.0])
    x_corr, p_corr = Linear_KF_update(x_new, p_new, z, Q, C)
    
    print("x_update is : \n", x_corr)
    print("p_update : \n", p_corr)

