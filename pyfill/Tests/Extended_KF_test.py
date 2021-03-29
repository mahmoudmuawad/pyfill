# -*- coding: utf-8 -*-


if __name__ == "__main__":
    
    import pyfill.Filters
    
        ######################### testing predict function ############################
    print("\n ######################### testing predict function ############################ \n")        
    
    def ROT(u_theta) : 
        rt = np.matrix([[cos(u_theta), sin(u_theta)],
                       [-sin(u_theta), cos(u_theta)]])
        return rt
    
    def pre_model(states, u) : 
        B = np.matrix([[cos(u[1]), 0.],
                        [sin(u[1]), 0.]])
        pre_states = ROT(u[1]) * ((states) - B * u)
        return pre_states
    
    def compute_fj(x, u) :
        F = np.matrix([[cos(u[1]), sin(u[1])],
                      [-sin(u[1]), cos(u[1])]])
        return F
    
    
    x = np.matrix(np.zeros((2 ,1)) + 0.5)
    p = np.array([[2., 0.],
                  [0., 2.]])
    u = np.matrix([[0.5*sqrt(2)],
                  [0.25*pi]])
    
    input_noise = np.array([[0.1, 0.0],
                            [0.0, 0.01*pi]])
    x_monte, p_monte = monteCarloPrediction(x, p, u, input_noise, pre_model, 5000)
    
    # Note : an important way to test predict must have u values
    x_e_pre, P_e_pre = Extended_KF_predict(x, p, u, Q, pre_model, compute_fj)
    
    print("filter prediction : ")
    print(x_e_pre)
    print(P_e_pre)
    print("\n")
    print("monte carlo prediction : ")
    print(x_monte)
    print(p_monte)
    x = np.matrix(np.zeros((2 ,1)) + 0.5)
    p = np.array([[2., 0.],
                  [0., 2.]])
    plot_state(x, p ,x_e_pre, P_e_pre, x_monte, p_monte, 'k','b', 'r')
    
    
    
    ############################ testing update ###################################
    print("\n ############################ testing update ################################### \n")
    
    def compute_h(x1, x2) :
        return x1 - x2
    
    def compute_H1(x, u) :
        H_1 = np.matrix([[1., 0.],
                         [0., 1.]])
        return H_1
    
    def compute_H2(x, u) : 
        return np.matrix([[-1., 0.],
                          [0., -1.]])
        
    z = np.matrix([[1.012],
                   [1.03]])
    R = np.matrix([[0.5, 0],
                   [0, 0.2]])    
    x = np.matrix(np.zeros((2 ,1)) + 0.5)
    p = np.array([[0.5, 0],
                  [0, 0.7]])
    x_up, p_up = Extended_KF_update(x, p, z, R, compute_h, compute_H1, compute_H2)
    print("filter update \n")
    print("x update : \n", x_up)
    print("p update : \n", p_up)
    
    z = np.matrix([[1.012],
                   [1.03]])
    R = np.matrix([[0.5, 0],
                   [0, 0.2]])    
    x = np.matrix(np.zeros((2 ,1)) + 0.5)
    p = np.array([[0.5, 0],
                  [0, 0.7]])
    x_lin, p_lin = linearMerge(x, p, z, R)
    print("l x update : \n", x_lin)
    print("l p update : \n", p_lin)
    
    plot_state(x, p ,x_up, p_up, x_lin, p_lin, 'k','b', 'r')