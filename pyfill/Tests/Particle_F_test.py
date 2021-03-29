    
######################### testing particle filter ##################################
if __name__ == "__main__":
    import pyfill.Filters
    
    def ROT(u_theta) : 
        rt = np.matrix([[cos(u_theta), sin(u_theta)],
                       [-sin(u_theta), cos(u_theta)]])
        return rt
    
    def pre_model(states, u) : 
        B = np.matrix([[cos(u[1]), 0.],
                        [sin(u[1]), 0.]])
        pre_states = ROT(u[1]) * ((states) - B * u)
        return pre_states   

    x=np.matrix([[1.],
                 [1.]])
    p=np.matrix([[0.01],
                 [0.08]])
        
    u = np.matrix([[1*sqrt(2)],
                   [0.25*pi]])
    z = np.matrix([[0.012],
                   [-0.003]])
    sense_std=np.matrix([0.01, 0.2]).T
    pc= predict_update(x, p, u, pre_model, z, sense_std, 10, 0.5)
    print(pc)

