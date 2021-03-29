
import numpy
import matplotlib.patches as patches
import numpy as np
from numpy import matrix
from numpy import linalg
import math
import matplotlib.pyplot as plt
import sys
import traceback


def GetDistBtwPts(point1 , point2):
    point1 = numpy.matrix(point1)
    point2 = numpy.matrix(point2)
    
    dist = numpy.sqrt( (point1[0,0]- point2[0,0])**2 + (point1[1,0]- point2[1,0])**2 )
    
    return dist 
    
# Creates an ellipse from its position and its covariance matrix
def Create_ellipse(Pos,Cov,color,style):
    
    vals, vecs = numpy.linalg.eigh(Cov)
    order = vals.argsort()[::-1]
    vals=vals[order]
    vecs=vecs[:,order]
    theta = numpy.degrees(numpy.arctan2(*vecs[:,0][::-1]))
    
    return patches.Ellipse(Pos,2*numpy.sqrt(vals[0]),2*numpy.sqrt(vals[1]),theta,fill=False, color=color, linestyle=style)
    
#def PlotEllipse(ax, mean, cov, color, style):
#    
#    ax.add_patch(Create_ellipse(mean, cov, color, style))
    
def DrawPoseWithEllipse(ax, pose, cov_pose, color_mean, color_cov , style = None, label=None):
    
    ax.add_patch(Create_ellipse(pose, cov_pose[0:2, 0:2], color_cov, style))
#    PlotEllipse(ax, pose, cov_pose[0:2, 0:2], color, style)
    return ax.plot(pose[0], pose[1], color_mean + 'o', label=label)
    
def DrawUnsharpEllipse(ax, center, a , b , theta, color_ellipse,style = None, label=None ):
    
    ax.add_patch(patches.Ellipse(center,2*a,2*b,theta,fill=False, color=color_ellipse, linestyle=style))
    
    return ax.plot(center[0], center[1], color_ellipse + 'o', label=label)
    
def DrawUnsharpMeas(ax, meas_dist,f1,f2,color_ellipse,style = None, label=None):
    center = (f1 + f2)/2 
    e = GetDistBtwPts(center,f1)
    a = meas_dist/2
    b = numpy.sqrt(a**2 - e**2)
    theta = numpy.degrees(numpy.arctan2((f1[1,0]-f2[1,0]),(f1[0,0]-f2[0,0])))
    
    ax.add_patch(patches.Ellipse(center,2*a,2*b,theta,fill=False, color=color_ellipse, linestyle=style))
    
    return ax.plot(f1[0], f1[1], color_ellipse + 'o', label=label),ax.plot(f2[0], f2[1], color_ellipse + 'o', label=label)
    
def Create_CircularArc(center, radius, theta1, theta2, color, style):
    lx = 2*radius
    ly = 2*radius
    return patches.Arc(center, lx, ly, 0.0, theta1, theta2, color=color, linestyle=style)
   
def DrawCircularArcWithStd(ax, center, radius, std, theta1, theta2, color, style= None):
    
    ax.add_patch(Create_CircularArc(center, radius, theta1, theta2, color, style))
    ax.add_patch(Create_CircularArc(center, (radius+std/2), theta1, theta2, color, style))
    ax.add_patch(Create_CircularArc(center, (radius-std/2), theta1, theta2, color, style))
    
    ax.plot(center[0], center[1], color + 'o')
    
    start_pt_x = (radius+std/2)*numpy.cos(math.radians(theta1)) + numpy.asscalar(center[0])
    start_pt_y = (radius+std/2)*numpy.sin( math.radians(theta1)) + numpy.asscalar(center[1])
    end_pt_x = (radius+std/2)*numpy.cos(math.radians(theta2)) + numpy.asscalar(center[0])
    end_pt_y = (radius+std/2)*numpy.sin(math.radians(theta2)) + numpy.asscalar(center[1])
    
    ax.plot([center[0], start_pt_x],[center[1], start_pt_y], color,style)
    ax.plot([center[0], end_pt_x],[center[1], end_pt_y], color,style)
    
def DrawCircularArcWithStdWithoutLines(ax, center, radius, std, theta1, theta2, color, style= None):
    
    ax.add_patch(Create_CircularArc(center, radius, theta1, theta2, color, style))
    ax.add_patch(Create_CircularArc(center, (radius+std/2), theta1, theta2, color, 'dashed'))
    ax.add_patch(Create_CircularArc(center, (radius-std/2), theta1, theta2, color, 'dashed'))
    
   
def matrix_square(mat):
    return (mat * mat.T)   
    
    
def JoinMatsOnDiag(mat1, mat2):
        
    #mat = numpy.vstack([mat1, numpy.matrix(numpy.zeros((mat2.shape[0], mat1.shape[1])))])
    #mat = numpy.column_stack([mat, numpy.matrix(numpy.zeros((mat1.shape[0], mat2.shape[1]))), mat2])
        
    mat = numpy.matrix(numpy.zeros((mat1.shape[0] + mat2.shape[0], mat1.shape[1] + mat2.shape[1])))
        
    mat[0 : mat1.shape[0], 0 : mat1.shape[1]] = mat1
    mat[mat1.shape[0] : mat1.shape[0] + mat2.shape[0], mat1.shape[1] : mat1.shape[1] + mat2.shape[1]] = mat2
        
    return mat

def GetSubStatesAsMat(state, n_sub_state):
    
    num_sub_states = state.shape[0] / n_sub_state
    return numpy.reshape(state, (n_sub_state, num_sub_states), 'F')    

def PlotPoseCosy(ax, pose, color, style=None):

    len_scale = 0.25
    
    ax.add_patch(patches.FancyArrowPatch((pose[0,0], pose[1,0]),
                                         (pose[0,0] + len_scale * numpy.cos(pose[2,0]), pose[1,0] + len_scale * numpy.sin(pose[2,0])),
                                          arrowstyle='-|>', mutation_scale=15, color=color, linestyle=style))
                                          
    ax.add_patch(patches.FancyArrowPatch((pose[0,0], pose[1,0]),
                                         (pose[0,0] - len_scale * numpy.sin(pose[2,0]), pose[1,0] + len_scale * numpy.cos(pose[2,0])),
                                          arrowstyle='-|>', mutation_scale=15, color=color, linestyle=style))
                                          

def PlotPosArrow(ax, pose, angle, color, style=None):
    
    len_scale = 0.25
    
    ax.add_patch(patches.FancyArrowPatch((pose[0,0], pose[1,0]),
                                         (pose[0,0] + len_scale * numpy.cos(angle), pose[1,0] + len_scale * numpy.sin(angle)),
                                          arrowstyle='-|>', mutation_scale=15, color=color, linestyle=style))

def rot(theta):
    sin_val = math.sin(math.radians(theta))
    cos_val = math.cos(math.radians(theta))
    rot1 = matrix ([[cos_val, -sin_val],[sin_val, cos_val]])    
    
    return rot1
    
def rot_cov_mat(cov,theta):
    return (rot(theta) * (cov) * (rot(theta)).T)
    
def rot_sqrt_cov_mat(sqrt_cov,theta):
    sqrt_cov_rotated = sqrt_cov_matrices((sqrt_cov.Q_sqrt_mat.shape[0]))
    
    sqrt_cov_rotated.Q_sqrt_mat = (rot(theta) * (sqrt_cov.Q_sqrt_mat))
    sqrt_cov_rotated.R0_sqrt_mat = (rot(theta) * (sqrt_cov.R0_sqrt_mat))
    sqrt_cov_rotated.R1_sqrt_mat = (rot(theta) * (sqrt_cov.R1_sqrt_mat))
    sqrt_cov_rotated.R2_sqrt_mat = (rot(theta) * (sqrt_cov.R2_sqrt_mat))
    
    return sqrt_cov_rotated
    
def sign(x):
    
    if x < 0:
        
        return -1
    
    elif x > 0:
        
        return 1
    
    else:
        
        return 1
        
        
#veh position is defined in fixed world coordinate system 
# need to transform the state position in world coordinates
def TransformLocalToGlobal(x_local, veh_pos, veh_angle):
    
    sin_val = math.sin(math.radians(veh_angle))
    cos_val = math.cos(math.radians(veh_angle))
    
    x_global = np.asmatrix(np.zeros(x_local.shape))
    
    x_global[0,0] = x_local[0,0] * cos_val - x_local[1,0] * sin_val + veh_pos[0,0]
    x_global[1,0] = x_local[0,0] * sin_val + x_local[1,0] * cos_val + veh_pos[1,0]
    
    return(x_global)
    
    
        
#rotation then translation from local to global (Transform)
def rot_trans (xi_k, veh_input):
    # motion model
    ds = veh_input[0,0]
    dangle = veh_input[1,0]
    
    dalpha_rad = math.radians(dangle)
    sin_alpha = math.sin(dalpha_rad)
    cos_alpha = math.cos(dalpha_rad)
    
    vx = ds * cos_alpha
    vy = ds * sin_alpha
    
    m,n = xi_k.shape
    xi_k1 = np.asmatrix(np.zeros((m,n)))
    
    xi_k1[0,0] = vx + xi_k[0,0] * cos_alpha - xi_k[1,0] * sin_alpha
    xi_k1[1,0] = vy + xi_k[0,0] * sin_alpha + xi_k[1,0] * cos_alpha
    
    return xi_k1
    
    
#translation then rotation from global to local (Untransform)
def trans_rot (xi_k, veh_input):
    # motion model
    ds = veh_input[0,0]
    dangle = veh_input[1,0]
    
#    dalpha_rad = math.radians(dangle)
    dalpha_rad = dangle
    
    sin_alpha = math.sin(dalpha_rad)
    cos_alpha = math.cos(dalpha_rad)
    
    vx = ds * cos_alpha
    vy = ds * sin_alpha
    
    dx = xi_k[0,0] - vx
    dy = xi_k[1,0] - vy
    
    m,n = xi_k.shape
    xi_k1 = np.asmatrix(np.zeros((m,n)))
    
    xi_k1[0,0] = dx * cos_alpha + dy * sin_alpha
    xi_k1[1,0] = -dx * sin_alpha + dy * cos_alpha
    
    return xi_k1


def trans_rot_sigmas(Xi_x, Xi_u):
    n = Xi_x.shape[1]
    Xi_predicted = np.asmatrix(np.zeros((2,n)))
    
    for i in range(n):
        Xi_predicted[:,i]= trans_rot(Xi_x[:,i], Xi_u[:,i])    
        
    return Xi_predicted
#unknown speed motion model for pedestrian
def unkown_speed_model(x,  veh_input, pd_speed_vector = np.asmatrix(np.zeros((2,1))), delta_time= 0.2):
    
    #predicted pedestrian position in vehicle frame (Assuming it will have the same position)
    x1 = trans_rot(x , veh_input)
    
    return x1
    
#constant speed motion model 
#Assume PD_SPEED = 500 mm/s
#Assume delta_time = 0.2 sec
def const_speed_model(x, veh_input, pd_speed_vector, delta_time= 0.2):
    
    #calc predicted PD position in previous vehicle coordinates
    x1 = np.asmatrix(np.zeros(x.shape))

    x1[0,0] = x[0,0] + delta_time * pd_speed_vector[0,0]    
    x1[1,0] = x[1,0] + delta_time * pd_speed_vector[1,0]  
    
    #calc predicted position in current vehicle coordinates
    x2 = trans_rot(x1, veh_input)
    
    return x2
    
#unknown speed motion model for pedestrian; state here contains only pos in x and y
def unkown_speed_model_pd(x, delta_time= 0.2):
    
    #predicted pedestrian position in world frame is the same
    x1 = x
    
    return x1
    
#constant speed motion model; state here contains pos in x and y , vel in x and y
def const_speed_model_pd(x, delta_time= 0.04):
    
    #calc predicted PD position in wrold frame
    x1 = np.asmatrix(np.zeros(x.shape))

    x1[0,0] = x[0,0] + delta_time * x[2,0]    
    x1[1,0] = x[1,0] + delta_time * x[3,0]  
    x1[2,0] = x[2,0]
    x1[3,0] = x[3,0]
    
    return x1
    
    
#unknown speed motion model for pedestrian; state here contains only pos in x and y
def unkown_speed_model3(x,  veh_input, delta_time= 0.2):
    
    #predicted pedestrian position in vehicle frame (Assuming it will have the same position)
    x1 = trans_rot(x , veh_input)
    
    return x1
    
#constant speed motion model; state here contains pos in x and y , vel in x and y
def const_speed_model3(x, veh_input, delta_time= 0.2):
    
    #calc predicted PD position in previous vehicle coordinates
    x1 = np.asmatrix(np.zeros(x.shape))

    x1[0,0] = x[0,0] + delta_time * x[2,0]    
    x1[1,0] = x[1,0] + delta_time * x[3,0]  
    
    #calc predicted position in current vehicle coordinates
    x2 = trans_rot(x1, veh_input)
    
    velocity_mag = np.sqrt((x[2,0])**2+(x[3,0])**2)
    velocity_ang = math.atan2(x[3,0], x[2,0])
    
    veh_movement_angle_rad = math.radians(veh_input[1,0])
    
    x2[2,0] = velocity_mag * np.cos(velocity_ang - veh_movement_angle_rad)
    x2[3,0] = velocity_mag * np.sin(velocity_ang - veh_movement_angle_rad)  
    
    return x2
    
#PD Measurement Models
def pt_meas_model (state, meas, sens_pos = matrix([[0.0],[0.0]])):
    
    innovation = meas - state[0:2]
    
    return innovation
    
def pt_meas_model_sigmas(Xi_x, Xi_meas):
    n = Xi_x.shape[1]
    
    #Calculate the innovation sigma points from the joint sigma points
    Vi =  np.asmatrix(np.zeros((2,n)))
    
    for j in range(n):
        Vi[:,j] = pt_meas_model(Xi_x[0:2,j], Xi_meas[:,j])
        
    return Vi
    
def unsharp_meas_model(state, meas, sens_pos = matrix([[0.0],[0.0]])):
    innovation = meas - np.sqrt((state[0,0]-sens_pos[0,0])**2 + (state[1,0]-sens_pos[1,0])**2)
    
    return innovation

def unsharp_meas_model_sigmas(Xi_x, Xi_meas):
    n = Xi_x.shape[1]
    
    #Calculate the innovation sigma points from the joint sigma points
    Vi =  np.asmatrix(np.zeros((1,n)))
    
    for j in range(n):
        Vi[:,j] = unsharp_meas_model(Xi_x[0:2,j], Xi_meas[:,j])
        
    return Vi   
    
def ellipse_meas_model(state,meas,uls_focal1,uls_focal2):
    
    dist_x_f1 = GetDistBtwPts(state[0:2,:],uls_focal1)
    dist_x_f2 = GetDistBtwPts(state[0:2,:],uls_focal2)
    v = meas - ((dist_x_f1 + dist_x_f2 )/2)

    return v
    
def PointEllipseInnovationEKF(state,meas,uls_focal1,uls_focal2):
    n_state = state.size
    
    H2= -1.0
    H1=numpy.matrix(numpy.zeros((1,n_state)))
    dist_x_f1 = GetDistBtwPts(state[0:2,:],uls_focal1)
    dist_x_f2 = GetDistBtwPts(state[0:2,:],uls_focal2)
    v = ((dist_x_f1 + dist_x_f2 )/2) - meas
    H1[0,0]= (((state[0,0]- uls_focal1[0,0])/dist_x_f1) + ((state[0,0]- uls_focal2[0,0])/dist_x_f2)) * 0.5
    H1[0,1]= (((state[1,0]- uls_focal1[1,0])/dist_x_f1) + ((state[1,0]- uls_focal2[1,0])/dist_x_f2)) * 0.5
    
    return v,H1,H2
    
def PointAngleInnovationEKF(state,meas):
    n_state = state.size
    
    state_angle= math.atan2(state[0,0],state[1,0])
    meas_angle= math.atan2(meas[0,0],meas[1,0])
    
    state_dist_sq =  state[0,0] * state[0,0] + state[1,0] * state[1,0]
    meas_dist_sq =  meas[0,0] * meas[0,0] + meas[1,0] * meas[1,0]
    
    v = numpy.matrix([[( meas_angle - state_angle)]])
    
    H1= numpy.matrix(numpy.zeros((1,n_state)))
    H1[0,0]= (-state[1,0] )/state_dist_sq
    H1[0,1]= (state[0,0] )/state_dist_sq
    
    H2= numpy.matrix(numpy.zeros((1,2)),dtype=np.double)
    H2[0,0]= (meas[1,0] )/meas_dist_sq
    H2[0,1]= (-meas[0,0] )/meas_dist_sq
        
    return v,H1,H2
    
    
def pts_innovation(x1,x2, sensor_pos):
    return(x2-x1)
    
def pt_unsharp_innovation(pt, unsharp_dist, sensor_pos):
    return(np.sqrt((pt[0,0]-sensor_pos[0,0])**2 + (pt[1,0]-sensor_pos[1,0])**2) - unsharp_dist)
    
def pt_dist_to_sensor(pt, sensor_pos):
    return(np.sqrt((pt[0,0]-sensor_pos[0,0])**2 + (pt[1,0]-sensor_pos[1,0])**2))
    
def CalcPdf(x, mean, cov):
    k = x.size
    
    if np.isscalar(x):
        x = matrix([[x]])
        
    if np.isscalar(mean):
        mean = matrix([[mean]])
        
    if np.isscalar(cov):
        cov = matrix([[cov]])
        
    # (x-mean) is transposed here as it is represented as one column at the beginning 
    return((((2*math.pi)**k * linalg.det(cov))**-0.5) * np.exp(-0.5 * (x-mean).T * (cov.I) * (x-mean)))
    
    
# Calculates a QR decomposition of given rectangular matrix (only R matrix is computed) 
# and returns upper triangular part of R
# Inspired by https://de.wikipedia.org/wiki/Householdertransformation#Pseudocode
def GetR(A):
    
    m = A.shape[0]
    n = A.shape[1]
    
    for k in range(0, n):
        
        # uk is a reference to the column! Do not modify uk!
        uk = A[k:, k]
        
        norm = np.sqrt(uk.T * uk)
        norm = norm[0, 0]
        
        if norm > 0.0:
            
            vk = np.matrix(np.zeros((m, 1)))
            vk[k:, 0] = uk
            vk[k, 0] = vk[k, 0] + sign(vk[k, 0]) * norm
        
            norm2 = vk.T * vk
            norm2 = norm2[0, 0]
            assert(norm2 > 0.0)
            A = A - (2.0 / norm2) * vk * vk.T * A
        
        else:
        
            # Skip column with only zero elements
            pass
            
    return A[0:n, :]
    

def CalcPredJac(f,state, veh_input):
    
    dangle_deg = veh_input[1,0]
    pt_x = state[0,0]
    pt_y = state[1,0]
    dalpha_rad = math.radians(dangle_deg)
    sin_alpha = math.sin(dalpha_rad)
    cos_alpha = math.cos(dalpha_rad)
        
    if (f == trans_rot):
        #jacobian of the state from the motion model
        Fx = matrix([[cos_alpha, sin_alpha],[-sin_alpha, cos_alpha]])
        #jacobian of the veh_input from the motion model
        Fu = matrix([[-1.0, pt_y*cos_alpha - pt_x*sin_alpha],[0.0, -pt_y*sin_alpha - pt_x*cos_alpha]]) 
        
    else:
        Fx = 1
        Fu = 1
        
    return (Fx, Fu)
    


# calculates the sqrt matrix which fulfil this relation C_sqrt = sqrt(A_sqrt^2 + B_sqrt^2) in a numrically stable way
# using QR decompostion by building a compound matrix of the two sqrt matrices and perform the QR decomposition on it
def AddSqrtMats1(A_sqrt, B_sqrt):
    
    try:
        assert(A_sqrt.shape[0] == B_sqrt.shape[0])
    except AssertionError:
        
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        
        print('\n Inputs dimensions of {}() are not consistent \n'.format(func))
        sys.exit()
    

    M = np.asmatrix(np.zeros((2 * max(A_sqrt.shape[1] , B_sqrt.shape[1]) , A_sqrt.shape[0])))
    
    M[0 : A_sqrt.shape[1] , :] = A_sqrt.T
    
    M[(M.shape[0]/2) : ((M.shape[0]/2)+B_sqrt.shape[1]) , :] = B_sqrt.T
            
    R_mat = GetR(M)
    
    C_sqrt = R_mat.T
    
    return C_sqrt
    

# calculates the sqrt matrix which fulfil this relation C_sqrt = sqrt(A_sqrt^2 + B_sqrt^2) in a numrically stable way
# using QR decompostion by building a compound matrix of the two sqrt matrices and perform the QR decomposition on it
def AddSqrtMats(A_sqrt, B_sqrt):
    
    try:
        assert(A_sqrt.shape[0] == B_sqrt.shape[0])
    except AssertionError:
        
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        
        print('\n Inputs dimensions of {}() are not consistent \n'.format(func))
        sys.exit()
        
    A_rows = A_sqrt.shape[0]
    A_cols = A_sqrt.shape[1]
    B_cols = B_sqrt.shape[1]
    
    M = np.asmatrix(np.zeros((A_cols + B_cols , A_rows)))
    
    M[0 : A_cols , :] = A_sqrt.T
    
    M[A_cols : (A_cols +  B_cols), :] = B_sqrt.T
            
    R_mat = GetR(M)
    
    C_sqrt = R_mat.T
    
    return C_sqrt
    
def AddFourSqrtMats(A_sqrt, B_sqrt,C_sqrt,D_sqrt):
    
    try:
        assert(A_sqrt.shape[0] == B_sqrt.shape[0])
    except AssertionError:
        
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        
        print('\n Inputs dimensions of {}() are not consistent \n'.format(func))
        sys.exit()
        
    A_rows = A_sqrt.shape[0]
    A_cols = A_sqrt.shape[1]
    B_cols = B_sqrt.shape[1]
    C_cols = C_sqrt.shape[1]
    D_cols = D_sqrt.shape[1]
    
    M = np.asmatrix(np.zeros((A_cols + B_cols + C_cols + D_cols , A_rows)))
    
    M[0 : A_cols , :] = A_sqrt.T
    
    M[A_cols : (A_cols +  B_cols), :] = B_sqrt.T
    
    M[(A_cols +  B_cols) : (A_cols +  B_cols + C_cols), :] = C_sqrt.T
    
    M[(A_cols +  B_cols + C_cols) : (A_cols +  B_cols + C_cols + D_cols), :] = D_sqrt.T
            
    R_mat = GetR(M)
    
    C_sqrt = R_mat.T
    
    return C_sqrt
    
def AddThreeSqrtMats(A_sqrt, B_sqrt,C_sqrt):
    
    try:
        assert(A_sqrt.shape[0] == B_sqrt.shape[0])
    except AssertionError:
        
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]
        
        print('\n Inputs dimensions of {}() are not consistent \n'.format(func))
        sys.exit()
        
    A_rows = A_sqrt.shape[0]
    A_cols = A_sqrt.shape[1]
    B_cols = B_sqrt.shape[1]
    C_cols = C_sqrt.shape[1]
    
    
    M = np.asmatrix(np.zeros((A_cols + B_cols + C_cols , A_rows)))
    
    M[0 : A_cols , :] = A_sqrt.T
    
    M[A_cols : (A_cols +  B_cols), :] = B_sqrt.T
    
    M[(A_cols +  B_cols) : (A_cols +  B_cols + C_cols), :] = C_sqrt.T
    
            
    R_mat = GetR(M)
    
    C_sqrt = R_mat.T
    
    return C_sqrt
    
    
def CalcPdMeasNoise(meas, sens_pos = matrix([[0.0],[0.0]])):
    
    meas_sqrt_cov = sqrt_cov_matrices(meas.size)
    
    delta_x = meas[0,0]-sens_pos[0,0]
    delta_y = meas[1,0]-sens_pos[1,0]
    d = np.sqrt((delta_x)**2 + (delta_y)**2)
    
    alpha_rad = (math.pi)/2.0 - math.atan2(delta_y, delta_x)
    
    k_alpha = (math.fabs(alpha_rad) / ((math.pi)/2)) + 1.0
    k_lat = 0.03
    k_long = 0.1
    k_lat_dash = 0.03
    k_long_dash = 0.06
    
    rot_alpha = np.matrix ([[math.cos(alpha_rad), -math.sin(alpha_rad)] , [math.sin(alpha_rad), math.cos(alpha_rad)]])
    
    Q_mat = rot_alpha * np.matrix([[(k_alpha * d * k_lat)**2 , 0.0], [0.0 , (k_alpha * d * k_long)**2]]) * rot_alpha.T
    
    R_mat = rot_alpha * np.matrix([[(k_alpha * d * k_lat_dash)**2 , 0.0], [0.0 , (k_alpha * d * k_long_dash)**2]]) * rot_alpha.T
    
    meas_sqrt_cov.Q_sqrt_mat = np.linalg.cholesky(Q_mat)
    
    meas_sqrt_cov.R0_sqrt_mat = np.linalg.cholesky(R_mat)
    
    return meas_sqrt_cov
    
def CalcPdMeasNoise1(meas, sens_pos = matrix([[0.0],[0.0]])):
    
    meas_sqrt_cov = sqrt_cov_matrices(meas.size)
    
    delta_x = meas[0,0]-sens_pos[0,0]
    delta_y = meas[1,0]-sens_pos[1,0]
    d = np.sqrt((delta_x)**2 + (delta_y)**2)
    
    alpha_rad = (math.pi)/2.0 - math.atan2(delta_y, delta_x)
    
    k_alpha = (math.fabs(alpha_rad) / ((math.pi)/2)) + 1.0
    k_lat = 0.03
    k_long = 0.1
    k_lat_dash = 0.03
    k_long_dash = 0.06
    
    rot_alpha = matrix ([[math.cos(alpha_rad), -math.sin(alpha_rad)] , [math.sin(alpha_rad), math.cos(alpha_rad)]])
    
    Q_mat = rot_alpha * matrix([[(k_alpha * d * k_lat)**2 , 0.0], [0.0 , (k_alpha * d * k_long)**2]]) * rot_alpha.T
    
    R_mat = rot_alpha * matrix([[(k_alpha * d * k_lat_dash)**2 , 0.0], [0.0 , (k_alpha * d * k_long_dash)**2]]) * rot_alpha.T
    
    meas_sqrt_cov.Q_sqrt_mat = np.linalg.cholesky(Q_mat)
    
    meas_sqrt_cov.R0_sqrt_mat = np.linalg.cholesky(R_mat)
    
    return Q_mat, R_mat
    
    
def AugmentPdStateWithVelocity(state, state_sqrt_cov):
    
    if state.size == 2 :
        augmented_state = matrix([[state[0,0],state[1,0], 0.0, 0.0]]).T
        
        #TODO: Tunable
        a = 5.0
        b = -a
        
        uniform_dist_std = np.sqrt(((b - a) ** 2) / 12.0)
        
        uniform_dist_sqrt_cov = matrix([[uniform_dist_std, 0.0],[0.0, uniform_dist_std]])
        
        zero_cov_matrix = np.asmatrix(np.zeros((2,2)))
        
        augmented_state_sqrt_cov = sqrt_cov_matrices(4)
        augmented_state_sqrt_cov.Q_sqrt_mat =  JoinMatsOnDiag(state_sqrt_cov.Q_sqrt_mat, uniform_dist_sqrt_cov)
        augmented_state_sqrt_cov.R0_sqrt_mat =  JoinMatsOnDiag(state_sqrt_cov.R0_sqrt_mat, zero_cov_matrix)
        augmented_state_sqrt_cov.R1_sqrt_mat =  JoinMatsOnDiag(state_sqrt_cov.R1_sqrt_mat, zero_cov_matrix)
        augmented_state_sqrt_cov.R2_sqrt_mat =  JoinMatsOnDiag(state_sqrt_cov.R2_sqrt_mat, zero_cov_matrix)
        
    elif state.size == 4:
        augmented_state = state
        
        augmented_state_sqrt_cov = state_sqrt_cov
    
    return augmented_state, augmented_state_sqrt_cov
    
    
def ConvertTo2x2Mat(in_mat):
    out_mat = in_mat [0:2,0:2]
    
    return out_mat

def ConvertTo2x1Vec(in_mat):
    out_mat = in_mat[0:2]
    
    return out_mat

class cov_matrices:
    def __init__(self,n):
        self.Q_mat = np.asmatrix(np.zeros((n,n)))
        self.R0_mat = np.asmatrix(np.zeros((n,n)))
        self.R1_mat = np.asmatrix(np.zeros((n,n)))
        self.R2_mat = np.asmatrix(np.zeros((n,n)))
    def get_pmat(self):
        return (self.Q_mat + self.R0_mat + self.R1_mat + self.R2_mat)
        
class sqrt_cov_matrices:
    def __init__(self,n):
        self.Q_sqrt_mat = np.asmatrix(np.zeros((n,n)))
        self.R0_sqrt_mat = np.asmatrix(np.zeros((n,n)))
        self.R1_sqrt_mat = np.asmatrix(np.zeros((n,n)))
        self.R2_sqrt_mat = np.asmatrix(np.zeros((n,n)))
        
    def get_pmat(self):
        return (matrix_square(self.Q_sqrt_mat) + 
        matrix_square(self.R0_sqrt_mat) + 
        matrix_square(self.R1_sqrt_mat) + 
        matrix_square(self.R2_sqrt_mat))
        
class sqrt_var:
    def __init__(self):
        self.Q_sqrt = 0.0
        self.R0_sqrt = 0.0
        self.R1_sqrt = 0.0
        self.R2_sqrt = 0.0
    def get_p(self):
        return ((self.Q_sqrt)**2 + (self.R0_sqrt)**2 + (self.R1_sqrt)**2 + (self.R2_sqrt)**2)
        
class params:
    def __init__(self):
        self.pt1_state = np.asmatrix(np.zeros((2,1)))
        self.pt2_state = np.asmatrix(np.zeros((2,1)))
        self.ptn_state = np.asmatrix(np.zeros((2,1)))
        self.pt1_cov_mats = cov_matrices(self.pt1_state.size)
        self.pt2_cov_mats = cov_matrices(self.pt2_state.size)
        self.ptn_cov_mats = cov_matrices(self.pt3_state.size)

        
class params_sqrt:
    def __init__(self):
        self.pt1_state = np.asmatrix(np.zeros((2,1)))
        self.pt2_state = np.asmatrix(np.zeros((2,1)))
        self.ptn_state = np.asmatrix(np.zeros((2,1)))
        self.pt1_sqrt_cov_mats = sqrt_cov_matrices(self.pt1_state.size)
        self.pt2_sqrt_cov_mats = sqrt_cov_matrices(self.pt2_state.size)
        self.ptn_sqrt_cov_mats = sqrt_cov_matrices(self.pt3_state.size)
        
class pt_data:
    def __init__(self):
        self.mean = np.asmatrix(np.zeros((2,1)))
        self.cov_sqrt = sqrt_cov_matrices(self.mean.size)
        self.is_unsharp = False
    def get_unsharp_dist(self):
        return (np.sqrt(self.mean[0,0] * self.mean[0,0] + self.mean[1,0] * self.mean[1,0]))
        
class unsharp_data:
    def __init__(self):
        self.unsharp_dist = 0.0
        self.var_sqrt = sqrt_var()
        self.sens_pos = np.asmatrix(np.zeros((2,1)))
        self.is_unsharp = True
        
        
class pd_data:
    def __init__(self):
        self.pos = np.asmatrix(np.zeros((2,1)))
        self.time_stamp = 0.0
        self.id = 0
        self.speed_vector = np.asmatrix(np.zeros((2,1)))
        self.delta_time = 0.0
        
        self.is_unsharp = False
        
        self.unsharp_distance = 0.0
        self.unsharp_angle = 0.0
        
        self.sens_pos = np.asmatrix(np.zeros((2,1)))
        
    def get_unsharp_dist(self):
        return (np.sqrt((self.pos[0,0] - self.sens_pos[0,0])**2 + (self.pos[1,0] - self.sens_pos[1,0])**2))
        
    def GetSpeed(self):
        return (np.sqrt((self.speed_vector[0,0])**2 + (self.speed_vector[1,0])**2 ))
        
    def GetMoveDir(self):
        return (math.atan2(self.speed_vector[1,0], self.speed_vector[0,0]))
        
    def get_unsharp_ang_deg(self):
        return(math.degrees(math.atan2((self.pos[1,0] - self.sens_pos[1,0]), (self.pos[0,0] - self.sens_pos[0,0]))))
        
        
    def update_unsharp_data(self):
        self.unsharp_distance = self.get_unsharp_dist()
        self.unsharp_angle = self.get_unsharp_ang_deg()
        
		
		
		
		
###################################################################
#moniem added section
def AugmentPDStateCovForSCIEKF(state, state_cov):
    
    if state.size == 2 :
        augmented_state = matrix([[state[0,0],state[1,0], 0.0, 0.0]]).T
        
        #TODO: Tunable
        a = 5.0
        b = -a
        
        uniform_dist_std = (((b - a) ** 2) / 12.0)
        
        uniform_dist_cov = matrix([[uniform_dist_std, 0.0],[0.0, uniform_dist_std]])
        
        zero_cov_matrix = np.asmatrix(np.zeros((2,2)))
        
        augmented_state_cov = cov_matrices(4)
        augmented_state_cov.Q_mat =  JoinMatsOnDiag(state_cov.Q_mat, uniform_dist_cov)
        augmented_state_cov.R0_mat = JoinMatsOnDiag(state_cov.R0_mat, zero_cov_matrix)
        augmented_state_cov.R1_mat =  JoinMatsOnDiag(state_cov.R1_mat, zero_cov_matrix)
        augmented_state_cov.R2_mat =  JoinMatsOnDiag(state_cov.R2_mat, zero_cov_matrix)
        
    elif state.size == 4:
        augmented_state = state
        
        augmented_state_cov = state_cov
    
    return augmented_state, augmented_state_cov

def UnAugmentPDStateCovForSCIEKF(state, state_cov):
    unaug_state= ConvertTo2x1Vec(state)
    unaug_cov= cov_matrices(4)
    
    unaug_cov.Q_mat= ConvertTo2x2Mat(state_cov.Q_mat)
    unaug_cov.R0_mat= ConvertTo2x2Mat(state_cov.R0_mat)
    unaug_cov.R1_mat= ConvertTo2x2Mat(state_cov.R1_mat)
    unaug_cov.R2_mat= ConvertTo2x2Mat(state_cov.R2_mat)
    
    return unaug_state,unaug_cov



def CalcPdMeasNoise_ekf(meas, sens_pos = matrix([[0.0],[0.0]])):
    
    meas_cov = cov_matrices(meas.size)
    
    delta_x = meas[0,0]-sens_pos[0,0]
    delta_y = meas[1,0]-sens_pos[1,0]
    d = np.sqrt((delta_x)**2 + (delta_y)**2)
    
    alpha_rad = (math.pi)/2.0 - math.atan2(delta_y, delta_x)
    
    k_alpha = (math.fabs(alpha_rad) / ((math.pi)/2)) + 1.0
    k_lat = 0.03
    k_long = 0.1
    k_lat_dash = 0.03
    k_long_dash = 0.06
    
    rot_alpha = matrix ([[math.cos(alpha_rad), -math.sin(alpha_rad)] , [math.sin(alpha_rad), math.cos(alpha_rad)]])
    
    Q_mat = rot_alpha * matrix([[(k_alpha * d * k_lat)**2 , 0.0], [0.0 , (k_alpha * d * k_long)**2]]) * rot_alpha.T
    
    R_mat = rot_alpha * matrix([[(k_alpha * d * k_lat_dash)**2 , 0.0], [0.0 , (k_alpha * d * k_long_dash)**2]]) * rot_alpha.T
    
    meas_cov.Q_mat = Q_mat
    
    meas_cov.R0_mat = R_mat
    
    return meas_cov


def rot_cov_obj(cov,theta):
    cov_rotated = cov_matrices((cov.Q_mat.shape[0]))
    
    cov_rotated.Q_mat= rot(theta) * (cov.Q_mat) * (rot(theta)).T
    cov_rotated.R0_mat= rot(theta) * (cov.R0_mat) * (rot(theta)).T
    cov_rotated.R1_mat= rot(theta) * (cov.R1_mat) * (rot(theta)).T
    cov_rotated.R2_mat= rot(theta) * (cov.R2_mat) * (rot(theta)).T
    
    return cov_rotated
    
def unsharp_state_jac(state, sens_pos = matrix([[0.0],[0.0]])):
    n_state= state.shape[0]
    H1=np.matrix(np.zeros((1,n_state)))
    dv_dx = -1*(state[0,0]-sens_pos[0,0])/np.sqrt((state[0,0]-sens_pos[0,0])**2 + (state[1,0]-sens_pos[1,0])**2)
    dv_dy = -1*(state[1,0]-sens_pos[1,0])/np.sqrt((state[0,0]-sens_pos[0,0])**2 + (state[1,0]-sens_pos[1,0])**2) 
    
    H1[0,0:2]=matrix([[dv_dx,dv_dy]])
    return H1
    
def point_state_jac(state):
    n_state= state.shape[0]
    H1=np.matrix(np.zeros((2,n_state)))
    H1[:,0:2]=matrix([[-1.,0.],[0.,-1.]])
    
    return H1