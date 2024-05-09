#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sympy as sm
import sympy.physics.mechanics as me
import numpy as np
from visualiser import Visualiser



# Global variables
g = sm.symbols('g')
N = me.ReferenceFrame('N')
O = me.Point('O') # global origin

t = me.dynamicsymbols._t # time

# constraint?
O.set_vel(N, 0) # lock the origin by setting the vel to zero (required for v2pt_theory later)

# object A
m_a = sm.symbols('m_a') # mass
a_psi, a_theta, a_phi = me.dynamicsymbols('a_psi, a_theta, a_varphi') # angle from the vertical in N <- This must be more general
a_u0,a_u1,a_u2 = me.dynamicsymbols('a_u0, a_u1, a_u2') # q1 prime

A1 = me.ReferenceFrame('A1')
A2 = me.ReferenceFrame('A2')
A3 = me.ReferenceFrame('A3')
A = me.ReferenceFrame('A')
#A.orient_quaternion(N, (1.0, a_q0,a_q1,a_q2)) # A's reference frame is rotated through angle q1 around N.z <- This should be more general
A.orient_body_fixed(N, (a_psi, a_theta, a_phi), 'XYZ')
#A1.orient_axis(N, N.y, a_psi)
#A2.orient_axis(A1, A1.x, a_theta)
#A3.orient_axis(A2, A2.z, a_phi)

# Put the orientation and angular speed into matrices
a_q = sm.Matrix([a_psi, a_theta, a_phi])
a_u = sm.Matrix([a_u0, a_u1, a_u2])

# Define the time differentials of orientation and angular speed
a_qd = a_q.diff(t)
a_ud = a_u.diff(t)

N_w_A = A.ang_vel_in(N)
N_w_A = N_w_A.xreplace(dict(zip(a_qd, a_u)))
A.set_ang_vel(N, N_w_A)

N_w_A = A.ang_acc_in(N)
N_w_A = N_w_A.xreplace(dict(zip(a_qd, a_u)))
ud_zerod = {udr: 0 for udr in a_ud}
N_w_A = N_w_A.xreplace(ud_zerod)
A.set_ang_acc(N, N_w_A)

Ao = me.Point('A_O') # A centre of mass
Ao.set_pos(O, 0.5*A.x) # Set the location of the com in relation to the origin

I = m_a**2/12
I_A_Ao = I*me.outer(A.y, A.y) + I*me.outer(A.z, A.z) # Inertia of A

# Constraint?
Ao.v2pt_theory(O, N, A) # The velocity of the com (in frame N) changes with A (with respect to O)

# N_v_A = Ao.vel(N)
# N_v_A = N_v_A.xreplace(dict(zip(a_qd, a_u)))
#Ao.set_vel(N, N_v_A)

N_v_A = Ao.acc(N)
N_v_A = N_v_A.xreplace(dict(zip(a_qd, a_u)))

Ao.set_acc(N, N_v_A)

# Force of gravity on A (applied at the com)
R_Ao = m_a*g*N.x

# Spring Forces
kt = sm.symbols('k_t')
T_A = -kt*N.z

Fr_bar = [] # This defines the total "Partial forces" (Generalised Active Forces) for each object: the force component of each angular speed
Frs_bar = [] # This defines the time-derivative solutions to the partial forces (Generalised Inertia Forces) for each object in terms of the mass, acceleration, inertia, etc.
# Effectively we're setting up the formulae here and then sympy will solve them to give us the required speeds etc.
# Fr is the "left-hand side" of the Newton-Euler formulae for the equations of motion
# Frs is the "right-hand side"
# Fr = (u.R) + (w.T) = (-m*a.u) + (a.I + wxI.w) = Frs



# Now we loop through each object and get its angular speed
for ui in [a_u0, a_u1, a_u2]:
    Fr = 0
    Frs = 0
    # The angular speed of each object is dependent on the forces applied to all connected objects
    # First, deal with the active forces (that move the centre of mass without rotation)
    for Pi, Ri, mi in zip([Ao], [R_Ao], [m_a]):
        vr = Pi.vel(N).diff(ui, N)
        Fr += vr.dot(Ri)
        Rs = -mi*Pi.acc(N)
        Frs += vr.dot(Rs)
        
    # Now deal with the inertia forces (that caus rotations) - in this example, we have no forces but this is where muscle forces will be applied
    for Bi, Ti, Ii in zip([A], [T_A], [I_A_Ao]):
        wr = Bi.ang_vel_in(N).diff(ui, N)
        Fr += wr.dot(Ti)
        Ts = -(Bi.ang_acc_in(N).dot(Ii) +
               me.cross(Bi.ang_vel_in(N), Ii).dot(Bi.ang_vel_in(N)))
        Frs += wr.dot(Ts)
        
        
    Fr_bar.append(Fr)
    Frs_bar.append(Frs)
    
# Put the total right-hand and left-hand sides into matrices
Fr = sm.Matrix(Fr_bar)
Frs = sm.Matrix(Frs_bar)

# Put the orientation and angular speed into matrices
q = sm.Matrix([a_psi, a_theta, a_phi])
u = sm.Matrix([a_u0, a_u1, a_u2])

# Define the time differentials of orientation and angular speed
qd = q.diff(t)
ud = u.diff(t)

print(ud)

# Set the acceleration to 0
ud_zerod = {udr: 0 for udr in ud}

# This is the site of forward kinematics vs inverse kinematics
# Initialise Mk (kinematics) - For forward kinematics, we know the current speeds
Mk = -sm.eye(3)
gk = u

# Initialise Md (dynamics) - For forward kinematics, we know the component forces and time derivatives 
Md = Frs.jacobian(ud)
gd = Frs.xreplace(ud_zerod) + Fr

# Now we define a function that evaluates the equations of motion (eom) - returning the calculated Mk, gk, Md, and gd
# The function takes the current orientations or the objects, the current angular speeds, masses, g
m =  sm.Matrix([m_a])
p =  sm.Matrix([g, kt])
eval_eom = sm.lambdify((q, u, m, p), [Mk, gk, Md, gd])

# get reference frames
# x,y,z = sm.symbols('x,y,z')
# p_in_A = x*A.x + y*A.y + z*A.z
# p_in_A = p_in_A.to_matrix(N)
# get_A_pos_in_N = sm.lambdify((x,y,z,a_psi, a_theta, a_phi), p_in_A)

# Initialise the inputs with starting values
q_vals = np.array([0.0,1.0,0.0  # q2, rad
])

u_vals = np.array([0.0,0.1,0.0
])

m_vals = np.array([
    1.0  # m_a, kg
])

p_vals = np.array([
     9.81,  # g, m/s**2
     0.0, # kt, Nm/rad
])

# Now find the initial mass matrix components
Mk_vals, gk_vals, Md_vals, gd_vals = eval_eom(q_vals, u_vals, m_vals, p_vals)

print("Mk", Mk_vals)
print("gk", gk_vals)
print("Md", Md_vals)
print("gd", gd_vals)

# Now the hard work must be done: find the speeds and accelerations from the
# system of equations defined by the mass matrix.

# calculate the angular speed - this apparently may not always equal u but I don't know why...
qd_vals = np.linalg.solve(-Mk_vals, np.squeeze(gk_vals))

# Now the angular acceleration
ud_vals = np.linalg.solve(-Md_vals, np.squeeze(gd_vals))

# Now we simply time step forward in whatever fashion we like. We must resolve the motion right-hand side each iteration.

vis = Visualiser()
vis.setupVisualiser()

# initial state: q1, q2, u1,u2
state = [np.deg2rad(25.0), 0.0]

for i in range(1000):

    q_vals = np.array([
        0.0,0.0,state[0] # q2, rad
    ])

    u_vals = np.array(
        [0.0,0.0,state[1]  # u2, rad/s
    ])
    
    Mk_vals, gk_vals, Md_vals, gd_vals = eval_eom(q_vals, u_vals, m_vals, p_vals)
    qd_vals = np.linalg.solve(-Mk_vals, np.squeeze(gk_vals))
    ud_vals = np.linalg.solve(-Md_vals, np.squeeze(gd_vals))
    
    state[0] += 0.01 * (qd_vals[0])
    state[1] += 0.01 * (ud_vals[0])
    
    vis.beginRendering()
    print(get_A_pos_in_N(0.0,-1.0,0.0,0.0,0.0,state[0]))
    vis.drawLine(np.array([0.0,0.0,0.0]), np.matmul(get_A(0.0,0.0,state[0]),np.array([0.0,-1.0,0.0])))
    vis.drawCube(matrix=np.identity(4), model_pos=np.array([0.0,0.0,0.0]), scale=0.02, col=(1,0,0,1))
    vis.drawCube(matrix=np.identity(4), model_pos=np.matmul(get_A(0.0,0.0,state[0]),np.array([0.0,-1.0,0.0])), scale=0.02, col=(1,0,0,1))
    vis.endRendering()