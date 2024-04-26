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
q1 = me.dynamicsymbols('q1') # angle from the vertical in N <- This must be more general
u1 = me.dynamicsymbols('u1') # q1 prime

A = me.ReferenceFrame('A')
A.orient_axis(N, q1, N.z) # A's reference frame is rotated through angle q1 around N.z <- This should be more general
A.set_ang_vel(N, u1*N.z) # A's reference frame has angular velocity u1 around N.z

Ao = me.Point('A_O') # A centre of mass
Ao.set_pos(O, 0.5*A.x) # Set the location of the com in relation to the origin

I = m_a**2/12
I_A_Ao = I*me.outer(A.y, A.y) + I*me.outer(A.z, A.z) # Inertia of A

# Constraint?
Ao.v2pt_theory(O, N, A) # The velocity of the com (in frame N) changes with A (with respect to O)

# object B
m_b = sm.symbols('m_b') 
q2 = me.dynamicsymbols('q2') # angle from the horizontal in A
u2  = me.dynamicsymbols('u2') # q2 prime

B = me.ReferenceFrame('B')
B.orient_axis(A, q2, A.x) # B's reference frame is rotated through angle q2 around A.x
B.set_ang_vel(A, u2*A.x) # B's reference frame has angular velocity u2 around A.x

Bo = me.Point('B_O') # B centre of mass
Bo.set_pos(O, A.x) # Set the location of the com in relation to the origin

I = m_b**2/12
I_B_Bo = I*me.outer(B.x, B.x) + I*me.outer(B.z, B.z) # Inertia of B 

# Constraint?
Bo.v2pt_theory(O, N, A) # The velocity of the com (in frame N) changes with A (with respect to O) <- this should be changes with B to be more general

# Force of gravity on A (applied at the com)
R_Ao = m_a*g*N.x

# Force of gravity on B (applied at the com)
R_Bo = m_b*g*N.x

# Spring Forces
kt = sm.symbols('k_t')
T_A = -kt*q1*N.z + kt*q2*A.x
T_B = -kt*q2*A.x

Fr_bar = [] # This defines the total "Partial forces" (Generalised Active Forces) for each object: the force component of each angular speed
Frs_bar = [] # This defines the time-derivative solutions to the partial forces (Generalised Inertia Forces) for each object in terms of the mass, acceleration, inertia, etc.
# Effectively we're setting up the formulae here and then sympy will solve them to give us the required speeds etc.
# Fr is the "left-hand side" of the Newton-Euler formulae for the equations of motion
# Frs is the "right-hand side"
# Fr = (u.R) + (w.T) = (-m*a.u) + (a.I + wxI.w) = Frs

# Now we loop through each object and get its angular speed
for ui in [u1, u2]:
    Fr = 0
    Frs = 0
    # The angular speed of each object is dependent on the forces applied to all connected objects
    # First, deal with the active forces (that move the centre of mass without rotation)
    for Pi, Ri, mi in zip([Ao, Bo], [R_Ao, R_Bo], [m_a, m_b]):
        vr = Pi.vel(N).diff(ui, N)
        Fr += vr.dot(Ri)
        Rs = -mi*Pi.acc(N)
        Frs += vr.dot(Rs)
        
    # Now deal with the inertia forces (that caus rotations) - in this example, we have no forces but this is where muscle forces will be applied
    for Bi, Ti, Ii in zip([A, B], [T_A, T_B], [I_A_Ao, I_B_Bo]):
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
q = sm.Matrix([q1, q2])
u = sm.Matrix([u1, u2])

# Define the time differentials of orientation and angular speed
qd = q.diff(t)
ud = u.diff(t)

# Set the acceleration to 0
ud_zerod = {udr: 0 for udr in ud}

# This is the site of forward kinematics vs inverse kinematics
# Initialise Mk (kinematics) - For forward kinematics, we know the current speeds
Mk = -sm.eye(2)
gk = u

# Initialise Md (dynamics) - For forward kinematics, we know the component forces and time derivatives 
Md = Frs.jacobian(ud)
gd = Frs.xreplace(ud_zerod) + Fr

# Now we define a function that evaluates the equations of motion (eom) - returning the calculated Mk, gk, Md, and gd
# The function takes the current orientations or the objects, the current angular speeds, masses, g
m =  sm.Matrix([m_a,m_b])
p =  sm.Matrix([g, kt])
eval_eom = sm.lambdify((q, u, m, p), [Mk, gk, Md, gd])

# Initialise the inputs with starting values
q_vals = np.array([
    np.deg2rad(25.0),  # q1, rad
    np.deg2rad(5.0),  # q2, rad
])

u_vals = np.array([
    0.1,  # u1, rad/s
    2.2,  # u2, rad/s
])

m_vals = np.array([
    1.0,  # m_a, kg
    1.0,  # m_b, kg
])

p_vals = np.array([
     9.81,  # g, m/s**2
     0.01, # kt, Nm/rad
])

# Now find the initial mass matrix components
Mk_vals, gk_vals, Md_vals, gd_vals = eval_eom(q_vals, u_vals, m_vals, p_vals)

# Now the hard work must be done: find the speeds and accelerations from the
# system of equations defined by the mass matrix.

# calculate the angular speed - this apparently may not always equal u but I don't know why...
qd_vals = np.linalg.solve(-Mk_vals, np.squeeze(gk_vals))

# Now the angular acceleration
ud_vals = np.linalg.solve(-Md_vals, np.squeeze(gd_vals))

# Now we simply time step forward in whatever fashion we like. We must resolve the motion right-hand side each iteration.

vis = Visualiser()
vis.setupVisualiser()

for i in range(1000):
    bone.forces = []
    vis.beginRendering()
    ground.draw(vis)
    bone.draw(vis)
    vis.endRendering()