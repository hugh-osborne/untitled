import numpy as np
from visualiser import Visualiser
import sympy as sm
import sympy.physics.mechanics as me

class Object:
    def __init__(self, sym_parentOrigin, sym_parentFrame, dofs, name="bone", mass=1.0, inertia=np.identity(3)):
        self.name = name
        
        self.forces = []
        self.torques = []
        
        self.state_mass = mass
        self.state_inertia = inertia
        self.state_orientation = np.zeros(3) # This will be r1,r2,r3
        self.state_ang_vel = np.zeros(3) # This will be u0,u1,u2
        self.state_vel = np.zeros(3) # This will be v0,v1,v2
        self.state_com = np.zeros(3) # This will be q0,q1,q2
        # Declare which degrees of freedom we want to calculate for this object.
        # Any dofs not declared will be locked to the parent frame
        # full set is ['rot_x','rot_y','rot_z','pos_x','pos_y','pos_z']
        self.dofs = dofs
        
        # symbols
        self.parentFrame = sym_parentFrame
        self.parentOrigin = sym_parentOrigin
        self.N = self.parentFrame
        self.O = self.parentOrigin

        self.mass = sm.symbols('m_' + name) # mass
        self.dof_orientation = []
        self.dof_ang_vel = []
        self.dof_position = []
        self.dof_vel = []
        self.frame = me.ReferenceFrame('F_' + name) # reference frame
        order = ''
        if 'rot_x' in self.dofs:
            self.dof_orientation += [me.dynamicsymbols('r1_' + name)] # 
            order += 'X'
        if 'rot_y' in self.dofs:
            self.dof_orientation += [me.dynamicsymbols('r2_' + name)] # 
            order += 'Y'
        if 'rot_z' in self.dofs:
            self.dof_orientation += [me.dynamicsymbols('r3_' + name)] # 
            order += 'Z'
        self.frame.orient_body_fixed(self.N, self.dof_orientation, order)
        vector = 0
        if 'rot_x' in self.dofs:
            self.dof_ang_vel += [me.dynamicsymbols('u0_' + name)] # angular velocity
            vector += self.dof_ang_vel[-1]*self.N.x
        if 'rot_y' in self.dofs:
            self.dof_ang_vel += [me.dynamicsymbols('u1_' + name)] # 
            vector += self.dof_ang_vel[-1]*self.N.y
        if 'rot_z' in self.dofs:
            self.dof_ang_vel += [me.dynamicsymbols('u2_' + name)] # 
            vector += self.dof_ang_vel[-1]*self.N.z
        self.frame.set_ang_vel(self.N, vector) # Reference frame has angular velocity u
        
        self.com = me.Point('com_' + name) # centre of mass
        vector = 0
        if 'pos_x' in self.dofs:
            self.dof_position += [me.dynamicsymbols('q0_' + name)] # com position in world coords
            vector += self.dof_position[-1]*self.frame.x
        if 'pos_y' in self.dofs:        
            self.dof_position += [me.dynamicsymbols('q1_' + name)] #
            vector += self.dof_position[-1]*self.frame.y
        if 'pos_z' in self.dofs:
            self.dof_position += [me.dynamicsymbols('q2_' + name)] #
            vector += self.dof_position[-1]*self.frame.z
        self.com.set_pos(self.O, vector) 
        vector = 0
        if 'pos_x' in self.dofs:
            self.dof_vel += [me.dynamicsymbols('v0_' + name)] # velocity
            vector += self.dof_vel[-1]*self.frame.x
        if 'pos_y' in self.dofs:        
            self.dof_vel += [me.dynamicsymbols('v1_' + name)] # 
            vector += self.dof_vel[-1]*self.frame.y
        if 'pos_z' in self.dofs:    
            self.dof_vel += [me.dynamicsymbols('v2_' + name)] # 
            vector += self.dof_vel[-1]*self.frame.z
        self.com.set_vel(self.N, vector)
        #self.com.set_vel(self.frame, 0)
        #self.com.v1pt_theory(self.com, self.N, self.frame)
        
        t = me.dynamicsymbols._t
        # Put the orientation and angular speed into matrices
        a_q = sm.Matrix(self.dof_position + self.dof_orientation)
        a_u = sm.Matrix(self.dof_vel + self.dof_ang_vel)

        # Define the time differentials of orientation and angular speed
        a_qd = a_q.diff(t)
        a_ud = a_u.diff(t)
        
        # N_w_A = self.frame.ang_vel_in(self.N)
        # N_w_A = N_w_A.xreplace(dict(zip(a_qd, a_u)))
        # self.frame.set_ang_vel(self.N, N_w_A)

        # N_w_A = self.frame.ang_acc_in(self.N)
        # N_w_A = N_w_A.xreplace(dict(zip(a_qd, a_u)))
        # ud_zerod = {udr: 0 for udr in a_ud}
        # N_w_A = N_w_A.xreplace(ud_zerod)
        # self.frame.set_ang_acc(self.N, N_w_A)
        
        # N_v_A = self.com.vel(self.N)
        # N_v_A = N_v_A.xreplace(dict(zip(a_qd, a_u)))
        # self.com.set_vel(self.N, N_v_A)

        # N_v_A = self.com.acc(self.N)
        # N_v_A = N_v_A.xreplace(dict(zip(a_qd, a_u)))
        # ud_zerod = {udr: 0 for udr in a_ud}
        # N_v_A = N_v_A.xreplace(ud_zerod)
        
        #self.com.v2pt_theory(self.O, self.N, self.frame)
        

        # Inertia matrix
        self.Ixx = sm.symbols('I_xx_' + name)
        self.Iyy = sm.symbols('I_yy_' + name)
        self.Izz = sm.symbols('I_zz_' + name)
        self.Ixy = sm.symbols('I_xy_' + name)
        self.Iyz = sm.symbols('I_yz_' + name)
        self.Ixz = sm.symbols('I_xz_' + name)
        
        self.I = me.inertia(self.frame, self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Iyz, self.Ixz)
        
        # symbol functions
        self.get_A = sm.lambdify(self.dof_orientation, self.N.dcm(self.frame))
        
    def getFrFrsFromForce(self, force):
        Fr = []
        Frs = []
        Rs = -self.mass*self.com.acc(self.N)
        
        for s in self.dof_vel + self.dof_ang_vel:
            v_com_0 = self.com.vel(self.N).diff(s, self.N)
            Fr += [v_com_0.dot(force)]
            Frs += [v_com_0.dot(Rs)]
        
        return Fr, Frs

    def getFrFrsFromTorque(self, torque):
        Fr = []
        Frs = []
        Ts = -(self.frame.ang_acc_in(self.N).dot(self.I) + me.cross(self.frame.ang_vel_in(self.N), self.I).dot(self.frame.ang_vel_in(self.N)))
        
        for s in self.dof_vel + self.dof_ang_vel:
            w_com_0 = self.frame.ang_vel_in(self.N).diff(s, self.N)
            Fr += [w_com_0.dot(torque)]
            Frs += [w_com_0.dot(Ts)]
        
        return Fr, Frs

    def setStateMass(self, m):
        self.state_mass = m
        
    def setStateOrientation(self, r):
        self.state_orientation = r
        
    def setStateCom(self, c):
        self.state_com = c
        
    def updateState(self, qd, ud, dt):
        self.state_orientation += dt * qd[3:] # This will be r1,r2,r3
        self.state_ang_vel += dt * ud[3:] # This will be u0,u1,u2
        self.state_vel += dt * ud[:3] # This will be v0,v1,v2
        self.state_com += dt * qd[:3] # This will be q0,q1,q2
        print(self.state_com, self.state_vel, self.state_orientation, self.state_ang_vel, ud[:3])
        
    def draw(self, vis):
        vis.drawLine(np.array([0.0,0.0,0.0]), np.matmul(self.get_A(self.state_orientation[0],self.state_orientation[1],self.state_orientation[2]),self.state_com))
        vis.drawCube(matrix=np.identity(4), model_pos=np.array([0.0,0.0,0.0]), scale=0.02, col=(1,0,0,1))
        vis.drawCube(matrix=np.identity(4), model_pos=np.matmul(self.get_A(self.state_orientation[0],self.state_orientation[1],self.state_orientation[2]),self.state_com), scale=0.02, col=(1,0,0,1))

    # @classmethod
    # def buildObjectFromInertiaMatrix(cls, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, referenceFrame=np.identity(3), com=np.zeros((3,1)), name="object", mass=1.0):
    #     new_object = cls(com, referenceFrame, name, mass)
    #     new_object.I = np.matrix([[I_xx, -I_xy, -I_xz],[-I_xy, I_yy, -I_yz],[-I_xz, -I_yz, I_zz]]) # Inertia Matrix
    #     return new_object
    
    def addTorque(self, torque):
        self.torques += [torque]
        
    def addForce(self, force):
        self.forces += [force]
        
class Model:
    def __init__(self):
        self.t = me.dynamicsymbols._t
        self.origin = me.Point('O')
        self.referenceFrame = me.ReferenceFrame('N')
        
        # the origin mustn't move
        self.origin.set_vel(self.referenceFrame, 0)
        
        self.objects = {}
        
        # calculation matrices
        self.qd = None
        self.ud = None
        self.Mk = None
        self.gk = None
        self.Md = None
        self.gd = None
        
        # equations of motion function
        self.eval_eom = None
        
    def addObject(self, obj):
        self.objects[obj.name] = obj
        
    def setup(self):
        Fr_bar = [] # This defines the total "Partial forces" (Generalised Active Forces) for each object: the force component of each angular speed
        Frs_bar = [] # This defines the time-derivative solutions to the partial forces (Generalised Inertia Forces) for each object in terms of the mass, acceleration, inertia, etc.
        # Effectively we're setting up the formulae here and then sympy will solve them to give us the required speeds etc.
        # Fr is the "left-hand side" of the Newton-Euler formulae for the equations of motion
        # Frs is the "right-hand side"
        # Fr = (u.R) + (w.T) = (-m*a.u) + (a.I + wxI.w) = Frs

        total_num_dofs = np.sum([len(o.dofs) for o in self.objects.values()])
        print("Total number of DOFs:", total_num_dofs)

        # Now we loop through each object and get its angular speed
        for name, obj in self.objects.items():
            Fr = [0,0,0,0,0,0]
            Frs = [0,0,0,0,0,0]
            # The angular speed of each object is dependent on the forces applied to all connected objects
            # First, deal with the active forces (that move the centre of mass without rotation)
            for fname, force_obj in self.objects.items(): # can this be reduced so we're not getting forces for all objects?
                for force in force_obj.forces:
                    fr, frs = force_obj.getFrFrsFromForce(force)
                    Fr = [Fr[i] + fr[i] for i in range(6)]
                    Frs = [Frs[i] + frs[i] for i in range(6)]
            
            for tname, torque_obj in self.objects.items(): # can this be reduced so we're not getting forces for all objects?
                for torque in torque_obj.torques:
                    fr, frs = torque_obj.getFrFrsFromTorque(torque)
                    Fr = [Fr[i] + fr[i] for i in range(6)]
                    Frs = [Frs[i] + frs[i] for i in range(6)]
        
            Fr_bar.append(Fr)
            Frs_bar.append(Frs)
    
        # Put the total right-hand and left-hand sides into matrices
        Fr = sm.Matrix(Fr_bar)
        Frs = sm.Matrix(Frs_bar)

        # Put the orientation and angular speed into matrices
        Is = sm.Matrix([[self.objects[o].Ixx,self.objects[o].Iyy,self.objects[o].Izz,self.objects[o].Ixy,self.objects[o].Iyz,self.objects[o].Ixz] for o in self.objects])
        qs = sm.Matrix([self.objects[o].dof_position + self.objects[o].dof_orientation for o in self.objects])
        us = sm.Matrix([self.objects[o].dof_vel + self.objects[o].dof_ang_vel for o in self.objects])
        ms = sm.Matrix([self.objects[o].mass for o in self.objects])

        # Define the time differentials of orientation and angular speed
        self.qd = qs.diff(self.t)
        self.ud = us.diff(self.t)

        # Set the acceleration to 0
        ud_zerod = {udr: 0 for udr in self.ud}

        # This is the site of forward kinematics vs inverse kinematics
        # Initialise Mk (kinematics) - For forward kinematics, we don't know the velocities but want to find them
        self.Mk = -sm.eye(total_num_dofs)
        self.gk = us

        # Initialise Md (dynamics) - For forward kinematics, we know the component forces and time derivatives 
        self.Md = Frs.jacobian(self.ud)
        self.gd = Frs.xreplace(ud_zerod) + Fr

        # Now build the equations of motion function that will be applied each time step
        self.eval_eom = sm.lambdify((qs, us, ms, Is), [self.Mk, self.gk, self.Md, self.gd])
        
    def solve(self):
        q_vals = []
        u_vals = []
        m_vals = []
        i_vals = []
        for name, obj in self.objects.items():
            q_vals += [obj.state_com[0], obj.state_com[1], obj.state_com[2], obj.state_orientation[0], obj.state_orientation[1], obj.state_orientation[2]]
            u_vals += [obj.state_vel[0], obj.state_vel[1], obj.state_vel[2], obj.state_ang_vel[0], obj.state_ang_vel[1], obj.state_ang_vel[2]]
            m_vals += [obj.state_mass]
            i_vals += [obj.state_inertia[0,0], obj.state_inertia[1,1], obj.state_inertia[2,2], obj.state_inertia[0,1], obj.state_inertia[1,2], obj.state_inertia[0,2]]
            
        q_vals = np.array(q_vals)
        u_vals = np.array(u_vals)
        m_vals = np.array(m_vals)
        i_vals = np.array(i_vals)
        
        Mk_vals, gk_vals, Md_vals, gd_vals = self.eval_eom(q_vals, u_vals, m_vals, i_vals)

        # Now the hard work must be done: find the speeds and accelerations from the
        # system of equations defined by the mass matrix.

        # calculate the angular speed - this apparently may not always equal u but I don't know why...
        qd_vals = np.linalg.solve(-Mk_vals, np.squeeze(gk_vals))

        # Now the angular acceleration
        ud_vals = np.linalg.solve(-Md_vals, np.squeeze(gd_vals))
        
        return qd_vals, ud_vals
        

class Muscle:
    def __init__(self):
        self.name = "muscle"
        

gravity_constant = sm.symbols('g')
groundFrame = me.ReferenceFrame('N')
groundOrigin = me.Point('O')
groundOrigin.set_vel(groundFrame, 0) # lock the origin by setting the vel to zero (required for v2pt_theory later)

obj = Object(groundOrigin, groundFrame, ['rot_x','rot_y','rot_z','pos_x','pos_y','pos_z'])
obj.setStateCom(np.array([0.0,-0.1,0.0]))
obj.setStateOrientation(np.array([0.0,0.0,0.3]))
obj.addForce(obj.mass*-9.81*groundFrame.y)
obj.addTorque(0.0*groundFrame.z)
mod = Model()

mod.addObject(obj)
mod.setup()

vis = Visualiser()
vis.setupVisualiser()

for i in range(1000):
    qd, ud = mod.solve()
    obj.updateState(qd, ud, 0.01)

    vis.beginRendering()
    obj.draw(vis)
    vis.endRendering()

# ground = Bone(1.0,1.0,1.0,0.0,0.0,0.0, com_world=np.reshape(np.array([0.0,0.5,0.0]), (3,1)))
# bone = Bone(0.0001,0.0001,0.0001,0,0,0, parent=ground, pivot_local=np.array([0.0,0.5,0.0]))
# bone.addLinearForce(np.transpose(np.array([[0.0,-9.8,0.0]])))
# bone.addForce(np.transpose(np.array([[0.0,0.5,0.0]])), np.transpose(np.array([[0.3,0.0,0.0]]))) # Add a force that acts upwards with 1N 0.3 across from the origin 

# vis = Visualiser()
# vis.setupVisualiser()

# for i in range(1000):
#     bone.euler(0.001)
#     bone.forces = []
#     vis.beginRendering()
#     ground.draw(vis)
#     bone.draw(vis)
#     vis.endRendering()