import numpy as np
from visualiser import Visualiser
import sympy as sm
import sympy.physics.mechanics as me

class Object:
    def __init__(self, sym_parentOrigin, sym_parentFrame, name="bone", mass=1.0, inertia=np.identity(3)):
        self.name = name
        
        self.forces = []
        self.torques = []
        
        self.state_mass = mass
        self.state_inertia = inertia
        self.state_orientation = np.zeros(3) # This will be r1,r2,r3
        self.state_ang_vel = np.zeros(3) # This will be u0,u1,u2
        self.state_vel = np.zeros(3) # This will be v0,v1,v2
        self.state_com = np.zeros(3) # This will be q0,q1,q2
        
        # symbols
        self.parentFrame = sym_parentFrame
        self.parentOrigin = sym_parentOrigin
        self.N = self.parentFrame
        self.O = self.parentOrigin

        self.mass = sm.symbols('m_' + name) # mass
        
        self.frame = me.ReferenceFrame('F_' + name) # reference frame
        self.r0 = me.dynamicsymbols('r0_' + name) # orientation quaternion
        self.r1 = me.dynamicsymbols('r1_' + name) # 
        self.r2 = me.dynamicsymbols('r2_' + name) # 
        self.r3 = me.dynamicsymbols('r3_' + name) # 
        self.frame.orient_quaternion(self.N, (1.0,self.r1,self.r2,self.r3) ) # Reference frame is rotated
        self.u0 = me.dynamicsymbols('u0_' + name) # angular velocity
        self.u1 = me.dynamicsymbols('u1_' + name) # 
        self.u2 = me.dynamicsymbols('u2_' + name) #
        self.frame.set_ang_vel(self.N, self.u0*self.N.x + self.u1*self.N.y + self.u2*self.N.z) # Reference frame has angular velocity u
        
        self.com = me.Point('com_' + name) # centre of mass
        self.q0 = me.dynamicsymbols('q0_' + name) # com position in world coords
        self.q1 = me.dynamicsymbols('q1_' + name) #
        self.q2 = me.dynamicsymbols('q2_' + name) #
        self.com.set_pos(self.O, self.q0*self.frame.x + self.q1*self.frame.y + self.q2*self.frame.z) 
        self.v0 = me.dynamicsymbols('v0_' + name) # velocity
        self.v1 = me.dynamicsymbols('v1_' + name) # 
        self.v2 = me.dynamicsymbols('v2_' + name) # 
        self.com.set_vel(self.N, self.v0*self.frame.x + self.v1*self.frame.y + self.v2*self.frame.z)

        # Inertia matrix
        self.Ixx = sm.symbols('I_xx_' + name)
        self.Iyy = sm.symbols('I_yy_' + name)
        self.Izz = sm.symbols('I_zz_' + name)
        self.Ixy = sm.symbols('I_xy_' + name)
        self.Iyz = sm.symbols('I_yz_' + name)
        self.Ixz = sm.symbols('I_xz_' + name)
        
        self.I = me.inertia(self.frame, self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Iyz, self.Ixz)
        
        # symbol functions
        self.get_A = sm.lambdify((self.r0, self.r1, self.r2, self.r3), self.frame.dcm(self.N))
        
    def getFrFrsFromForce(self, force):
        v_com_0 = self.com.vel(self.N).diff(self.v0, self.N, var_in_dcm=False)
        v_com_1 = self.com.vel(self.N).diff(self.v1, self.N, var_in_dcm=False)
        v_com_2 = self.com.vel(self.N).diff(self.v2, self.N, var_in_dcm=False)
        v_com_3 = self.com.vel(self.N).diff(self.u0, self.N, var_in_dcm=False)
        v_com_4 = self.com.vel(self.N).diff(self.u1, self.N, var_in_dcm=False)
        v_com_5 = self.com.vel(self.N).diff(self.u2, self.N, var_in_dcm=False)
        
        F1 = v_com_0.dot(force)
        F2 = v_com_1.dot(force)
        F3 = v_com_2.dot(force)
        F4 = v_com_3.dot(force)
        F5 = v_com_4.dot(force)
        F6 = v_com_5.dot(force)
        
        Fr = [F1, F2, F3, F4, F5, F6]
        
        Rs = -self.mass*self.com.acc(self.N)
        Ts = -(self.frame.ang_acc_in(self.N).dot(self.I) + me.cross(self.frame.ang_vel_in(self.N), self.I).dot(self.frame.ang_vel_in(self.N)))

        F1s = v_com_0.dot(Rs)
        F2s = v_com_1.dot(Rs)
        F3s = v_com_2.dot(Rs)
        F4s = v_com_3.dot(Rs)
        F5s = v_com_4.dot(Rs)
        F6s = v_com_5.dot(Rs)
        
        Frs = [F1s, F2s, F3s, F4s, F5s, F6s]
        
        return Fr, Frs

    def getFrFrsFromTorque(self, torque):
        w_com_0 = self.frame.ang_vel_in(self.N).diff(self.v0, self.N, var_in_dcm=False)
        w_com_1 = self.frame.ang_vel_in(self.N).diff(self.v1, self.N, var_in_dcm=False)
        w_com_2 = self.frame.ang_vel_in(self.N).diff(self.v2, self.N, var_in_dcm=False)
        w_com_3 = self.frame.ang_vel_in(self.N).diff(self.u0, self.N, var_in_dcm=False)
        w_com_4 = self.frame.ang_vel_in(self.N).diff(self.u1, self.N, var_in_dcm=False)
        w_com_5 = self.frame.ang_vel_in(self.N).diff(self.u2, self.N, var_in_dcm=False)
        
        F1 = w_com_0.dot(torque)
        F2 = w_com_1.dot(torque)
        F3 = w_com_2.dot(torque)
        F4 = w_com_3.dot(torque)
        F5 = w_com_4.dot(torque)
        F6 = w_com_5.dot(torque)
        
        Fr = [F1, F2, F3, F4, F5, F6]
        
        Rs = -self.mass*self.com.acc(self.N)
        Ts = -(self.frame.ang_acc_in(self.N).dot(self.I) + me.cross(self.frame.ang_vel_in(self.N), self.I).dot(self.frame.ang_vel_in(self.N)))

        F1s = w_com_0.dot(Ts)
        F2s = w_com_1.dot(Ts)
        F3s = w_com_2.dot(Ts)
        F4s = w_com_3.dot(Ts)
        F5s = w_com_4.dot(Ts)
        F6s = w_com_5.dot(Ts)
        
        Frs = [F1s, F2s, F3s, F4s, F5s, F6s]
        
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
        
    def draw(self, vis):
        vis.drawLine(np.array([0.0,0.0,0.0]), np.matmul(self.get_A(1.0,self.state_orientation[0],self.state_orientation[1],self.state_orientation[2]),np.array([0.0,-1.0,0.0])))
        vis.drawCube(matrix=np.identity(4), model_pos=np.array([0.0,0.0,0.0]), scale=0.02, col=(1,0,0,1))
        vis.drawCube(matrix=np.identity(4), model_pos=np.matmul(self.get_A(1.0,self.state_orientation[0],self.state_orientation[1],self.state_orientation[2]),np.array([0.0,-1.0,0.0])), scale=0.02, col=(1,0,0,1))

    # @classmethod
    # def buildObjectFromInertiaMatrix(cls, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, referenceFrame=np.identity(3), com=np.zeros((3,1)), name="object", mass=1.0):
    #     new_object = cls(com, referenceFrame, name, mass)
    #     new_object.I = np.matrix([[I_xx, -I_xy, -I_xz],[-I_xy, I_yy, -I_yz],[-I_xz, -I_yz, I_zz]]) # Inertia Matrix
    #     return new_object
    
    def addTorque(self, torque):
        self.torques += [torque]
        
    def addForce(self, force):
        self.forces += [force]


class Bone(Object):
    def __init__(self, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, referenceFrame=np.identity(3), com_world=np.zeros((3,1)), name="bone", mass=1.0, parent=None, pivot_local=None):
        super().__init__(I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, referenceFrame, com_world, name, mass)
        self.parent = parent
        self.children = []
        
    def draw(self, vis):
        if self.parent is not None:
            if self.child is not None: # we have a parent bone and a child bone so draw a line between the pivot points
                vis.drawLine(self.local2world(self.parent_pivot), self.local2world(self.child_pivot))
                vis.drawCube(matrix=self.getMatrix(), model_pos=self.parent_pivot, scale=0.02, col=(1,0,0,1))
                vis.drawCube(matrix=self.getMatrix(), model_pos=self.child_pivot, scale=0.02, col=(1,0,0,1))
            else: # we only have a parent bone so draw a line between the parent pivot and the centre of mass
                print(self.local2world(self.parent_pivot).shape)
                vis.drawLine(self.local2world(self.parent_pivot), self.x[:,0])
                vis.drawCube(matrix=self.getMatrix(), model_pos=self.parent_pivot, scale=0.02, col=(1,0,0,1))
                vis.drawCube(matrix=self.getMatrix(), scale=0.02, col=(1,0,0,1))
        else: # we have no parent or child so just draw a cube at the centre of mass (for now the size is arbitrary)
            vis.drawCube(matrix=self.getMatrix(), scale=0.02, col=(1,0,0,1))
        
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
        qs = sm.Matrix([[self.objects[o].q0,self.objects[o].q1,self.objects[o].q2,self.objects[o].r1,self.objects[o].r2,self.objects[o].r3] for o in self.objects])
        us = sm.Matrix([[self.objects[o].v0,self.objects[o].v1,self.objects[o].v2,self.objects[o].u0,self.objects[o].u1,self.objects[o].u2] for o in self.objects])
        ms = sm.Matrix([self.objects[o].mass for o in self.objects])

        # Define the time differentials of orientation and angular speed
        self.qd = qs.diff(self.t)
        self.ud = us.diff(self.t)

        # Set the acceleration to 0
        ud_zerod = {udr: 0 for udr in self.ud}

        # This is the site of forward kinematics vs inverse kinematics
        # Initialise Mk (kinematics) - For forward kinematics, we don't know the velocities but want to find them
        self.Mk = -sm.eye(6)
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

obj = Object(groundOrigin, groundFrame)
obj.setStateOrientation(np.array([0.0,0.0,0.3]))
obj.addForce(obj.mass*9.81*groundFrame.x)
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
