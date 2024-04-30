import numpy as np
from visualiser import Visualiser
import sympy as sm
import sympy.physics.mechanics as me

class Object:
    def __init__(self, com=np.zeros(3), sym_parentFrame=None, name="bone", mass=1.0):
        self.name = name
        
        self.forces = []
        self.torques = []
        
        # symbols
        self.parentFrame = sym_parentFrame
        self.N = self.parentFrame

        self.mass = sm.symbols('m_' + name) # mass
        self.q0 = me.dynamicsymbols('q0_' + name) # orientation quaternion
        self.q1 = me.dynamicsymbols('q1_' + name) # orientation quaternion
        self.q2 = me.dynamicsymbols('q2_' + name) # orientation quaternion
        self.q3 = me.dynamicsymbols('q3_' + name) # orientation quaternion
        self.u0 = me.dynamicsymbols('u0_' + name) # q prime
        self.u1 = me.dynamicsymbols('u1_' + name) # q prime
        self.u2 = me.dynamicsymbols('u2_' + name) # q prime

        self.frame = me.ReferenceFrame('F_' + name) # reference frame
        self.frame.orient_quaternion(self.N, (self.q0,self.q1,self.q2,self.q3)) # Reference frame is rotated according to quaternion q
        self.frame.set_ang_vel(self.N, self.u0*self.N.x + self.u1*self.N.y + self.u2*self.N.z) # Reference frame has angular velocity u

        self.com = com # centre of mass
        #self.c.set_pos(O, 0.5*A.x) # Set the location of the com in relation to the origin

        # Inertia matrix
        self.Ixx = sm.symbols('I_xx_' + name)
        self.Iyy = sm.symbols('I_yy_' + name)
        self.Izz = sm.symbols('I_zz_' + name)
        self.Ixy = sm.symbols('I_xy_' + name)
        self.Iyz = sm.symbols('I_yz_' + name)
        self.Ixz = sm.symbols('I_xz_' + name)
        
        self.I = me.inertia(self.frameB, self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Iyz, self.Ixz)
        
    def getFrFrs(self, force, torque):
        v_com_0 = self.com.vel(self.N).diff(self.u0, self.N, var_in_dcm=False)
        v_com_1 = self.com.vel(self.N).diff(self.u1, self.N, var_in_dcm=False)
        v_com_2 = self.com.vel(self.N).diff(self.u2, self.N, var_in_dcm=False)
        w_com_0 = self.com.ang_vel_in(self.N).diff(self.u0, self.N, var_in_dcm=False)
        w_com_1 = self.com.ang_vel_in(self.N).diff(self.u1, self.N, var_in_dcm=False)
        w_com_2 = self.com.ang_vel_in(self.N).diff(self.u2, self.N, var_in_dcm=False)
        
        F1 = v_com_0.dot(force) + w_com_0.dot(torque)
        F2 = v_com_1.dot(force) + w_com_1.dot(torque)
        F3 = v_com_2.dot(force) + w_com_2.dot(torque)
        
        Fr = sm.Matrix([F1, F2, F3])
        
        Rs = -self.mass*self.com.acc(self.N)
        Ts = -(self.frame.ang_acc_in(self.N).dot(self.I) + me.cross(self.frame.ang_vel_in(self.N), self.I).dot(self.frame.ang_vel_in(self.N)))

        F1s = v_com_0.dot(Rs) + w_com_0.dot(Ts)
        F2s = v_com_1.dot(Rs) + w_com_1.dot(Ts)
        F3s = v_com_2.dot(Rs) + w_com_2.dot(Ts)
        
        Frs = sm.Matrix([F1s, F2s, F3s])
        
        return Fr, Frs

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
        
    def draw(self, visualiser):
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
    def __init_(self):
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
        for obj in self.objects:
            ui = obj.u
            Fr = 0
            Frs = 0
            # The angular speed of each object is dependent on the forces applied to all connected objects
            # First, deal with the active forces (that move the centre of mass without rotation)
            for force_obj in self.objects: # can this be reduced so we're not getting forces for all objects?
                for force in force_obj.forces:
                    Pi = force_obj.com
                    Ri = force
                    mi = force_obj.mass
                    N = self.referenceFrame
                
                    vr = Pi.vel(N).diff(ui, N)
                    Fr += vr.dot(Ri)
                    Rs = -mi*Pi.acc(N)
                    Frs += vr.dot(Rs)
            
            for torque_obj in self.objects: # can this be reduced so we're not getting forces for all objects?
                for torque in torque_obj.torques:
                    Bi = torque_obj.referenceFrame
                    Ti = torque
                    Ii = torque_obj.I
               
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
        qs = sm.Matrix([o.q for o in self.objects])
        us = sm.Matrix([o.u for o in self.objects])
        ms = sm.Matrix([o.mass for o in self.objects])

        # Define the time differentials of orientation and angular speed
        self.qd = qs.diff(self.t)
        self.ud = us.diff(self.t)

        # Set the acceleration to 0
        ud_zerod = {udr: 0 for udr in self.ud}

        # This is the site of forward kinematics vs inverse kinematics
        # Initialise Mk (kinematics) - For forward kinematics, we know the current speeds
        self.Mk = -sm.eye(len(self.objects))
        self.gk = us

        # Initialise Md (dynamics) - For forward kinematics, we know the component forces and time derivatives 
        self.Md = Frs.jacobian(self.ud)
        self.gd = Frs.xreplace(ud_zerod) + Fr

        self.eval_eom = sm.lambdify((qs, us, ms), [self.Mk, self.gk, self.Md, self.gd])
        
        

class Muscle:
    def __init__(self):
        self.name = "muscle"
        
        
ground = Bone(1.0,1.0,1.0,0.0,0.0,0.0, com_world=np.reshape(np.array([0.0,0.5,0.0]), (3,1)))
bone = Bone(0.0001,0.0001,0.0001,0,0,0, parent=ground, pivot_local=np.array([0.0,0.5,0.0]))
bone.addLinearForce(np.transpose(np.array([[0.0,-9.8,0.0]])))
bone.addForce(np.transpose(np.array([[0.0,0.5,0.0]])), np.transpose(np.array([[0.3,0.0,0.0]]))) # Add a force that acts upwards with 1N 0.3 across from the origin 

vis = Visualiser()
vis.setupVisualiser()

for i in range(1000):
    bone.euler(0.001)
    bone.forces = []
    vis.beginRendering()
    ground.draw(vis)
    bone.draw(vis)
    vis.endRendering()
