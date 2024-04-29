import numpy as np
from visualiser import Visualiser
import sympy as sm
import sympy.physics.mechanics as me

class Object:
    def __init__(self, com=np.zeros((3,1)), sym_parentFrame=None, name="bone", mass=1.0):
        self.name = name
        self.mass = mass
        self.I = np.identity(3)
        self.com = com
        self.frame = np.identity(3)
        
        self.ang = np.zeros(3)
        self.ang_vel = np.zeros(3)
        
        self.forces = []
        self.torques = []
        
        # symbols
        self.N = sym_parentFrame

        self.m = sm.symbols('m_' + name) # mass
        self.q = me.dynamicsymbols('q_' + name) # orientation quaternion
        self.u = me.dynamicsymbols('u_' + name) # q prime

        self.F = me.ReferenceFrame('F_' + name) # reference frame
        self.F.orient_axis(self.N, 'Quaternion', self.q) # Reference frame is rotated according to quaternion q
        self.F.set_ang_vel(self.N, self.u) # Reference frame has angular velocity u

        self.c = me.Point('c_' + name) # centre of mass
        self.c.set_pos(O, 0.5*A.x) # Set the location of the com in relation to the origin

        self.i = self.I

    @classmethod
    def buildObjectFromInertiaMatrix(cls, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, referenceFrame=np.identity(3), com=np.zeros((3,1)), name="object", mass=1.0):
        new_object = cls(com, referenceFrame, name, mass)
        new_object.I = np.matrix([[I_xx, -I_xy, -I_xz],[-I_xy, I_yy, -I_yz],[-I_xz, -I_yz, I_zz]]) # Inertia Matrix
        return new_object
    
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
        
    def addObject(self, obj):
        self.objects[obj.name] = obj
        
    def setup(self):
        
        

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
