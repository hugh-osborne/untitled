import numpy as np
from visualiser import Visualiser
import sympy as sm
import sympy.physics.mechanics as me



class Object:
    def __init__(self, com_world=np.zeros((3,1)), name="bone", mass=1.0):
        self.name = name
        self.mass = mass
        self.local_I = np.identity(3)
        self.local_I_i =  np.identity(3)
        self.x = com_world # world position of the centre of mass
        self.q = np.zeros((4,1)) # orientation quaternion - assuming index 0 is the real part then i,j,k
        self.q[0,0] = 1
        self.P = np.zeros((3,1)) # linear momentum
        self.L = np.zeros((3,1)) # angular momentum
        self.mat_I_i = np.identity(3)
        self.R = np.identity(3) # orientation
        self.v = np.zeros((3,1)) # linear velocity
        self.omega = np.zeros((3,1)) # angular velocity
        self.force = np.zeros((3,1)) # net linear force
        self.torque = np.zeros((3,1)) # new angular force
        
        # Each force is defined with a (force vector, point of contact)
        self.forces = []
        # Linear forces impart no spin and act through the centre of mass (like gravity)
        # so are defined only with a force vector
        self.linear_forces = []

    @classmethod
    def buildObjectFromInertiaMatrix(cls, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, referenceFrame, com_world=np.zeros((3,1)), name="bone", mass=1.0):
        new_object = cls(com_world, name, mass)
        new_object.local_I = np.matrix([[I_xx, -I_xy, -I_xz],[-I_xy, I_yy, -I_yz],[-I_xz, -I_yz, I_zz]]) # Inertia Matrix
        new_object.local_I_i = np.linalg.inv(new_object.local_I) # Inverse Inertia Matrix
        return new_object
    

    def addForce(self, force, contact):
        self.forces += [(force, contact)]
        
    def addLinearForce(self, force):
        self.linear_forces += [force]
        
    def calculate_v(self):
        self.v = self.P / self.mass
        
    def calculate_mat_I_i(self):
        self.mat_I_i = np.matmul(np.matmul(self.R, self.local_I_i),np.transpose(self.R))
        
    def calculate_omega(self):
        self.omega = np.matmul(self.mat_I_i, self.L)
        
    def quaternion_to_matrix(self, q):
        return [[1-(2*(q[2,0]**2))-(2*(q[3,0]**2)), (2*q[1,0]*q[2,0]) - (2*q[0,0]*q[3,0]), (2*q[1,0]*q[3,0]) + (2*q[0,0]*q[2,0])],
                [(2*q[1,0]*q[2,0]) + (2*q[0,0]*q[3,0]), 1-(2*(q[1,0]**2))-(2*(q[3,0]**2)), (2*q[2,0]*q[3,0]) - (2*q[0,0]*q[1,0])],
                [(2*q[1,0]*q[3,0]) - (2*q[0,0]*q[2,0]), (2*q[2,0]*q[3,0]) + (2*q[0,0]*q[1,0]), 1-(2*(q[1,0]**2))-(2*(q[2,0]**2))]]
    
    def quatmul(self, q1, q2):
        q1v = q1[1:,0]
        q2v = q2[1:,0]
        q1s = q1[0,0]
        q2s = q2[0,0]
        result_s = (q1s*q2s) - (np.dot(q1v, q2v))
        result_v = (q1s*q2v) + (q2s*q1v) + np.cross(q1v, q2v)
        result = np.reshape(np.concatenate([[result_s], result_v], axis=0), (4,1))
        return result
    
    def matrix_to_quaternion(self, m):
        q = np.zeros((4,1))
        tr = m[0,0] + m[1,1] + m[2,2]
        
        if tr >= 0:
            s = np.sqrt(tr + 1)
            q[0,0] = 0.5 * s
            s = 0.5 / s
            q[1,0] = (m[2,1] - m[1,2]) * s
            q[2,0] = (m[0,2] - m[2,0]) * s
            q[3,0] = (m[1,0] - m[0,1]) * s
        else:
            i = 0
            if m[1,1] > m[0,0]:
                i = 1
            if m[2,2] > m[i,i]:
                i = 2
                
            if i == 0:
                s = np.sqrt((m[0,0] - (m[1,1] + m[2,2])) + 1)
                q[1,0] = 0.5 * s
                s = 0.5 / s
                q[2,0] = (m[0,1] + m[1,0]) * s
                q[3,0] = (m[2,0] + m[0,2]) * s
                q[0,0] = (m[2,1] - m[1,2]) * s
            elif i == 1:
                s = np.sqrt((m[1,1] - (m[2,2] + m[0,0])) + 1)
                q[2,0] = 0.5 * s
                s = 0.5 / s
                q[3,0] = (m[1,2] + m[2,1]) * s
                q[1,0] = (m[0,1] + m[1,0]) * s
                q[0,0] = (m[0,2] - m[2,0]) * s
            elif i ==2:
                s = np.sqrt((m[2,2] - (m[0,0] + m[1,1])) + 1)
                q[3,0] = 0.5 * s
                s = 0.5 / s
                q[1,0] = (m[2,0] + m[0,2]) * s
                q[2,0] = (m[1,2] + m[2,1]) * s
                q[0,0] = (m[1,0] - m[0,1]) * s
                
        return q
                
        
    def calculate_R(self):
        self.q = self.q / self.q[0,0]
        self.R = self.quaternion_to_matrix(self.q)
    
    def calculateNetForceTorque(self):
        self.force = np.sum(self.linear_forces, axis=0) + np.sum([f[0] for f in self.forces], axis=0) # linear force is just the sum of all forces
        self.torque = np.sum([np.reshape(np.cross(f[0][:,0], (f[1] - self.x)[:,0]),(3,1)) for f in self.forces], axis=0) # torques are calculated with the cross product
        
    def euler(self, dt):
        # update the position
        self.x += dt*self.v
        
        # update the orientation quaternion
        self.q += dt * (0.5 * self.quatmul(np.reshape(np.array([0,self.omega[0,0], self.omega[1,0], self.omega[2,0]]), (4,1)), self.q))
        
        # update linear momentum
        self.P += dt * self.force
        
        # update angular momentum
        self.L += dt * self.torque

        # Recalculate Forces etc
        self.calculateNetForceTorque()
        self.calculate_R()
        self.calculate_v()
        self.calculate_mat_I_i()
        self.calculate_omega()
        
    def getMatrix(self):
        # build the transform matrix
        mat = np.vstack((self.R, np.transpose(self.x)))
        mat = np.hstack((mat, np.zeros((4,1))))
        mat[3,3] = 1.0
        return np.transpose(mat)
    
    def local2world(self, point_3d):
        point_4d = np.reshape(np.array([point_3d[0], point_3d[1], point_3d[2], 1.0]), (4,1))
        return np.matmul(self.getMatrix(), point_4d)[:3,0]

class Bone(Object):
    def __init__(self, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, com_world=np.zeros((3,1)), name="bone", mass=1.0, parent=None, pivot_local=None):
        super().__init__(I_xx, I_yy, I_zz, I_xy, I_xz, I_yz, com_world, name, mass)
        self.parent = parent
        self.parent_pivot = pivot_local
        self.child = None
        self.child_pivot = None
        
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
        
        
        

class Muscle:
    def __init__(self):
        self.name = "muscle"
        
class Joint:
    def __init__(self):
        self.name = "joint"
        
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
