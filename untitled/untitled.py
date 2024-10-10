from csv import QUOTE_ALL
import numpy as np
from visualiser import Visualiser
import sympy as sm
import sympy.physics.mechanics as me

class Object:
    def __init__(self, name, dofs, parent=None, mass=100.0, inertia=np.identity(3)):
        self.name = name
        
        self.parent = parent
        if parent != None:
            self.parent.child = self
        self.child = None
        
        self.forces = []
        self.torques = []
        self.force_symbols = []
        self.torque_symbols = []
        self.state_forces = []
        self.state_torques = []
        
        self.state_mass = mass
        self.state_inertia = inertia
        self.state_orientation = np.zeros(3) # This will be r1,r2,r3
        self.state_ang_vel = np.zeros(3) # This will be u0,u1,u2
        self.state_vel = np.zeros(3) # This will be v0,v1,v2
        self.state_com = np.zeros(3) # This will be q0,q1,q2
        self.state_parent_pivot = np.zeros(3)
        # Declare which degrees of freedom we want to calculate for this object.
        # Any dofs not declared will be locked to the parent frame
        # full set is ['rot_x','rot_y','rot_z','pos_x','pos_y','pos_z']
        self.dofs = dofs
        self.num_dofs = len(self.dofs)
        self.num_position_dofs = len([1 for a in ['pos_x','pos_y','pos_z'] if a in self.dofs])
        self.num_rotation_dofs = len([1 for a in ['rot_x','rot_y','rot_z'] if a in self.dofs])
        
        self.dof_dynamic_inds = {}
        self.dof_static_inds = {}
        dindex = 0
        sindex = 0
        for f in ['rot_x','rot_y','rot_z']:
            if f in self.dofs:
                self.dof_dynamic_inds[f] = dindex
                dindex += 1
            else:
                self.dof_static_inds[f] = sindex
                sindex += 1
        dindex = 0
        sindex = 0
        for f in ['pos_x','pos_y','pos_z']:
            if f in self.dofs:
                self.dof_dynamic_inds[f] = dindex
                dindex += 1
            else:
                self.dof_static_inds[f] = sindex
                sindex += 1
                
        # limits on dofs - a min/max pair for each dof
        self.dof_limits = {}
        
        # symbols
        if self.parent == None:
            self.parentFrame = me.ReferenceFrame('N')
            self.parentPivot = me.Point('O')                 
            self.parentPivot.set_vel(self.parentFrame, 0)
            self.groundFrame = self.parentFrame
        else:
            self.parentFrame = self.parent.frame
            self.parentPivot = me.Point('P_' + name)
            self.parentPivotSymbols = [sm.symbols('p0_' + name), sm.symbols('p1_' + name), sm.symbols('p2_' + name)]
            self.parentPivot.set_pos(self.parent.com, self.parentPivotSymbols[0]*self.parent.frame.x + self.parentPivotSymbols[1]*self.parent.frame.y + self.parentPivotSymbols[2]*self.parent.frame.z)
            self.groundFrame = self.parent.groundFrame
        self.N = self.parentFrame
        self.O = self.parentPivot

        self.mass = sm.symbols('m_' + name) # mass
        self.dynamic_orientation = []
        self.dynamic_ang_vel = []
        self.dynamic_position = []
        self.dynamic_vel = []
        self.static_orientation = []
        self.static_ang_vel = []
        self.static_position = []
        self.static_vel = []
        self.frame = me.ReferenceFrame('F_' + name) # reference frame
        # Orientation
        orientation = []
        if 'rot_x' in self.dofs:
            self.dynamic_orientation += [me.dynamicsymbols('r0_' + name)] # 
            orientation += [self.dynamic_orientation[-1]]
        else:
            self.static_orientation += [sm.symbols('r0_' + name)]
            orientation += [self.static_orientation[-1]]
        if 'rot_y' in self.dofs:
            self.dynamic_orientation += [me.dynamicsymbols('r1_' + name)] # 
            orientation += [self.dynamic_orientation[-1]]
        else:
            self.static_orientation += [sm.symbols('r1_' + name)]
            orientation += [self.static_orientation[-1]]
        if 'rot_z' in self.dofs:
            self.dynamic_orientation += [me.dynamicsymbols('r2_' + name)] # 
            orientation += [self.dynamic_orientation[-1]]
        else:
            self.static_orientation += [sm.symbols('r3_' + name)]
            orientation += [self.static_orientation[-1]]
            
        self.frame.orient_body_fixed(self.N, orientation, 'XYZ')
        

        # Angular Velocity
        ang_vel = 0
        dynamic_ang_vel = 0
        if 'rot_x' in self.dofs:
            self.dynamic_ang_vel += [me.dynamicsymbols('u0_' + name)] #
            ang_vel += self.dynamic_ang_vel[-1]*self.N.x
            dynamic_ang_vel += self.dynamic_ang_vel[-1]*self.N.x
        else:
            self.static_ang_vel += [sm.symbols('u0_' + name)]
            ang_vel += self.static_ang_vel[-1]*self.N.x
        if 'rot_y' in self.dofs:
            self.dynamic_ang_vel += [me.dynamicsymbols('u1_' + name)] # 
            ang_vel += self.dynamic_ang_vel[-1]*self.N.y
            dynamic_ang_vel += self.dynamic_ang_vel[-1]*self.N.y
        else:
            self.static_ang_vel += [sm.symbols('u1_' + name)]
            ang_vel += self.static_ang_vel[-1]*self.N.y
        if 'rot_z' in self.dofs:
            self.dynamic_ang_vel += [me.dynamicsymbols('u2_' + name)] # 
            ang_vel += self.dynamic_ang_vel[-1]*self.N.z
            dynamic_ang_vel += self.dynamic_ang_vel[-1]*self.N.z
        else:
            self.static_ang_vel += [sm.symbols('u2_' + name)]
            ang_vel += self.static_ang_vel[-1]*self.N.z
        
        self.frame.set_ang_vel(self.N, dynamic_ang_vel)
        
        # Centre of mass
        self.com = me.Point('com_' + name) #
        com_pos = 0
        if 'pos_x' in self.dofs:
            self.dynamic_position += [me.dynamicsymbols('q0_' + name)] #
            com_pos += self.dynamic_position[-1]*self.frame.x
        else:
            self.static_position += [sm.symbols('q0_' + name)]
            com_pos += self.static_position[-1]*self.frame.x
        if 'pos_y' in self.dofs:        
            self.dynamic_position += [me.dynamicsymbols('q1_' + name)] #
            com_pos += self.dynamic_position[-1]*self.frame.y
        else:
            self.static_position += [sm.symbols('q1_' + name)]
            com_pos += self.static_position[-1]*self.frame.y
        if 'pos_z' in self.dofs:
            self.dynamic_position += [me.dynamicsymbols('q2_' + name)] #
            com_pos += self.dynamic_position[-1]*self.frame.z
        else:
            self.static_position += [sm.symbols('q2_' + name)]
            com_pos += self.static_position[-1]*self.frame.z
        
        self.com.set_pos(self.O, com_pos) 

        # Centre of mass velocity
        com_vel = 0
        dynamic_com_vel = 0
        if 'pos_x' in self.dofs:
            self.dynamic_vel += [me.dynamicsymbols('v0_' + name)] # velocity
            com_vel += self.dynamic_vel[-1]*self.frame.x
            dynamic_com_vel += self.dynamic_vel[-1]*self.frame.x
        else:
            self.static_vel += [sm.symbols('v0_' + name)]
            com_vel += self.static_vel[-1]*self.frame.x
        if 'pos_y' in self.dofs:        
            self.dynamic_vel += [me.dynamicsymbols('v1_' + name)] # 
            com_vel += self.dynamic_vel[-1]*self.frame.y
            dynamic_com_vel += self.dynamic_vel[-1]*self.frame.y
        else:
            self.static_vel += [sm.symbols('v1_' + name)]
            com_vel += self.static_vel[-1]*self.frame.y
        if 'pos_z' in self.dofs:    
            self.dynamic_vel += [me.dynamicsymbols('v2_' + name)] # 
            com_vel += self.dynamic_vel[-1]*self.frame.z
            dynamic_com_vel += self.dynamic_vel[-1]*self.frame.z
        else:
            self.static_vel += [sm.symbols('v2_' + name)]
            com_vel += self.static_vel[-1]*self.frame.z
        
        self.com.set_vel(self.frame, dynamic_com_vel)
        self.com.v1pt_theory(self.O, self.groundFrame, self.frame)

        # Inertia matrix
        self.Ixx = sm.symbols('I_xx_' + name)
        self.Iyy = sm.symbols('I_yy_' + name)
        self.Izz = sm.symbols('I_zz_' + name)
        self.Ixy = sm.symbols('I_xy_' + name)
        self.Iyz = sm.symbols('I_yz_' + name)
        self.Ixz = sm.symbols('I_xz_' + name)
        
        self.I = me.inertia(self.frame, self.Ixx, self.Iyy, self.Izz, self.Ixy, self.Iyz, self.Ixz)
        
        # symbol functions
        self.get_A = sm.lambdify(orientation, self.N.dcm(self.frame))
        
    def addDOFLimits(self, dof, _min, _max):
        if dof not in self.dofs:
            print(dof, "is not a degree of freedom for this object (", self.name, ").")
            return
        
        self.dof_limits[dof] = (_min, _max)
        
    def getFrameWorldMatrix(self):
        if self.parent == None:
            return self.get_A(self.state_orientation[0],self.state_orientation[1],self.state_orientation[2])
        return np.matmul(self.parent.getFrameWorldMatrix(), self.get_A(self.state_orientation[0],self.state_orientation[1],self.state_orientation[2]))
    
    def getComWorld(self):
        if self.parent == None:
            return np.matmul(self.getFrameWorldMatrix(), self.state_com)    
        return self.getParentPivotWorld() + np.matmul(self.getFrameWorldMatrix(), self.state_com)
    
    def getPointWorld(self, point):
        if self.parent == None:
            return np.matmul(self.getFrameWorldMatrix(), point)    
        return self.getParentPivotWorld() + np.matmul(self.getFrameWorldMatrix(), point)
    
    def getPointVelocityFrame(self, point):
        # velocity of a point in the frame is the cross of the frame rotational_vel and the point vector from pivot
        v = point - self.state_com
        w = self.state_ang_vel
        return np.cross(v,w)
    
    def getParentPivotWorld(self):
        if self.parent == None:
            return np.matmul(self.getFrameWorldMatrix(), self.state_parent_pivot)    
        return self.parent.getComWorld() + np.matmul(self.parent.getFrameWorldMatrix(), self.state_parent_pivot)
        
    def getFrFrsFromForce(self, force, symbol):
        Rs = -self.mass*self.com.acc(self.groundFrame)
        v_com_0 = self.com.vel(self.groundFrame).diff(symbol, self.groundFrame)
        return v_com_0.dot(force), v_com_0.dot(Rs)

    def getFrFrsFromTorque(self, torque, symbol):
        Ts = -(self.frame.ang_acc_in(self.groundFrame).dot(self.I) + me.cross(self.frame.ang_vel_in(self.groundFrame), self.I).dot(self.frame.ang_vel_in(self.groundFrame)))
        w_com_0 = self.frame.ang_vel_in(self.groundFrame).diff(symbol, self.groundFrame)
        
        return w_com_0.dot(torque), w_com_0.dot(Ts)

    def setStateMass(self, m):
        self.state_mass = m
        
    def setStateOrientation(self, r):
        self.state_orientation = r
        
    def setStateCom(self, c):
        self.state_com = c
        
    def setStateParentPivot(self, p):
        self.state_parent_pivot = p
        
    def getStateParentPivot(self):
        return self.state_parent_pivot
       
    def getDofOrientationValues(self):
        dvals = []
        svals = []
        if 'rot_x' in self.dofs:
            dvals += [self.state_orientation[0]]
        else:
            svals += [self.state_orientation[0]]
        if 'rot_y' in self.dofs:
            dvals += [self.state_orientation[1]]
        else:
            svals += [self.state_orientation[1]]
        if 'rot_z' in self.dofs:
            dvals += [self.state_orientation[2]]
        else:
            svals += [self.state_orientation[2]]
        return np.array(dvals), np.array(svals)
    
    def getDynamicOrientationValues(self):
        dvals,_ = self.getDofOrientationValues()
        return dvals
    
    def getStaticOrientationValues(self):
        _,svals = self.getDofOrientationValues()
        return svals
    
    def getDofAngVelValues(self):
        dvals = []
        svals = []
        if 'rot_x' in self.dofs:
            dvals += [self.state_ang_vel[0]]
        else:
            svals += [self.state_ang_vel[0]]
        if 'rot_y' in self.dofs:
            dvals += [self.state_ang_vel[1]]
        else:
            svals += [self.state_ang_vel[1]]
        if 'rot_z' in self.dofs:
            dvals += [self.state_ang_vel[2]]
        else:
            svals += [self.state_ang_vel[2]]
        return np.array(dvals), np.array(svals)
    
    def getDynamicAngVelValues(self):
        dvals,_ = self.getDofAngVelValues()
        return dvals
    
    def getStaticAngVelValues(self):
        _,svals = self.getDofAngVelValues()
        return svals
    
    def getDofComValues(self):
        dvals = []
        svals = []
        if 'pos_x' in self.dofs:
            dvals += [self.state_com[0]]
        else:
            svals += [self.state_com[0]]
        if 'pos_y' in self.dofs:
            dvals += [self.state_com[1]]
        else:
            svals += [self.state_com[1]]
        if 'pos_z' in self.dofs:
            dvals += [self.state_com[2]]
        else:
            svals += [self.state_com[2]]
        return np.array(dvals), np.array(svals)
    
    def getDynamicComValues(self):
        dvals,_ = self.getDofComValues()
        return dvals
    
    def getStaticComValues(self):
        _,svals = self.getDofComValues()
        return svals
    
    def getDofVelValues(self):
        dvals = []
        svals = []
        if 'pos_x' in self.dofs:
            dvals += [self.state_vel[0]]
        else:
            svals += [self.state_vel[0]]
        if 'pos_y' in self.dofs:
            dvals += [self.state_vel[1]]
        else:
            svals += [self.state_vel[1]]
        if 'pos_z' in self.dofs:
            dvals += [self.state_vel[2]]
        else:
            svals += [self.state_vel[2]]
        return np.array(dvals), np.array(svals)
    
    def getDynamicVelValues(self):
        dvals,_ = self.getDofVelValues()
        return dvals
    
    def getStaticVelValues(self):
        _,svals = self.getDofVelValues()
        return svals
    
    def setDynamicOrientationValues(self, vals):
        counter = 0
        if 'rot_x' in self.dofs:
            self.state_orientation[0] = vals[counter]
            counter += 1
        if 'rot_y' in self.dofs:
            self.state_orientation[1] = vals[counter]
            counter += 1
        if 'rot_z' in self.dofs:
            self.state_orientation[2] = vals[counter]
            counter += 1
    
    def setDynamicAngVelValues(self, vals):
        counter = 0
        if 'rot_x' in self.dofs:
            self.state_ang_vel[0] = vals[counter]
            counter += 1
        if 'rot_y' in self.dofs:
            self.state_ang_vel[1] = vals[counter]
            counter += 1
        if 'rot_z' in self.dofs:
            self.state_ang_vel[2] = vals[counter]
            counter += 1    

    def setDynamicComValues(self, vals):
        counter = 0
        if 'pos_x' in self.dofs:
            self.state_com[0] = vals[counter]
            counter += 1
        if 'pos_y' in self.dofs:
            self.state_com[1] = vals[counter]
            counter += 1
        if 'pos_z' in self.dofs:
            self.state_com[2] = vals[counter]
            counter += 1
    
    def setDynamicVelValues(self, vals):
        counter = 0
        if 'pos_x' in self.dofs:
            self.state_vel[0] = vals[counter]
            counter += 1
        if 'pos_y' in self.dofs:
            self.state_vel[1] = vals[counter]
            counter += 1
        if 'pos_z' in self.dofs:
            self.state_vel[2] = vals[counter]
            counter += 1
        
    def updateState(self, qds, uds, dt):
        com_vals = self.getDynamicComValues() + dt * np.array(qds[:self.num_position_dofs])
        vel_vals = self.getDynamicVelValues() + dt * np.array(uds[:self.num_position_dofs])
        orientation_vals = self.getDynamicOrientationValues() + dt * np.array(qds[self.num_position_dofs:])
        ang_vel_vals = self.getDynamicAngVelValues() + dt * np.array(uds[self.num_position_dofs:])
        
        for d in ["rot_x", "rot_y", "rot_z"]:
            if d in self.dofs and d in self.dof_limits:
                if orientation_vals[self.dof_dynamic_inds[d]] < self.dof_limits[d][0]:
                    orientation_vals[self.dof_dynamic_inds[d]] = self.dof_limits[d][0]
                    ang_vel_vals[self.dof_dynamic_inds[d]] = 0.0
                if orientation_vals[self.dof_dynamic_inds[d]] > self.dof_limits[d][1]:
                    orientation_vals[self.dof_dynamic_inds[d]] = self.dof_limits[d][1]
                    ang_vel_vals[self.dof_dynamic_inds[d]] = 0.0
                    
        for d in ["pos_x", "pos_y", "pos_z"]:
            if d in self.dofs and d in self.dof_limits:
                if com_vals[self.dof_dynamic_inds[d]] < self.dof_limits[d][0]:
                    com_vals[self.dof_dynamic_inds[d]] = self.dof_limits[d][0]
                    vel_vals[self.dof_dynamic_inds[d]] = 0.0
                if com_vals[self.dof_dynamic_inds[d]] > self.dof_limits[d][1]:
                    com_vals[self.dof_dynamic_inds[d]] = self.dof_limits[d][1]
                    vel_vals[self.dof_dynamic_inds[d]] = 0.0

        self.setDynamicComValues(com_vals)
        self.setDynamicOrientationValues(orientation_vals)
        self.setDynamicVelValues(vel_vals)
        self.setDynamicAngVelValues(ang_vel_vals)
        
    def draw(self, vis):
        if self.parent == None: # Draw the ground object with just a single green cube
            vis.drawCube(matrix=np.identity(4), model_pos=self.getComWorld(), scale=0.02, col=(0,1,0,1))
            return
        if self.parent != None:
            if self.child != None: # This object has both a parent and child so draw a line between the pivots
                vis.drawLine(self.getParentPivotWorld(), self.child.getParentPivotWorld())
                vis.drawCube(matrix=np.identity(4), model_pos=self.getParentPivotWorld(), scale=0.02, col=(1,0,0,1))
                #vis.drawCube(matrix=np.identity(4), model_pos=self.child.getParentPivotWorld(), scale=0.02, col=(1,0,0,1))
                vis.drawCube(matrix=np.identity(4), model_pos=self.getComWorld(), scale=0.01, col=(0,1,0,1))
            else : # This is the end of the chain so just draw
                vis.drawLine(self.getParentPivotWorld(), self.getComWorld())
                vis.drawCube(matrix=np.identity(4), model_pos=self.getParentPivotWorld(), scale=0.02, col=(1,0,0,1))
                vis.drawCube(matrix=np.identity(4), model_pos=self.getComWorld(), scale=0.01, col=(0,1,0,1))
                
    # Force and Point are given in the locel reference frame (not ground)
    def addTorqueAsForcePoint(self, force_x, force_y, force_z, point_x, point_y, point_z, frame):
        torque = np.cross(np.array([point_x, point_y, point_z]), np.array([force_x, force_y, force_z]))
        return self.addTorque(torque[0], torque[1], torque[2], frame)
    
    def addTorqueAsForcePointInFrame(self, force_x, force_y, force_z, point_x, point_y, point_z):
        return self.addTorqueAsForcePoint(force_x, force_y, force_z, point_x, point_y, point_z, self.frame)
        
    def addRotationalDamping(self, coeff):
        torque = -coeff * self.frame.ang_vel_in(self.parentFrame)
        self.state_torques += [[]]
        self.torque_symbols += [[]]
        self.torques += [torque]
        return len(self.torques)-1
        
    def addTranslationalDamping(self, coeff):
        force = -coeff * self.com.vel(self.parentFrame)
        self.state_forces += [[]]
        self.force_symbols += [[]]
        self.forces += [force]
        return len(self.forces)-1
    
    def addTorqueInFrame(self, x, y, z):
        return self.addTorque(x, y, z, self.frame)
        
    def addForceInFrame(self, x, y, z):
        return self.addForce(x, y, z, self.frame)
    
    def addTorque(self, x, y, z, frame):
        self.state_torques += [np.array([x, y, z])]
        self.torque_symbols += [[me.dynamicsymbols('t_x_' + str(len(self.torques)) + '_' + self.name), me.dynamicsymbols('t_y_' + str(len(self.torques)) + '_' + self.name), me.dynamicsymbols('t_z_' + str(len(self.torques)) + '_' + self.name)]]
        self.torques += [self.torque_symbols[-1][0]*frame.x + self.torque_symbols[-1][1]*frame.y + self.torque_symbols[-1][2]*frame.z]
        return len(self.torques)-1
        
    def addForce(self, x, y, z, frame):
        self.state_forces += [np.array([x, y, z])]
        self.force_symbols += [[me.dynamicsymbols('f_x_' + str(len(self.forces)) + '_' + self.name), me.dynamicsymbols('f_y_' + str(len(self.forces)) + '_' + self.name), me.dynamicsymbols('f_z_' + str(len(self.forces)) + '_' + self.name)]]
        self.forces += [self.force_symbols[-1][0]*frame.x + self.force_symbols[-1][1]*frame.y + self.force_symbols[-1][2]*frame.z]
        return len(self.forces)-1
    
    def updateTorque(self, torque_id, x, y, z):
        self.state_torques[torque_id] = [x, y, z]
        
    def updateForce(self, force_id, x, y, z):
        self.state_forces[force_id] = [x, y, z]
        
    def updateTorqueAsForcePoint(self, torque_id, force_x, force_y, force_z, point_x, point_y, point_z):
        torque = np.cross(np.array([point_x, point_y, point_z]), np.array([force_x, force_y, force_z]))
        self.state_torques[torque_id] = [torque[0], torque[1], torque[2]]
        
class MillardMuscle:
    def __init__(self):
        # muscle activation
        self.neural_excitation = sm.symbols('u') # input
        self.muscle_activation_tc = sm.symbols('tau_a') # time constant
        self.muscle_deactivation_tc = sm.symbols('tau_d') # time constant
        self.muscle_activation = me.dynamicsymbols('a')
        
        # equilibrium model
        self.musculotendon_length = me.dynamicsymbols('l_MT') # input from bone model
        self.musculotendon_velocity = me.dynamicsymbols('v_MT') # input from bone model
        
        self.max_active_force = sm.symbols('f_o_M')
        self.max_active_velocity = sm.symbols('v_max_M')
        self.max_active_muscle_length = sm.symbols('l_o_M')
        self.tendon_slack_length = sm.symbols('l_s_T')
        
        self.pennation_angle = me.dynamicsymbols('alpha')
        
        self.force_length = me.dynamicsymbols('f_L') # This is a predefined function
        self.force_velocity = me.dynamicsymbols('f_V') # This is a predefined function
        self.passive_force = me.dynamicsymbols('f_PE')
        self.tendon_force = me.dynamicsymbols('f_T')
        self.muscle_force = me.dynamicsymbols('f_M') # final output
        
        print("not implemented")
        
class RiveraMuscle:
    def __init__(self, name, _a, _b, _c, _L_0, _L_min, _origin_obj, _origin_point, _insertion_obj, _insertion_point):
        self.name = name
        self.a = _a
        self.b = _b
        self.c = _c
        self.L_0 = _L_0 # Muscle equilibrium length (spring not under tension)
        self.L_min = _L_min # minimum length
        
        self.origin_obj = _origin_obj
        self.origin_point = _origin_point
        self.insertion_obj = _insertion_obj
        self.insertion_point = _insertion_point
        self.insertion_force_id = self.insertion_obj.addTorqueAsForcePointInFrame(0.0,0.0,0.0,0.0,0.0,0.0)
        
    def calculateL_t(self, activation):
        return self.L_0 - (activation*(self.L_0 - self.L_min))
    
    def calculateForce(self, length, length_velocity, activation):
        L_t = self.calculateL_t(activation)
        L_m = length
        L_m_dot = length_velocity
        f_a = self.a*(L_t - L_m) - L_m_dot # active force
        f_p = self.b*(self.L_0 - L_m) - L_m_dot # passive force
        return (self.c*activation*f_a) + f_p # total force
    
    def getPrincipleVector(self):
        return self.insertion_obj.getPointWorld(self.insertion_point) - self.origin_obj.getPointWorld(self.origin_point)
        
    def calculateForceBetweenObjectPoints(self, activation):
        v = self.getPrincipleVector()
        L_m = np.linalg.norm(v)
        L_m_dot = np.dot(self.insertion_obj.getPointVelocityFrame(self.insertion_point), v) # getPointVelocityFrame gets the velocity with respect to the parent bone - so this assumes origin_obj is the parent! Uh, Biarticular much?
        return self.calculateForce(L_m, L_m_dot, activation)
    
    def updateForce(self, activation):
        prin_vector = self.getPrincipleVector()
        force_unit_vector = prin_vector / np.linalg.norm(prin_vector)
        force_vector = force_unit_vector * -self.calculateForceBetweenObjectPoints(activation)
        self.insertion_obj.updateTorqueAsForcePoint(self.insertion_force_id, force_vector[0], force_vector[1], force_vector[2], self.insertion_point[0], self.insertion_point[1], self.insertion_point[2])
        
    def draw(self, vis):
        vis.drawLine(self.insertion_obj.getPointWorld(self.insertion_point), self.origin_obj.getPointWorld(self.origin_point), col=(0,0,1,1))
        vis.drawCube(matrix=np.identity(4), model_pos=self.insertion_obj.getPointWorld(self.insertion_point), scale=0.01, col=(0,0,1,1))
        vis.drawCube(matrix=np.identity(4), model_pos=self.origin_obj.getPointWorld(self.origin_point), scale=0.01, col=(0,0,1,1))
        
from collada import Collada
class Mannequin:
    def __init__(self):
        self.mesh = Collada('BaseMesh_Anim2.dae')
        self.geometry = self.mesh.geometries[0]
        self.triset = self.geometry.primitives[0]
        self.trilist = list(self.triset)
        self.vertices = self.triset.vertex[self.triset.vertex_index]
        self.normals = self.triset.normal[self.triset.normal_index]
        
        # Eventually, these values need to be calculated based on the bones
        self.scale = [0.005,0.005,0.005] # shrink the bloke
        self.rot = [-90.0*(np.pi/180),0.0,0.0] # rotate the bloke to face the camera
        self.pos = [0.0,0.0,0.0] # shift the bloke to 0.0
        
        self.rot_x_mat = np.array([[1, 0, 0, 0],
                          [0, np.cos(self.rot[0]), -np.sin(self.rot[0]), 0],
                          [0, np.sin(self.rot[0]), np.cos(self.rot[0]), 0],
                          [0, 0, 0, 1]])

        self.rot_y_mat = np.array([[np.cos(self.rot[1]), 0, np.sin(self.rot[1]), 0],
                          [0, 1, 0, 0],
                          [-np.sin(self.rot[1]), 0, np.cos(self.rot[1]), 0],
                          [0, 0, 0, 1]])
        
        self.rot_z_mat = np.array([[np.cos(self.rot[2]), -np.sin(self.rot[2]), 0, 0],
                          [np.sin(self.rot[2]), np.cos(self.rot[2]), 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])
        
        self.trans_mat = [[1, 0, 0, self.pos[0]],
                          [0, 1, 0, self.pos[1]],
                          [0, 0, 1, self.pos[2]],
                          [0, 0, 0, 1]]
        
        self.scale_mat = [[self.scale[0], 0, 0, 0],
                          [0, self.scale[1], 0, 0],
                          [0, 0, self.scale[2], 0],
                          [0, 0, 0, 1]]
        
        self.transform_mat = np.identity(4)
        self.transform_mat = np.matmul(self.transform_mat, self.scale_mat)
        self.transform_mat = np.matmul(self.transform_mat, self.rot_x_mat)
        self.transform_mat = np.matmul(self.transform_mat, self.rot_y_mat)
        self.transform_mat = np.matmul(self.transform_mat, self.rot_z_mat)
        self.transform_mat = np.matmul(self.transform_mat, self.trans_mat)
        
        # Currently this model has three figures (man woman child) in a single mesh.
        # Man is on the left so remove all verts beyond a certain point.
        # Later, move only the verts we're interested in to its own file (using pycollada or in blender or something)
        # Also transform the points
        self.vertices = self.vertices.tolist()
        for tri in range(len(self.vertices)):
            for v in range(len(self.vertices[tri])):
                self.vertices[tri][v] = self.vertices[tri][v] + [1.0]
        
        for tri in range(len(self.vertices)):
            self.vertices[tri] = np.matmul(self.transform_mat, np.array(self.vertices[tri]).T).T
        
        self.vertices = np.array(self.vertices)
        # Load the skin controller (for the bones and vertex weights)
        self.controller = list(self.mesh.scene.objects('controller'))
        
        self.weights = []
        for w in range(28010):
            self.weights += [w]
            
        # vcounts gives the number of bones that affect each vertex
        # v gives pairs of values: the first of each pair is the bone index (up to 47 in this case)
        # the second is the index to the weights loaded above
        # so for each count in vcounts, we associate that many pairs from v with each vertex in turn

        print(self.controller[0].skin.nindices)
        
        
    def setVis(self, vis):
        self.vis = vis
        verts = np.reshape(self.vertices, (self.vertices.shape[0]*self.vertices.shape[1], self.vertices.shape[2]))
        self.vis.addModel(verts)
        

class Model:
    def __init__(self):
        self.t = me.dynamicsymbols._t
        
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

        total_num_dofs = np.sum([o.num_dofs for o in self.objects.values()])
        print("Total number of DOFs:", total_num_dofs)

        # Now we loop through each object and get its angular speed
        for _, obj in self.objects.items():
            for dof_o in obj.dynamic_vel + obj.dynamic_ang_vel:
                Fr = 0
                Frs = 0
                
                for _, force_obj in self.objects.items(): # can this be reduced so we're not getting forces for all objects?
                    for force in force_obj.forces:
                        fr, frs = force_obj.getFrFrsFromForce(force, dof_o)
                        Fr += fr
                        Frs += frs
            
                for _, torque_obj in self.objects.items(): # can this be reduced so we're not getting forces for all objects?
                    for torque in torque_obj.torques:
                        fr, frs = torque_obj.getFrFrsFromTorque(torque, dof_o)
                        Fr += fr
                        Frs += frs
                
                Fr_bar.append(Fr)
                Frs_bar.append(Frs)    
    
        # Put the total right-hand and left-hand sides into matrices
        Fr = sm.Matrix(Fr_bar)
        Frs = sm.Matrix(Frs_bar)

        # Put the orientation and angular speed into matrices
        Is = []
        for o in self.objects:
            Is += [self.objects[o].Ixx,self.objects[o].Iyy,self.objects[o].Izz,self.objects[o].Ixy,self.objects[o].Iyz,self.objects[o].Ixz]
        Is = sm.Matrix([Is])
        statics = []
        for o in self.objects:
            statics += self.objects[o].parentPivotSymbols + self.objects[o].static_position + self.objects[o].static_orientation + self.objects[o].static_vel + self.objects[o].static_ang_vel
        statics = sm.Matrix([statics])
        forces_torques = []
        for o in self.objects:
            for f in self.objects[o].force_symbols:
                forces_torques += f
            for t in self.objects[o].torque_symbols:
                forces_torques += t
        forces_torques = sm.Matrix([forces_torques])
        qs = []
        for o in self.objects:
            qs += self.objects[o].dynamic_position + self.objects[o].dynamic_orientation
        qs = sm.Matrix(qs)
        us = []
        for o in self.objects:
            us += self.objects[o].dynamic_vel + self.objects[o].dynamic_ang_vel
        us = sm.Matrix(us)
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
        
        self.gd = self.gd.xreplace(dict(zip(self.qd, us)))

        # Now build the equations of motion function that will be applied each time step
        self.eval_eom = sm.lambdify((forces_torques, statics, qs, us, ms, Is), [self.Mk, self.gk, self.Md, self.gd])
        
    def solve(self, dt):
        forces_torques_vals = []
        static_vals = []
        q_vals = []
        u_vals = []
        m_vals = []
        i_vals = []
        for name, obj in self.objects.items():
            for f in obj.state_forces:
                forces_torques_vals = np.concatenate([forces_torques_vals, f], axis=0)
            for t in obj.state_torques:
                forces_torques_vals = np.concatenate([forces_torques_vals, t], axis=0)
            static_vals = np.concatenate([static_vals, obj.getStateParentPivot(), obj.getStaticComValues(), obj.getStaticOrientationValues(), obj.getStaticVelValues(), obj.getStaticAngVelValues()], axis=0)
            q_vals = np.concatenate([q_vals, obj.getDynamicComValues(),obj.getDynamicOrientationValues()], axis=0)
            u_vals = np.concatenate([u_vals, obj.getDynamicVelValues(),obj.getDynamicAngVelValues()], axis=0)
            m_vals += [obj.state_mass]
            i_vals += [obj.state_inertia[0,0], obj.state_inertia[1,1], obj.state_inertia[2,2], obj.state_inertia[0,1], obj.state_inertia[1,2], obj.state_inertia[0,2]]
            
        forces_torques_vals = np.array(forces_torques_vals)
        static_vals = np.array(static_vals)
        q_vals = np.array(q_vals)
        u_vals = np.array(u_vals)
        m_vals = np.array(m_vals)
        i_vals = np.array(i_vals)
        
        Mk_vals, gk_vals, Md_vals, gd_vals = self.eval_eom(forces_torques_vals, static_vals, q_vals, u_vals, m_vals, i_vals)

        # Now the hard work must be done: find the speeds and accelerations from the
        # system of equations defined by the mass matrix.

        # calculate the angular speed - this apparently may not always equal u but I don't know why...
        qd_vals = np.linalg.solve(-Mk_vals, gk_vals)

        # Now the angular acceleration
        ud_vals = np.linalg.solve(-Md_vals, gd_vals)
        dof_counter = 0
        for _, obj in self.objects.items():
            obj.updateState(qd_vals[dof_counter:dof_counter+obj.num_dofs], ud_vals[dof_counter:dof_counter+obj.num_dofs], dt)
            dof_counter += obj.num_dofs
        
        return qd_vals, ud_vals
        

gravity_constant = sm.symbols('g')
ground = Object("ground", [], parent=None, mass=0.0)

obj = Object("bone1", ['rot_z'], parent=ground, mass=10)
obj.setStateParentPivot(np.array([0.0,-0.1, 0.0])) # pivot is -0.1 below ground
obj.setStateCom(np.array([0.0,-0.1,0.0])) # com is -0.1 below the pivot
obj.setStateOrientation(np.array([0.0,0.0,0.0]))
obj.addForce(0.0, 0.0*obj.state_mass*-9.81, 0.0, ground.frame)
obj.addTorque(0.0, 0.0, 0.0, ground.frame)
obj.addRotationalDamping(5)

obj2 = Object("bone2", ['rot_z','pos_y'], parent=obj, mass=10)
obj2.setStateParentPivot(np.array([0.0,-0.1, 0.0]))
obj2.setStateCom(np.array([0.0,-0.1,0.0]))
obj2.addDOFLimits("pos_y", -0.2, 0.2)
obj2.setStateOrientation(np.array([0.0,0.0,0.1]))
obj2.addForce(0.0, 0.0*obj.state_mass*-9.81, 0.0, ground.frame)
obj2.addTorque(0.0, 0.0, 0.0, ground.frame)
obj2.addRotationalDamping(5)
#obj2_torque = obj2.addTorqueAsForcePointInFrame(100.0,0.0,0.0, 0.0,-0.5,0.0)

muscle = RiveraMuscle("musc", 10.0, 100.0, 1.0, 0.7, 0.3, obj, [0.0,0.0,0.0], obj2, [0.0,-0.5,0.0])

man = Mannequin()


mod = Model()
mod.addObject(obj)
mod.addObject(obj2)
mod.setup()

vis = Visualiser()

man.setVis(vis)

vis.setupVisualiser()

for i in range(1000):
    muscle.updateForce(1.0)
    qd, ud = mod.solve(0.01)

    vis.beginRendering()
    ground.draw(vis)
    obj.draw(vis)
    obj2.draw(vis)
    muscle.draw(vis)
    #man.draw(vis)
    vis.endRendering()
