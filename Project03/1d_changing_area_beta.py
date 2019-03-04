"""
Python script for Part 1d.ii of Project 2a

Original Author:	Vinamra Agrawal
Date:				January 25, 2019

Edited By:			Omkar Mulekar
Date:				February 28, 2019

"""

from __future__ import print_function
from fenics import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ufl import nabla_div
import math
import numpy as np
from scipy.signal import argrelextrema

#==============================================================
# Define System Properties
#==============================================================
length = 1;
W = 0.2;
H1 = 0.2;
beta = np.linspace(0.2,1,num=5)
print("beta = ",beta)
H = beta * H1
print("H = ", H)




youngs = 200e9 # Youngs
nu = 0.3 # Poisson
rho = 7800 # Density


# Lame parameters
mu = (youngs)/(2*(1+nu))
lambda_ = (nu*youngs)/((1+nu)*(1-2*nu))

g = 10



nat_freq = [0] * len(beta)

print("Beginning Loop...")
for i in range(len(beta)):
	print("beta = ", beta[i])

	#==============================================================
	#	Dimensionless parameters
	#==============================================================
	l_nd = length/length
	w_nd = W/length
	h_nd = H[i]/length

	bar_speed = math.sqrt(youngs/rho)
	t_char = length/bar_speed
	t = 0
	t_i = 0.5
	dt = 0.1
	num_steps = 275

	mu_nd = mu/youngs
	lambda_nd = lambda_/youngs
	
	F = -100
	a = 0.04*length;
	b = 0.4*H[i];
	traction_applied = F/(a*b)
	traction_nd  = traction_applied/youngs

	#============================================================
	# Boundaries and Geometry
	#============================================================
	mesh = BoxMesh(Point(0,0,0),Point(l_nd,w_nd,h_nd),20,6,6)
	V = VectorFunctionSpace(mesh,'P',1)

	tol = 1E-14

	def boundary_left(x,on_boundary):
		return (on_boundary and near(x[0],0,tol))

	def boundary_right(x,on_boundary):
		return on_boundary and near(x[0],l_nd,tol)

	bc_left = DirichletBC(V,Constant((0,0,0)),boundary_left)
	bc_right = DirichletBC(V,Constant((0,0,0)),boundary_right)


	#============================================================
	def epsilon(u):
		return 0.5*(nabla_grad(u) + nabla_grad(u).T)

	def sigma(u):
		return lambda_nd*nabla_div(u)*Identity(d) + mu_nd*(epsilon(u) + epsilon(u).T)

	#============================================================
	# First we solve the problem of a cantelever beam under fixed
	# load. 
	#============================================================
	u_init = TrialFunction(V)
	d = u_init.geometric_dimension()
	v = TestFunction(V)
	f = Constant((0.0,0.0,0.0))
	T_init = Expression(('0.0', 'x[0] >= 0.48*l && x[0] <= 52 && near(x[1],w) && x[2] >= 0.3*h && x[2] <= 0.7*h? A : 0.0' ,'0.0'), degree=1, l=l_nd, w=w_nd,h=h_nd, A=traction_nd)
	F_init = inner(sigma(u_init),epsilon(v))*dx - dot(f,v)*dx - dot(T_init,v)*ds
	a_init, L_init = lhs(F_init), rhs(F_init)

	print("Solving the initial cantelever problem")
	u_init = Function(V)
	solve(a_init==L_init,u_init,[bc_left,bc_right])
	down_nd = u_init(l_nd/2.0,w_nd/2.0,h_nd/2.0)
	w = down_nd * length
	print("Initial Displacement: ",w[1])

	#============================================================
	# Next we use this as initial condition, let the force go and 
	# study the vertical vibrations of the beam
	#============================================================
	u_n = interpolate(Constant((0.0,0.0,0.0)),V)
	u_n_1 = interpolate(Constant((0.0,0.0,0.0)),V)
	u_n.assign(u_init)
	u_n_1.assign(u_init)

	T_n = Constant((0.0,0.0,0.0))

	u = TrialFunction(V)
	d = u.geometric_dimension()
	v = TestFunction(V)

	F = (dt*dt)*inner(sigma(u),epsilon(v))*dx + dot(u,v)*dx - (dt*dt)*dot(f,v)*dx - (dt*dt)*dot (T_n,v)*ds - 2.0*dot(u_n,v)*dx + dot(u_n_1,v)*dx
	a,L = lhs(F), rhs(F)


	u_store = [0] * num_steps
	time = [0] * num_steps

	index = 0
	for n in range(num_steps):
		print("time = %.2f" % t)
		T_n.t = t
		u = Function(V)
		solve(a == L, u, [bc_left,bc_right])
		u_grab = u(l_nd/2.0,w_nd/2.0,h_nd/2.0)
		u_store[n] = u_grab[1]

		if(abs(t-index)<0.01):
			W_ = TensorFunctionSpace(mesh, "Lagrange", 1)
			stress =  lambda_*nabla_div(u)*Identity(d) + mu*(epsilon(u) + epsilon(u).T)
			index += 1

		time[n] = t
		t+=dt
		u_n_1.assign(u_n)
		u_n.assign(u)

	plt.figure(1)
	plt.plot(time,u_store)
	plt.xlabel('time [s]')
	plt.ylabel('Vertical Deflection [m]')
	plt.savefig('1dfig_test2.png')

	# Get period of oscillation
	u_np = np.array(u_store)
	min_args = argrelextrema(u_np,np.greater)
	print("min_args", min_args[0])
	period = (time[min_args[0][1]] - time[min_args[0][0]])*t_char
	nat_freq[i] = 2*math.pi /period
	print("Period of Oscillation", period, " seconds")
	print("Natural Frequency:   ", nat_freq," rad/s")

plt.figure(2)
plt.plot(H,nat_freq,'b-x')
plt.xlabel('Beam Height [m]')
plt.ylabel('Natural Frequency [rad/s]')
plt.savefig('1dfig_beta.png')

