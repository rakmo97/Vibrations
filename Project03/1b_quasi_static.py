"""
Python script for Part 1b of Project 2a

Original Author:	Vinamra Agrawal
Date:				January 25, 2019

Edited By:			Omkar Mulekar
Date:				February 10, 2019

"""

from __future__ import print_function
from fenics import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ufl import nabla_div
import math

#==============================================================
# Define System Properties
#==============================================================
length = 1;
W = 0.2;
H = 0.2;

a = 0.04;
b = 0.4;
area = a*b;

F = -100

youngs = 200e9 # Youngs
nu = 0.3 # Poisson
rho = 7800 # Density


# Lame parameters
mu = (youngs)/(2*(1+nu))
lambda_ = (nu*youngs)/((1+nu)*(1-2*nu))

g = 10

traction_applied = F/area

#==============================================================
#	Dimensionless parameters
#==============================================================
l_nd = length/length
w_nd = W/length
h_nd = H/length

mu_nd = mu/youngs
lambda_nd = lambda_/youngs

traction_nd = traction_applied/youngs

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
w_nd = u_init(l_nd/2.0,w_nd/2.0,h_nd/2.0)
w = w_nd * length
print(w[1])

vtkfile_u = File('deflection.pvd')
vtkfile_u << u_init

## END ##

# mesh = BoxMesh(Point(0,0,0),Point(length,W,H),10,3,3)
# V = VectorFunctionSpace(mesh,'P',1)

# tol = 1E-14

# def boundary_left(x,on_boundary):
# 	return (on_boundary and near(x[0],0))

# def boundary_right(x,on_boundary):
# 	return on_boundary and near(x[0],length)

# bc_left = DirichletBC(V,Constant((0,0,0)),boundary_left)
# bc_right = DirichletBC(V,Constant((0,0,0)),boundary_right)

# def epsilon(u):
# 	return 0.5*(nabla_grad(u) + nabla_grad(u).T)

# def sigma(u):
# 	return lambda_*nabla_div(u)*Identity(d) + mu*(epsilon(u) + epsilon(u).T)

# u = TrialFunction(V)
# d = u.geometric_dimension()
# v = TestFunction(V)

# f = Constant((0,0,0))
# T = Expression(('0.0','(x[0] >= a && x[0] <= (L-a) && near(x[1],W)) ? (F/(H*(L-2*a))) : 0.0','0.0'),L=length,a=A,F=F,W=W,H=H,degree=1)

# a = inner(sigma(u),epsilon(v))*dx
# L = dot(f,v)*dx + dot (T,v)*ds

# u = Function(V)
# solve(a == L, u, [bc_left,bc_right])

# w = u(length/2.0,W/2.0,H/2.0)
# print(w[1])

# vtkfile_u = File('deflection.pvd')
# vtkfile_u << u

# W = TensorFunctionSpace(mesh, "Lagrange", 1)
# stress =  lambda_*nabla_div(u)*Identity(d) + mu*(epsilon(u) + epsilon(u).T)

# vtkfile_s = File('stress.pvd')
# vtkfile_s << project(stress,W)

# s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
# von_Mises = sqrt(3./2*inner(s, s))
# X = FunctionSpace(mesh, 'P', 1)
# vtkfile_von = File('von_Mises.pvd')
# vtkfile_von << project(von_Mises, X)
