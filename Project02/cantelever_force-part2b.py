"""
Python script for Part 1b of Project 2b

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

length = .150;
W = 0.025;
H = 0.025;

F = -10

n = [3,4,5,6]
# vertical_deflection = [] * len(n)
w = [0] * len(n)

for i in range(len(n)):
	print(i)
	A = length / n[i]

	mu = 7.69e10
	rho = 7800
	lambda_ = 1.153e11
	g = 10

	mesh = BoxMesh(Point(0,0,0),Point(length,W,H),10,3,3)
	V = VectorFunctionSpace(mesh,'P',1)

	tol = 1E-14

	def boundary_left(x,on_boundary):
		return (on_boundary and near(x[0],0))

	def boundary_right(x,on_boundary):
		return on_boundary and near(x[0],length)

	bc_left = DirichletBC(V,Constant((0,0,0)),boundary_left)
	bc_right = DirichletBC(V,Constant((0,0,0)),boundary_right)

	def epsilon(u):
		return 0.5*(nabla_grad(u) + nabla_grad(u).T)

	def sigma(u):
		return lambda_*nabla_div(u)*Identity(d) + mu*(epsilon(u) + epsilon(u).T)

	u = TrialFunction(V)
	d = u.geometric_dimension()
	v = TestFunction(V)

	f = Constant((0,0,0))
	T = Expression(('0.0','(x[0] >= a && x[0] <= (L-a) && near(x[1],W)) ? (F/(H*(L-2*a))) : 0.0','0.0'),L=length,a=A,F=F,W=W,H=H,degree=1)

	a = inner(sigma(u),epsilon(v))*dx
	L = dot(f,v)*dx + dot (T,v)*ds

	u = Function(V)
	solve(a == L, u, [bc_left,bc_right])

	w[i] = u(length/2.0,W/2.0,H/2.0)[1]
	print(w[1])
	# vertical_deflection[i] = w[1]

	vtkfile_u = File('deflection.pvd')
	vtkfile_u << u

	W = TensorFunctionSpace(mesh, "Lagrange", 1)
	stress =  lambda_*nabla_div(u)*Identity(d) + mu*(epsilon(u) + epsilon(u).T)

	vtkfile_s = File('stress.pvd')
	vtkfile_s << project(stress,W)

	s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
	von_Mises = sqrt(3./2*inner(s, s))
	X = FunctionSpace(mesh, 'P', 1)
	vtkfile_von = File('von_Mises.pvd')
	vtkfile_von << project(von_Mises, X)
	W = 0.025

plt.figure(1)
plt.plot(n,w,'b-x')
plt.xlabel('N')
plt.ylabel('Vertical Deflection [m]')
plt.savefig('2afig1.png')