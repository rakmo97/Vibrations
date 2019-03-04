"""
	Heat Lab
	Author:		Vinamra Agrawal

	Edited by:	Omkar Mulkear
"""

from __future__ import print_function
from fenics import *
import numpy as np

final_time = 2.0
num_steps = 50
dt = final_time/num_steps

D = 0.01

nx = ny = 50
mesh = UnitSquareMesh(nx,ny)
V = FunctionSpace(mesh,'P',2)

u_d = Constant(2.0)

def boundary(x,on_boundary):
	return on_boundary

bc = DirichletBC(V,u_d,boundary)

#u_n = interpolate(u_d,V)
u_n = interpolate(Constant(0.0),V)

u = TrialFunction(V)
v = TestFunction(V)
#f = Constant(beta-2-2*alpha)
f = Constant(0.0)

a = D*dt*dot(grad(u),grad(v))*dx + u*v*dx
L = (u_n + dt*f)*v*dx

u = Function(V)
vtkfile = File('results/solution.pvd')

t = 0

for n in range(num_steps):
	t += dt
	#u_d.t = t

	solve(a==L,u,bc)

	vtkfile << (u,t)

	u_e = interpolate(u_d,V)
	vertex_values_ue = u_e.compute_vertex_values(mesh);
	vertex_values_u = u.compute_vertex_values(mesh);
	error = np.abs(vertex_values_u - vertex_values_ue).max()
	print('t = %.2f: error = %.3g' % (t, error))

	u_n.assign(u)