from __future__ import print_function
from fenics import *
from ufl import nabla_div

length = 1.0
W = 0.2
H = 0.2

mu = 3000
rho = 100
lambda_ = 5000
g = 10

mesh = BoxMesh(Point(0,0,0),Point(length,W,H),10,3,3)
V = VectorFunctionSpace(mesh,'P',1)

tol = 1E-14

def boundary_left(x,on_boundary):
	return (on_boundary and near(x[0],0))

def boundary_right(x,on_boundary):
	return on_boundary and near(x[0],length)

bc_left = DirichletBC(V,Constant((0,0,0)),boundary_left)
bc_right = DirichletBC(V,Constant((0.01,0,0)),boundary_right)

def epsilon(u):
	return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
	return lambda_*nabla_div(u)*Identity(d) + mu*(epsilon(u) + epsilon(u).T)

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)

f = Constant((0,0,0))
T = Constant((0,0,0))

a = inner(sigma(u),epsilon(v))*dx
L = dot(f,v)*dx + dot (T,v)*ds

u = Function(V)
solve(a == L, u, [bc_left,bc_right])

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