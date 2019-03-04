from __future__ import print_function
from fenics import *
from ufl import nabla_div

L = 1;
W = 0.2;
H = 0.2;

rho = 100;
g = 10;
lambda_ = 5000;
mu_ = 3000;

mesh = BoxMesh(Point(0,0,0), Point(L,W,H),10,3,3)
V = VectorFunctionSpace(mesh,'P',1)

tol = 1E-14

def clamped_boundary(x, on_boundary):
	return on_boundary and x[0] < tol

bc = DirichletBC(V, Constant((0,0,0)), clamped_boundary)

def epsilon(u):
	return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
	return lambda_*nabla_div(u)*Identity(d) + mu_*(epsilon(u) + epsilon(u).T)

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
f = Constant((0,0,-rho*g))
T = Constant((0,0,0))

a = inner(sigma(u),epsilon(v))*dx
L = dot(f,v)*dx + dot(T,v)*ds

u = Function(V)
solve(a==L, u, bc)

vtkfile_u = File('deflection.pvd')
vtkfile_u << u

W = TensorFunctionSpace(mesh, "Lagrange", 1)
stress =  lambda_*nabla_div(u)*Identity(d) + mu_*(epsilon(u) + epsilon(u).T)
vtkfile_s = File('stress.pvd')
vtkfile_s << project(stress,W)

s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d) # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
X = FunctionSpace(mesh, 'P', 1)
vtkfile_von = File('von_Mises.pvd')
vtkfile_von << project(von_Mises, X)