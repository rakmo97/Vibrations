from __future__ import print_function
from fenics import *
from ufl import nabla_div
import math

#==============================================================
#	Dimensional parameters
#==============================================================
length = 1.0
W = 0.1
H = 0.1

mu = 300e6
rho = 5e3
lambda_ = 400e6

traction_applied = -1e4


#==============================================================
#	Dimensionless parameters
#==============================================================
youngs = (mu*(3.0*lambda_+2.0*mu))/(lambda_+mu)
bar_speed = math.sqrt(youngs/rho)

l_nd = length/length
w_nd = W/length
h_nd = H/length

t_char = length/bar_speed
t = 0
t_i = 0.5
dt = 0.1
num_steps = 500

mu_nd = mu/youngs
lambda_nd = lambda_/youngs

traction_nd = traction_applied/youngs

#============================================================
mesh = BoxMesh(Point(0,0,0),Point(l_nd,w_nd,h_nd),20,6,6)
V = VectorFunctionSpace(mesh,'P',1)

boundary_left = 'near(x[0],0)'
bc_left = DirichletBC(V,Constant((0,0,0)),boundary_left)

tol = 1E-14

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
T_init = Expression(('0.0', 'x[0] >= 0.8*l && x[0] <= l && near(x[1],w)? A : 0.0' ,'0.0'), degree=1, l=l_nd, w=w_nd, A=traction_nd)
F_init = inner(sigma(u_init),epsilon(v))*dx - dot(f,v)*dx - dot(T_init,v)*ds
a_init, L_init = lhs(F_init), rhs(F_init)

print("First solving the initial cantelever problem")
u_init = Function(V)
solve(a_init==L_init,u_init,bc_left)

#============================================================
# Next we use this as initial condition, let the force go and 
# study the vertical vibrations of the beam
#============================================================
u_n = interpolate(Constant((0.0,0.0,0.0)),V)
u_n_1 = interpolate(Constant((0.0,0.0,0.0)),V)
u_n.assign(u_init)
u_n_1.assign(u_init)

#T_n = Expression(('near(x[0],l) ? (t <= t_i ? A : 0.0) : 0.0','0.0','0.0'), degree=1, l=l_nd, A=traction_nd, t=t, t_i=t_i)
T_n = Constant((0.0,0.0,0.0))

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)

F = (dt*dt)*inner(sigma(u),epsilon(v))*dx + dot(u,v)*dx - (dt*dt)*dot(f,v)*dx - (dt*dt)*dot (T_n,v)*ds - 2.0*dot(u_n,v)*dx + dot(u_n_1,v)*dx
a,L = lhs(F), rhs(F)

xdmffile_u = XDMFFile('results/solution.xdmf')
xdmffile_s = XDMFFile('results/stress.xdmf')

u = Function(V)

index = 0
for n in range(num_steps):
	print("time = %.2f" % t)
	T_n.t = t
	solve(a == L, u, bc_left)

	if(abs(t-index)<0.01):
		print("Writing output files...")
		xdmffile_u.write(u*length,t)
		W = TensorFunctionSpace(mesh, "Lagrange", 1)
		stress =  lambda_*nabla_div(u)*Identity(d) + mu*(epsilon(u) + epsilon(u).T)
		xdmffile_s.write(project(stress,W),t)
		index += 1

	t+=dt
	u_n_1.assign(u_n)
	u_n.assign(u)
