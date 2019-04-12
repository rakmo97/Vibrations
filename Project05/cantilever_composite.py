from __future__ import print_function
from dolfin import *
from ufl import nabla_div
import math

#==============================================================
#	Dimensional parameters
#==============================================================
length = 1.0
W = 0.1
H = 0.1

mu_l = 300e6
rho_l = 5e3
lambda_l = 400e6

mu_r = 200e5
rho_r = 5e3
lambda_r = 300e5

traction_applied = -1e4


#==============================================================
#	Dimensionless parameters
#==============================================================
youngs = (mu_l*(3.0*lambda_l+2.0*mu_l))/(lambda_l+mu_l)
bar_speed = math.sqrt(youngs/rho_l)

l_nd = length/length
w_nd = W/length
h_nd = H/length

t_char = length/bar_speed
t = 0
t_i = 0.5
dt = 0.1
num_steps = 500

mu_l_nd = mu_l/youngs
lambda_l_nd = lambda_l/youngs
mu_r_nd = mu_r/youngs
lambda_r_nd = lambda_r/youngs

#mu_nd = Expression('x[0]<=l_nd ? mu_l_nd:mu_r_nd',l_nd=l_nd,mu_l_nd=mu_l_nd,mu_r_nd=mu_r_nd,degree=1)
#lambda_nd = Expression('x[0]<=l_nd ? lambda_l_nd:lambda_r_nd',l_nd=l_nd,lambda_l_nd=lambda_l_nd,lambda_r_nd=lambda_r_nd,degree=1)

traction_nd = traction_applied/youngs

#============================================================
mesh = BoxMesh(Point(0,0,0),Point(l_nd,w_nd,h_nd),40,12,12)
S = FunctionSpace(mesh,'P',1)
V = VectorFunctionSpace(mesh,'P',1)

boundary_left = 'near(x[0],0)'
bc_left = DirichletBC(V,Constant((0,0,0)),boundary_left)

mu_nd = interpolate(Expression('x[0]<=0.5*l_nd ? mu_l_nd:mu_r_nd',l_nd=l_nd,mu_l_nd=mu_l_nd,mu_r_nd=mu_r_nd,degree=1),S)
lambda_nd = interpolate(Expression('x[0]<=0.5*l_nd ? lambda_l_nd:lambda_r_nd',l_nd=l_nd,lambda_l_nd=lambda_l_nd,lambda_r_nd=lambda_r_nd,degree=1),S)
rho_nd = interpolate(Expression('x[0]<=0.5*l_nd ? 1.0:rho_r/rho_l',l_nd=l_nd,rho_l=rho_l,rho_r=rho_r,degree=1),S)

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

F = (dt*dt/rho_nd)*inner(sigma(u),epsilon(v))*dx \
	+ dot(u,v)*dx \
	- (dt*dt/rho_nd)*dot(f,v)*dx \
	- (dt*dt/rho_nd)*dot (T_n,v)*ds \
	- 2.0*dot(u_n,v)*dx \
	+ dot(u_n_1,v)*dx
a,L = lhs(F), rhs(F)

xdmffile_u = XDMFFile('results/solution.xdmf')
xdmffile_s = XDMFFile('results/stress.xdmf')

u = Function(V)
Q = TensorFunctionSpace(mesh, "Lagrange", 1)
stress_proj = Function(Q)

index = 0
for n in range(num_steps):
	print("time = %.2f" % t)
	T_n.t = t
	solve(a == L, u, bc_left)

	if(abs(t-index)<0.01):
		print("Writing output files...")
		xdmffile_u.write(u*length,t)
		stress =  sigma(u)
		stress_proj.vector()[:] = project(stress,Q).vector()
		xdmffile_s.write(stress_proj,t)
		index += 1

	t+=dt
	u_n_1.assign(u_n)
	u_n.assign(u)
