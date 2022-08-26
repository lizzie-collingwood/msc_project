"""Python file written to run on CX1, producing Williamson5 test case
simulation, with energy conserving space discretisation including
upwinding for u, D, and semi-implicit time discretisation"""
import logging
import json
from time import ctime, time
start = time()
from distutils.util import strtobool
from petsc4py import PETSc
from numpy import float64, zeros, arctan2, arcsin
from numpy import sqrt as np_sqrt
from netCDF4 import Dataset
from firedrake import IcosahedralSphereMesh, SpatialCoordinate, exp, \
    as_vector, pi, FunctionSpace, FiniteElement, MixedFunctionSpace, \
    Function, TestFunction, TrialFunction, inner, dx, triangle, grad, \
    LinearVariationalProblem, LinearVariationalSolver, File, lhs, rhs, \
    FacetNormal, jump, sign, dot, dS, TestFunctions, CellNormal, cross, \
    TrialFunctions, div, assemble, sqrt, conditional, Constant, \
    DumbCheckpoint, FILE_READ, FILE_CREATE, COMM_WORLD, \
    Min, Max, atan_2, asin, cos, sin, VectorFunctionSpace, Mesh, \
    functionspaceimpl, READ, WRITE, par_loop, op2, ge, le, interpolate
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for augmented Lagrangian solver.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid.')
parser.add_argument('--dmax', type=float, default=15, help='Final time in days.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='w5aug')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--maxk', type=int, default=4, help='Degree of finite element space (the DG space).')
parser.add_argument('--atol', type=str, default=1e-8, help='The absolute size of the residual norm which is used as stopping criterion for Newton iterations.')
parser.add_argument('--rtol', type=str, default=1e-8, help='The relative size of the residual norm which is used as stopping criterion for Newton iterations.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
args = parser.parse_known_args()
args = args[0]

# some domain, parameters and FS setup
name = args.filename
dir_name = 'Test'
file_name = 'W5_u-ad_D-ad'

# discretisation parameters
ref_level = args.ref_level
# dt = 120.
dt = 60*60*args.dt
tmax = 24*60*60*args.dmax
init_t = 0.
maxk = args.maxk
field_dumpfreq = 24*30*5 # Dump every 5 days
t = 0.
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

R, Omega = 6371220., 7.292e-5
mesh = IcosahedralSphereMesh(radius=R, refinement_level=ref_level,
                             degree=2)
mesh.init_cell_orientations(SpatialCoordinate(mesh))

# Logger
logger = logging.getLogger("W5")
def set_log_handler(comm):
    """Function to set log handler using mesh comm"""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(\
        fmt="%(name)s:%(levelname)s %(message)s"))
    if logger.hasHandlers():
        logger.handlers.clear()
    if comm.rank == 0:
        logger.addHandler(handler)
    else:
        logger.addHandler(logging.NullHandler())

logger.setLevel('INFO')
set_log_handler(mesh.comm)
logger.info("Simulation for Williamson5 test case, model u-ad_D-ad")
logger.info("Starting Initial condition, and function setup at {0}" \
            .format(ctime()))

x = SpatialCoordinate(mesh)
f = Function(FunctionSpace(mesh, "CG", 1))
f.interpolate(2*Omega*x[2]/R)
g, H = 9.8, 5960.
u_0 = 20.

def latlon_coords(mesh):
    """Compute latitude-longitude coordinates given Cartesian ones"""
    x0, y0, z0 = SpatialCoordinate(mesh)
    unsafe = z0/sqrt(x0*x0 + y0*y0 + z0*z0)
    safe = Min(Max(unsafe, -1.0), 1.0)  # avoid silly roundoff errors
    theta = asin(safe)  # latitude
    lamda = atan_2(y0, x0)  # longitude
    return theta, lamda

theta, lamda = latlon_coords(mesh)
R0, lamda_c, theta_c = pi/9., -pi/2., pi/6.
R0sq, Rsq = R0**2, R**2
lsq, thsq = (lamda - lamda_c)**2, (theta - theta_c)**2
r = sqrt(Min(R0sq, lsq + thsq))

bexpr = 2000*(1 - r/R0)
uexpr = u_0*as_vector([-x[1], x[0], 0.0])/R
Dexpr = H - ((R*Omega*u_0 + 0.5*u_0**2)*x[2]**2/Rsq)/g - bexpr

# Build function spaces
degree = args.degree + 1
family = ("DG", "BDM", "CG")
W0 = FunctionSpace(mesh, family[0], degree-1, family[0])
W1_elt = FiniteElement(family[1], triangle, degree, variant='integral')
W1 = FunctionSpace(mesh, W1_elt, name="HDiv")
W2 = FunctionSpace(mesh, family[2], degree+1)
M = MixedFunctionSpace((W1, W0))

# Set up functions
xn = Function(M)
un, Dn = xn.split()
un.rename('u')
Dn.rename('D')
fields = {'u': un, 'D': Dn}

# Vorticity field
qn = Function(W2, name='potential vorticity')
fields['q'] = qn
# Vorticity field
vortn = Function(W2, name='vorticity')
# Enstrophy field
q2Dn = Function(W2, name='enstrophy')
# Topography field
b = Function(W0, name='topography')
eta_out = Function(W0, name='eta')

# Solver fields
uad = Function(W1)
uf = Function(W1)

# Time scheme fields
xp = Function(M)
up, Dp = xp.split()

xnk = Function(M)
unk, Dnk = xnk.split()

ubar = 0.5*(un + unk)
Dbar = 0.5*(Dn + Dnk)

xd = Function(M)

# Hamiltonian variations
F = Function(W1)
P = Function(W0)
u_rec = Function(W1)

# Load initial conditions onto fields
un.project(uexpr, form_compiler_parameters={'quadrature_degree': 12})
Dn.interpolate(Dexpr)
b.interpolate(bexpr)

logger.info("Finished setting up functions at {0}".format(ctime()))

# Build perp and upwind perp (including 2D version for test purposes)
n = FacetNormal(mesh)
s = lambda u: 0.5*(sign(dot(u, n)) + 1)
uw = lambda u, v: (s(u)('+')*v('+') + s(u)('-')*v('-'))

if mesh.geometric_dimension() == 2:
    perp = lambda u: as_vector([-u[1], u[0]])
    p_uw = lambda u, v: perp(uw(u, v))
else:
    perp = lambda u: cross(CellNormal(mesh), u)
    out_n = CellNormal(mesh)
    p_uw = lambda u, v: (s(u)('+')*cross(out_n('+'), v('+'))
                         +s(u)('-')*cross(out_n('-'), v('-')))

# Build advection, forcing forms
Frhs = unk*Dnk/3. + un*Dnk/6. + unk*Dn/6. + un*Dn/3.
K = inner(un, un)/3. + inner(un, unk)/3. + inner(unk, unk)/3.
Prhs = g*(Dbar + b) + 0.5*K

# Compute conserved quantities.
mass = Dn*dx
energy = (Dn*inner(un, un)/2 + g*inner(Dn+b,Dn+b)/2)*dx

# Compute absolute vorticity and enstrophy
Q = Dbar*qn*dx
Z = Dbar*qn**2*dx

gm_prms = {'ksp_type': 'gmres', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}
lu_prms = {"ksp_type":"preonly", "pc_type":"lu"}
cg_prms = {'ksp_type': 'cg', 'pc_type': 'bjacobi', 'sub_pc_type': 'ilu'}

# D advection solver
phi = TestFunction(W0)
D_ = TrialFunction(W0)

D_ad = (inner(grad(phi), Dbar*u_rec)*dx
        - jump(phi*u_rec, n)*uw(ubar, Dbar)*dS)
D_eqn = (D_ - Dn)*phi*dx - dt*D_ad
Dad_p = LinearVariationalProblem(lhs(D_eqn), rhs(D_eqn), Dp,
                                 constant_jacobian=True)
D_ad_solver = LinearVariationalSolver(Dad_p, solver_parameters=cg_prms)

# u advection solver
w = TestFunction(W1)
u_ = TrialFunction(W1)
u_bar = 0.5*(un + u_)

u_ad = (inner(perp(grad(inner(Dbar*w, perp(u_rec)))), u_bar)*dx
        + inner(jump(inner(Dbar*w, perp(u_rec)), n),
                p_uw(ubar, u_bar))*dS)
u_eqn = inner(u_ - un, Dbar*w)*dx - dt*u_ad
uad_p = LinearVariationalProblem(lhs(u_eqn), rhs(u_eqn), uad)
u_ad_solver = LinearVariationalSolver(uad_p, solver_parameters=gm_prms)

# u forcing solver
u_f = (jump(P*w, n)*uw(ubar, Dbar)*dS - inner(Dbar*w, grad(P))*dx
       - f*inner(perp(u_rec), Dbar*w)*dx)

f_eqn = inner(u_, Dbar*w)*dx - dt*u_f
uf_p = LinearVariationalProblem(lhs(f_eqn), rhs(f_eqn), uf)
f_u_solver = LinearVariationalSolver(uf_p, solver_parameters=cg_prms)

# Auxiliary solvers
Peqn = inner(phi, D_ - Prhs)*dx
Pproblem = LinearVariationalProblem(lhs(Peqn), rhs(Peqn), P,
                                    constant_jacobian=True)
Psolver = LinearVariationalSolver(Pproblem, solver_parameters=cg_prms)

u_rec_eqn = inner(w, Dbar*u_ - Frhs)*dx
u_rec_problem = LinearVariationalProblem(lhs(u_rec_eqn), rhs(u_rec_eqn),
                                         u_rec)
u_rec_solver = LinearVariationalSolver(u_rec_problem, solver_parameters=cg_prms)

# Output solvers
eta = TestFunction(W2)
q_ = TrialFunction(W2)
q_eqn = eta*q_*Dn*dx + inner(perp(grad(eta)), un)*dx - eta*f*dx
q_p = LinearVariationalProblem(lhs(q_eqn), rhs(q_eqn), qn)
qsolver = LinearVariationalSolver(q_p, solver_parameters=cg_prms)

vrt_eqn = eta*q_*dx + inner(perp(grad(eta)), un)*dx
vort_problem = LinearVariationalProblem(lhs(vrt_eqn), rhs(vrt_eqn), vortn,
                                        constant_jacobian=True)
vortsolver = LinearVariationalSolver(vort_problem, solver_parameters=cg_prms)

# Linear solver
u, D = TrialFunctions(M)
w, phi = TestFunctions(M)

eqn = (inner(w, u - (up - unk))
       +phi*(D - (Dp - Dnk))
       -0.5*dt*(g*div(w)*D - f*inner(w, perp(u)) - H*phi*div(u)))*dx

params = {'mat_type': 'matfree',
          'ksp_type': 'preonly',
          'pc_type': 'python',
          'pc_python_type': 'firedrake.HybridizationPC',
          'hybridization': {'ksp_type': 'gmres',
                            'pc_type': 'gamg',
                            'ksp_rtol': 1e-8,
                            'mg_levels': {'ksp_type': 'chebyshev',
                                          'ksp_max_it': 2,
                                          'pc_type': 'bjacobi',
                                          'sub_pc_type': 'ilu'}}}
#          'hybridization': {'ksp_type': 'preonly',
#                            'pc_type': 'lu',
#                            'pc_factor_mat_solver_type': 'mumps'}}

uD_problem = LinearVariationalProblem(lhs(eqn), rhs(eqn), xd)
uD_solver = LinearVariationalSolver(uD_problem, solver_parameters=params)

logger.info("Finished setting up solvers at {0}".format(ctime()))

# Setup output
outfile = File('{0}.pvd'.format(args.filename))
field_output = [un, eta_out, vortn, qn, q2Dn]

# output function
def write_output(t, counter, dfr, outf, fld_out):
    """Function to write vtu output"""
    # Output vtu file
    if (counter % dfr) == 0:
        qsolver.solve()
        vortsolver.solve()
        q2Dn.project(qn**2*Dn)
        eta_out.interpolate(Dn + b)        
        outf.write(*fld_out)

# Store the conserved properties data
energy0 = assemble(energy)
simdata = {t: [assemble(mass), energy0, assemble(Q), assemble(Z), 0, 0, 0]}

write_output(init_t, 0, field_dumpfreq, outfile, field_output)

logger.info("Finished setting up output at {0}".format(ctime()))

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
nonlin_itcount = 0

# Timeloop
xnk.assign(xn)
t = init_t
count = 0
while t < tmax - 0.5*dt:
    PETSc.Sys.Print(t)
    PETSc.Sys.Print('Percentage complete: ', t/tmax)
    logger.info("Timestep nr {0} at {1}".format(count, ctime()))
    t += dt
    tdump += dt

    # Compute and print quantities that should be conserved
    _mass = assemble(mass)
    _energy = assemble(energy)
    _Q = assemble(Q)
    _Z = assemble(Z)
    print("mass:", _mass)
    print("energy:", (energy0 - _energy) / energy0)
    print("abs vorticity:", _Q)
    print("enstrophy:", _Z)

    # Run solvers
    its = 0
    nonlin_its = 0
    for _ in range(maxk):
        nonlin_its += 1
        et0 = time()
        u_rec_solver.solve()
        Psolver.solve()
        D_ad_solver.solve()
        u_ad_solver.solve()
        up.assign(uad)
        f_u_solver.solve()
        up += uf

        uD_solver.solve()
        xnk += xd
        extime = time() - et0

        # Get the number of linear iterations
        its += uD_solver.snes.getLinearSolveIterations()
        # TODO: increase maxk so energy conservation is the same as ours (check in paper)

    # Update field
    xn.assign(xnk)

    simdata.update({t: [_mass, _energy, _Q, _Z, its, nonlin_its, extime]})

    # Write output
    count += 1
    write_output(t, count, field_dumpfreq, outfile, field_output)

# Print execution time
extime = time() - start
print('execution_time:', extime)

# Save the performance and solution data to json.
argz = {'base_level': args.base_level, 'ref_level': args.ref_level, 'dmax': args.dmax, 'dumpt': args.dumpt, 'dt': args.dt, 'filename': args.filename, 'coords_degree': args.coords_degree, 'degree': args.degree, 'upwind': True, 'softsign': 0, 'poisson': 'sw_im', 'snes_rtol': 1e-8, 'atol': args.atol, 'rtol': args.rtol, 'show_args': args.show_args, 'maxk':args.maxk}
argdict = str(argz)
with open(name+'.json', 'w') as f:
    json.dump({'options': argdict, 'data': simdata}, f)

# Write options to text file.
with open(name+'_options.txt', 'w') as f:
    f.write(argdict)

# Print performance metrics
PETSc.Sys.Print("Iterations", count,
                "dt", args.dt,
                "ref_level", args.ref_level,
                "dmax", args.dmax)