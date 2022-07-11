import firedrake as fd
import json
#get command arguments
from petsc4py import PETSc
import numpy as np
PETSc.Sys.popErrorHandler()
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for augmented Lagrangian solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.') # : change default to default=1 (deffo less than 3)
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 15.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--filename', type=str, default='w5aug')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--upwind', type=str, default=True, help='Calculation of an approximation of u: "avg" or "upwind".')
parser.add_argument('--snes_rtol', type=str, default=1e-8, help='The absolute size of the residual norm which is used as stopping criterion for Newton iterations.')
parser.add_argument('--atol', type=str, default=1e-8, help='The absolute size of the residual norm which is used as stopping criterion for Newton iterations.')
parser.add_argument('--rtol', type=str, default=1e-8, help='The relative size of the residual norm which is used as stopping criterion for Newton iterations.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
args = parser.parse_known_args()
args = args[0]

# numpy save using tofile - method of array, function fromfile to read in array, sep (default saves as binary, use sep=' ' saves as text file)

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = 6371220. # radius of earth [m]
H = fd.Constant(5960.) # mean depth [m]
base_level = args.base_level # TODO: what is base_level representing?
nrefs = args.ref_level - base_level # number of refinements
name = args.filename
deg = args.coords_degree # degree of coordinate field? 
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

def high_order_mesh_hierarchy(mh, degree, R0): # works out multigrid mesh
    meshes = []
    for m in mh:
        X = fd.VectorFunctionSpace(m, "Lagrange", degree)
        new_coords = fd.interpolate(m.coordinates, X)
        x, y, z = new_coords
        r = (x**2 + y**2 + z**2)**0.5
        new_coords.assign(R0*new_coords/r)
        new_mesh = fd.Mesh(new_coords)
        meshes.append(new_mesh)

    return fd.HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)

basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=base_level,
                                    degree=1,
                                    distribution_parameters = distribution_parameters)
del basemesh._radius
mh = fd.MeshHierarchy(basemesh, nrefs)
mh = high_order_mesh_hierarchy(mh, deg, R0)
for mesh in mh:
    xf = mesh.coordinates
    mesh.transfer_coordinates = fd.Function(xf)
    x = fd.SpatialCoordinate(mesh)
    r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
    xf.interpolate(R0*xf/r)
    mesh.init_cell_orientations(x)
mesh = mh[-1]

R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh) # extract Cartesian coords

outward_normals = fd.CellNormal(mesh) # set up orientation of global normals


def perp(u):
    """ Define the perp operator by taking cross products of velocities with cell normals.
    """
    return fd.cross(outward_normals, u)


# === Set up function spaces and mixed function space
degree = args.degree # degree of FE space / complex
V1 = fd.FunctionSpace(mesh, "BDM", degree+1) # set up velocity space
V2 = fd.FunctionSpace(mesh, "DG", degree) # set up depth space (discontinuous galerkin)
V0 = fd.FunctionSpace(mesh, "CG", degree+2) # set up space for pv

W = fd.MixedFunctionSpace((V1, V2, V1)) # create mixed space
# :: velocity, depth, potential vorticity, momentum
# BDM - vector valued, linear components, \
# compatible spaces, deg => second order, 

Omega = fd.Constant(7.292e-5)  # angular rotation rate [rads]
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant [ms^-2]
b = fd.Function(V2, name="Topography") # bathymetry from depth space
c = fd.sqrt(g*H)

# Initialise test functions
v, phi, w = fd.TestFunctions(W)

dx = fd.dx

Un = fd.Function(W)
Unp1 = fd.Function(W)

u0, h0, F0 = fd.split(Un)
u1, h1, F1 = fd.split(Unp1)
uh = 0.5*(u0 + u1)
hh = 0.5*(h0 + h1)

K = 0.5*fd.inner(uh, uh)
dT = fd.Constant(0.)

# ========= Equations
def dHdu(u0, u1, D0, D1):
    """Compute the u-derivative of Hamiltonian."""
    return (D0*u0 + D0*u1/2 + D1*u0/2 + D1*u1)/3

def dHdD(u0, u1, D0, D1, g, b):
    """Compute the D-derivative of Hamiltonian."""
    a = fd.inner(u0, u0) + fd.inner(u0, u1) + fd.inner(u1, u1)
    return a/6 + g*((D1 + D0)/2 + b)

# Finite element variational forms of the 3-variable shallow water equations
def u_energy_op(v, u, h, F, dD):
    """"""
    dS = fd.dS
    n = fd.FacetNormal(mesh)

    def both(x):
        return 2*fd.avg(x)

    # Compute approximation of u according to arg approx_type.
    if args.upwind:
        up = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
        uappx = both(up*u) # FIXME: check this is right
    else:
        uappx = fd.avg(u)

    return (fd.inner(v, f*perp(F)/h)*dx
            - fd.inner(perp(fd.grad(fd.inner(v, perp(F)/h))), u)*dx
            + fd.inner(both(perp(n)*fd.inner(v, perp(F)/h)), uappx)*dS
            - fd.div(v)*dD*dx)

# Construct components of poisson integrator
du = dHdu(u0, u1, h0, h1)
dD = dHdD(u0, u1, h0, h1, g, b)

# Poisson integrator
p_vel_eqn = (
    fd.inner(v, u1 - u0)*dx
    + phi*(h1 - h0)*dx
    + dT*u_energy_op(v, uh, hh, F1, dD)
    + phi*dT*fd.div(F1)*dx
    + fd.inner(w, F1 - du)*dx
    )

# Compute conserved quantities.
mass = h0*dx
energy = (h0*fd.inner(u0, u0)/2 + g*h0*(h0/2 + b))*dx

# Tell petsce how to solve nonlinear equations
mg_parameters = {
    "snes_monitor": None, # monitor the nonlinear solver's iterations from the web browser.
    "mat_type": "matfree", # works with matrix as a linear operator rather than with its representation as a matrix
    "ksp_type": "fgmres", # ksp is the package of linear solvers - flexible GMRES
    "ksp_monitor_true_residual": None, # print the residual after each iteration
    "ksp_converged_reason": None, # print reason for convergence
    "snes_converged_reason": None, # print reason for convergence
    "snes_rtol": args.snes_rtol, # set convergence criterion; relative size of residual norm for nonlinear iterations
    "snes_atol": 1e-50,
    "snes_stol": 1e-50,
    "ksp_atol": args.atol, # conv test: measure of the absolute size of the residual norm
    "ksp_rtol": args.rtol, # conv test: the decrease of the residual norm relative to the norm of the right hand side
    "ksp_max_it": 40, # cap the number of iterations
    "pc_type": "mg", # precontitioning method - geometric multigrid preconditioner (Newton-Krylov-multigrid method)
    "pc_mg_cycle_type": "v", # V-cycle
    "pc_mg_type": "multiplicative", # one of additive multiplicative full cascade
    "mg_levels_ksp_type": "gmres", # linear solver for the mg levels
    "mg_levels_ksp_max_it": 5, # max iterations for the levels of multigrid
    "mg_levels_pc_type": "python", #
    "mg_levels_pc_python_type": "firedrake.PatchPC", #
    "mg_levels_patch_pc_patch_save_operators": True, #
    "mg_levels_patch_pc_patch_partition_of_unity": True, #
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense", #
    "mg_levels_patch_pc_patch_construct_codim": 0, #
    "mg_levels_patch_pc_patch_construct_type": "vanka", #
    "mg_levels_patch_pc_patch_local_type": "additive", #
    "mg_levels_patch_pc_patch_precompute_element_tensors": True, #
    "mg_levels_patch_pc_patch_symmetrise_sweep": False, #
    "mg_levels_patch_sub_ksp_type": "preonly", # applies only pc exactly once
    "mg_levels_patch_sub_pc_type": "lu", # LU preconditioner
    "mg_coarse_ksp_type": "preonly", # on coarse level use only pc exactly once
    "mg_coarse_pc_type": "python", #
    "mg_coarse_pc_python_type": "firedrake.AssembledPC", #
    "mg_coarse_assembled_pc_type": "lu", # coarsest level not matrix free
    "mg_coarse_assembled_ksp_type": "preonly", #
    "mg_coarse_assembled_pc_factor_mat_solver_type": "superlu_dist", #
}

# Time step size [s]
dt = 60*60*args.dt
dT.assign(dt)
t = 0.

# Nonlinear solver
nprob = fd.NonlinearVariationalProblem(p_vel_eqn, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob, solver_parameters=mg_parameters)

dmax = args.dmax 
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

# --- set up test case
x = fd.SpatialCoordinate(mesh)
u_0 = 20.0
u_max = fd.Constant(u_0) # maximum amplitude of the zonal wind [m/s]
u_expr = fd.as_vector([-u_max*x[1]/R0, u_max*x[0]/R0, 0.0])
eta_expr = - ((R0 * Omega * u_max + u_max*u_max/2.0)*(x[2]*x[2]/(R0*R0)))/g
un = fd.Function(V1, name="Velocity").project(u_expr)
etan = fd.Function(V2, name="Elevation").project(eta_expr)

# Topography.
rl = fd.pi/9.0
lambda_x = fd.atan_2(x[1]/R0, x[0]/R0)
lambda_c = -fd.pi/2.0
phi_x = fd.asin(x[2]/R0)
phi_c = fd.pi/6.0
minarg = fd.Min(pow(rl, 2),
                pow(phi_x - phi_c, 2) + pow(lambda_x - lambda_c, 2))
bexpr = 2000.0*(1 - fd.sqrt(minarg)/rl)
b.interpolate(bexpr)

# Initial conditions
u0, h0, F0 = Un.split()
u0.assign(un)
h0.assign(etan + H - b)

# Compute potential vorticity using solvers
q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Potential Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx - p*f*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

# Compute absolute vorticity and enstrophy
Q = hh*qn*dx
Z = hh*qn**2*dx

# Write initial fields into a file which can be interpreted by software ParaView
file_sw = fd.File(name+'.pvd')
etan.assign(h0 - H + b)

# # Write new file to hold solver data
# file_sw_data = fd.File(name+'.JSON') 
# with open(name+'.json', 'w') as f:
#     json.dump(mass, f)
simdata = {t: [fd.assemble(mass), fd.assemble(energy), fd.assemble(Q), fd.assemble(Z), 0, 0]}
energy0 = fd.assemble(energy)

# Store initial conditions in functions to be used later on
un.assign(u0)
qsolver.solve()
F0.project(u0*h0)
file_sw.write(un, etan, qn)
Unp1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
nonlin_itcount = 0
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(t)
    t += dt
    tdump += dt

    # Solve for updated fields
    nsolver.solve()

    # Get the number of linear iterations
    its = nsolver.snes.getLinearSolveIterations()
    nonlin_its = nsolver.snes.getIterationNumber()

    # Compute and print quantities that should be conserved
    _mass = fd.assemble(mass)
    _energy = fd.assemble(energy)
    _Q = fd.assemble(Q)
    _Z = fd.assemble(Z)
    print("mass:", _mass)
    print("energy:", (energy0 - _energy)/energy0)
    print("abs vorticity:", _Q)
    print("enstrophy:", _Z)

    # Update field
    Un.assign(Unp1)

    simdata.update({t: [_mass, _energy, _Q, _Z, its, nonlin_its]})

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt

    itcount += its

with open(name+'.json', 'w') as f:
    json.dump(simdata, f)

# Write options to text file.
# allargs = parser.parse_args()
# options = vars(allargs)
with open(name+'_options.txt', 'w') as f:
    f.write('\n'.join(args))

PETSc.Sys.Print("Iterations", itcount,
                "dt", dt,
                "ref_level", args.ref_level,
                "dmax", args.dmax)