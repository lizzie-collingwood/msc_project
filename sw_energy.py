import firedrake as fd
#get command arguments
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for augmented Lagrangian solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 5.') # : change default to default=1 (deffo less than 3)
parser.add_argument('--dmax', type=float, default=15, help='Final time in days. Default 15.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--gamma', type=float, default=1.0e5, help='Augmented Lagrangian scaling parameter. Default 10000.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--filename', type=str, default='w5aug')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--kspschur', type=int, default=3, help='Number of KSP iterations on the Schur complement.')
parser.add_argument('--kspmg', type=int, default=3, help='Number of KSP iterations in the MG levels.')
parser.add_argument('--tlblock', type=str, default='mg', help='Solver for the velocity-velocity block. mg==Multigrid with patchPC, lu==direct solver with MUMPS, patch==just do a patch smoother. Default is mg')
parser.add_argument('--schurpc', type=str, default='mass', help='Preconditioner for the Schur complement. mass==mass inverse, helmholtz==helmholtz inverse * laplace * mass inverse. Default is mass')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = 6371220.
H = fd.Constant(5960.)
base_level = args.base_level
nrefs = args.ref_level - base_level
name = args.filename
deg = args.coords_degree
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}
#distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.FACET, 2)}
if args.tlblock == "mg":
    basemesh = fd.IcosahedralSphereMesh(radius=R0,
                                        refinement_level=base_level, degree=deg,
                                        distribution_parameters = distribution_parameters)
    mh = fd.MeshHierarchy(basemesh, nrefs)
    for mesh in mh:
        x = fd.SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    mesh = mh[-1]
else:
    mesh = fd.IcosahedralSphereMesh(radius=R0,
                                    refinement_level=args.ref_level, degree=deg,
                                    distribution_parameters = distribution_parameters)
    x = fd.SpatialCoordinate(mesh)
    mesh.init_cell_orientations(x)
R0 = fd.Constant(R0)
cx, cy, cz = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree # degree of FE space
V1 = fd.FunctionSpace(mesh, "BDM", degree+1) # can be BDM instead
V2 = fd.FunctionSpace(mesh, "DG", degree)
W = fd.MixedFunctionSpace((V1, V2, V1)) # TODO: velocity, depth, momentum

Omega = fd.Constant(7.292e-5)  # rotation rate
f = 2*Omega*cz/fd.Constant(R0)  # Coriolis parameter
g = fd.Constant(9.8)  # Gravitational constant
b = fd.Function(V2, name="Topography")
c = fd.sqrt(g*H)
gamma0 = args.gamma
gamma = fd.Constant(gamma0)

# D = eta + b

v, phi, w = fd.TestFunctions(W) # TODO:look at camassa holm example

dx = fd.dx

Un = fd.Function(W)
Unp1 = fd.Function(W)

u0, h0, F0 = fd.split(Un)
u1, h1, F1 = fd.split(Unp1)
uh = 0.5*(u0 + u1)
hh = 0.5*(h0 + h1)


def both(u):
    return 2*fd.avg(u)


K = 0.5*fd.inner(uh, uh)
dT = fd.Constant(0.)


eqn = (
    fd.inner(v, u1 - u0)*dx + dT*fd.inner(v, q1*perp(F1))*dx
    - dT*fd.div(v)*(g*(hh + b) + K)*dx
    + phi*(h1 - h0 + dT*fd.div(F1))*dx
    #+ fd.div(v)*gamma*(h1 - h0 + dT*fd.div(F1))*dx
    # + p*q1*hh*dx + fd.inner(perp(fd.grad(p)), uh)*dx - p*f*dx
    + fd.inner(w, F1 - hh*uh)*dx
    )


mass = h0*dx
energy = (h0*u0**2 + g*h0*(h0/2 - b))*dx
Q = hh*q1*dx # FIXME: how to compute initial value?
Z = hh*q1**2*dx


# U_t + N(U) = 0
# IMPLICIT MIDPOINT
# U^{n+1} - U^n + dt*N( (U^{n+1}+U^n)/2 ) = 0.

# TRAPEZOIDAL RULE
# U^{n+1} - U^n + dt*( N(U^{n+1}) + N(U^n) )/2 = 0.
    
# Newton's method
# f(x) = 0, f:R^M -> R^M
# [Df(x)]_{i,j} = df_i/dx_j
# x^0, x^1, ...
# Df(x^k).xp = -f(x^k)
# x^{k+1} = x^k + xp.


solver_dict = {
    'mat_type':'aij',
    'snes_monitor':None,
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solve_type':'mumps'
}

sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
    'ksp_monitor': None,
    "pc_type": "fieldsplit", # TODO: petsc manual about this
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
    "pc_fieldsplit_0_fields": '0,2,3',
    "pc_fieldsplit_1_fields": '1',
    "fieldsplit_0_ksp_type": 'gmres',
    "fieldsplit_0_ksp_max_it": 3,
    "fieldsplit_0_ksp_monitor": None,
    "fieldsplit_0_pc_type": 'python',
    'fieldsplit_0_pc_python_type': 'firedrake.PatchPC',
    "fieldsplit_0_patch_pc_patch_save_operators": True,
    "fieldsplit_0_patch_pc_patch_partition_of_unity": False,
    "fieldsplit_0_patch_pc_patch_sub_mat_type": "seqaij",
    "fieldsplit_0_patch_pc_patch_construct_type": "star",
    "fieldsplit_0_patch_pc_patch_multiplicative": False,
    "fieldsplit_0_patch_pc_patch_symmetrise_sweep": False,
    "fieldsplit_0_patch_pc_patch_construct_dim": 0,
    "fieldsplit_0_patch_sub_ksp_type": "preonly",
    "fieldsplit_0_patch_sub_pc_type": "lu",
    'fieldsplit_0_patch_sub_pc_factor_mat_solve_type':'mumps',
    "fieldsplit_1_ksp_type": 'preonly',
    "fieldsplit_1_pc_type": 'python',
    "fieldsplit_1_pc_python_type": 'firedrake.MassInvPC',
    "fieldsplit_1_Mp_pc_type": 'ilu',
}

sparameters_lu = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "fgmres",
    "ksp_view": None,
    "ksp_gmres_modifiedgramschmidt": None,
    'ksp_monitor': None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
    "pc_fieldsplit_0_fields": '0,2,3',
    "pc_fieldsplit_1_fields": '1',
    "fieldsplit_0_ksp_type": 'preonly',
    "fieldsplit_0_pc_type": 'python',
    "fieldsplit_0_pc_python_type": 'firedrake.AssembledPC',
    "fieldsplit_0_assembled_mat_type":"aij",
    "fieldsplit_0_assembled_pc_type":"lu",
    "fieldsplit_0_assembled_pc_factor_mat_solver_type":"mumps",
    "fieldsplit_1_ksp_type": 'gmres',
    "fieldsplit_1_ksp_max_it": 3,
    "fieldsplit_1_pc_type": 'python',
    "fieldsplit_1_pc_python_type": 'firedrake.MassInvPC',
    "fieldsplit_1_Mp_ksp_type": 'preonly',
    "fieldsplit_1_Mp_pc_type": 'lu',
}

sparameters = {
    "snes_monitor": None,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
    #"mg_levels_ksp_convergence_test": "skip",
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": True,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_construct_codim": 0,
    "mg_levels_patch_pc_patch_construct_type": "vanka",
    "mg_levels_patch_pc_patch_local_type": "additive",
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
    "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
}
 
dt = 60*60*args.dt
dT.assign(dt)
t = 0.

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
ctx = {"mu":g*dt/gamma/2}

#ctx = {}
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=solver_dict,
                                        appctx=ctx)
dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

x = fd.SpatialCoordinate(mesh)
u_0 = 20.0  # maximum amplitude of the zonal wind [m/s]
u_max = fd.Constant(u_0)
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

u0, h0, F0 = Un.split()
u0.assign(un)
h0.assign(etan + H - b)

# q = fd.TrialFunction(V0)
# p = fd.TestFunction(V0)

# qn = fd.Function(V0, name="Potential Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx - p*f*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = fd.File(name+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
# q0.assign(qn)
F0.project(u0*h0)
file_sw.write(un, etan, qn) # FIXME: what is this line doing?
Unp1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(t)
    t += dt
    tdump += dt

    nsolver.solve()

    print("mass:", fd.assemble(mass))
    print("energy:", fd.assemble(energy))
    print("abs vorticity:", fd.assemble(Q))
    print("enstrophy:", fd.assemble(Z))

    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("Iterations", itcount, "dt", dt, "tlblock", args.tlblock, "ref_level", args.ref_level, "dmax", args.dmax)
