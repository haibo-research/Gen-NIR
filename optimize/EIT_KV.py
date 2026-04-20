import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append('..')
sys.path.append('../Utils')
sys.path.append('../data')
sys.path.append('../data/SdfSamples')
import argparse
import signal
from tqdm import tqdm
from Utils import *
import bempp
from bempp.api.linalg import gmres
from networks.NET import NET
from Utils.integrate import integrate_over_surface



root = join(os.path.dirname(os.path.realpath(__file__)), '..')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="")
parser.add_argument("--sdf_log_dir", default=root + '/logs/DeepSDF_EIT')
parser.add_argument("--optimization_dir", default=root + '/logs/opt_spheres_' + time.strftime('%m_%d_%H%M%S'))
parser.add_argument("--LogFrequency", default=1)
parser.add_argument("--iterations", default=2000)
parser.add_argument("--Nr", type=int, default=32)
parser.add_argument("--level", default=0.05)
parser.add_argument("--grid_r", default=1.2)
parser.add_argument("--grid_h", default=0.3)
parser.add_argument("--num_current", default=1)
parser.add_argument("--tol", default=1E-4)
parser.add_argument("--lr", default=1e-2)
parser.add_argument("--delta", default=0)
parser.add_argument("--CodeRegularization", default=False)
parser.add_argument("--CodeRegularizationLambda", default=5e-1)



# 设置日志和目录
def setup_logging(args):
    """
    创建所需的目录并设置日志文件。
    """
    optimization_meshes_dir = args.optimization_dir
    os.makedirs(optimization_meshes_dir, exist_ok=True)
    images_dir = optimization_meshes_dir + '/intermediate'
    os.makedirs(images_dir, exist_ok=True)
    label_dir = optimization_meshes_dir + '/label'
    os.makedirs(label_dir, exist_ok=True)
    epoch_log = open(os.path.join(optimization_meshes_dir, 'optimization.csv'), 'w')
    print('epoch,loss,indicator error,hausdorff distance,volume difference', file=epoch_log, flush=True)
    with open(os.path.join(optimization_meshes_dir, 'args.json'), 'w', newline='\n') as f:
        json.dump(dict(vars(args)), f, indent=1)
    return optimization_meshes_dir, epoch_log, images_dir, label_dir




def load_model(args):
    # load decoder and latent
    decoder = NET(latent_size=15).to(device)
    saved_model_state = torch.load(args.sdf_log_dir + '/spheres.pth', map_location=torch.device('cpu'))
    decoder.load_state_dict(saved_model_state['weights'])

    # load decoder and latent
    decoder_total = NET(latent_size=15).to(device)
    saved_model_state_total = torch.load(args.sdf_log_dir + '/spheres.pth', map_location=torch.device('cpu'))
    decoder_total.load_state_dict(saved_model_state_total['weights'])

    return decoder, decoder_total




def initialize_grid_and_boundary(decoder, args, images_dir, label_dir, decoder_total):
    # initialize and visualize initialization
    spheres = []
    spheres.append(np.array([0.6, 0, 0, 0.1]))
    spheres.append(np.array([0, 0.6, 0, 0.1]))
    spheres.append(np.array([0, 0, 0.6, 0]))
    latent_init = np.array(spheres)
    # latent_init, f = generate_overlapping_spheres_within_bounds(n_spheres=3, max_radius=0.4)
    latent_init = torch.from_numpy(latent_init.reshape(-1)).double().to(device)
    latent_init.requires_grad = True
    verts_init, faces_init, normals_init = create_mesh_with_edge(decoder, latent_init.detach(), N=args.Nr, l=args.level)
    image_filename = os.path.join(images_dir, "init.html")
    write_verts_faces_to_file(verts_init, faces_init, image_filename)
    torch.save(latent_init, label_dir + '/' + "init.pt")
    print('init vert shape:', verts_init.shape, '\ninit faces shape:', faces_init.shape)

    # target stuff
    spheres = []
    spheres.append(np.array([0.1, 0, 0, 0.2]))
    spheres.append(np.array([0, 0.4, 0, 0]))
    spheres.append(np.array([0, 0, 0.7, 0.2]))
    latent_target = np.array(spheres)
    # latent_target, f = generate_overlapping_spheres_within_bounds(n_spheres=3, max_radius=0.4)
    latent_target = torch.from_numpy(latent_target.reshape(-1)).double().to(device)
    latent_target.requires_grad = False
    verts_target, faces_target, normals_target = create_mesh_with_edge(decoder_total, latent_target, N=args.Nr,
                                                                       l=args.level)
    image_filename = os.path.join(images_dir, "target.html")
    write_verts_faces_to_file(verts_target, faces_target, image_filename)
    torch.save(latent_target, label_dir + '/' + "target.pt")
    print('target vert shape:', verts_target.shape, '\ntarget faces shape:', faces_target.shape)
    indicator_target = indicator_plane(decoder_total, latent_target, N=args.Nr, l=args.level)


    # dirichlet boundary and neumann grid
    grid_target = bempp.api.Grid(verts_target.transpose(), faces_target.transpose())
    print('grid_target: {} elements, {} vertices'.format(grid_target.elements.shape[1], grid_target.vertices.shape[1]))
    grid_obs = bempp.api.shapes.sphere(r=args.grid_r, h=args.grid_h)
    print('grid_obs: {} elements, {} vertices'.format(grid_obs.elements.shape[1], grid_obs.vertices.shape[1]))


    return latent_init, grid_target, grid_obs, indicator_target




def obs_data(args, grid_target, grid_obs):
    vertices = np.concatenate((grid_target.vertices, grid_obs.vertices), axis=1)
    elements = np.concatenate((grid_target.elements, grid_obs.elements + grid_target.vertices.shape[1]), axis=1)
    domain_indices = np.ones(grid_target.elements.shape[1] + grid_obs.elements.shape[1])
    domain_indices[grid_target.elements.shape[1]:] = 2
    grid = bempp.api.Grid(vertices, elements, domain_indices=domain_indices)

    dp0_space = bempp.api.function_space(grid, "DP", 0)
    p1_space = bempp.api.function_space(grid, "P", 1)

    identity = bempp.api.operators.boundary.sparse.identity(
        p1_space, p1_space, dp0_space)
    dlp = bempp.api.operators.boundary.laplace.double_layer(
        p1_space, p1_space, dp0_space)
    slp = bempp.api.operators.boundary.laplace.single_layer(
        dp0_space, p1_space, dp0_space)


    obs_dirichlet_element = np.zeros((args.num_current, grid_obs.elements.shape[1]))
    obs_neumann_element = np.zeros((args.num_current, grid_obs.elements.shape[1]))
    obs_dirichlet_vertices = np.zeros((args.num_current, grid_obs.vertices.shape[1]))
    obs_neumann_vertices = np.zeros((args.num_current, grid_obs.vertices.shape[1]))
    for kk in range(args.num_current):

        # the functions of the Dirichlet and Neumann data and their discretisations on the corresponding segments.
        @bempp.api.real_callable
        def dirichlet_data(x, n, domain_index, result):
            if domain_index == 1:
                result[0] = 0
            if domain_index == 2:
                result[0] = np.cos(kk * np.pi * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5 )

        dirichlet_fun = bempp.api.GridFunction(p1_space, fun=dirichlet_data)

        rhs = (.5 * identity + dlp) * dirichlet_fun

        neumann_fun, info = bempp.api.linalg.cg(slp, rhs, tol=1E-3)

        # observing data
        obs_dirichlet_element[kk, :] = dirichlet_fun.evaluate_on_element_centers().squeeze()[
                             grid_target.elements.shape[1]:]
        obs_neumann_element[kk, :] = neumann_fun.evaluate_on_element_centers().squeeze()[
                            grid_target.elements.shape[1]:]

        obs_dirichlet_vertices[kk, :] = dirichlet_fun.evaluate_on_vertices().squeeze()[
                                       grid_target.vertices.shape[1]:]
        obs_neumann_vertices[kk, :] = neumann_fun.evaluate_on_vertices().squeeze()[
                                     grid_target.vertices.shape[1]:]

    return obs_dirichlet_vertices, obs_neumann_element, obs_neumann_vertices




def Shape_Derivative(verts, faces, args, grid_obs, obs_dirichlet_vertices, obs_neumann_element):
    grid_init = bempp.api.Grid(verts.transpose(), faces.transpose())
    vertices = np.concatenate((grid_init.vertices, grid_obs.vertices), axis=1)
    elements = np.concatenate((grid_init.elements, grid_obs.elements + grid_init.vertices.shape[1]), axis=1)
    domain_indices = np.ones(grid_init.elements.shape[1] + grid_obs.elements.shape[1])
    domain_indices[grid_init.elements.shape[1]:] = 2
    grid = bempp.api.Grid(vertices, elements, domain_indices=domain_indices)
    int_segments = [1]
    out_segments = [2]

    # boundary element space
    neumann_space_int_segment = bempp.api.function_space(
        grid, "DP", 0, segments=int_segments)
    neumann_space_out_segment = bempp.api.function_space(
        grid, "DP", 0, segments=out_segments)
    dirichlet_space_int_segment = bempp.api.function_space(
        grid, "P", 1, segments=int_segments, include_boundary_dofs=True,
        truncate_at_segment_edge=False)
    dirichlet_space_out_segment = bempp.api.function_space(
        grid, "P", 1, segments=out_segments)


    shape_derivative = torch.zeros(verts.shape[0]).to(device)
    for kk in range(args.num_current):
        '''
            adjoint equation
        '''
        # potential operator
        slp_II = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_int_segment,
            neumann_space_int_segment,
            neumann_space_int_segment)
        slp_OI = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_out_segment,
            neumann_space_int_segment,
            neumann_space_int_segment)
        slp_IO = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_int_segment,
            neumann_space_out_segment,
            neumann_space_out_segment)
        slp_OO = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_out_segment,
            neumann_space_out_segment,
            neumann_space_out_segment)
        dlp_OI = bempp.api.operators.boundary.laplace.double_layer(
            dirichlet_space_out_segment,
            neumann_space_int_segment,
            neumann_space_int_segment)
        id_OO = bempp.api.operators.boundary.sparse.identity(
            dirichlet_space_out_segment,
            neumann_space_out_segment,
            neumann_space_out_segment)
        dlp_OO = bempp.api.operators.boundary.laplace.double_layer(
            dirichlet_space_out_segment,
            neumann_space_out_segment,
            neumann_space_out_segment)

        # block operator
        blocked = bempp.api.BlockedOperator(2, 2)
        blocked[0, 0] = slp_II
        blocked[0, 1] = slp_OI
        blocked[1, 0] = slp_IO
        blocked[1, 1] = slp_OO

        # RHS
        dirichlet_grid_fun = bempp.api.GridFunction(
            dirichlet_space_out_segment,
            coefficients=obs_dirichlet_vertices[kk, :])

        rhs_fun1 = dlp_OI * dirichlet_grid_fun

        rhs_fun2 = (.5 * id_OO + dlp_OO) * dirichlet_grid_fun

        # solving
        (adjoint_neumann_init, adjoint_neumann_obs), _ = \
            bempp.api.linalg.gmres(blocked, [rhs_fun1, rhs_fun2], tol=args.tol)


        '''
            state equation
        '''
        # potential operator
        slp_II = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_int_segment,
            dirichlet_space_int_segment,
            neumann_space_int_segment)
        dlp_OI = bempp.api.operators.boundary.laplace.double_layer(
            dirichlet_space_out_segment,
            dirichlet_space_int_segment,
            neumann_space_int_segment)
        slp_IO = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_int_segment,
            neumann_space_out_segment,
            dirichlet_space_out_segment)
        id_OO = bempp.api.operators.boundary.sparse.identity(
            dirichlet_space_out_segment,
            neumann_space_out_segment,
            dirichlet_space_out_segment)
        dlp_OO = bempp.api.operators.boundary.laplace.double_layer(
            dirichlet_space_out_segment,
            neumann_space_out_segment,
            dirichlet_space_out_segment)
        slp_OI = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_out_segment,
            dirichlet_space_int_segment,
            neumann_space_int_segment)
        slp_OO = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_out_segment,
            neumann_space_out_segment,
            neumann_space_out_segment)

        # block operator
        blocked = bempp.api.BlockedOperator(2, 2)
        blocked[0, 0] = slp_II
        blocked[0, 1] = -dlp_OI
        blocked[1, 0] = -slp_IO
        blocked[1, 1] = 0.5 * id_OO + dlp_OO

        # RHS
        neumann_grid_fun = bempp.api.GridFunction(
            neumann_space_out_segment,
            coefficients=obs_neumann_element[kk, :])

        rhs_fun1 = - slp_OI * neumann_grid_fun
        rhs_fun2 = slp_OO * neumann_grid_fun

        # solving
        (state_neumann_init, state_dirichlet_obs), _ = \
            bempp.api.linalg.gmres(blocked, [rhs_fun1, rhs_fun2], tol=args.tol)

        # shape_derivative
        shape_derivative = shape_derivative + torch.from_numpy(
            np.power(state_neumann_init.evaluate_on_vertices().squeeze()[0:grid_init.vertices.shape[1]], 2) -
            np.power(adjoint_neumann_init.evaluate_on_vertices().squeeze()[0:grid_init.vertices.shape[1]], 2)
        ).to(device)

    shape_derivative = shape_derivative / args.num_current

    return shape_derivative




def compute_loss(args, verts, faces, grid_obs, obs_dirichlet_vertices, obs_neumann_element, obs_neumann_vertices):
    grid_init = bempp.api.Grid(verts.transpose(), faces.transpose())
    vertices = np.concatenate((grid_init.vertices, grid_obs.vertices), axis=1)
    elements = np.concatenate((grid_init.elements, grid_obs.elements + grid_init.vertices.shape[1]), axis=1)
    domain_indices = np.ones(grid_init.elements.shape[1] + grid_obs.elements.shape[1])
    domain_indices[grid_init.elements.shape[1]:] = 2
    grid = bempp.api.Grid(vertices, elements, domain_indices=domain_indices)
    int_segments = [1]
    out_segments = [2]



    loss_coe = np.zeros(grid.vertices.shape[1])
    for kk in range(args.num_current):

        dp0_space = bempp.api.function_space(grid, "DP", 0)
        p1_space = bempp.api.function_space(grid, "P", 1)

        identity = bempp.api.operators.boundary.sparse.identity(
            p1_space, p1_space, dp0_space)
        dlp = bempp.api.operators.boundary.laplace.double_layer(
            p1_space, p1_space, dp0_space)
        slp = bempp.api.operators.boundary.laplace.single_layer(
            dp0_space, p1_space, dp0_space)

        @bempp.api.real_callable
        def dirichlet_data(x, n, domain_index, result):
            if domain_index == 1:
                result[0] = 0
            if domain_index == 2:
                result[0] = np.cos(kk * np.pi * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5 )
        dirichlet_fun = bempp.api.GridFunction(p1_space, fun=dirichlet_data)


        rhs = (.5 * identity + dlp) * dirichlet_fun

        neumann_fun, info = bempp.api.linalg.cg(slp, rhs, tol=1E-3)

        # boundary element space
        neumann_space_int_segment = bempp.api.function_space(
            grid, "DP", 0, segments=int_segments)
        neumann_space_out_segment = bempp.api.function_space(
            grid, "DP", 0, segments=out_segments)
        dirichlet_space_int_segment = bempp.api.function_space(
            grid, "P", 1, segments=int_segments, include_boundary_dofs=True,
            truncate_at_segment_edge=False)
        dirichlet_space_out_segment = bempp.api.function_space(
            grid, "P", 1, segments=out_segments)
        dual_int_space = bempp.api.function_space(
            grid, "P", 1, segments=int_segments,
            include_boundary_dofs=True)

        slp_II = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_int_segment,
            dirichlet_space_int_segment,
            neumann_space_int_segment)

        dlp_OI = bempp.api.operators.boundary.laplace.double_layer(
            dirichlet_space_out_segment,
            dirichlet_space_int_segment,
            neumann_space_int_segment)

        adlp_IO = bempp.api.operators.boundary.laplace.adjoint_double_layer(
            neumann_space_int_segment,
            neumann_space_out_segment,
            dirichlet_space_out_segment)

        hyp_OO = bempp.api.operators.boundary.laplace.hypersingular(
            dirichlet_space_out_segment,
            neumann_space_out_segment,
            dirichlet_space_out_segment)

        slp_OI = bempp.api.operators.boundary.laplace.single_layer(
            neumann_space_out_segment,
            dirichlet_space_int_segment,
            neumann_space_int_segment)

        dlp_II = bempp.api.operators.boundary.laplace.double_layer(
            dirichlet_space_int_segment,
            dirichlet_space_int_segment,
            neumann_space_int_segment)

        id_II = bempp.api.operators.boundary.sparse.identity(
            dirichlet_space_int_segment,
            dirichlet_space_int_segment,
            neumann_space_int_segment)

        adlp_OO = bempp.api.operators.boundary.laplace.adjoint_double_layer(
            neumann_space_out_segment,
            neumann_space_out_segment,
            dirichlet_space_out_segment)

        id_OO = bempp.api.operators.boundary.sparse.identity(
            neumann_space_out_segment,
            neumann_space_out_segment,
            dirichlet_space_out_segment)

        hyp_IO = bempp.api.operators.boundary.laplace.hypersingular(
            dirichlet_space_int_segment,
            neumann_space_out_segment,
            dirichlet_space_out_segment)

        blocked = bempp.api.BlockedOperator(2, 2)

        blocked[0, 0] = slp_II
        blocked[0, 1] = -dlp_OI
        blocked[1, 0] = adlp_IO
        blocked[1, 1] = hyp_OO

        @bempp.api.real_callable
        def dirichlet_data(x, n, domain_index, res):
            res[0] = 0
        dirichlet_grid_fun = bempp.api.GridFunction(
            dirichlet_space_int_segment,
            fun=dirichlet_data,
            dual_space=dual_int_space)

        neumann_grid_fun = bempp.api.GridFunction(
            neumann_space_out_segment,
            coefficients=obs_neumann_element[kk, :],
            dual_space=dirichlet_space_out_segment)

        rhs_fun1 = (.5 * id_II + dlp_II) * dirichlet_grid_fun \
                   - slp_OI * neumann_grid_fun
        rhs_fun2 = - hyp_IO * dirichlet_grid_fun \
                   + (.5 * id_OO - adlp_OO) * neumann_grid_fun

        (neumann_solution, dirichlet_solution), _ = bempp.api.linalg.gmres(blocked, [rhs_fun1, rhs_fun2])

        loss_coe[grid_init.vertices.shape[1]:] = loss_coe[grid_init.vertices.shape[1]:] \
                + (obs_neumann_vertices[kk, :] - neumann_fun.evaluate_on_vertices().squeeze()[grid_init.vertices.shape[1]:]) \
                * (dirichlet_solution.evaluate_on_vertices().squeeze()[grid_init.vertices.shape[1]:] - obs_dirichlet_vertices[kk, :])


    loss_epoch = integrate_over_surface(grid.vertices, grid.elements, loss_coe)

    return loss_epoch




def optimize(latent_init, args, decoder, grid_obs, grid_target, obs_dirichlet_vertices, obs_neumann_element, obs_neumann_vertices, indicator_target,
             epoch_log, optimization_meshes_dir, images_dir, label_dir):
    # optimizer
    optimizer = torch.optim.Adam([latent_init], lr=args.lr)


    do_code_regularization = args.CodeRegularization
    code_reg_lambda = args.CodeRegularizationLambda

    print("Starting optimization:")
    training_loop = tqdm(range(args.iterations))
    for epoch in training_loop:

        optimizer.zero_grad()

        # first extract iso-surface
        verts, faces, normals = create_mesh_with_edge(decoder, latent_init.detach(), N=args.Nr, l=args.level)

        # shape derivative
        shape_derivative = Shape_Derivative(verts, faces, args, grid_obs, obs_dirichlet_vertices, obs_neumann_element)

        # loss Back-propagating to mesh vertices
        normals_upstream = torch.tensor(normals.astype(float), requires_grad=False, dtype=torch.float64,
                                        device=device)
        dL_dx_i = shape_derivative.unsqueeze(1) * normals_upstream

        # compute loss
        loss_epoch = compute_loss(args, verts, faces, grid_obs, obs_dirichlet_vertices, obs_neumann_element, obs_neumann_vertices)

        # compute indicator_error and hausdorff_distance
        indicator_init = indicator_plane(decoder, latent_init, N=args.Nr, l=args.level)
        indicator_error = torch.norm(indicator_target - indicator_init).squeeze().detach().cpu().numpy()
        hausdorff_distance = hausdorff(verts, faces, grid_target.vertices.T, grid_target.elements.T)
        volume_difference = volume_diff(verts, faces, grid_target.vertices.T, grid_target.elements.T)
        print('%d,%.5f,%.5f,%.5f,%.5f' % (epoch, loss_epoch, indicator_error, hausdorff_distance, volume_difference),
              file=epoch_log, flush=True)
        training_loop.set_description('%.4f' % (float(loss_epoch)))

        """
            mesh vertices Back-propagating to label
        """
        # first compute normals
        optimizer.zero_grad()
        verts_dr = torch.tensor(verts.astype(float), requires_grad=True, dtype=torch.float64, device=device)
        latent_inputs = latent_init.expand(verts_dr.shape[0], -1)
        pred_sdf = decoder(torch.cat([latent_inputs, verts_dr], 1).double())
        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph=True)
        # normalization to take into account for the fact sdf is not perfect...
        normals = verts_dr.grad / torch.norm(verts_dr.grad, 2, 1).unsqueeze(-1)
        # now assemble inflow derivative
        optimizer.zero_grad()
        dL_ds_i = -torch.matmul(dL_dx_i.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)
        # refer to Equation (4) in the main paper
        if do_code_regularization:
            loss_backward = torch.sum(dL_ds_i * pred_sdf) + code_reg_lambda * torch.norm(latent_init)
        else:
            loss_backward = torch.sum(dL_ds_i * pred_sdf)
        loss_backward.backward()
        # and update params
        optimizer.step()

        # log stuff
        if epoch % args.LogFrequency == 0:
            plot_error(optimization_meshes_dir)
            image_filename = images_dir + '/' + str(epoch) + '.html'
            field = torch.real(shape_derivative.unsqueeze(1))
            write_verts_faces_fields_to_file(verts, faces, field.cpu(), image_filename)
            torch.save(latent_init.detach(), label_dir + '/' + str(epoch) + ".pt")

    epoch_log.close()




def main_function():

    signal.signal(signal.SIGINT, signal_handler)

    # Parse input arguments
    args = parser.parse_args()

    # Create logging files/folders for losses
    optimization_meshes_dir, epoch_log, images_dir, label_dir = setup_logging(args)

    # load model and latent
    decoder, decoder_total = load_model(args)

    # initialize_grid_and_boundary
    latent_init, grid_target, grid_obs, indicator_target = \
        initialize_grid_and_boundary(decoder, args, images_dir, label_dir, decoder_total)

    # obs_data
    obs_dirichlet_vertices, obs_neumann_element, obs_neumann_vertices = obs_data(args, grid_target, grid_obs)

    # optimize
    optimize(latent_init, args, decoder, grid_obs,grid_target, obs_dirichlet_vertices, obs_neumann_element, obs_neumann_vertices, indicator_target,
             epoch_log, optimization_meshes_dir, images_dir, label_dir)

    print("Done.")




if __name__ == "__main__":

    main_function()


