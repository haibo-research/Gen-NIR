import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append('..')
sys.path.append('../Utils')
import argparse
import signal
from tqdm import tqdm
from Utils import *
import bempp
from bempp.api.linalg import gmres
from Utils.integrate import integrate_over_surface_gaussian_vectorized
import deep_sdf.workspace as ws

root = join(os.path.dirname(os.path.realpath(__file__)), '..')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="")
parser.add_argument("--optimization_dir", default=root + '/logs/opt_pancreas_CT_' + time.strftime('%m_%d_%H%M%S'))
parser.add_argument("--LogFrequency", default=1)
parser.add_argument("--iterations", default=2000)
parser.add_argument("--Nr", type=int, default=32)
parser.add_argument("--level", default=0.005)
parser.add_argument("--grid_r", default=2)
parser.add_argument("--grid_h", default=0.25)
parser.add_argument("--num_current", default=1)
parser.add_argument("--tol", default=1E-4)
parser.add_argument("--lr", default=0.005)
parser.add_argument("--z_i", type=int, default=1)
parser.add_argument("--z_t", type=int, default=1)
parser.add_argument("--delta", default=0)
parser.add_argument("--CodeRegularization", default=False)
parser.add_argument("--CodeRegularizationLambda", default=5e-1)
parser.add_argument(
    "--experiment",
    "-e",
    dest="experiment_directory",
    default='../pretrained/PANCREAS_CT',
    # default='../pretrained/ellipsoid',
    # default='../pretrained/spheres',
    # default='../pretrained/mask',
    help="The experiment directory which includes specifications and saved model "
         + "files to use for reconstruction",
)
parser.add_argument(
    "--checkpoint",
    "-c",
    dest="checkpoint",
    # default= str(100),
    default= 'latest',
    help="The checkpoint weights to use. This can be a number indicated an epoch "
         + "or 'latest' for the latest weights (this is the default)",
)

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
    specs_filename = os.path.join("E:\inv_EIT\pretrained\PANCREAS_CT", "specs.json")
    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(args.experiment_directory, ws.model_params_subdir, args.checkpoint + ".pth"), map_location=device
    )

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.to(device)

    latent_vectors = ws.load_pre_trained_latent_vectors(args.experiment_directory, args.checkpoint)
    latent_vectors = latent_vectors.to(device)

    print(" len of latent_vectors: ", len(latent_vectors))

    return decoder, latent_vectors

def initialize_grid_and_boundary(args, images_dir, label_dir, decoder, latent):
    decoder.eval()


    # for i in range(60):
    #     latent_init = latent[i]
    #
    #     verts_init, faces_init, normals_init = create_mesh_with_edge(decoder, latent_init.detach(), N=args.Nr,
    #                                                                  l=args.level)
    #     image_filename = os.path.join('/home/hbliu/Desktop/DeepSDF_EIT/pretrained/mask/reconstruction/latent/', str(i) + ".html")
    #     write_verts_faces_to_file(verts_init, faces_init, image_filename)



    # initialize and visualize initialization
    latent_init = latent[args.z_i]
    latent_init.requires_grad = True

    verts_init, faces_init, normals_init = create_mesh_with_edge(decoder, latent_init.detach(), N=args.Nr,
                                                                       l=args.level)
    image_filename = os.path.join(images_dir, "init.html")
    write_verts_faces_to_file(verts_init, faces_init, image_filename)
    torch.save(latent_init, label_dir + '/' + "init.pt")
    print('init grid: {} elements, {} vertices'.format(faces_init.shape[0], verts_init.shape[0]))

    # target stuff
    latent_target = latent[args.z_t]
    latent_target.requires_grad = False
    verts_target, faces_target, normals_target = create_mesh_with_edge(decoder, latent_target, N=args.Nr,
                                                                             l=args.level)
    image_filename = os.path.join(images_dir, "target.html")
    write_verts_faces_to_file(verts_target, faces_target, image_filename)
    torch.save(latent_target, label_dir + '/' + "target.pt")
    indicator_target = indicator_plane(decoder, latent_target, N=args.Nr, l=args.level)
    print('target grid: {} elements, {} vertices'.format(faces_target.shape[0], verts_target.shape[0]))

    # dirichlet boundary and neumann grid
    grid_target = bempp.api.Grid(verts_target.transpose(), faces_target.transpose())
    grid_obs = bempp.api.shapes.sphere(r=args.grid_r, h=args.grid_h)
    print('obs grid: {} elements, {} vertices'.format(grid_obs.elements.shape[1], grid_obs.vertices.shape[1]))

    return latent_init, grid_target, grid_obs, indicator_target

def obs_data(args, grid_target, grid_obs):
    vertices = np.concatenate((grid_target.vertices, grid_obs.vertices), axis=1)
    elements = np.concatenate((grid_target.elements, grid_obs.elements + grid_target.vertices.shape[1]), axis=1)
    domain_indices = np.ones(grid_target.elements.shape[1] + grid_obs.elements.shape[1])
    domain_indices[grid_target.elements.shape[1]:] = 2
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

    # potential operator
    slp_DD = bempp.api.operators.boundary.laplace.single_layer(
        neumann_space_int_segment,
        dirichlet_space_int_segment,
        neumann_space_int_segment)

    dlp_DN = bempp.api.operators.boundary.laplace.double_layer(
        dirichlet_space_out_segment,
        dirichlet_space_int_segment,
        neumann_space_int_segment)

    adlp_ND = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        neumann_space_int_segment,
        neumann_space_out_segment,
        dirichlet_space_out_segment)

    hyp_NN = bempp.api.operators.boundary.laplace.hypersingular(
        dirichlet_space_out_segment,
        neumann_space_out_segment,
        dirichlet_space_out_segment)

    slp_DN = bempp.api.operators.boundary.laplace.single_layer(
        neumann_space_out_segment,
        dirichlet_space_int_segment,
        neumann_space_int_segment)

    adlp_NN = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        neumann_space_out_segment,
        neumann_space_out_segment,
        dirichlet_space_out_segment)

    id_NN = bempp.api.operators.boundary.sparse.identity(
        neumann_space_out_segment,
        neumann_space_out_segment,
        dirichlet_space_out_segment)


    # block operator
    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = slp_DD
    blocked[0, 1] = -dlp_DN
    blocked[1, 0] = adlp_ND
    blocked[1, 1] = hyp_NN


    dirichlet_data_obs_element = np.zeros((args.num_current, grid_obs.elements.shape[1]))
    neumann_data_obs_element = np.zeros((args.num_current, grid_obs.elements.shape[1]))
    dirichlet_data_obs_vertices = np.zeros((args.num_current, grid_obs.vertices.shape[1]))
    for kk in range(args.num_current):
        # the functions of the Dirichlet and Neumann data and their discretisations on the corresponding segments.
        @bempp.api.real_callable
        def neumann_data(x, n, domain_index, result):
            if domain_index == 1:
                result[0] = 0
            if domain_index == 2:
                result[0] = np.cos(kk * 8 * np.pi * (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5)


        # RHS
        neumann_grid_fun = bempp.api.GridFunction(
            neumann_space_out_segment,
            fun=neumann_data)

        rhs_fun1 = - slp_DN * neumann_grid_fun
        rhs_fun2 = (.5 * id_NN - adlp_NN) * neumann_grid_fun

        # solving
        (neumann_fun_init, dirichlet_fun_obs), _ = \
            bempp.api.linalg.gmres(blocked, [rhs_fun1, rhs_fun2], tol=args.tol, return_residuals=False, return_iteration_count=False)


        # observing data
        dirichlet_data_obs_element[kk, :] = dirichlet_fun_obs.evaluate_on_element_centers().squeeze()[
                             grid_target.elements.shape[1]:]
        neumann_data_obs_element[kk, :] = neumann_grid_fun.evaluate_on_element_centers().squeeze()[
                             grid_target.elements.shape[1]:]
        dirichlet_data_obs_vertices[kk, :] = dirichlet_fun_obs.evaluate_on_vertices().squeeze()[
                                       grid_target.vertices.shape[1]:]

    normal = 2 * np.random.random((args.num_current, grid_obs.elements.shape[1])) - 1
    dirichlet_data_obs_element = dirichlet_data_obs_element + args.delta * normal * dirichlet_data_obs_element
    normal = 2 * np.random.random((args.num_current, grid_obs.vertices.shape[1])) - 1
    dirichlet_data_obs_vertices = dirichlet_data_obs_vertices + args.delta * normal * dirichlet_data_obs_vertices

    return neumann_data_obs_element, dirichlet_data_obs_element, dirichlet_data_obs_vertices

def Shape_Derivative(verts, faces, args, grid_obs, neumann_data_obs_element, dirichlet_data_obs_element, dirichlet_data_obs_vertices):
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


    # potential operator
    slp_DD = bempp.api.operators.boundary.laplace.single_layer(
        neumann_space_int_segment,
        dirichlet_space_int_segment,
        neumann_space_int_segment)

    dlp_DN = bempp.api.operators.boundary.laplace.double_layer(
        dirichlet_space_out_segment,
        dirichlet_space_int_segment,
        neumann_space_int_segment)

    adlp_ND = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        neumann_space_int_segment,
        neumann_space_out_segment,
        dirichlet_space_out_segment)

    hyp_NN = bempp.api.operators.boundary.laplace.hypersingular(
        dirichlet_space_out_segment,
        neumann_space_out_segment,
        dirichlet_space_out_segment)

    slp_DN = bempp.api.operators.boundary.laplace.single_layer(
        neumann_space_out_segment,
        dirichlet_space_int_segment,
        neumann_space_int_segment)

    adlp_NN = bempp.api.operators.boundary.laplace.adjoint_double_layer(
        neumann_space_out_segment,
        neumann_space_out_segment,
        dirichlet_space_out_segment)

    id_NN = bempp.api.operators.boundary.sparse.identity(
        neumann_space_out_segment,
        neumann_space_out_segment,
        dirichlet_space_out_segment)


    # block operator
    blocked = bempp.api.BlockedOperator(2, 2)
    blocked[0, 0] = slp_DD
    blocked[0, 1] = -dlp_DN
    blocked[1, 0] = adlp_ND
    blocked[1, 1] = hyp_NN


    shape_derivative = torch.zeros(verts.shape[0]).to(device)
    loss_vert = np.zeros(grid.vertices.shape[1])
    loss_ele = np.zeros(grid.elements.shape[1])
    for kk in range(args.num_current):
        '''
            state equation
        '''
        # RHS
        neumann_grid_fun = bempp.api.GridFunction(
            neumann_space_out_segment,
            coefficients=neumann_data_obs_element[kk, :])

        rhs_fun1 = - slp_DN * neumann_grid_fun
        rhs_fun2 = (.5 * id_NN - adlp_NN) * neumann_grid_fun

        # solving
        (state_neumann_init, state_dirichlet_obs), _ = \
            bempp.api.linalg.gmres(blocked, [rhs_fun1, rhs_fun2], tol=args.tol, return_residuals=False, return_iteration_count=False)


        # compute loss
        loss_vert[grid_init.vertices.shape[1]:] = \
            loss_vert[grid_init.vertices.shape[1]:] \
            + np.power(dirichlet_data_obs_vertices[kk,:]
                       - state_dirichlet_obs.evaluate_on_vertices().squeeze()[grid_init.vertices.shape[1]:], 2)

        loss_ele[grid_init.elements.shape[1]:] = \
            loss_ele[grid_init.elements.shape[1]:] \
            + np.power(dirichlet_data_obs_element[kk,:]
                       - state_dirichlet_obs.evaluate_on_element_centers().squeeze()[grid_init.elements.shape[1]:], 2)

        '''
            adjoint equation
        '''
        # RHS
        neumann_grid_fun = bempp.api.GridFunction(
            neumann_space_out_segment,
            coefficients= state_dirichlet_obs.evaluate_on_element_centers().squeeze()[grid_init.elements.shape[1]:]
                          - dirichlet_data_obs_element[kk, :])


        rhs_fun1 = - slp_DN * neumann_grid_fun
        rhs_fun2 = (.5 * id_NN - adlp_NN) * neumann_grid_fun


        # solving
        (adjoint_neumann_init, adjoint_dirichlet_obs), _ = \
            bempp.api.linalg.gmres(blocked, [rhs_fun1, rhs_fun2], tol=args.tol, return_residuals=False, return_iteration_count=False)


        # shape_derivative
        shape_derivative = shape_derivative + torch.from_numpy(
            state_neumann_init.evaluate_on_vertices().squeeze()[0:grid_init.vertices.shape[1]] *
            adjoint_neumann_init.evaluate_on_vertices().squeeze()[0:grid_init.vertices.shape[1]]).to(device)


    shape_derivative = shape_derivative / args.num_current

    # shape_derivative = smooth_function_torch(verts, faces, shape_derivative)

    loss_epoch = integrate_over_surface_gaussian_vectorized(grid.vertices, grid.elements, loss_vert, loss_ele)
    return shape_derivative, loss_epoch

def optimize(latent_init, args, decoder, grid_obs, grid_target,
             neumann_data_obs_element, dirichlet_data_obs_element, dirichlet_data_obs_vertices,
             indicator_target, epoch_log, optimization_meshes_dir, images_dir, label_dir):
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
        shape_derivative, loss_epoch = Shape_Derivative(verts, faces, args, grid_obs,
                                                        neumann_data_obs_element, dirichlet_data_obs_element,
                                                        dirichlet_data_obs_vertices)

        # loss Back-propagating to mesh vertices
        normals_upstream = torch.tensor(normals.astype(float), requires_grad=False, dtype=torch.float64,
                                        device=device)
        dL_dx_i = shape_derivative.unsqueeze(1) * normals_upstream

        # compute loss of far field pattern
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
    decoder, latent = load_model(args)

    # initialize_grid_and_boundary
    latent_init, grid_target, grid_obs, indicator_target = \
        initialize_grid_and_boundary(args, images_dir, label_dir, decoder, latent)

    # obs_data
    obs_dirichlet_vertices, obs_neumann_element, obs_neumann_vertices = obs_data(args, grid_target, grid_obs)

    # optimize
    optimize(latent_init, args, decoder, grid_obs, grid_target, obs_dirichlet_vertices, obs_neumann_element, obs_neumann_vertices, indicator_target,
             epoch_log, optimization_meshes_dir, images_dir, label_dir)

    print("Done.")

if __name__ == "__main__":

    main_function()


