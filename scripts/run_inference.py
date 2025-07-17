# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
from pathlib import Path
import tqdm

def inference():

  # Load the arguments
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_file_folder', default='/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-04-extract/left_rectified', type=str)
  parser.add_argument('--right_file_folder', default='/media/dbutterfield3/T75/Graco_Datasets/ground/ros2/ground-04-extract/right_rectified', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/GRaCo_ground/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default=f'{code_dir}/../outputs/GRaCo/ground_04', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=0.25, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=100, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=16, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  args = parser.parse_args()

  # Set logging, seed, and other parameters
  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  # Make the output directory
  os.makedirs(args.out_dir, exist_ok=True)
  os.makedirs(args.out_dir + "/depth_vis", exist_ok=True)
  os.makedirs(args.out_dir + "/depth", exist_ok=True)

  # Load checkpoint configuration
  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  # Create the modal and initialize with checkpoint weights
  model = FoundationStereo(args)
  ckpt = torch.load(ckpt_dir)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  # Set the model to the GPU and to evaluation mode
  model.cuda()
  model.eval()
  code_dir = os.path.dirname(os.path.realpath(__file__))

  # Load the scale parameter and make sure it is valid
  scale = args.scale
  assert scale<=1, "scale must be <=1"

  # Get paths to all images
  left_image_paths = sorted(list(Path(args.left_file_folder).glob('*.png')))
  right_image_paths = sorted(list(Path(args.right_file_folder).glob('*.png')))
   
  # Ensure each image is properly paired with its partner
  for i in range(len(left_image_paths)):
    assert left_image_paths[i].name == right_image_paths[i].name

  # For each image, run inference
  pbar = tqdm.tqdm(total=len(left_image_paths), desc="Predicting Depth...", unit=" images")
  for i in range(len(left_image_paths)):
    left_file = left_image_paths[i]
    right_file = right_image_paths[i]

    # Read in the two images
    img0 = imageio.imread(left_file)
    img1 = imageio.imread(right_file)

    # Resize the images based on the provided scale
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)

    # If loading mono images, fake extend into RGB, where R=G=B
    if len(img0.shape) == 2:
      assert len(img0.shape) == len(img1.shape)
      img0 = np.stack([img0, img0, img0], axis=-1)
      img1 = np.stack([img1, img1, img1], axis=-1)

    # Extract the Height and Width
    H,W = img0.shape[:2]
    img0_ori = img0.copy()
    logging.info(f"img0: {img0.shape}")

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.cuda.amp.autocast(True):
      if not args.hiera:
        disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
      else:
        disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H,W)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)

    # Write the visualized disparity
    imageio.imwrite(f'{args.out_dir}/depth_vis/' + left_file.name[:-4] + '.png', vis)
    logging.info(f"Output saved to {args.out_dir}")
    pbar.update()

    if args.remove_invisible:
      yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
      us_right = xx-disp
      invalid = us_right<0
      disp[invalid] = np.inf

    if args.get_pc:
      with open(args.intrinsic_file, 'r') as f:
        lines = f.readlines()
        K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
        baseline = float(lines[1])
      K[:2] *= scale

      # Calculate the depth and save it
      depth = K[0,0]*baseline/disp
      np.save(f'{args.out_dir}/depth/' + left_file.name[:-4] + '.npy', depth)

      # === Disable Cloud Calculations for Inference ===
      # xyz_map = depth2xyzmap(depth, K)
      # pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
      # keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
      # keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
      # pcd = pcd.select_by_index(keep_ids)
      # o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
      # logging.info(f"PCL saved to {args.out_dir}")

      # if args.denoise_cloud:
      #   logging.info("[Optional step] denoise point cloud...")
      #   cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      #   inlier_cloud = pcd.select_by_index(ind)
      #   o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
      #   pcd = inlier_cloud

      # === Disable Visualizaiton for inference ===
      # logging.info("Visualizing point cloud. Press ESC to exit.")
      # vis = o3d.visualization.Visualizer()
      # vis.create_window()
      # vis.add_geometry(pcd)
      # vis.get_render_option().point_size = 1.0
      # vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
      # vis.run()
      # vis.destroy_window()

if __name__=="__main__":
  inference()