from DrawingInterface import DrawingInterface

import pydiffvg
import torch
from torch.nn import functional as F
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import PIL.Image

from util import str2bool

from scipy.spatial.transform import Rotation as R

class Projector:
   def __init__(self, canvas_width, num_rows, num_cols):
       self.canvas_width = canvas_width 
       self.num_rows = num_rows
       self.num_cols = num_cols

       self.scale_factor = int(math.floor(math.sqrt(2) * canvas_width / (num_rows + num_cols)))
       self.center_point = np.array([
           canvas_width // 2, 
           canvas_width // 2, 
       ])

   def __call__(self, r:int, c:int, phi: int, theta: int):
       r_centered = r - self.num_rows // 2
       c_centered = c - self.num_cols // 2
       base_point = np.array([r_centered, c_centered, 0])

       rotation = R.from_euler('ZX', [phi, theta], degrees=True)
       rotated = rotation.apply(base_point) 
       rotated_xy = np.roll(rotated, 2)[:2]  # (x, y, z) -> (y, z, x) -> (y, z) 

       return rotated_xy * self.scale_factor + self.center_point


def npsin(x):
    return np.sin(x * np.pi / 180)

def npcos(x):
    return np.cos(x * np.pi / 180)

def get_point_base(r, c, attack, canvas_width, num_rows, num_cols):
    point = [
             canvas_width * npcos(attack) * (r + c) / (npcos(attack) * (num_rows + num_cols)), 
             canvas_width * npsin(attack) * (r - c) / (npcos(attack) * (num_rows + num_cols)) + \
                 canvas_width * npsin(attack) * (num_cols) / (npcos(attack) * (num_rows + num_cols))
    ]
    return point

def make_path(point_base, height):
    pass

def rect_from_corners(pp00, p1):
    x1, y1 = p0
    x2, y2 = p1
    pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return pts

# canonical interpolation function, like https://p5js.org/reference/#/p5/map
def map_number(n, start1, stop1, start2, stop2):
  return ((n-start1)/(stop1-start1))*(stop2-start2)+start2;

def diamond_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    n = 1
    hyA = map_number(-2, -n, n, y1, y2)
    hyB = map_number( 2, -n, n, y1, y2)
    hyH = map_number(     0, -n, n, y1, y2)
    hxH = map_number(     0, -n, n, x1, x2)
    pts = [[hxH, hyA], [x1, hyH], [hxH, hyB], [x2, hyH]]
    return pts

def tri_from_corners(p0, p1, is_up):
    x1, y1 = p0
    x2, y2 = p1
    n = 1
    hxA = map_number( 2, -n, n, x1, x2)
    hxB = map_number(-2, -n, n, x1, x2)
    hxH = map_number(     0, -n, n, x1, x2)
    if is_up:
        pts = [[hxH, y1], [hxB, y2], [hxA, y2]]
    else:
        pts = [[hxH, y2], [hxA, y1], [hxB, y1]]
    return pts

def hex_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    n  = 3
    hyA = map_number(4, -n, n, y1, y2)
    hyB = map_number(2, -n, n, y1, y2)
    hyC = map_number(-2, -n, n, y1, y2)
    hyD = map_number(-4, -n, n, y1, y2)
    hxH = map_number(0, -n, n, x1, x2)
    pts = [[hxH, hyA], [x1, hyB], [x1, hyC], [hxH, hyD], [x2, hyC], [x2, hyB]]
    return pts

def knit_from_corners(p0, p1):
    x1, y1 = p0
    x2, y2 = p1
    xm = (x1 + x2) / 2.0
    lean_up = 0.45
    slump_down = 0.30
    fall_back = 0.2
    y_up1 = map_number(lean_up, 0, 1, y2, y1)
    y_up2 = map_number(1 + lean_up, 0, 1, y2, y1)
    y_down1 = map_number(slump_down, 0, 1, y1, y2)
    y_down2 = map_number(1 + slump_down, 0, 1, y1, y2)
    x_fall_back1 = map_number(fall_back, 0, 1, x2, xm)
    x_fall_back2 = map_number(fall_back, 0, 1, x1, xm)

    pts = []
    # center bottom
    pts.append([xm, y_down2])
    # vertical line on right side
    pts.extend([[x2, y_up1], [x2, y_up2]])
    # horizontal line back
    pts.append([x_fall_back1, y_up2])
    # center top
    pts.append([xm, y_down1])
    # back up to top
    pts.append([x_fall_back2, y_up2])
    # vertical line on left side
    pts.extend([[x1, y_up2], [x1, y_up1]])
    return pts

shift_pixel_types = ["hex", "rectshift", "diamond"]

class PixelDrawer(DrawingInterface):
    VERTICAL_BRICK = torch.tensor([[0., 0.], [0., 1.]], requires_grad=False)

    @staticmethod
    def add_settings(parser):
        parser.add_argument("--pixel_size", nargs=2, type=int, help="Pixel size (width height)", default=None, dest='pixel_size')
        parser.add_argument("--pixel_scale", type=float, help="Pixel scale", default=None, dest='pixel_scale')
        parser.add_argument("--pixel_type", type=str, help="rect, rectshift, hex, tri, diamond, knit", default="rect", dest='pixel_type')
        parser.add_argument("--pixel_edge_check", type=str2bool, help="ensure grid is symmetric", default=True, dest='pixel_edge_check')
        parser.add_argument("--pixel_iso_check", type=str2bool, help="ensure tri and hex shapes are w/h scaled", default=True, dest='pixel_iso_check')
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()

        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]

        # current logic: assume 16x9, or 4x5, but check for 1x1 (all others must be provided explicitly)
        # TODO: could compute this based on output size instead?
        if settings.pixel_size is not None:
            self.num_cols, self.num_rows = settings.pixel_size
        elif self.canvas_width == self.canvas_height:
            self.num_cols, self.num_rows = [80, 80]
            # self.num_cols, self.num_rows = [40, 40]
        elif self.canvas_width < self.canvas_height:
            self.num_cols, self.num_rows = [40, 50]
        else:
            self.num_cols, self.num_rows = [80, 45]

        self.pixel_type = settings.pixel_type

        if settings.pixel_iso_check and settings.pixel_size is None:
            if self.pixel_type == "tri":
                # auto-adjust triangles to be wider if not specified
                self.num_cols = int(1.414 * self.num_cols)
            elif self.pixel_type == "hex":
                # auto-adjust hexes to be narrower if not specified
                self.num_rows = int(1.414 * self.num_rows)
            elif self.pixel_type == "diamond":
                self.num_rows = int(2 * self.num_rows)
 
        # we can also "scale" pixels -- scaling "up" meaning fewer rows/cols, etc.
        if settings.pixel_scale is not None and settings.pixel_scale > 0:
            self.num_cols = int(self.num_cols / settings.pixel_scale)
            self.num_rows = int(self.num_rows / settings.pixel_scale)


        shrink = False
        if self.num_cols>self.canvas_width:
            shrink = True
            self.num_cols = self.canvas_width
        if self.num_rows>self.canvas_height:
            shrink = True
            self.num_rows = self.canvas_height
        if shrink:
            print('pixel grid size should not be larger than output pixel size: reducing pixel grid')

        print(f"Running pixeldrawer with {self.num_cols}x{self.num_rows} grid")

        if settings.pixel_edge_check:
            if self.pixel_type in shift_pixel_types:
                if self.num_cols % 2 == 0:
                    self.num_cols = self.num_cols + 1
                if self.num_rows % 2 == 0:
                    self.num_rows = self.num_rows + 1
            elif self.pixel_type == "tri":
                if self.num_cols % 2 == 0:
                    self.num_cols = self.num_cols + 1
                if self.num_rows % 2 == 1:
                    self.num_rows = self.num_rows + 1

    def load_model(self, settings, device):
        # gamma = 1.0

        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        pydiffvg.set_device(device)
        self.device = device

    def get_opts(self):
        return self.opts

    def rand_init(self, toksX, toksY):
        self.init_from_tensor(None)

    def init_from_tensor(self, init_tensor):
        # print("----> SHAPE", self.num_rows, self.num_cols)
        canvas_width, canvas_height = self.canvas_width, self.canvas_height
        num_rows, num_cols = self.num_rows, self.num_cols
        cell_width = canvas_width / num_cols
        cell_height = canvas_height / num_rows

        self.projector = Projector(canvas_width, num_rows, num_cols) 

        tensor_cell_height = 0
        tensor_cell_width = 0
        max_tensor_num_subsamples = 4
        tensor_subsamples_x = []
        tensor_subsamples_y = []
        if init_tensor is not None:
            tensor_shape = init_tensor.shape
            tensor_cell_width = tensor_shape[3] / num_cols
            tensor_cell_height = tensor_shape[2] / num_rows
            if int(tensor_cell_width) < max_tensor_num_subsamples:
                tensor_subsamples_x = [i for i in range(int(tensor_cell_width))]
            else:
                step_size_x = tensor_cell_width / max_tensor_num_subsamples
                tensor_subsamples_x = [int(i*step_size_x) for i in range(max_tensor_num_subsamples)]
            if int(tensor_cell_height) < max_tensor_num_subsamples:
                tensor_subsamples_y = [i for i in range(int(tensor_cell_height))]
            else:
                step_size_y = tensor_cell_height / max_tensor_num_subsamples
                tensor_subsamples_y = [int(i*step_size_y) for i in range(max_tensor_num_subsamples)]

            # print(tensor_shape, tensor_cell_width, tensor_cell_height,tensor_subsamples_x,tensor_subsamples_y)

        # Initialize Random Pixels
        color_vars = [] # common
        points_vars = [] # common

        pts_bases = []
#        pts_bases_30 = []
#        pts_bases_45 = []
#        pts_bases_30r90 = []
        many = 3
        many_pts_base_map = [list() for _ in range(many)]
        many_shapes = [list() for _ in range(many)]
        many_shape_groups = [list() for _ in range(many)]

        shapes = []
#        shapes_30 = []
#        shapes_45 = []
#        shapes_30r90 = []

        shape_groups = []
#        shape_groups_30 = []
#        shape_groups_45 = []
#        shape_groups_30r90 = []

        # colors = []
        scaled_init_tensor = (init_tensor[0] + 1.0) / 2.0
        for r in range(num_rows):
            tensor_cur_y = int(r * tensor_cell_height)
            cur_y = r * cell_height
            num_cols_this_row = num_cols
            col_offset = 0
            if self.pixel_type in shift_pixel_types and r % 2 == 0:
                num_cols_this_row = num_cols - 1
                col_offset = 0.5
            for c in range(num_cols_this_row):
                tensor_cur_x =  (col_offset + c) * tensor_cell_width
                cur_x = (col_offset + c) * cell_width
                if init_tensor is None:
                    cell_color = torch.tensor([random.random(), random.random(), random.random(), 1.0])
                else:
                    try:
                        rgb_sum = [0, 0, 0]
                        rgb_count = 0
                        for t_x in tensor_subsamples_x:
                            cur_subsample_x = tensor_cur_x + t_x
                            for t_y in tensor_subsamples_y:
                                cur_subsample_y = tensor_cur_y + t_y
                                if(cur_subsample_x < tensor_shape[3] and cur_subsample_y < tensor_shape[2]):
                                    rgb_count += 1
                                    rgb_sum[0] += scaled_init_tensor[0][int(cur_subsample_y)][int(cur_subsample_x)]
                                    rgb_sum[1] += scaled_init_tensor[1][int(cur_subsample_y)][int(cur_subsample_x)]
                                    rgb_sum[2] += scaled_init_tensor[2][int(cur_subsample_y)][int(cur_subsample_x)]
                                else:
                                    print(f"Ignoring out of bounds entry: {cur_subsample_x},{cur_subsample_y}")
                        if rgb_count == 0:
                            print("init cell count is 0, this shouldn't happen. but it did?")
                            rgb_count = 1
                        cell_color = torch.tensor([rgb_sum[0]/rgb_count, rgb_sum[1]/rgb_count, rgb_sum[2]/rgb_count, 1.0])
                    except BaseException as error:
                        print("WTF", error)
                        mono_color = random.random()
                        cell_color = torch.tensor([mono_color, mono_color, mono_color, 1.0])
                # colors.append(cell_color)

#                p30 = get_point_base(r, c, 30, canvas_width, num_rows, num_cols)
#                p45 = get_point_base(r, c, 45, canvas_width, num_rows, num_cols)
#
#                p30r90 = get_point_base(c, r, 30, canvas_width, num_rows, num_cols)
#                many_points = [p30, p45, p30r90]

                angles = [(75, -5), (75, -10), (90, -5)]
                many_points = [self.projector(r, c, phi, theta) for phi, theta in angles]


#                pts_base_30 = torch.tensor([p30, p30], dtype=torch.float32, requires_grad=False)
#                pts_base_45 = torch.tensor([p45, p45], dtype=torch.float32, requires_grad=False)
#                pts_base_30r90 = torch.tensor([p30r90, p30r90], dtype=torch.float32, requires_grad=False)

                many_pts_base = [
                    torch.tensor([point, point], dtype=torch.float32, requires_grad=False)
                    for point in many_points
                ]

                for base_map, a_pts_base in zip(many_pts_base_map, many_pts_base):
                    base_map.append(a_pts_base)
                    

# [pts_base_30, pts_base_45, pts_base_30r90]

                height_tensor = torch.tensor(cell_height, dtype=torch.float32, requires_grad=True)

		# testing
                pts_base = many_pts_base[1]
                pts = pts_base - torch.abs(height_tensor) * self.VERTICAL_BRICK

#                pts_30 = pts_base_30 - torch.abs(height_tensor) * self.VERTICAL_BRICK
#                pts_45 = pts_base_45 - torch.abs(height_tensor) * self.VERTICAL_BRICK
#                pts_30r90 = pts_base_30r90 - torch.abs(height_tensor) * self.VERTICAL_BRICK
#                
                many_pts = [
                    a_pts_base - torch.abs(height_tensor) * self.VERTICAL_BRICK
                    for a_pts_base in many_pts_base
#                    pts_base_30 - torch.abs(height_tensor) * self.VERTICAL_BRICK,
#                    pts_base_45 - torch.abs(height_tensor) * self.VERTICAL_BRICK,
#                    pts_base_30r90 - torch.abs(height_tensor) * self.VERTICAL_BRICK,
                ]

                points_vars.append(height_tensor)

                path = pydiffvg.Polygon(pts, False, stroke_width = torch.tensor(2)) # TODO: remove

                many_paths = [
                    pydiffvg.Polygon(a_pts, False, stroke_width = torch.tensor(2))
                    for a_pts in many_pts
                ]

                pts_bases.append(pts_base)  # TODO: remove

                shapes.append(path)  # TODO: remove

                for a_shapes, a_paths in zip(many_shapes, many_paths):
                    a_shapes.append(a_paths)

                path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), stroke_color = cell_color, fill_color = None)
                many_path_group = [
                    pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(a_shapes) - 1]), stroke_color = cell_color, fill_color = None)
                    for a_shapes in many_shapes
                ]

                shape_groups.append(path_group) # TODO: remove

#                many_shape_groups = [
#                    shape_groups_30,
#                    shape_groups_45,
#                    shape_groups_30r90,
#                ]
                for a_shape_groups, path_group in zip(many_shape_groups, many_path_group):
                    a_shape_groups.append(path_group)
        # exit()
        # Just some diffvg setup

#        scene_args = pydiffvg.RenderFunction.serialize_scene(\
#            canvas_width, canvas_height, shapes, shape_groups)

        render = pydiffvg.RenderFunction.apply

        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, many_shapes[0], many_shape_groups[0])
#        scene_args_45 = pydiffvg.RenderFunction.serialize_scene(\
#            canvas_width, canvas_height, shapes_45, shape_groups_45)
#        scene_args_30r90 = pydiffvg.RenderFunction.serialize_scene(\
#            canvas_width, canvas_height, shapes_45, shape_groups_30r90)
#

        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
#        img_30 = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args_30)
#        img_45 = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args_45)
#        img_30r90 = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args_30r90)
#
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

        self.color_vars = color_vars
        self.points_vars = points_vars
        self.img = img

   #     self.pts_bases = pts_bases
        self.pre_voxels = many_pts_base_map
#        self.pre_voxels = [
#            pts_bases_30,
#            pts_bases_45,
#            pts_bases_30r90,
#        ]
   
#        self.pts_bases_30 = pts_bases_30
#        self.pts_bases_45 = pts_bases_45
#        self.pts_bases_30r90 = pts_bases_30r90

        self.shapes = shapes  # TODO: remove
#        self.shapes_30 = shapes_30 
#        self.shapes_45 = shapes_45 
#        self.shapes_30r90 = shapes_30r90 

        self.many_shapes = many_shapes 
#        [
#            shapes_30,
#            shapes_45,
#            shapes_30r90,
#        ]

        self.shape_groups = shape_groups # TODO: remove

        self.many_shape_groups = many_shape_groups
#        [
#            shape_groups_30,
#            shape_groups_45,
#            shape_groups_30r90,
#        ]
#        self.shape_groups_30 = shape_groups_30
#        self.shape_groups_45 = shape_groups_45
#        self.shape_groups_30r90 = shape_groups_30r90

    def get_opts(self, decay_divisor=1):
        # Optimizers
        points_optim = torch.optim.Adam(self.points_vars, lr=1.0)
        # width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1)
        color_optim = torch.optim.Adam(self.color_vars, lr=0.03/decay_divisor)
        # self.opts = [points_optim] #, color_optim]
        self.opts = [points_optim, color_optim]
        return self.opts

    def reapply_from_tensor(self, new_tensor):
        self.init_from_tensor(new_tensor)

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        return None

    def synth(self, cur_iteration):
        if cur_iteration < 0:
            return self.img

        render = pydiffvg.RenderFunction.apply

#### re-assign
        pts_bases = self.pre_voxels[(cur_iteration % len(self.pre_voxels))]
        shapes = self.many_shapes[(cur_iteration % len(self.many_shapes))]
        shape_groups = self.many_shape_groups[(cur_iteration % len(self.many_shape_groups))]

#        if cur_iteration % 3 == 0: # 45, 30r90, 30, 45, 30r90, 30 ....
#            pts_bases = self.pts_bases_30
#            shapes = self.shapes_30
#            shape_groups = self.shape_groups_30
#        elif cur_iteration % 3 == 1:
#            pts_bases = self.pts_bases_30r90
#            shapes = self.shapes_30r90
#            shape_groups = self.shape_groups_30r90
#        else:
#            pts_bases = self.pts_bases_45
#            shapes = self.shapes_45
#            shape_groups = self.shape_groups_45
  

        for pts_base, height_tensor, path in zip(pts_bases, self.points_vars, shapes):
            pts = pts_base - torch.abs(height_tensor) * self.VERTICAL_BRICK
            path.points = pts
####
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, shapes, shape_groups)
        img = render(self.canvas_width, self.canvas_height, 2, 2, cur_iteration, None, *scene_args)
        img_h, img_w = img.shape[0], img.shape[1]
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = self.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        # if cur_iteration == 0:
        #     print("SHAPE", img.shape)

        self.img = img
        return img

    @torch.no_grad()
    def to_image(self):
        img = self.img.detach().cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = np.uint8(img * 255)
        pimg = PIL.Image.fromarray(img, mode="RGB")
        return pimg

    def clip_z(self):
        with torch.no_grad():
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0.0, 1.0)
                group.stroke_color.data[3].clamp_(1.0, 1.0)
        pass

    def get_z(self):
        return None

    def get_z_copy(self):
        shape_groups_copy = []
        # for group in self.shape_groups:
        #     group_copy = torch.clone(group.stroke_color.data)
        #     shape_groups_copy.append(group_copy)
        return shape_groups_copy

    def set_z(self, new_z):
        # l = len(new_z)
        # for l in range(len(new_z)):
        #     active_group = self.shape_groups[l]
        #     new_group = new_z[l]
        #     active_group.stroke_color.data.copy_(new_group)
        return None
