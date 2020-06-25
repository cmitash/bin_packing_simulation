import cv2
from PIL import Image, ImageDraw
import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import OpenEXR, Imath
import open3d as o3d
import matplotlib.pyplot as plt
from mathutils import Vector, Matrix, Quaternion

class Label:
    def draw_bboxes(self, syn_images_folder, num_of_images, frame_number):
        dataset_file = "rendered_images/dataset_info.txt"
        obj_classes = ['dove', 'toothpaste']
        with open(dataset_file, "r") as dFile:
            scenes = dFile.readlines()
            for scene in scenes:
                scene_num, num_instances = scene.split(',')
                scene_num = int(scene_num)
                num_instances = int(num_instances)

                print ('scene_num: ', scene_num)
                img_filepath = osp.join(syn_images_folder, 'image_%05d/rgb/image.png' % scene_num)
                im = Image.open(img_filepath)
                draw = ImageDraw.Draw(im)

                class_filepath = osp.join(syn_images_folder, 'debug/class_id_%05d.txt' % scene_num)
                class_list = []
                with open(class_filepath, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        class_list.append(int(line))
                file.close()

                for instance_num in range(0, num_instances):
                    if class_list[instance_num] == -1:
                        print ('Invalid instance::not present in scene\n')
                        continue

                    mask_img_filepath = osp.join(syn_images_folder, 'debug/image_%05d_%02d_%04d.png' % (scene_num, instance_num, frame_number))
                    mask_image = cv2.imread(mask_img_filepath, cv2.IMREAD_GRAYSCALE)
                    x1, y1, w1, h1 = cv2.boundingRect(mask_image)
                    
                    box_filepath = osp.join(syn_images_folder, 'image_%05d/labels/bbox.txt' % scene_num)
                    box_file = open(box_filepath,'a')
                    box_file.write("%s %i %i %i %i\n" % (obj_classes[int(class_list[instance_num])], int(x1), int(y1), int(x1)+int(w1), int(y1)+int(h1)))
                    draw.rectangle([(x1, y1), (x1 + w1, y1 + h1)], outline=(255,0,0,255))
                    box_file.close()

                    os.remove(mask_img_filepath)  

                # save debug image showing bounding box
                del draw
                new_img_filepath = osp.join(syn_images_folder, 'debug/dbg_img_%05d.png' % scene_num)
                im.save(new_img_filepath)

    def save_pointcloud_data(self, syn_images_folder, scene_num, frame_number):
        print ('scene_num: ', scene_num)
        rgb_img_filepath = osp.join(syn_images_folder, 'image_%05d/rgb/image.png' % scene_num)
        depth_img_filepath = osp.join(syn_images_folder, 'image_%05d/depth/image_%04d.exr' % (scene_num, frame_number))
        des_depth_path = osp.join(syn_images_folder, 'image_%05d/depth/image.png' % scene_num)
        des_pcl_path = osp.join(syn_images_folder, 'image_%05d/depth/pointcloud.pcd' % scene_num)
        des_pcl_path_viz = osp.join(syn_images_folder, 'image_%05d/depth/pointcloud.ply' % scene_num)

        # Extracting depth image from EXR file format 
        depth_image_raw = OpenEXR.InputFile(depth_img_filepath)

        point_type = Imath.PixelType(Imath.PixelType.FLOAT)
        depthstr = depth_image_raw.channel('R', point_type)
        depth = np.fromstring(depthstr, dtype = np.float32)

        dw = depth_image_raw.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        depth.shape = (size[1], size[0]) # np arrays are (row, col)
        depth[depth < 0] = 0
        depth = depth*1000
        depth = depth.astype(np.uint16)
        cv2.imwrite(des_depth_path, depth)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        pinhole_camera_intrinsic.set_intrinsics(640, 480, 615.95776367187, 615.95776367187, 320, 240)

        source_color = o3d.io.read_image(rgb_img_filepath)
        source_depth = o3d.io.read_image(des_depth_path)

        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth, 1000, 3, False)
        source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)

        cam_matrix = Matrix(((0.9657,  0.1838, -0.1833,  0.3500),
        (0.2596, -0.6834,  0.6822, -0.6000),
        (0.0001, -0.7064, -0.7077,  0.6000),
        (0.0000,  0.0000,  0.0000,  1.0000)))
        source_pcd.transform(cam_matrix)
        o3d.io.write_point_cloud(des_pcl_path, source_pcd, True)
        o3d.io.write_point_cloud(des_pcl_path_viz, source_pcd, True)

    def save_pointcloud_data_dir(self, dir_inout, frame_number):
        rgb_img_filepath = osp.join(dir_inout, 'rgb/image.png')
        depth_img_filepath = osp.join(dir_inout, 'depth/image_%04d.exr' % frame_number)
        des_depth_path = osp.join(dir_inout, 'depth/image.png')
        des_pcl_path = osp.join(dir_inout, 'depth/pointcloud.pcd')
        des_pcl_path_viz = osp.join(dir_inout, 'depth/pointcloud.ply')

        # Extracting depth image from EXR file format 
        depth_image_raw = OpenEXR.InputFile(depth_img_filepath)

        point_type = Imath.PixelType(Imath.PixelType.FLOAT)
        depthstr = depth_image_raw.channel('R', point_type)
        depth = np.fromstring(depthstr, dtype = np.float32)

        dw = depth_image_raw.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        depth.shape = (size[1], size[0]) # np arrays are (row, col)
        depth[depth < 0] = 0
        depth = depth*1000
        depth = depth.astype(np.uint16)
        cv2.imwrite(des_depth_path, depth)

        pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        pinhole_camera_intrinsic.set_intrinsics(640, 480, 615.95776367187, 615.95776367187, 320, 240)

        source_color = o3d.io.read_image(rgb_img_filepath)
        source_depth = o3d.io.read_image(des_depth_path)

        source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth, 1000, 3, False)
        source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)

        cam_matrix = Matrix(((0.9657,  0.1838, -0.1833,  0.3500),
        (0.2596, -0.6834,  0.6822, -0.6000),
        (0.0001, -0.7064, -0.7077,  0.6000),
        (0.0000,  0.0000,  0.0000,  1.0000)))
        source_pcd.transform(cam_matrix)
        o3d.io.write_point_cloud(des_pcl_path, source_pcd, True)
        o3d.io.write_point_cloud(des_pcl_path_viz, source_pcd, True)

    def get_segmentation_labels(self, syn_images_folder, num_of_images, frame_number):
        dataset_file = "rendered_images/dataset_info.txt"
        with open(dataset_file, "r") as dFile:
            scenes = dFile.readlines()
            for scene in scenes:
                scene_num, num_instances = scene.split(',')
                scene_num = int(scene_num)
                num_instances = int(num_instances)

                print ('scene_num: ', scene_num)
                img_filepath = osp.join(syn_images_folder, 'image_%05d/rgb/image.png' % scene_num)
                im = cv2.imread(img_filepath)

                class_filepath = osp.join(syn_images_folder, 'debug/class_id_%05d.txt' % scene_num)
                class_list = []
                with open(class_filepath, "r") as file:
                    lines = file.readlines()
                    for line in lines:
                        class_list.append(int(line))
                file.close()
                height, width, channels = im.shape
                seg_img = np.zeros((height,width,1), np.uint8)
                edge_img = np.zeros((height,width), np.uint8)

                for instance_num in range(0, num_instances):
                    if class_list[instance_num] == -1:
                        print ('Invalid instance::not present in scene\n')
                        continue

                    mask_img_filepath = osp.join(syn_images_folder, 'debug/image_%05d_%02d_%04d.png' % (scene_num, instance_num, frame_number))
                    mask_image = cv2.imread(mask_img_filepath, cv2.IMREAD_GRAYSCALE)
                    active_indices = np.nonzero(mask_image)

                    # semantic label
                    seg_img[active_indices] = np.uint8(class_list[instance_num] + 1)

                    # instance label
                    ins_seg_img = np.zeros((height,width,1), np.uint8)
                    ins_seg_img[active_indices] = np.uint8(class_list[instance_num] + 1)
                    
                    # boundary label
                    edgex = cv2.Sobel(ins_seg_img, cv2.CV_64F,1,0,ksize=1)
                    edgey = cv2.Sobel(ins_seg_img, cv2.CV_64F,0,1,ksize=1)
                    edge = np.hypot(edgex, edgey)
                    edge *= 255.0/np.max(edge)
                    edge = np.uint8(edge)
                    edge_img = cv2.bitwise_or(edge_img, edge)    

                    os.remove(mask_img_filepath)           

                # save segmentation image
                seg_img_filepath = osp.join(syn_images_folder, 'image_%05d/labels/seg_img.png' % scene_num)
                cv2.imwrite(seg_img_filepath, seg_img)
                seg_img_plt = Image.open(seg_img_filepath).convert('P')
                seg_img_plt.putpalette([
                    0, 0, 0,
                    128, 0, 0, 
                    0, 128, 0,
                    128, 128, 0,
                    0, 128, 128,
                    128, 128, 128,
                    64, 0, 0,
                    192, 0, 0,
                    64, 128, 0,
                    192, 128, 0,
                    64, 0, 128,
                    192, 0, 128,
                    64, 128, 128,
                    192, 128, 128,
                    0, 64, 0,
                    128, 64, 0,
                    0, 192, 0,
                    128, 192, 0, # defined for 18 classes currently
                ])
                seg_img_plt.save(seg_img_filepath)

                # save edge image
                edge_img_filepath = osp.join(syn_images_folder, 'image_%05d/labels/edge_img.png' % scene_num)
                ret, edge_img = cv2.threshold(edge_img, 10, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3,3), np.uint8)
                edge_img = cv2.dilate(edge_img, kernel, iterations = 1)

                cv2.imwrite(edge_img_filepath, edge_img)