"""
@file simulate_and_render.py
@copyright Software License Agreement (BSD License).
Copyright (c) 2017, Rutgers the State University of New Jersey, New Brunswick,
All Rights Reserved. For a full description see the file named LICENSE.
Authors: Chaitanya Mitash, Kostas Bekris, Abdeslam Boularias.
"""

import sys, os, tempfile, glob, shutil, time
import bpy
import math, random, numpy
from scipy.spatial import distance
from mathutils import Vector, Matrix, Quaternion

# Verify if repository path is set in bashrc
if os.environ.get('PHYSIM_GENDATA') == None:
    print("Please set PHYSIM_GENDATA in bashrc!")
    sys.exit()

g_repo_path = os.environ['PHYSIM_GENDATA']
sys.path.append(g_repo_path)

from Environment import Bin, Light
from ConfigParser import ConfigParser
from Camera import Camera

def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True

    """
    M = numpy.array(matrix, dtype=numpy.float64, copy=False)[:4, :4]
    if isprecise:
        q = numpy.empty((4, ))
        t = numpy.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = numpy.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                         [m01+m10,     m11-m00-m22, 0.0,         0.0],
                         [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                         [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = numpy.linalg.eigh(K)
        q = V[[3, 0, 1, 2], numpy.argmax(w)]
    if q[0] < 0.0:
        numpy.negative(q, q)
    return q

if __name__ == "__main__":

    argv = sys.argv
    argv = argv[argv.index("--") + 1:]
    num = int(argv[0])

    ## read configuration file
    cfg = ConfigParser("config.yml", "camera_info.yml")
    framesIter = cfg.getNumSimulationSteps()

    ## initial condition
    if num == 0:
        env = cfg.getSurfaceType()
        surface = Bin('3d_models/bin/shelf.obj')
        sPose = cfg.getSurfacePose()
        surface.setPose(sPose)
        camIntrinsic = cfg.getCamIntrinsic()
        maxCamViews, camExtrinsic = cfg.getCamExtrinsic()  # maxCamViews: num of poses
        numViews = cfg.getNumViews()
        cam = Camera(camIntrinsic, camExtrinsic, numViews)
        cam_pose = cam.placeCamera(0)
        pLight = Light()
        light_range_x = cfg.getLightRangeX()
        light_range_y = cfg.getLightRangeY()
        light_range_z = cfg.getLightRangeZ()
        pLight.placePointLight(light_range_x, light_range_y, light_range_z)

    
    if num > 0:
        scene_path = os.path.join(g_repo_path, 'rendered_images/image_%05i/labels/obj_poses.yml' % num)
        numObjectsInScene, o1_poses = cfg.getObjPoses(scene_path, 1)
        for i in range(0, numObjectsInScene):
            ## sample initial pose for each of the selected object
            bpy.ops.import_scene.obj(filepath="3d_models/dove/dove.obj")
            shape_file = bpy.context.selected_objects[0].name
            bpy.data.objects[shape_file].hide = False
            bpy.data.objects[shape_file].hide_render = False
            bpy.data.objects[shape_file].pass_index = 1
            bpy.data.objects[shape_file].location = (o1_poses[i][0], o1_poses[i][1], o1_poses[i][2])
            bpy.data.objects[shape_file].rotation_mode = 'QUATERNION'
            bpy.data.objects[shape_file].rotation_quaternion = (o1_poses[i][3], o1_poses[i][4], o1_poses[i][5], o1_poses[i][6])

            bpy.context.scene.objects.active = bpy.context.scene.objects[shape_file]

            #### ADD RIGID BODY
            bpy.ops.rigidbody.object_add(type='ACTIVE')
            bpy.ops.object.modifier_add(type = 'COLLISION')
            bpy.context.scene.objects[shape_file].rigid_body.mass = 10.0
            bpy.context.scene.objects[shape_file].rigid_body.use_margin = True
            # bpy.context.scene.objects[shape_file].rigid_body.enabled = False
            bpy.context.scene.objects[shape_file].rigid_body.collision_shape = 'MESH'
            bpy.context.scene.objects[shape_file].rigid_body.collision_margin = 0.001
            bpy.context.scene.objects[shape_file].rigid_body.linear_damping = 0.9
            bpy.context.scene.objects[shape_file].rigid_body.angular_damping = 0.9

        ## performing simulation
        for i in range(1, framesIter):
            bpy.context.scene.frame_set(i)

    output_pose = "rendered_images/image_%05i/labels/obj_poses.yml" % num
    if os.path.exists(output_pose):
        os.remove(output_pose)

    ## rendering configuration
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.viewport_shade = 'TEXTURED'

    # FLAT RENDERING BLANDER INTERNAL
    if cfg.getRenderer() == 'flat':
        for item in bpy.data.materials:
            item.use_shadeless = True
            item.use_cast_buffer_shadows = False

    # BLENDER RENDER::TODO: Look into the effect of these parameters
    if cfg.getRenderer() == 'blender':
        bpy.context.scene.render.use_shadows = True
        bpy.context.scene.render.use_raytrace = False
        for item in bpy.data.materials:
            item.emit = random.uniform(0.1, 0.6)

    # Use nodes for rendering depth images and object masks
    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_object_index = True
    bpy.context.scene.use_nodes = True

    tree = bpy.context.scene.node_tree
    links = tree.links

    for n in tree.nodes:
        tree.nodes.remove(n)

    render_node = tree.nodes.new('CompositorNodeRLayers')

    # For depth rendering
    bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
    depth_node = tree.nodes.new('CompositorNodeOutputFile') 
    depth_node.base_path = "rendered_images/image_%05i/depth/" % num
    depth_node.file_slots[0].path = "image_"
    links.new(render_node.outputs[2], depth_node.inputs[0])

    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    tmp_node = tree.nodes.new('CompositorNodeIDMask')
    tmp_node.index = 1
    links.new(render_node.outputs[14], tmp_node.inputs[0])

    tmp_out = tree.nodes.new('CompositorNodeOutputFile') 
    tmp_out.base_path = "rendered_images/debug/"
    tmp_out.file_slots[0].path = "image_%05i_%02i_" % (num, 0)

    links.new(tmp_node.outputs[0], tmp_out.inputs[0])

    output_img = "rendered_images/image_%05i/rgb/image.png" % num
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.filepath = os.path.join(g_repo_path, output_img) 
    bpy.ops.render.render(write_still=True)

    for soap in bpy.data.objects:
        if 'dove' in soap.name:
            bpy.context.scene.objects.active = soap
            with open(output_pose, "a+") as file:
                q = quaternion_from_matrix(soap.matrix_world)
                print ('position: ', soap.location[0], soap.location[1], soap.location[2])
                file.write("- rotation : [%f, %f, %f, %f]\n  translation : [%f, %f, %f]\n" % (q[0], q[1], q[2], q[3],
                                                                                              soap.matrix_world[0][3], soap.matrix_world[1][3], soap.matrix_world[2][3]))

    output_filepath = "rendered_images/dataset_info.txt"
    with open(output_filepath, "a+") as file:
        file.write("%i,%i\n" % (num, 1))

    # save to temp.blend
    if cfg.saveDebugFile() == True: 
        mainfile_path = "rendered_images/debug/blend_curr_%02d.blend" % num
        bpy.ops.file.autopack_toggle()
        bpy.ops.wm.save_as_mainfile(filepath=mainfile_path)