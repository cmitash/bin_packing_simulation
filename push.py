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
    init_pose = [float(argv[1]), float(argv[2]), float(argv[3]), float(argv[4]), float(argv[5]), float(argv[6]), float(argv[7])]
    target_vector = [float(argv[8]), float(argv[9]), float(argv[10])]

    ## read configuration file
    cfg = ConfigParser("config.yml", "camera_info.yml")
    framesIter = cfg.getNumSimulationSteps()
    
    # From second object onwards, we need to load previous objects
    if num > 1: 
        scene_path = os.path.join(g_repo_path, 'rendered_images/image_%05i/labels/obj_poses.yml' % (num - 1))
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
            bpy.context.scene.objects[shape_file].rigid_body.enabled = True
            bpy.context.scene.objects[shape_file].rigid_body.use_margin = True
            bpy.context.scene.objects[shape_file].rigid_body.collision_shape = 'MESH'
            bpy.context.scene.objects[shape_file].rigid_body.collision_margin = 0.001
            bpy.context.scene.objects[shape_file].rigid_body.linear_damping = 0.9
            bpy.context.scene.objects[shape_file].rigid_body.angular_damping = 0.9

    ## sample initial pose for each of the selected object
    bpy.ops.import_scene.obj(filepath="3d_models/dove/dove.obj")
    shape_file = bpy.context.selected_objects[0].name
    bpy.data.objects[shape_file].hide = False
    bpy.data.objects[shape_file].hide_render = False
    bpy.data.objects[shape_file].pass_index = 1
    bpy.context.scene.objects.active = bpy.context.scene.objects[shape_file]

    #### ADD RIGID BODY
    bpy.ops.rigidbody.object_add(type='ACTIVE')
    bpy.ops.object.modifier_add(type = 'COLLISION')
    bpy.context.scene.objects[shape_file].rigid_body.mass = 10.0
    bpy.context.scene.objects[shape_file].rigid_body.use_margin = True
    bpy.context.scene.objects[shape_file].rigid_body.enabled = True
    bpy.context.scene.objects[shape_file].rigid_body.collision_shape = 'MESH'
    bpy.context.scene.objects[shape_file].rigid_body.collision_margin = 0.001
    bpy.context.scene.objects[shape_file].rigid_body.linear_damping = 0.9
    bpy.context.scene.objects[shape_file].rigid_body.angular_damping = 0.9

    bpy.data.objects[shape_file].location = (init_pose[0], init_pose[1], init_pose[2])
    bpy.data.objects[shape_file].rotation_mode = 'QUATERNION'
    bpy.data.objects[shape_file].rotation_quaternion = (init_pose[3], init_pose[4], init_pose[5], init_pose[6])

    #########################

     ## sample initial pose for each of the selected object
    bpy.ops.import_scene.obj(filepath="3d_models/ee/ee.obj")
    ee_shape_file = bpy.context.selected_objects[0].name
    bpy.data.objects[ee_shape_file].hide = False
    bpy.data.objects[ee_shape_file].hide_render = False
    bpy.data.objects[ee_shape_file].pass_index = 1
    bpy.context.scene.objects.active = bpy.context.scene.objects[ee_shape_file]

    bpy.ops.rigidbody.object_add(type='ACTIVE')
    bpy.ops.object.modifier_add(type = 'COLLISION')
    bpy.context.scene.objects[ee_shape_file].rigid_body.mass = 10.0
    bpy.context.scene.objects[ee_shape_file].rigid_body.use_margin = True
    bpy.context.scene.objects[ee_shape_file].rigid_body.enabled = True
    bpy.context.scene.objects[ee_shape_file].rigid_body.kinematic = True
    bpy.context.scene.objects[ee_shape_file].rigid_body.collision_shape = 'MESH'
    bpy.context.scene.objects[ee_shape_file].rigid_body.collision_margin = 0.001
    bpy.context.scene.objects[ee_shape_file].rigid_body.linear_damping = 0.9
    bpy.context.scene.objects[ee_shape_file].rigid_body.angular_damping = 0.9

    bpy.data.objects[ee_shape_file].location = (bpy.data.objects[shape_file].location[0] + 0.05 * bpy.data.objects[shape_file].matrix_world[0][2],
                                                bpy.data.objects[shape_file].location[1] + 0.05 * bpy.data.objects[shape_file].matrix_world[1][2],
                                                bpy.data.objects[shape_file].location[2] + 0.05 * bpy.data.objects[shape_file].matrix_world[2][2])

    init_pose_euler = bpy.data.objects[ee_shape_file].rotation_euler
    bpy.data.objects[ee_shape_file].rotation_mode = 'XYZ'
    bpy.data.objects[ee_shape_file].rotation_euler = (bpy.data.objects[shape_file].rotation_euler[0],
                                                      bpy.data.objects[shape_file].rotation_euler[1],
                                                      bpy.data.objects[shape_file].rotation_euler[2])

    bpy.data.objects[ee_shape_file].keyframe_insert(data_path="location", frame=1) 
    bpy.data.objects[ee_shape_file].location = (bpy.data.objects[ee_shape_file].location[0] + target_vector[0], 
                                                bpy.data.objects[ee_shape_file].location[1] + target_vector[1], 
                                                bpy.data.objects[ee_shape_file].location[2] + target_vector[2])
    bpy.data.objects[ee_shape_file].keyframe_insert(data_path="location", frame=framesIter)

    bpy.ops.rigidbody.constraint_add()
    bpy.data.objects[ee_shape_file].rigid_body_constraint.type = 'GENERIC_SPRING'
    bpy.data.objects[ee_shape_file].rigid_body_constraint.object1 = bpy.data.objects[ee_shape_file]
    bpy.data.objects[ee_shape_file].rigid_body_constraint.object2 = bpy.data.objects[shape_file]

    # bpy.data.objects[ee_shape_file].rigid_body_constraint.use_limit_ang_z = True

    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_limit_lin_x = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_limit_lin_y = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_limit_lin_z = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_limit_ang_x = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_limit_ang_y = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_limit_ang_z = True

    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_lin_x_lower = -0.005
    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_lin_y_lower = -0.005
    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_lin_z_lower = -0.005

    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_lin_x_upper = 0.005
    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_lin_y_upper = 0.005
    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_lin_z_upper = 0.005

    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_ang_x_lower = -0.5
    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_ang_y_lower = -0.5
    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_ang_z_lower = -0.5

    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_ang_x_upper = 0.5
    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_ang_y_upper = 0.5
    bpy.data.objects[ee_shape_file].rigid_body_constraint.limit_ang_z_upper = 0.5

    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_spring_x = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_spring_y = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_spring_z = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_spring_ang_x = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_spring_ang_y = True
    bpy.data.objects[ee_shape_file].rigid_body_constraint.use_spring_ang_z = True

    stiffness_coeff = 0.05
    damping_coeff = 0.5

    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_stiffness_x = stiffness_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_stiffness_y = stiffness_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_stiffness_z = stiffness_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_stiffness_ang_x = stiffness_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_stiffness_ang_y = stiffness_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_stiffness_ang_z = stiffness_coeff

    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_damping_x = damping_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_damping_y = damping_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_damping_z = damping_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_damping_ang_x = damping_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_damping_ang_y = damping_coeff
    bpy.data.objects[ee_shape_file].rigid_body_constraint.spring_damping_ang_z = damping_coeff

    # bpy.context.scene.rigidbody_world.steps_per_second = 500
    # bpy.context.scene.rigidbody_world.solver_iterations = 500

    ##############################
    ## performing simulation
    for i in range(1, framesIter):
        bpy.context.scene.frame_set(i)

     ## rendering configuration
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.viewport_shade = 'TEXTURED'

    for item in bpy.data.materials:
        item.use_shadeless = True
        item.use_cast_buffer_shadows = False

    # output_video = "rendered_images/image_%05i/pushing.avi" % num
    # bpy.context.scene.render.resolution_percentage = 100
    # bpy.context.scene.render.image_settings.file_format = 'AVI_JPEG'
    # bpy.context.scene.render.image_settings.color_mode = 'RGB'
    # bpy.context.scene.render.image_settings.color_depth = '8'
    # bpy.context.scene.frame_start = 1
    # bpy.context.scene.frame_end = 20
    # bpy.context.scene.render.filepath = os.path.join(g_repo_path, output_video) 
    # bpy.ops.render.render(animation=True)

    os.makedirs("rendered_images/image_%05i/labels" % num)
    output_pose = "rendered_images/image_%05i/labels/obj_poses.yml" % num

    for soap in bpy.data.objects:
        if 'dove' in soap.name:
            bpy.context.scene.objects.active = soap
            with open(output_pose, "a+") as file:
                q = quaternion_from_matrix(soap.matrix_world)
                print ('position: ', soap.location[0], soap.location[1], soap.location[2])
                file.write("- rotation : [%f, %f, %f, %f]\n  translation : [%f, %f, %f]\n" % (q[0], q[1], q[2], q[3],
                                                                                              soap.matrix_world[0][3], soap.matrix_world[1][3], soap.matrix_world[2][3]))

    # save to temp.blend
    if cfg.saveDebugFile() == True: 
        mainfile_path = "rendered_images/debug/push_%02d.blend" % num
        bpy.ops.file.autopack_toggle()
        bpy.ops.wm.save_as_mainfile(filepath=mainfile_path, relative_remap = False)