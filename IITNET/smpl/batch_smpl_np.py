import numpy as np
import os
import pickle
import sys

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl


class BATCHSMPLModel():
    def __init__(self, model_path, batch_size):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        self.batch_size = batch_size
        with open(model_path, 'rb') as f:
            params = pickle.load(f,encoding='iso-8859-1')

            self.J_regressor = params['J_regressor'].toarray()
            self.weights = params['weights']
            self.posedirs = params['posedirs']
            self.v_template = params['v_template']
            self.shapedirs = params['shapedirs']
            self.faces = params['f']
            self.kintree_table = params['kintree_table']
            self.Displayment = np.tile(np.zeros_like(self.v_template), [self.batch_size, 1, 1])  # N, 6890, 3

        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        # self.pose_shape = [24, 3]
        # self.beta_shape = [10]
        # self.trans_shape = [3]
        #
        self.pose = np.zeros((self.batch_size, 24, 3))
        self.beta = np.zeros((self.batch_size, 10))
        self.trans = np.zeros((self.batch_size, 3))
        #
        self.verts = None
        self.J = None
        self.R = None
        #
        self.update()

    def set_params(self, pose=None, beta=None, trans=None, Displayment=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Parameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        if Displayment is not None:
            self.Displayment = Displayment
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.

        """

        # how beta affect body shape
        shapedirs_ = self.shapedirs.reshape(-1, 10)  # 20670*10
        v_shaped_scaled = np.dot(self.beta, shapedirs_.T).reshape(-1, self.v_template.shape[0],
                                                                  self.v_template.shape[1]) + self.v_template
        self.v_shaped_scaled = v_shaped_scaled  # N, 6890, 3

        body_height = (v_shaped_scaled[:, 2802, 1] + v_shaped_scaled[:, 6262, 1]) - (
                v_shaped_scaled[:, 2237, 1] + v_shaped_scaled[:, 6728, 1])

        scale = np.reshape(1.66 / body_height, (-1, 1, 1))  # N, 1, 1
        self.v_shaped = scale * v_shaped_scaled
        self.Displayment = scale * self.Displayment
        self.v_shaped_personal = self.v_shaped + self.Displayment

        # joints location
        J_x = np.expand_dims(np.dot(v_shaped_scaled[:, :, 0], self.J_regressor.T), -1)  # N*6980 dot 6890*24
        J_y = np.expand_dims(np.dot(v_shaped_scaled[:, :, 1], self.J_regressor.T), -1)  # N*6980 dot 6890*24
        J_z = np.expand_dims(np.dot(v_shaped_scaled[:, :, 2], self.J_regressor.T), -1)  # N*6980 dot 6890*24
        self.J = scale * np.concatenate([J_x, J_y, J_z], axis=2)  # N*24*3

        # pose_cube = self.pose.reshape((-1, 1, 3))  #N,24,3 -> N*24, 1, 3
        # rotation matrix for each joint
        # self.R = np.reshape(self.rodrigues(pose_cube) ,(self.batch_size, -1, 3, 3)) #N, 24, 3, 3
        self.R = np.reshape(
            self.batch_rodrigues(np.reshape(self.pose, [-1, 3])), [-1, 24, 3, 3])

        I_cube = np.broadcast_to(
            np.expand_dims(np.expand_dims(np.eye(3), axis=0), 0),
            (self.R.shape[0], self.R.shape[1] - 1, 3, 3)
        )  # N, 23, 3, 3

        lrotmin = (self.R[:, 1:, :, :] - I_cube).reshape(-1, 207)  # N*207,


        # how pose affect body shape in zero pose
        posedirs_ = self.posedirs.reshape(-1, 207)  # 6890*3, 207

        v_posed = self.v_shaped_personal + np.dot(lrotmin, posedirs_.T).reshape(-1, self.v_template.shape[0],
                                                                                self.v_template.shape[1])
        self.v_posed = v_posed

        root_rotation = self.R[:, 0, :, :]  # N, 3, 3
        Js = np.expand_dims(self.J, -1)  # N, 24, 3, 1

        def make_A(R, t, name=None):
            # Rs is N x 3 x 3, ts is N x 3 x 1
            R_homo = np.concatenate([R, np.zeros((self.batch_size, 1, 3))], 1)
            t_homo = np.concatenate([t, np.ones((self.batch_size, 1, 1))], 1)
            return np.concatenate([R_homo, t_homo], 2).reshape(self.batch_size, 1, 4, 4)  # N, 1 , 4, 4

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        for i in range(1, 24):
            j_here = Js[:, i] - Js[:, self.parent[i]]
            A_here = make_A(self.R[:, i], j_here)  # N, 1, 4, 4
            res_here = np.matmul(
                results[self.parent[i]], A_here)
            results.append(res_here)
        results = np.concatenate(results, axis=1)  # N, 24, 4, 4
        new_J = results[:, :, :3, 3]  # N, 24, 3

        Js_w0 = np.concatenate([Js, np.zeros([self.batch_size, 24, 1, 1])], 2)  # N, 24, 4, 1
        init_bone = np.matmul(results, Js_w0)  # N, 24, 4, 1
        # Append empty 4 x 3:
        init_bone = np.concatenate([np.zeros((self.batch_size, 24, 4, 3)), init_bone], axis=3)  # N, 24, 4, 4
        # init_bone = np.pad(init_bone, [[0, 0], [0, 0], [0, 0], [3, 0]])
        A = results - init_bone  # N, 24, 4, 4
        self.J_transformed = new_J + np.expand_dims(self.trans, axis=1)

        # 5. Do skinning:
        W = np.tile(self.weights, [self.batch_size, 1, 1])  # W is N x 6890 x 24

        # # (N x 6890 x 24) x (N x 24 x 16)
        T = np.reshape(
            np.matmul(W, np.reshape(A, [self.batch_size, 24, 16])),
            [self.batch_size, -1, 4, 4])  # T is N,6890, 4, 4

        v_posed_homo = np.concatenate(
            [self.v_posed, np.ones([self.batch_size, self.v_posed.shape[1], 1])], 2)  # N, 6890, 4
        v_homo = np.matmul(T, np.expand_dims(v_posed_homo, -1))  # N, 6890, 4, 1

        verts = v_homo[:, :, :3, 0]
        verts_t = verts + np.expand_dims(self.trans, axis=1)
        self.verts = verts_t  # N, 6890, 3

        '''
        # world transformation of each joint
        G = np.tile(np.empty((self.kintree_table.shape[1], 4, 4)), [self.batch_size, 1, 1, 1]) #N, 24, 4, 4
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
          G[i] = G[self.parent[i]].dot(
            self.with_zeros(
              np.hstack(
                [self.R[i],((self.J[i, :]-self.J[self.parent[i],:]).reshape([3,1]))]
              )
            )
          )
        G = G - self.pack(
          np.matmul(
            G,
            np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
          )
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])
        '''

    def batch_skew(self, vec, batch_size=None):
        """
        vec is N x 3, batch_size is int

        returns N x 3 x 3. Skew_sym version of each matrix.
        """
        if batch_size is None:
            batch_size = vec.shape[0]
        col_inds = np.array([1, 2, 3, 5, 6, 7])
        indices = np.reshape(
            np.reshape(np.array(range(0, batch_size)) * 9, [-1, 1]) + col_inds,
            [-1])
        updates = np.reshape(
            np.stack(
                [
                    -vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1],
                    vec[:, 0]
                ],
                axis=1), [-1])
        mini = abs(np.min(indices))
        maxi = abs(np.max(indices))
        output = np.zeros(maxi + mini + 1)
        for ai, bi in zip(updates, indices - 1):
            output[bi + mini] = ai

        res = np.reshape(output, [batch_size, 3, 3])

        return res

    def batch_rodrigues(self, theta):
        batch_size = theta.shape[0]
        angle = np.expand_dims(np.linalg.norm(theta + 1e-8, axis=1), -1)  # batchsize, 1
        r = np.expand_dims(np.divide(theta, angle), -1)  # batchsize, 3, 1
        angle = np.expand_dims(angle, -1)  # batchsize, 1, 1
        cos = np.cos(angle)
        sin = np.sin(angle)
        r2 = np.transpose(r, [0, 2, 1])  # batchsize, 1, 3
        outer = np.matmul(r, r2)  # batchsize, 3, 3

        eyes = np.tile(np.expand_dims(np.eye(3), 0), [batch_size, 1, 1])
        R = cos * eyes + (1 - cos) * outer + sin * self.batch_skew(
            r.reshape(batch_size, 3), batch_size=batch_size)
        return R

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        """
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def batch_save_to_obj(self, folder, type="vertices"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        vertices_array=[]
        if type=='vertices':
          vertices_array=self.verts
        elif type=='vertices_tposed':
          vertices_array=self.v_shaped_personal
        elif type=='vertices_naked':
          vertices_array=self.v_shaped

        num = vertices_array.shape[0]
        for i in range(num):
            vertices = vertices_array[i]
            with open(folder + str(i).zfill(5) + ".obj", 'w') as wf:
                for v in vertices:
                    wf.write('v %f %f %f\n' % (v[0], v[1], v[2]))
                for f in self.faces + 1:
                    wf.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def load_smplData(folder):
    pose = []
    trans = []
    shape = []
    Displayment = np.zeros((3, 6890))
    with open(folder + "pose00000.txt", 'r') as rf:
        data = rf.readlines()
        for i in range(3):
            trans.append(float(data[i]))
        for i in range(24 * 3):
            pose.append(float(data[i + 3]))
    with open(folder + "shape00000.txt", 'r') as rf:
        data = rf.readlines()
        for i in range(10):
            shape.append(float(data[i]))
    with open(folder + "Displayment00000.txt", 'r') as rf:
        data = rf.readlines()
        for i in range(3):
            temp = data[i].split(" ")
            for j in range(6890):
                Displayment[i, j] = float(temp[j])
    return np.array(pose).reshape(24, 3), np.array(shape), np.array(trans), Displayment.T


def load_joints_pkl(J_regressor_pkl, face_regressor_pkl):
    body_25_reg = pickle.load(open(J_regressor_pkl, 'rb'),encoding='iso-8859-1').T
    body_25_reg = body_25_reg.toarray()  # 25,6890
    face_reg = pickle.load(open(face_regressor_pkl, 'rb'),encoding='iso-8859-1').T
    face_reg = face_reg.toarray()  # 70, 6890
    return body_25_reg, face_reg


if __name__ == '__main__':
    batch_pose, batch_beta, batch_trans, batch_Displayment = [], [], [], []
    nameList = ['134211537199056', '134211550077666']
    for name in nameList:
        pose, beta, trans, Displayment = load_smplData("../data/train_data/params/" + name + "/paramFile_smpl/")
        batch_pose.append(pose)
        batch_beta.append(beta)
        batch_trans.append(trans)
        batch_Displayment.append(Displayment)
    batch_pose = np.array(batch_pose)
    batch_beta = np.array(batch_beta)
    batch_trans = np.array(batch_trans)
    batch_Displayment = np.array(batch_Displayment)

    print(batch_pose.shape, batch_beta.shape, batch_trans.shape, batch_Displayment.shape)
    smpl = BATCHSMPLModel('../assets/neutral_smpl.pkl', batch_pose.shape[0])
    smpl.set_params(beta=batch_beta, pose=batch_pose, trans=batch_trans, Displayment=batch_Displayment)
    smpl.batch_save_to_obj("result/vertices/", type='vertices')
    smpl.batch_save_to_obj("result/vertices_tposed/", type='vertices_tposed')
    smpl.batch_save_to_obj("result/vertices_naked/", type='vertices_naked')

    #for vertices, vertices_tposed, vertices_naked
    vertices=smpl.verts
    vertices_tposed=smpl.v_shaped_personal
    vertices_naked=smpl.v_shaped
    print(vertices.shape, vertices_tposed.shape, vertices_naked.shape) #N, 6890, 3

    # k3d
    body_25_reg, face_reg = load_joints_pkl("../assets/J_regressor.pkl", '../assets/face_regressor.pkl')
    #print("body_25_reg.shape", body_25_reg.shape, "face_reg.shape", face_reg.shape)
    smpl.set_params(beta=batch_beta, pose=batch_pose, trans=batch_trans, Displayment=np.zeros_like(smpl.Displayment))
    smpl.batch_save_to_obj("result/", type='vertices')
    body_25_reg=np.tile(body_25_reg, [smpl.batch_size, 1, 1])
    face_reg = np.tile(face_reg, [smpl.batch_size, 1, 1])
    #print("body_25_reg.shape", body_25_reg.shape, "face_reg.shape", face_reg.shape)
    gt_kps=np.concatenate([np.matmul(body_25_reg, smpl.verts),np.matmul(face_reg, smpl.verts)], axis=1)
    print(gt_kps.shape)


