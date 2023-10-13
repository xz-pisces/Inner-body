# -*- coding:utf-8 *-:
import numpy as np
import pickle


class SMPLModel():
  def __init__(self, model_path):
    """
    SMPL model.

    Parameter:
    ---------
    model_path: Path to the SMPL model parameters, pre-processed by
    `preprocess.py`.

    """
    with open(model_path, 'rb') as f:
      params = pickle.load(f)

      self.J_regressor = params['J_regressor']
      self.weights = params['weights']
      self.posedirs = params['posedirs']
      self.v_template = params['v_template']
      self.shapedirs = params['shapedirs']
      self.faces = params['f']
      self.kintree_table = params['kintree_table']
      self.Displacement=np.zeros_like(self.v_template)


    id_to_col = {
      self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
    }
    self.parent = {
      i: id_to_col[self.kintree_table[0, i]]
      for i in range(1, self.kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.J = None
    self.R = None

    self.update()

  def set_params(self, pose=None, beta=None, trans=None, Displacement=None):
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
    if Displacement is not None:
      self.Displacement = Displacement
    self.update()
    return self.verts

  def update(self):
    """
    Called automatically when parameters are updated.

    """
    # how beta affect body shape
    v_shaped = self.shapedirs.dot(self.beta) + self.v_template
    self.v_shaped=v_shaped
    # joints location
    self.J = self.J_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # rotation matrix for each joint
    self.R = self.rodrigues(pose_cube)
    I_cube = np.broadcast_to(
      np.expand_dims(np.eye(3), axis=0),
      (self.R.shape[0]-1, 3, 3)
    )
    lrotmin = (self.R[1:] - I_cube).ravel()
    # how pose affect body shape in zero pose
    v_posed = v_shaped + self.posedirs.dot(lrotmin)
    ### adding Displacement
    v_posed = v_posed + self.Displacement

    # world transformation of each joint
    G = np.empty((self.kintree_table.shape[1], 4, 4))
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
    np.seterr(divide='ignore', invalid='ignore')

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

class SmplTPoseLayer():
    def __init__(self, model='assets/neutral_smpl.pkl'):
        self.smpl = SMPLModel(model)
    
    def get_vertices(self, pose, betas, trans, Displacement):
        # print(pose.shape, betas.shape, trans.shape, Displacement.shape)
        verts = self.smpl.set_params(pose=pose,beta=betas,trans=trans,Displacement=Displacement)
        v_shaped_scaled=self.smpl.v_shaped
        body_height = (v_shaped_scaled[2802, 1] + v_shaped_scaled[6262, 1]) - (
            v_shaped_scaled[2237, 1] + v_shaped_scaled[6728, 1])

        scale = 1.66 / float(body_height)
        v_shaped = scale * v_shaped_scaled  #vertices_naked
        v_shaped_personal = v_shaped + self.smpl.Displacement  #获取vertices_tposed
        return [verts, v_shaped_personal, v_shaped]
    
    def load_joints_pkl(self, J_regressor_pkl, face_regressor_pkl):
        body_25_reg = pickle.load(open(J_regressor_pkl, 'rb')).T
        body_25_reg = body_25_reg.toarray()  # 25,6890
        face_reg = pickle.load(open(face_regressor_pkl, 'rb')).T
        face_reg = face_reg.toarray()  # 70, 6890
        return body_25_reg, face_reg
    
    def get_joints(self, pose, betas, trans):
        body_25_reg, face_reg = self.load_joints_pkl('assets/J_regressor.pkl', 'assets/face_regressor.pkl')
        # print(np.dot(body_25_reg, self.smpl.verts).shape)
        # print(np.dot(face_reg, self.smpl.verts).shape)
        gt_kps=np.vstack((np.dot(body_25_reg, self.smpl.verts),np.dot(face_reg, self.smpl.verts)))
        return gt_kps


def load_smplData(folder):
  pose=[]
  trans=[]
  shape=[]
  Displacement=np.zeros((3, 6890))
  with open(folder+"pose.txt", 'r') as rf:
    data=rf.readlines()
    for i in range(3):
      trans.append(float(data[i]))
    for i in range(24*3):
      pose.append(float(data[i+3]))
  with open(folder+"shape.txt", 'r') as rf:
    data = rf.readlines()
    for i in range(10):
      shape.append(float(data[i]))
  with open(folder+"Displacement.txt", 'r') as rf:
    data=rf.readlines()
    for i in range(3):
      temp=data[i].split(" ")
      for j in range(6890):
        Displacement[i, j]=float(temp[j])
  return np.array(pose).reshape(24, 3), np.array(shape), np.array(trans), Displacement.T

if __name__ == '__main__':
  pose, beta, trans, Displacement=load_smplData("126111535847796\\")
  print(pose.shape, beta.shape, trans.shape, Displacement.shape)
  smpl = SMPLModel('neutral_smpl.pkl')
  # np.random.seed(9608)
  # pose = (np.random.rand(*smpl.pose_shape) - 0.5) * 0.4
  # beta = (np.random.rand(*smpl.beta_shape) - 0.5) * 0.06
  # trans = np.zeros(smpl.trans_shape)
  smpl.set_params(beta=beta, pose=pose, trans=trans, Displacement=Displacement)
  smpl.save_to_obj('smpl_np.obj')

  #获取vertices_tposed, vertices_naked
  v_shaped_scaled=smpl.v_shaped
  body_height = (v_shaped_scaled[2802, 1] + v_shaped_scaled[6262, 1]) - (
            v_shaped_scaled[2237, 1] + v_shaped_scaled[6728, 1])

  scale = 1.66 / float(body_height)


  v_shaped = scale * v_shaped_scaled  #vertices_naked
  v_shaped_personal = v_shaped + smpl.Displacement  #获取vertices_tposed
  smpl.verts=v_shaped
  smpl.save_to_obj("v_shaped.obj")
  smpl.verts=v_shaped_personal
  smpl.save_to_obj("v_shaped_tposed.obj")
  smpl.verts=v_shaped_scaled+smpl.Displacement
  smpl.save_to_obj("v_scaled_tposed.obj")
  smpl.verts = v_shaped + smpl.Displacement*scale
  smpl.save_to_obj("x.obj")


