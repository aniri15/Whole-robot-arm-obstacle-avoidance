"""
Useful classes for supporting DeepMind MuJoCo binding.
"""

import gc
import os
from tempfile import TemporaryDirectory

# DIRTY HACK copied from mujoco-py - a global lock on rendering
from threading import Lock

import mujoco
import numpy as np

_MjSim_render_lock = Lock()

import ctypes
import ctypes.util
import os
import platform
import subprocess

class MjSimState:
    """
    A mujoco simulation state.
    """

    def __init__(self, time, qpos, qvel):
        self.time = time
        self.qpos = qpos
        self.qvel = qvel

    @classmethod
    def from_flattened(cls, array, sim):
        """
        Takes flat mjstate array and MjSim instance and
        returns MjSimState.
        """
        idx_time = 0
        idx_qpos = idx_time + 1
        idx_qvel = idx_qpos + sim.model.nq

        time = array[idx_time]
        qpos = array[idx_qpos : idx_qpos + sim.model.nq]
        qvel = array[idx_qvel : idx_qvel + sim.model.nv]
        assert sim.model.na == 0

        return cls(time=time, qpos=qpos, qvel=qvel)

    def flatten(self):
        return np.concatenate([[self.time], self.qpos, self.qvel], axis=0)


class _MjModelMeta(type):
    """
    Metaclass which allows MjModel below to delegate to mujoco.MjModel.

    Taken from dm_control: https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/wrapper/core.py#L244
    """

    def __new__(cls, name, bases, dct):
        for attr in dir(mujoco.MjModel):
            if not attr.startswith("_"):
                if attr not in dct:
                    # pylint: disable=protected-access
                    fget = lambda self, attr=attr: getattr(self._model, attr)
                    fset = lambda self, value, attr=attr: setattr(self._model, attr, value)
                    # pylint: enable=protected-access
                    dct[attr] = property(fget, fset)
        return super().__new__(cls, name, bases, dct)

class MjModel(metaclass=_MjModelMeta):
    """Wrapper class for a MuJoCo 'mjModel' instance.
    MjModel encapsulates features of the model that are expected to remain
    constant. It also contains simulation and visualization options which may be
    changed occasionally, although this is done explicitly by the user.
    """

    _HAS_DYNAMIC_ATTRIBUTES = True

    def __init__(self, model_ptr):
        """Creates a new MjModel instance from a mujoco.MjModel."""
        self._model = model_ptr

        # make useful mappings such as _body_name2id and _body_id2name
        self.make_mappings()

    def __del__(self):
        # free mujoco model
        del self._model

    """
    Some methods supported by sim.model in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """

    def _extract_mj_names(self, name_adr, num_obj, obj_type):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1127
        """

        ### TODO: fix this to use @name_adr like mujoco-py - more robust than assuming IDs are continuous ###

        # objects don't need to be named in the XML, so name might be None
        id2name = {i: None for i in range(num_obj)}
        name2id = {}
        for i in range(num_obj):
            name = mujoco.mj_id2name(self._model, obj_type, i)
            name2id[name] = i
            id2name[i] = name

        # # objects don't need to be named in the XML, so name might be None
        # id2name = { i: None for i in range(num_obj) }
        # name2id = {}
        # for i in range(num_obj):
        #     name = self.model.names[name_adr[i]]
        #     decoded_name = name.decode()
        #     if decoded_name:
        #         obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        #         assert (0 <= obj_id < num_obj) and (id2name[obj_id] is None)
        #         name2id[decoded_name] = obj_id
        #         id2name[obj_id] = decoded_name

        # sort names by increasing id to keep order deterministic
        return tuple(id2name[nid] for nid in sorted(name2id.values())), name2id, id2name

    def make_mappings(self):
        """
        Make some useful internal mappings that mujoco-py supported.
        """
        p = self
        self.body_names, self._body_name2id, self._body_id2name = self._extract_mj_names(
            p.name_bodyadr, p.nbody, mujoco.mjtObj.mjOBJ_BODY
        )
        self.joint_names, self._joint_name2id, self._joint_id2name = self._extract_mj_names(
            p.name_jntadr, p.njnt, mujoco.mjtObj.mjOBJ_JOINT
        )
        self.geom_names, self._geom_name2id, self._geom_id2name = self._extract_mj_names(
            p.name_geomadr, p.ngeom, mujoco.mjtObj.mjOBJ_GEOM
        )
        self.site_names, self._site_name2id, self._site_id2name = self._extract_mj_names(
            p.name_siteadr, p.nsite, mujoco.mjtObj.mjOBJ_SITE
        )
        self.light_names, self._light_name2id, self._light_id2name = self._extract_mj_names(
            p.name_lightadr, p.nlight, mujoco.mjtObj.mjOBJ_LIGHT
        )
        self.camera_names, self._camera_name2id, self._camera_id2name = self._extract_mj_names(
            p.name_camadr, p.ncam, mujoco.mjtObj.mjOBJ_CAMERA
        )
        self.actuator_names, self._actuator_name2id, self._actuator_id2name = self._extract_mj_names(
            p.name_actuatoradr, p.nu, mujoco.mjtObj.mjOBJ_ACTUATOR
        )
        self.sensor_names, self._sensor_name2id, self._sensor_id2name = self._extract_mj_names(
            p.name_sensoradr, p.nsensor, mujoco.mjtObj.mjOBJ_SENSOR
        )
        self.tendon_names, self._tendon_name2id, self._tendon_id2name = self._extract_mj_names(
            p.name_tendonadr, p.ntendon, mujoco.mjtObj.mjOBJ_TENDON
        )
        self.mesh_names, self._mesh_name2id, self._mesh_id2name = self._extract_mj_names(
            p.name_meshadr, p.nmesh, mujoco.mjtObj.mjOBJ_MESH
        )

    def body_id2name(self, id):
        """Get body name from mujoco body id."""
        if id not in self._body_id2name:
            raise ValueError("No body with id %d exists." % id)
        return self._body_id2name[id]

    def body_name2id(self, name):
        """Get body id from mujoco body name."""
        if name not in self._body_name2id:
            raise ValueError('No "body" with name %s exists. Available "body" names = %s.' % (name, self.body_names))
        return self._body_name2id[name]

    def joint_id2name(self, id):
        """Get joint name from mujoco joint id."""
        if id not in self._joint_id2name:
            raise ValueError("No joint with id %d exists." % id)
        return self._joint_id2name[id]

    def joint_name2id(self, name):
        """Get joint id from joint name."""
        if name not in self._joint_name2id:
            raise ValueError('No "joint" with name %s exists. Available "joint" names = %s.' % (name, self.joint_names))
        return self._joint_name2id[name]

    def geom_id2name(self, id):
        """Get geom name from  geom id."""
        if id not in self._geom_id2name:
            raise ValueError("No geom with id %d exists." % id)
        return self._geom_id2name[id]

    def geom_name2id(self, name):
        """Get geom id from  geom name."""
        if name not in self._geom_name2id:
            raise ValueError('No "geom" with name %s exists. Available "geom" names = %s.' % (name, self.geom_names))
        return self._geom_name2id[name]

    def site_id2name(self, id):
        """Get site name from site id."""
        if id not in self._site_id2name:
            raise ValueError("No site with id %d exists." % id)
        return self._site_id2name[id]

    def site_name2id(self, name):
        """Get site id from site name."""
        if name not in self._site_name2id:
            raise ValueError('No "site" with name %s exists. Available "site" names = %s.' % (name, self.site_names))
        return self._site_name2id[name]

    def light_id2name(self, id):
        """Get light name from light id."""
        if id not in self._light_id2name:
            raise ValueError("No light with id %d exists." % id)
        return self._light_id2name[id]

    def light_name2id(self, name):
        """Get light id from light name."""
        if name not in self._light_name2id:
            raise ValueError('No "light" with name %s exists. Available "light" names = %s.' % (name, self.light_names))
        return self._light_name2id[name]

    def camera_id2name(self, id):
        """Get camera name from camera id."""
        if id == -1:
            return "free"
        if id not in self._camera_id2name:
            raise ValueError("No camera with id %d exists." % id)
        return self._camera_id2name[id]

    def camera_name2id(self, name):
        """Get camera id from  camera name."""
        if name == "free":
            return -1
        if name not in self._camera_name2id:
            raise ValueError(
                'No "camera" with name %s exists. Available "camera" names = %s.' % (name, self.camera_names)
            )
        return self._camera_name2id[name]

    def actuator_id2name(self, id):
        """Get actuator name from actuator id."""
        if id not in self._actuator_id2name:
            raise ValueError("No actuator with id %d exists." % id)
        return self._actuator_id2name[id]

    def actuator_name2id(self, name):
        """Get actuator id from actuator name."""
        if name not in self._actuator_name2id:
            raise ValueError(
                'No "actuator" with name %s exists. Available "actuator" names = %s.' % (name, self.actuator_names)
            )
        return self._actuator_name2id[name]

    def sensor_id2name(self, id):
        """Get sensor name from sensor id."""
        if id not in self._sensor_id2name:
            raise ValueError("No sensor with id %d exists." % id)
        return self._sensor_id2name[id]

    def sensor_name2id(self, name):
        """Get sensor id from sensor name."""
        if name not in self._sensor_name2id:
            raise ValueError(
                'No "sensor" with name %s exists. Available "sensor" names = %s.' % (name, self.sensor_names)
            )
        return self._sensor_name2id[name]

    def tendon_id2name(self, id):
        """Get tendon name from tendon id."""
        if id not in self._tendon_id2name:
            raise ValueError("No tendon with id %d exists." % id)
        return self._tendon_id2name[id]

    def tendon_name2id(self, name):
        """Get tendon id from tendon name."""
        if name not in self._tendon_name2id:
            raise ValueError(
                'No "tendon" with name %s exists. Available "tendon" names = %s.' % (name, self.tendon_names)
            )
        return self._tendon_name2id[name]

    def mesh_id2name(self, id):
        """Get mesh name from  mesh id."""
        if id not in self._mesh_id2name:
            raise ValueError("No mesh with id %d exists." % id)
        return self._mesh_id2name[id]

    def mesh_name2id(self, name):
        """Get mesh id from mesh name."""
        if name not in self._mesh_name2id:
            raise ValueError('No "mesh" with name %s exists. Available "mesh" names = %s.' % (name, self.mesh_names))
        return self._mesh_name2id[name]

    # def userdata_id2name(self, id):
    #     if id not in self._userdata_id2name:
    #         raise ValueError("No userdata with id %d exists." % id)
    #     return self._userdata_id2name[id]

    # def userdata_name2id(self, name):
    #     if name not in self._userdata_name2id:
    #         raise ValueError("No \"userdata\" with name %s exists. Available \"userdata\" names = %s." % (name, self.userdata_names))
    #     return self._userdata_name2id[name]

    def get_xml(self):
        with TemporaryDirectory() as td:
            filename = os.path.join(td, "model.xml")
            ret = mujoco.mj_saveLastXML(filename.encode(), self._model)
            return open(filename).read()

    def get_joint_qpos_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1178

        Returns the qpos address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for pos[start:end] access.
        """
        joint_id = self.joint_name2id(name)
        joint_type = self.jnt_type[joint_id]
        joint_addr = self.jnt_qposadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    def get_joint_qvel_addr(self, name):
        """
        See https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L1202

        Returns the qvel address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for vel[start:end] access.
        """
        joint_id = self.joint_name2id(name)
        joint_type = self.jnt_type[joint_id]
        joint_addr = self.jnt_dofadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 3
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)


class _MjDataMeta(type):
    """
    Metaclass which allows MjData below to delegate to mujoco.MjData.

    Taken from dm_control.
    """

    def __new__(cls, name, bases, dct):
        for attr in dir(mujoco.MjData):
            if not attr.startswith("_"):
                if attr not in dct:
                    # pylint: disable=protected-access
                    fget = lambda self, attr=attr: getattr(self._data, attr)
                    fset = lambda self, value, attr=attr: setattr(self._data, attr, value)
                    # pylint: enable=protected-access
                    dct[attr] = property(fget, fset)
        return super().__new__(cls, name, bases, dct)


class MjData(metaclass=_MjDataMeta):
    """Wrapper class for a MuJoCo 'mjData' instance.
    MjData contains all of the dynamic variables and intermediate results produced
    by the simulation. These are expected to change on each simulation timestep.
    The properties without docstrings are defined in mujoco source code from https://github.com/deepmind/mujoco/blob/062cb53a4a14b2a7a900453613a7ce498728f9d8/include/mujoco/mjdata.h#L126.
    """

    def __init__(self, model):
        """Construct a new MjData instance.
        Args:
          model: An MjModel instance.
        """
        self._model = model
        self._data = mujoco.MjData(model._model)

    @property
    def model(self):
        """The parent MjModel for this MjData instance."""
        return self._model

    def __del__(self):
        # free mujoco data
        del self._data

    """
    Some methods supported by sim.data in mujoco-py.
    Copied from https://github.com/openai/mujoco-py/blob/ab86d331c9a77ae412079c6e58b8771fe63747fc/mujoco_py/generated/wrappers.pxi#L2611
    """

    @property
    def body_xpos(self):
        """
        Note: mujoco-py used to support sim.data.body_xpos but DM mujoco bindings requires sim.data.xpos,
              so we explicitly expose this as a property
        """
        return self._data.xpos

    @property
    def body_xquat(self):
        """
        Note: mujoco-py used to support sim.data.body_xquat but DM mujoco bindings requires sim.data.xquat,
              so we explicitly expose this as a property
        """
        return self._data.xquat

    @property
    def body_xmat(self):
        """
        Note: mujoco-py used to support sim.data.body_xmat but DM mujoco bindings requires sim.data.xmax,
              so we explicitly expose this as a property
        """
        return self._data.xmat

    def get_body_xpos(self, name):
        """
        Query cartesian position of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xpos (np.ndarray): The xpos value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xpos[bid]

    def get_body_xquat(self, name):
        """
        Query the rotation of a mujoco body in quaternion (in wxyz convention) using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xquat (np.ndarray): The xquat value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xquat[bid]

    def get_body_xmat(self, name):
        """
        Query the rotation of a mujoco body in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xmat (np.ndarray): The xmat value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        return self.xmat[bid].reshape((3, 3))

    def get_body_jacp(self, name):
        """
        Query the position jacobian of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model._model, self._data, jacp, None, bid)
        return jacp

    def get_body_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco body
        """
        bid = self.model.body_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model._model, self._data, None, jacr, bid)
        return jacr

    def get_body_xvelp(self, name):
        """
        Query the translational velocity of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco body.
        """
        jacp = self.get_body_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_body_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco body using a name string.

        Args:
            name (str): The name of a mujoco body
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco body.
        """
        jacr = self.get_body_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_geom_xpos(self, name):
        """
        Query the cartesian position of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            geom_xpos (np.ndarray): The cartesian position of the mujoco body.
        """
        gid = self.model.geom_name2id(name)
        return self.geom_xpos[gid]

    def get_geom_xmat(self, name):
        """
        Query the rotation of a mujoco geom in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            geom_xmat (np.ndarray): The 3x3 rotation matrix of the mujoco geom.
        """
        gid = self.model.geom_name2id(name)
        return self.geom_xmat[gid].reshape((3, 3))

    def get_geom_jacp(self, name):
        """
        Query the position jacobian of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco geom
        """
        gid = self.model.geom_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model._model, self._data, jacp, None, gid)
        return jacp

    def get_geom_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco geom
        """
        gid = self.model.geom_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacGeom(self.model._model, self._data, None, jacr, gid)
        return jacr

    def get_geom_xvelp(self, name):
        """
        Query the translational velocity of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco geom
        """
        jacp = self.get_geom_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_geom_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco geom using a name string.

        Args:
            name (str): The name of a mujoco geom
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco geom
        """
        jacr = self.get_geom_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_site_xpos(self, name):
        """
        Query the cartesian position of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            site_xpos (np.ndarray): The carteisan position of the mujoco site
        """
        sid = self.model.site_name2id(name)
        return self.site_xpos[sid]

    def get_site_xmat(self, name):
        """
        Query the rotation of a mujoco site in a rotation matrix using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            site_xmat (np.ndarray): The 3x3 rotation matrix of the mujoco site.
        """
        sid = self.model.site_name2id(name)
        return self.site_xmat[sid].reshape((3, 3))

    def get_site_jacp(self, name):
        """
        Query the position jacobian of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            jacp (np.ndarray): The jacp value of the mujoco site
        """
        sid = self.model.site_name2id(name)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self._data, jacp, None, sid)
        return jacp

    def get_site_jacr(self, name):
        """
        Query the rotation jacobian of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            jacr (np.ndarray): The jacr value of the mujoco site
        """
        sid = self.model.site_name2id(name)
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model._model, self._data, None, jacr, sid)
        return jacr

    def get_site_xvelp(self, name):
        """
        Query the translational velocity of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            xvelp (np.ndarray): The translational velocity of the mujoco site
        """
        jacp = self.get_site_jacp(name)
        xvelp = np.dot(jacp, self.qvel)
        return xvelp

    def get_site_xvelr(self, name):
        """
        Query the rotational velocity of a mujoco site using a name string.

        Args:
            name (str): The name of a mujoco site
        Returns:
            xvelr (np.ndarray): The rotational velocity of the mujoco site
        """
        jacr = self.get_site_jacr(name)
        xvelr = np.dot(jacr, self.qvel)
        return xvelr

    def get_camera_xpos(self, name):
        """
        Get the cartesian position of a camera using name

        Args:
            name (str): The name of a camera
        Returns:
            cam_xpos (np.ndarray): The cartesian position of a camera
        """
        cid = self.model.camera_name2id(name)
        return self.cam_xpos[cid]

    def get_camera_xmat(self, name):
        """
        Get the rotation of a camera in a rotation matrix using name

        Args:
            name (str): The name of a camera
        Returns:
            cam_xmat (np.ndarray): The 3x3 rotation matrix of a camera
        """
        cid = self.model.camera_name2id(name)
        return self.cam_xmat[cid].reshape((3, 3))

    def get_light_xpos(self, name):
        """
        Get cartesian position of a light source

        Args:
            name (str): The name of a lighting source
        Returns:
            light_xpos (np.ndarray): The cartesian position of the light source
        """
        lid = self.model.light_name2id(name)
        return self.light_xpos[lid]

    def get_light_xdir(self, name):
        """
        Get the direction of a light source using name

        Args:
            name (str): The name of a light
        Returns:
            light_xdir (np.ndarray): The direction vector of the lightsource
        """
        lid = self.model.light_name2id(name)
        return self.light_xdir[lid]

    def get_sensor(self, name):
        """
        Get the data of a sensor using name

        Args:
            name (str): The name of a sensor
        Returns:
            sensordata (np.ndarray): The sensor data vector
        """
        sid = self.model.sensor_name2id(name)
        return self.sensordata[sid]

    def get_mocap_pos(self, name):
        """
        Get the position of a mocap body using name.

        Args:
            name (str): The name of a joint
        Returns:
            mocap_pos (np.ndarray): The current position of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.mocap_pos[mocap_id]

    def set_mocap_pos(self, name, value):
        """
        Set the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint position of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.mocap_pos[mocap_id] = value

    def get_mocap_quat(self, name):
        """
        Get the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
        Returns:
            mocap_quat (np.ndarray): The current quaternion of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        return self.mocap_quat[mocap_id]

    def set_mocap_quat(self, name, value):
        """
        Set the quaternion of a mocap body using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint quaternion of a mocap body.
        """
        body_id = self.model.body_name2id(name)
        mocap_id = self.model.body_mocapid[body_id]
        self.mocap_quat[mocap_id] = value

    def get_joint_qpos(self, name):
        """
        Get the position of a joint using name.

        Args:
            name (str): The name of a joint

        Returns:
            qpos (np.ndarray): The current position of a joint.
        """
        addr = self.model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qpos[addr]
        else:
            start_i, end_i = addr
            return self.qpos[start_i:end_i]

    def set_joint_qpos(self, name, value):
        """
        Set the position of a joint using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint velocity of a joint.
        """
        addr = self.model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.qpos[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (end_i - start_i,), "Value has incorrect shape %s: %s" % (name, value)
            self.qpos[start_i:end_i] = value

    def get_joint_qvel(self, name):
        """
        Get the velocity of a joint using name.

        Args:
            name (str): The name of a joint

        Returns:
            qvel (np.ndarray): The current velocity of a joint.
        """
        addr = self.model.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qvel[addr]
        else:
            start_i, end_i = addr
            return self.qvel[start_i:end_i]

    def set_joint_qvel(self, name, value):
        """
        Set the velocities of a joint using name.

        Args:
            name (str): The name of a joint
            value (float): The desired joint velocity of a joint.
        """
        addr = self.model.get_joint_qvel_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            self.qvel[addr] = value
        else:
            start_i, end_i = addr
            value = np.array(value)
            assert value.shape == (end_i - start_i,), "Value has incorrect shape %s: %s" % (name, value)
            self.qvel[start_i:end_i] = value


class MjSim:
    """
    Meant to somewhat replicate functionality in mujoco-py's MjSim object
    (see https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjsim.pyx).
    """

    def __init__(self, model):
        """
        Args:
            model: should be an MjModel instance created via a factory function
                such as mujoco.MjModel.from_xml_string(xml)
        """
        self.model = MjModel(model)
        self.data = MjData(self.model)

        # offscreen render context object
        self._render_context_offscreen = None

    @classmethod
    def from_xml_string(cls, xml):
        model = mujoco.MjModel.from_xml_string(xml)
        return cls(model)

    @classmethod
    def from_xml_file(cls, xml_file):
        f = open(xml_file, "r")
        xml = f.read()
        f.close()
        return cls.from_xml_string(xml)

    def reset(self):
        """Reset simulation."""
        mujoco.mj_resetData(self.model._model, self.data._data)

    def forward(self):
        """Forward call to synchronize derived quantities."""
        mujoco.mj_forward(self.model._model, self.data._data)

    def step(self, with_udd=True):
        """Step simulation."""
        mujoco.mj_step(self.model._model, self.data._data)

    def step1(self):
        """Step1 (before actions are set)."""
        mujoco.mj_step1(self.model._model, self.data._data)

    def step2(self):
        """Step2 (after actions are set)."""
        mujoco.mj_step2(self.model._model, self.data._data)

    def render(
        self,
        width=None,
        height=None,
        *,
        camera_name=None,
        depth=False,
        mode="offscreen",
        device_id=-1,
        segmentation=False,
    ):
        """
        Renders view from a camera and returns image as an `numpy.ndarray`.
        Args:
        - width (int): desired image width.
        - height (int): desired image height.
        - camera_name (str): name of camera in model. If None, the free
            camera will be used.
        - depth (bool): if True, also return depth buffer
        - device (int): device to use for rendering (only for GPU-backed
            rendering).
        Returns:
        - rgb (uint8 array): image buffer from camera
        - depth (float array): depth buffer from camera (only returned
            if depth=True)
        """
        if camera_name is None:
            camera_id = None
        else:
            camera_id = self.model.camera_name2id(camera_name)

        assert mode == "offscreen", "only offscreen supported for now"
        assert self._render_context_offscreen is not None
        with _MjSim_render_lock:
            self._render_context_offscreen.render(
                width=width, height=height, camera_id=camera_id, segmentation=segmentation
            )
            return self._render_context_offscreen.read_pixels(width, height, depth=depth, segmentation=segmentation)

    def add_render_context(self, render_context):
        assert render_context.offscreen
        if self._render_context_offscreen is not None:
            # free context
            del self._render_context_offscreen
        self._render_context_offscreen = render_context

    def get_state(self):
        """Return MjSimState instance for current state."""
        return MjSimState(
            time=self.data.time,
            qpos=np.copy(self.data.qpos),
            qvel=np.copy(self.data.qvel),
        )

    def set_state(self, value):
        """
        Set internal state from MjSimState instance. Should
        call @forward afterwards to synchronize derived quantities.
        """
        self.data.time = value.time
        self.data.qpos[:] = np.copy(value.qpos)
        self.data.qvel[:] = np.copy(value.qvel)

    def set_state_from_flattened(self, value):
        """
        Set internal mujoco state using flat mjstate array. Should
        call @forward afterwards to synchronize derived quantities.

        See https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/mjsimstate.pyx#L54
        """
        state = MjSimState.from_flattened(value, self)

        # do this instead of @set_state to avoid extra copy of qpos and qvel
        self.data.time = state.time
        self.data.qpos[:] = state.qpos
        self.data.qvel[:] = state.qvel

    def free(self):
        # clean up here to prevent memory leaks
        del self._render_context_offscreen
        del self.data
        del self.model
        del self
        gc.collect()
