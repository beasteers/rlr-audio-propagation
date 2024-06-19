import os
import ctypes
import numpy as np
import _rlr_audio_propagation_v1 as RLRAudioPropagation
from _rlr_audio_propagation_v1 import Simulator, Vector3f, Quaternion, Configuration, ChannelLayout, as_vertex_data, as_index_data, IndexData

class AudioSensorSpec:
    def __init__(self):
        self.uuid = "audio"
        self.sensorType = "Audio"
        self.sensorSubType = "ImpulseResponse"
        self.acousticsConfig_ = Configuration()  # Initialize with Configuration
        self.channelLayout_ = ChannelLayout()
        self.outputDirectory_ = "/home/AudioSimulation"

    def sanity_check(self):
        assert self.sensorType == "Audio", "sensorType must be Audio"
        assert self.sensorSubType == "ImpulseResponse", "sensorSubType must be Audio"




import dataclasses
@dataclasses.dataclass
class SceneMesh:
    # Vertex positions
    vbo: np.ndarray
    #  Vertex normals
    nbo: np.ndarray
    # # Texture coordinates
    # tbo: np.ndarray
    # # Vertex colors
    # cbo: np.ndarray
    # Index buffer
    ibo: np.ndarray


def create_scene_mesh_from_trimesh(scene):
    vbo_list = []
    nbo_list = []
    ibo_list = []
    index_offset = 0

    for geom in scene.geometry.values():
        vbo_list.append(geom.vertices)
        nbo_list.append(geom.vertex_normals)
        faces = geom.faces + index_offset
        ibo_list.append(faces)
        index_offset += len(geom.vertices)
    
    vbo = np.vstack(vbo_list)
    nbo = np.vstack(nbo_list)
    ibo = np.vstack(ibo_list).flatten()

    return SceneMesh(vbo=vbo, nbo=nbo, ibo=ibo)

# import pybind11

# def as_vertex_data(vertices):
#     print(vertices.shape, vertices.dtype)
#     vertex_data = RLRAudioPropagation.VertexData()
#     vertex_data.vertices = vertices #ctypes.addressof(ctypes.c_float.from_buffer(vertices))
#     vertex_data.byteOffset = 0
#     vertex_data.vertexCount = len(vertices) // 3  # Assuming 3 floats per vertex
#     vertex_data.vertexStride = 0  # Assuming tightly packed vertices
#     return vertex_data

# def as_index_data(indices):
#     index_data = RLRAudioPropagation.IndexData()
#     index_data.indices = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
#     index_data.byteOffset = 0
#     index_data.indexCount = len(indices)
#     return index_data

class AudioSensor:
    def __init__(self, node, spec):
        self.node = node
        self.audioSensorSpec_ = spec
        self.audioSensorSpec_.sanity_check()
        self.audioSimulator_ = None
        self.impulseResponse_ = []
        self.buffer_ = None
        self.currentSimCount_ = 0
        self.newInitialization_ = False
        self.newSource_ = False
        self.lastSourcePos_ = None
        self.lastAgentPos_ = Vector3f(float('-inf'), float('-inf'), float('-inf'))
        self.lastAgentRot_ = None
        self.audioMaterialsJsonSet_ = False

    def reset(self):
        self.audioSimulator_ = None
        self.impulseResponse_.clear()

    def set_audio_source_transform(self, source_pos):
        self.lastSourcePos_ = source_pos
        self.newSource_ = True

    def set_audio_listener_transform(self, agent_pos, agent_rot_quat):
        self.create_audio_simulator()
        if self.newInitialization_ or self.lastAgentPos_ != agent_pos or self.lastAgentRot_ != agent_rot_quat:
            self.audioSimulator_.AddListener(
                Vector3f(agent_pos[0], agent_pos[1], agent_pos[2]),
                Quaternion(agent_rot_quat[0], agent_rot_quat[1], agent_rot_quat[2], agent_rot_quat[3]),
                self.audioSensorSpec_.channelLayout_
            )

    def run_simulation(self, mesh):
        if self.newInitialization_:
            self.newInitialization_ = False
            # , self.audioSensorSpec_.acousticsConfig_.enableMaterials
            self.load_mesh(mesh)

        if self.newSource_:
            self.newSource_ = False
            self.audioSimulator_.AddSource(Vector3f(self.lastSourcePos_[0], self.lastSourcePos_[1], self.lastSourcePos_[2]))

        sim_folder = self.get_simulation_folder()
        self.audioSimulator_.RunSimulation(sim_folder)
        self.impulseResponse_.clear()

    def set_audio_materials_json(self, json_path):
        if self.audioMaterialsJsonSet_:
            return
        self.audioMaterialsJsonSet_ = True
        self.audioSimulator_.LoadAudioMaterialJSON(json_path)

    def get_ir(self):
        if not self.impulseResponse_:
            obs_space = self.get_observation_space()
            self.impulseResponse_ = np.zeros((obs_space["shape"][0], obs_space["shape"][1]))
            for channel_index in range(obs_space["shape"][0]):
                ir = self.audioSimulator_.GetImpulseResponseForChannel(channel_index)
                self.impulseResponse_[channel_index, :] = ir
        return self.impulseResponse_

    def get_observation(self, sim, obs):
        if not self.audioSimulator_:
            return False

        obs_space = self.get_observation_space()
        if obs_space["shape"][0] == 0 or obs_space["shape"][1] == 0:
            return False

        if self.buffer_ is None:
            self.buffer_ = np.zeros((obs_space["shape"][0], obs_space["shape"][1]), dtype=np.float32)

        for channel_index in range(obs_space["shape"][0]):
            ir = self.audioSimulator_.GetImpulseResponseForChannel(channel_index)
            self.buffer_[channel_index, :] = ir

        obs["buffer"] = self.buffer_

        if self.audioSensorSpec_.acousticsConfig_.writeIrToFile:
            self.write_ir_file(obs)

        return True

    def get_observation_space(self):
        obs_space = {"spaceType": "Tensor", "shape": [0, 0], "dataType": "DT_FLOAT"}
        if self.audioSimulator_:
            obs_space["shape"] = [self.audioSimulator_.GetChannelCount(), self.audioSimulator_.GetSampleCount()]
        return obs_space

    def display_observation(self, sim):
        raise NotImplementedError("Display observation for audio sensor is not used.")

    def create_audio_simulator(self):
        self.currentSimCount_ += 1
        if self.audioSimulator_:
            return
        self.newInitialization_ = True
        self.audioSimulator_ = Simulator()
        self.audioSimulator_.Configure(self.audioSensorSpec_.acousticsConfig_)

    def load_semantic_mesh(self, scene_mesh):
        vertices = {"vertices": scene_mesh.vbo, "byteOffset": 0, "vertexCount": len(scene_mesh.vbo), "vertexStride": 0}
        self.audioSimulator_.LoadMeshVertices(vertices)

        category_name_to_indices = {}
        for ibo_idx in range(0, len(scene_mesh.ibo), 3):
            cat = self.get_category(ibo_idx)
            if cat not in category_name_to_indices:
                category_name_to_indices[cat] = []
            category_name_to_indices[cat].extend(scene_mesh.ibo[ibo_idx:ibo_idx + 3])

        for cat, indices in category_name_to_indices.items():
            index_data = {"indices": indices, "byteOffset": 0, "indexCount": len(indices)}
            self.audioSimulator_.LoadMeshIndices(index_data, cat)

        self.audioSimulator_.UploadMesh()

    def get_category(self, ibo_idx):
        cat1 = self.objects.get(ibo_idx[0], 'default')
        cat2 = self.objects.get(ibo_idx[1], 'default')
        cat3 = self.objects.get(ibo_idx[2], 'default')
        cat = None
        if len({cat1, cat2, cat3}) == 1:
            # If all 3 categories are the same, save the indices to cat1 in the categoryNameToIndices map
            cat = cat1
        elif cat1 not in {cat2, cat3}:
            # If cat1 != 2 and 3
            # then either all 3 are different, or cat1 is different while 2==3
            # Either case, use 1
            # reason : if all 3 are different, we cant determine which one is correct
            # if this is the odd one out, then the triangle is actually of this
            # category
            cat = cat1
        elif cat1 == cat2:
            cat = cat3
        else:
            cat = cat2
        return cat

    def load_mesh(self, scene_mesh):
        v = as_vertex_data(scene_mesh.vbo.astype(np.float32))
        print(v)
        i = as_index_data(scene_mesh.ibo)
        print(i)
        self.audioSimulator_.LoadMeshData(v, i)
        print("Loaded")

    def get_simulation_folder(self):
        return self.audioSensorSpec_.outputDirectory_ + str(self.currentSimCount_)

    def write_ir_file(self, obs):
        folder_path = self.get_simulation_folder()
        for channel_index in range(obs["buffer"].shape[0]):
            file_path = os.path.join(folder_path, f"ir{channel_index}.txt")
            np.savetxt(file_path, obs["buffer"][channel_index])


if __name__ == '__main__':
    import trimesh
    glb_file='/datasets/soundspaces/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb'
    glb_file="/datasets/soundspaces/scene_datasets/gibson_data/gibson/Oyens.glb"
    scene = trimesh.load(glb_file)
    mesh = create_scene_mesh_from_trimesh(scene)
    spec = AudioSensorSpec()
    sensor = AudioSensor(None, spec)
    sensor.set_audio_source_transform(Vector3f(1.0, 2.0, 3.0))
    sensor.set_audio_listener_transform(Vector3f(1.0, 2.0, 3.0), Quaternion(0.0, 0.0, 0.0, 1.0))
    sensor.run_simulation(mesh)
    ir = sensor.get_ir()