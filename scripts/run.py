import os
import numpy as np
from rlr_audio_propagation import Context, Config, ChannelLayout, ChannelLayoutType


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
    ibo = np.vstack(ibo_list)

    return SceneMesh(vbo=vbo, nbo=nbo, ibo=ibo)


def render(scene, ctx):
    pass

def plot(fname):
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    y, sr = librosa.load(fname, mono=False)
    plt.figure(figsize=(10, 6))
    for i, yi in enumerate(y):
        print((yi!=0).sum(), yi.shape)
        # ix=np.where(yi!=0)[0]
        # if len(ix)>1:
        #     yi = yi[ix[0]:ix[-1]]
        # print(ix)
        # yi = yi[:200]

        librosa.display.waveshow(yi, sr=sr, alpha=0.5, label=f'Channel {i} ({yi.max():.2f})')
    plt.legend()
    plt.savefig(f'ir.png')

# import ipdb
# @ipdb.iex
def main(glb_file='/datasets/soundspaces/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb', ply_file=None, out_dir='output'):
    ply_file = ply_file or glb_file.rsplit('.', 1)[0] + '_semantic.ply'
    name = os.path.basename(glb_file).rsplit('.', 1)[0]

    cfg = Config()
    print(cfg)
    ctx = Context(cfg)
    ctx.add_source()
    ctx.set_source_position(0, [0, 1.5, 0])
    ctx.set_source_radius(0, 0.1)

    ctx.add_listener(ChannelLayout(ChannelLayoutType.Ambisonics, 4))
    ctx.set_listener_position(0, [0, 1, 0])
    ctx.set_listener_radius(0, 0.1)

    ctx.add_object()
    ctx.set_object_position(0, [0, 0, 0])
    ctx.set_object_mesh_obj(0, glb_file, "")
    ctx.set_object_mesh_ply(0, ply_file, "")

    ctx.simulate()

    os.makedirs(out_dir, exist_ok=True)
    ctx.write_ir_wave(0, 0, f'{out_dir}/{name}.wav')
    ctx.write_ir_metrics(0, 0, f'{out_dir}/{name}')
    plot(f'{out_dir}/{name}.wav')
    
    import trimesh
    # scene = trimesh.load('/datasets/soundspaces/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb')
    # mesh = create_scene_mesh_from_trimesh(scene)
    

    
    # ctx.add_object()
    # ctx.set_object_position(0, [0, 0, 0])
    # ctx.set_object_mesh_obj(0, "/datasets/soundspaces/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb")
    # ctx.set_object_mesh_ply(0, "/datasets/soundspaces/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy_semantic.ply")
    # ctx.finalize_object_mesh(0)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
