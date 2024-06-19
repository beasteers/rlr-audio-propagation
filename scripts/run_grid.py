import pyglet
pyglet.options["headless"] = True
import os
import tqdm
import numpy as np
from rlr_audio_propagation import Context, Config, ChannelLayout, ChannelLayoutType

import trimesh
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf



def plot_wav(fname, out_fname):
    y, sr = librosa.load(fname, mono=False)
    plt.figure(figsize=(10, 6))
    for i, yi in enumerate(y):
        # print((yi!=0).sum(), yi.shape)
        ix=np.where(yi!=0)[0]
        if len(ix)>1:
            yi = yi[ix[0]:ix[-1]]
        librosa.display.waveshow(yi, sr=sr, alpha=0.5, label=f'Channel {i} ({np.abs(yi).max():.2g})')
    plt.legend()
    plt.savefig(out_fname)
    return y


def add_sphere(scene, pos, color=[0,0,0], r=0.4, **kw):
    sphere = trimesh.creation.uv_sphere(radius=r)
    sphere.apply_translation(pos)
    sphere.visual.face_colors = color
    name = scene.add_geometry(sphere)
    return sphere, name


# def draw_3d(fname, out_fname, mic_pos, source_pos):
#     scene = trimesh.load(fname)
#     add_sphere(scene, mic_pos,[255, 0, 0])
#     add_sphere(scene, source_pos,[0, 255, 0])
#     with open(out_fname, 'wb') as f:
#         f.write(scene.save_image())
def draw_3d(scene, out_fname, mic_pos, source_pos):
    mic_sphere, mic_name = add_sphere(scene, mic_pos,[255, 0, 0])
    source_sphere, source_name = add_sphere(scene, source_pos,[0, 255, 0])
    with open(out_fname, 'wb') as f:
        f.write(scene.save_image((100, 100)))
    scene.delete_geometry([mic_name, source_name])


def draw_grid(scene, xs, ys, zs, out_fname):
    spheres = []
    for x in xs:
        for y in ys:
            for z in zs:
                _, name = add_sphere(scene, [x, y, z], [0, 0, 255])
                spheres.append(name)
    with open(out_fname, 'wb') as f:
        f.write(scene.save_image())
    scene.delete_geometry(spheres)

MP3D_GLB="/datasets/soundspaces/scene_datasets/mp3d_example/17DRP5sb8fy/17DRP5sb8fy.glb"
GIBSON_GLB="/datasets/soundspaces/scene_datasets/gibson_data/gibson/Oyens.glb"

def main(glb_file=GIBSON_GLB, ch=4, n=12):
    ply_file = glb_file.rsplit('.', 1)[0] + '_semantic.ply'
    name = os.path.basename(glb_file).rsplit('.', 1)[0]
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    mic_pos = np.array([-4, 1, -2])
    source_pos = np.array([0, 1.5, -2])
    mic_pos = np.array([0, 0, 1])
    source_pos = np.array([0, 2, 1.5])

    cfg = Config()
    ctx = Context(cfg)
    ctx.add_source()
    ctx.add_listener(ChannelLayout(ChannelLayoutType.Ambisonics, ch))
    ctx.add_object()
    ctx.set_object_position(0, [0, 0, 0])
    ctx.set_object_mesh_obj(0, glb_file, "")
    ctx.set_object_mesh_ply(0, ply_file, "")

    scene = trimesh.load(glb_file)
    # rotation_matrix = trimesh.transformations.rotation_matrix(-np.radians(90), [1, 0, 0])
    # scene.apply_transform(rotation_matrix)
    for k,g in scene.geometry.items():
        scene.geometry[k].visual = g.visual.to_color()
        scene.geometry[k].visual.face_colors = [255, 235, 255]
    scene.add_geometry(trimesh.creation.axis(axis_length=1))

    xs = np.linspace(-14, 8, n)
    ys = np.linspace(-4, 8, n)
    zs = np.linspace(0, 2, n)
    # ys = [1.5]
    zs = [1.5]

    draw_grid(scene, xs, ys, zs, f'{out_dir}/3d_all_{name}.png')

    mic_pos=np.array([0, 1, 1])
    ineff = np.zeros((len(xs), len(ys), len(zs)))
    ymax = np.zeros((len(xs), len(ys), len(zs), ch))
    for i, x in enumerate(tqdm.tqdm(xs)):
        for j, y in enumerate(tqdm.tqdm(ys)):
            for k, z in enumerate(tqdm.tqdm(zs)):
                name_i = f'{name}_{x:.2f}_{y:.2f}_{z:.2f}'
                source_pos = np.array([x, y, z])
                ctx.set_source_position(0, source_pos)
                ctx.set_listener_position(0, mic_pos)
                ctx.simulate()
                # write to file
                ctx.write_ir_wave(0, 0, f'{out_dir}/{name_i}.wav')
                # ctx.write_ir_metrics(0, 0, f'{out_dir}/{name_i}')
                # draw
                audio=plot_wav(f'{out_dir}/{name_i}.wav', f'{out_dir}/wavplt_{name_i}.png')
                # draw_3d(glb_file, f'{out_dir}/{name_i}_3d.png', mic_pos, source_pos)
                draw_3d(scene, f'{out_dir}/3d_{name_i}.png', mic_pos, source_pos)
                ineff[i, j, k] = ctx.get_indirect_ray_efficiency()
                ymax[i, j, k] = audio.max()

    plt.figure(figsize=(10, 10))
    plt.imshow(ineff.squeeze().T)
    plt.savefig(os.path.join(out_dir, 'rayineff_grid.png'))
    plt.close()
    ymax = ymax.squeeze().T
    for i in range(ymax.shape[-1]):
        plt.figure(figsize=(10, 10))
        plt.imshow(ymax[:,:,i].T)
        plt.xticks(range(len(xs)), xs)
        plt.yticks(range(len(ys)), ys)
        plt.savefig(os.path.join(out_dir, 'ymax_grid.png'))
        plt.close()

if __name__ == '__main__':
    import fire
    fire.Fire(main)