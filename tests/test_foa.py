import trimesh
import numpy as np
from rlr_audio_propagation import Config, Context, ChannelLayout, ChannelLayoutType, AMBISONICS_XYZ_INDEX


def test_generation(height=6.0, width=6.0, depth=6.0):
    extent = np.array([width, depth, height])
    listener_position = extent / 2

    mesh = trimesh.creation.box(extents=extent)
    mesh.invert()

    # create audio context
    cfg = Config()
    ctx = Context(cfg)

    # set object
    ctx.add_object()
    ctx.add_mesh_vertices(mesh.vertices.flatten().tolist())
    ctx.add_mesh_indices(mesh.faces.flatten().tolist(), 3, "default")
    ctx.finalize_object_mesh(0)

    # set listener position
    ctx.add_listener(ChannelLayout(ChannelLayoutType.Ambisonics, 4))
    ctx.set_listener_position(0, listener_position)

    # set source position
    sources = np.array([
        + np.array([1, 0, 0]),
        - np.array([1, 0, 0]),
        + np.array([0, 1, 0]),
        - np.array([0, 1, 0]),
        + np.array([0, 0, 1]),
        - np.array([0, 0, 1]),
    ])
    source_positions = sources + listener_position
    for i, source_position in enumerate(source_positions):
        ctx.add_source()
        ctx.set_source_position(i, source_position)

    # render audio
    ctx.simulate()
    print(ctx.get_listener_count(), ctx.get_source_count())
    print(ctx.get_channel_counts())

    # get audio
    audio = ctx.get_audio()
    print(audio.shape)
    assert audio.shape[:-1] == (1, len(sources), 4)

    # check principal direction matches source position
    for i, source_true in enumerate(sources):
        source_pred = principal_direction(audio[0, i])
        source_pred = source_pred[AMBISONICS_XYZ_INDEX]
        print(i, source_true, source_pred)
        assert (source_true == source_pred).all()


def principal_direction(x):
    peaks = unsigned_peak(x)
    top_ch = np.argmax(np.abs(peaks))
    source_pred = np.zeros(len(x), dtype=int)
    source_pred[top_ch] = 1 if peaks[top_ch] >= 0 else -1
    return source_pred


def unsigned_peak(audio):
    return audio[np.arange(len(audio)), np.argmax(np.abs(audio), axis=-1)]