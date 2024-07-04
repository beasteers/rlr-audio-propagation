import numpy as np
import trimesh
import struct

def create_ply(obj_file, ids_file, out_file):
    # Load the OBJ file
    mesh = trimesh.load(obj_file)
    if mesh is None:
        print(f"Failed to load {obj_file}")
        return 1

    # Load the object IDs
    with open(ids_file, 'rb') as f:
        object_id = np.fromfile(f, dtype=np.uint16)
    
    num_verts = len(mesh.vertices)

    with open(out_file, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {num_verts}\n'.encode())
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property uchar red\n')
        f.write(b'property uchar green\n')
        f.write(b'property uchar blue\n')
        f.write(f'element face {len(mesh.faces)}\n'.encode())
        f.write(b'property list uchar int vertex_indices\n')
        f.write(b'property ushort object_id\n')
        f.write(b'end_header\n')

        # Rotate to match .glb where -Z is gravity
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.radians(-90), [1, 0, 0])
        mesh.apply_transform(rotation_matrix)

        gray = struct.pack('BBB', 0x80, 0x80, 0x80)

        for vertex in mesh.vertices:
            f.write(struct.pack('fff', *vertex))
            f.write(gray)

        for face, obj_id in zip(mesh.faces, object_id):
            f.write(struct.pack('B', len(face)))
            for idx in face:
                f.write(struct.pack('i', idx))
            f.write(struct.pack('H', obj_id))


if __name__ == "__main__":
    import fire
    fire.Fire(create_ply)