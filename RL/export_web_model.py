"""Export the neutral assembled STLv2 MJCF pose as a web-ready GLB.

From the repository root:
  python RL/export_web_model.py
"""
import argparse
import json
import os
import struct

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATERIALS = [
    {"name": "Porcelain white", "pbrMetallicRoughness": {"baseColorFactor": [0.86, 0.86, 0.86, 1], "metallicFactor": 0.0, "roughnessFactor": 0.62}},
    {"name": "Graphite black", "pbrMetallicRoughness": {"baseColorFactor": [0.12, 0.12, 0.12, 1], "metallicFactor": 0.05, "roughnessFactor": 0.58}},
    {"name": "Brushed grey", "pbrMetallicRoughness": {"baseColorFactor": [0.48, 0.48, 0.48, 1], "metallicFactor": 0.08, "roughnessFactor": 0.5}},
]


def material_for(name):
    if "Foot" in name:
        return 1
    if any(part in name for part in ("HipJoint", "ShoulderJoint", "LowerThigh", "LowerUpperArm", "Shin")):
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default=os.path.join(ROOT, "RL", "STLv2_13in.xml"))
    parser.add_argument("--output", default=os.path.join(ROOT, "assets", "mini-humanoid.glb"))
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    meshes = []
    world_points = []
    # MuJoCo visual meshes are already scaled and centered in mesh_vert. Apply each geom's
    # neutral-pose transform, then convert MuJoCo Z-up coordinates to glTF Y-up.
    z_up_to_y_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        vert_start, vert_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[vert_start:vert_start + vert_count].astype(np.float64, copy=True)
        faces = model.mesh_face[face_start:face_start + face_count].astype(np.uint32, copy=True)
        normal_start = int(model.mesh_normaladr[mesh_id])
        normals = model.mesh_normal[normal_start:normal_start + vert_count].astype(np.float64, copy=True)
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
        vertices = vertices @ z_up_to_y_up.T
        normals = normals @ rotation.T @ z_up_to_y_up.T
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"part_{geom_id}"
        meshes.append((geom_name, vertices, normals, faces, material_for(geom_name)))
        world_points.append(vertices)

    if not meshes:
        raise SystemExit(f"No STL-backed mesh geoms found in {args.xml}")

    all_points = np.concatenate(world_points, axis=0)
    bounds_min, bounds_max = all_points.min(axis=0), all_points.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    binary = bytearray()
    buffer_views = []
    accessors = []
    gltf_meshes = []
    nodes = []

    def append_aligned(array, target):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        raw = np.ascontiguousarray(array).tobytes()
        binary.extend(raw)
        view_id = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        return view_id

    for name, vertices, normals, faces, material_id in meshes:
        vertices = (vertices - center).astype("<f4")
        normals = normals.astype("<f4")
        faces = faces.astype("<u4", copy=False).reshape(-1)
        pos_view = append_aligned(vertices, 34962)
        pos_accessor = len(accessors)
        accessors.append({"bufferView": pos_view, "componentType": 5126, "count": len(vertices), "type": "VEC3",
                          "min": vertices.min(axis=0).astype(float).tolist(), "max": vertices.max(axis=0).astype(float).tolist()})
        normal_view = append_aligned(normals, 34962)
        normal_accessor = len(accessors)
        accessors.append({"bufferView": normal_view, "componentType": 5126, "count": len(normals), "type": "VEC3"})
        index_view = append_aligned(faces, 34963)
        index_accessor = len(accessors)
        accessors.append({"bufferView": index_view, "componentType": 5125, "count": len(faces), "type": "SCALAR"})
        gltf_meshes.append({"name": name, "primitives": [{"attributes": {"POSITION": pos_accessor, "NORMAL": normal_accessor},
                           "indices": index_accessor, "material": material_id, "mode": 4}]})
        nodes.append({"name": name, "mesh": len(gltf_meshes) - 1})

    document = {
        "asset": {"version": "2.0", "generator": "MiniHumanoid MuJoCo/STL web exporter",
                  "extras": {"source": os.path.basename(args.xml), "pose": "neutral", "units": "meters"}},
        "scene": 0,
        "scenes": [{"name": "MiniHumanoid neutral pose", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": MATERIALS,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    binary.extend(b"\0" * ((4 - len(binary) % 4) % 4))
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    glb = struct.pack("<4sII", b"glTF", 2, total_length)
    glb += struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
    glb += struct.pack("<I4s", len(binary), b"BIN\0") + binary

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "wb") as stream:
        stream.write(glb)
    print(f"Exported {len(meshes)} assembled STL parts to {output} ({len(glb) / 1024 / 1024:.2f} MiB)")
    print("Neutral model size (XYZ):", np.round(bounds_max - bounds_min, 3).tolist(), "meters")


if __name__ == "__main__":
    main()
