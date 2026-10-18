"""Take the scaled model (scaled_only.xml) and add: defaults, visual/collision split,
contact excludes, position actuators, neutral keyframe. Writes STLv2_13in.xml."""
import numpy as np, mujoco, xml.etree.ElementTree as ET

SRC, DST = "scaled_only.xml", "STLv2_13in.xml"
# ---- tunables (placeholders until servo specs are known) ----
KP, KV, FORCE = 30.0, 0.6, 3.0           # N*m/rad, N*m*s/rad, +/- N*m torque limit
DAMPING, ARMATURE, FRICTIONLOSS = 0.05, 0.002, 0.01
BOX_BODIES = {"base_link_1", "Torso_1", "LeftFoot_1", "RightFoot_1"}

m = mujoco.MjModel.from_xml_path(SRC); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
name = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
fmt = lambda a: " ".join(f"{x:.6g}" for x in np.atleast_1d(a))

def body_frame_verts(g):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return m.geom_pos[g] + v @ R.T

tree = ET.parse(SRC); root = tree.getroot()
bodies = {b.get("name"): b for b in root.iter("body")}
info = []
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
    b = m.geom_bodyid[g]; bn = name(b); V = body_frame_verts(g); el = bodies[bn]
    if bn in BOX_BODIES:   # world-aligned box (in neutral pose) so foot soles are flat
        Rb = d.xmat[b].reshape(3, 3); Vw = V @ Rb.T
        lo, hi = Vw.min(0), Vw.max(0)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, Rb.T.flatten())
        ET.SubElement(el, "geom", {"name": bn + "_col", "class": "collision", "type": "box",
            "pos": fmt(Rb.T @ ((lo + hi) / 2)), "quat": fmt(q), "size": fmt((hi - lo) / 2)})
        info.append((bn, "box", (hi - lo)))
    else:                  # capsule along principal axis
        c = V.mean(0); w, U = np.linalg.eigh(np.cov((V - c).T)); ax = U[:, -1]
        t = (V - c) @ ax; perp = np.linalg.norm((V - c) - np.outer(t, ax), axis=1)
        r = float(np.percentile(perp, 85)); mid_t = (t.max() + t.min()) / 2
        hl = max((t.max() - t.min()) / 2 - r, 1e-4); c0 = c + ax * mid_t
        ET.SubElement(el, "geom", {"name": bn + "_col", "class": "collision", "type": "capsule",
            "fromto": fmt(np.r_[c0 - ax * hl, c0 + ax * hl]), "size": f"{r:.6g}"})
        info.append((bn, "capsule", (r, 2 * (hl + r))))

# visual geoms: class only, no collision
for gel in root.iter("geom"):
    if gel.get("mesh"):
        gel.set("class", "visual")
        for a in ("contype", "conaffinity"): gel.attrib.pop(a, None)
for jel in root.iter("joint"): jel.set("class", "hinge")

# defaults
dflt = ET.Element("default")
ET.SubElement(dflt, "position", {"kp": str(KP), "kv": str(KV), "forcerange": f"-{FORCE} {FORCE}"})
c = ET.SubElement(dflt, "default", {"class": "hinge"})
ET.SubElement(c, "joint", {"damping": str(DAMPING), "armature": str(ARMATURE), "frictionloss": str(FRICTIONLOSS)})
c = ET.SubElement(dflt, "default", {"class": "visual"})
ET.SubElement(c, "geom", {"contype": "0", "conaffinity": "0", "group": "1"})
c = ET.SubElement(dflt, "default", {"class": "collision"})
ET.SubElement(c, "geom", {"group": "3", "contype": "1", "conaffinity": "1", "condim": "3",
                          "friction": "1 0.005 0.0001", "rgba": "0.9 0.3 0.3 0.3"})
root.insert(2, dflt)
opt = root.find("option"); opt.set("integrator", "implicitfast"); opt.set("timestep", "0.001")

# actuators (one position servo per hinge, ctrlrange = joint range)
act = ET.SubElement(root, "actuator")
hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
for j in hinges:
    jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j); lo, hi = m.jnt_range[j]
    ET.SubElement(act, "position", {"name": jn, "joint": jn, "ctrlrange": f"{lo:.6g} {hi:.6g}"})

# keyframe: neutral pose (all joints 0) at spawn height
kf = ET.SubElement(root, "keyframe")
ET.SubElement(kf, "key", {"name": "neutral", "qpos": fmt(m.qpos0), "ctrl": fmt(np.zeros(len(hinges)))})
ET.indent(tree); tree.write("stage1.xml", encoding="utf-8", xml_declaration=True)

# ---- contact excludes: tree distance <= 2, plus anything touching at neutral ----
m1 = mujoco.MjModel.from_xml_path("stage1.xml"); d1 = mujoco.MjData(m1)
par = m1.body_parentid
def anc(b):
    out = [b]
    while b != 0: b = par[b]; out.append(b)
    return out
def tdist(a, b):
    A, B = anc(a), anc(b); common = next(x for x in A if x in B)
    return A.index(common) + B.index(common)
excl = set()
for a in range(1, m1.nbody):
    for b in range(a + 1, m1.nbody):
        if tdist(a, b) <= 2: excl.add((a, b))
mujoco.mj_forward(m1, d1)
floor = mujoco.mj_name2id(m1, mujoco.mjtObj.mjOBJ_GEOM, "floor") if False else -1
def selfcontacts(dd):
    out = []
    for k in range(dd.ncon):
        c = dd.contact[k]; b1, b2 = m1.geom_bodyid[c.geom1], m1.geom_bodyid[c.geom2]
        if b1 and b2 and b1 != b2: out.append(tuple(sorted((b1, b2))))
    return out
neutral = set(selfcontacts(d1)); excl |= neutral
for a, b in sorted(excl):
    pass
con = ET.SubElement(root, "contact")
for a, b in sorted(excl):
    ET.SubElement(con, "exclude", {"body1": mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_BODY, a),
                                   "body2": mujoco.mj_id2name(m1, mujoco.mjtObj.mjOBJ_BODY, b)})
ET.indent(tree); tree.write(DST, encoding="utf-8", xml_declaration=True)
print("collision shapes:")
for n, k, s in info: print(f"  {n:22s} {k:8s} {np.round(np.array(s)*1000,1)} mm")
print("excluded pairs:", len(excl), "| neutral-pose self contacts:", len(neutral))
