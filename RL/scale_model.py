"""Scale the STLv2 MJCF model to a target height (default 13 in)."""
import sys, xml.etree.ElementTree as ET

SRC, DST = "STLv2.xml", "STLv2_13in.xml"
ORIG_HEIGHT_M = 3.412          # measured z-span of the original meshes
TARGET_HEIGHT_M = 13 * 0.0254  # 13 inches
DENSITY_RATIO = 1.0            # new material density / steel (7850). PLA ~ 0.16
SPAWN_Z = None                 # None = auto (feet just above floor)

s = TARGET_HEIGHT_M / ORIG_HEIGHT_M   # length scale
m = s**3 * DENSITY_RATIO              # mass scale
i = m * s**2                          # inertia scale (mass * length^2)
f = lambda txt, k: " ".join(repr(float(x) * k) for x in txt.split())

tree = ET.parse(SRC); root = tree.getroot()
for e in root.iter("mesh"):
    e.set("scale", f(e.get("scale"), s))
for tag in ("body", "geom", "joint", "inertial"):
    for e in root.iter(tag):
        if e.get("pos"): e.set("pos", f(e.get("pos"), s))
for e in root.iter("inertial"):
    e.set("mass", repr(float(e.get("mass")) * m))
    e.set("fullinertia", f(e.get("fullinertia"), i))

# spawn height: original feet were 2.1 m below base origin at spawn z=2 (-0.1 m lowest point)
spawn = next(b for b in root.iter("body") if b.get("name") == "spawn")
z_feet_below = 2.1 * s
spawn.set("pos", "0 0 %r" % (SPAWN_Z if SPAWN_Z else z_feet_below + 0.01))

# timestep can stay; tiny robot is stiffer in time, so tighten a bit
root.find("option").set("timestep", "0.001")
ET.indent(tree); tree.write(DST, encoding="utf-8", xml_declaration=True)
print(f"length scale {s:.5f} -> mesh scale {0.001*s:.3e}, mass x{m:.3e}, inertia x{i:.3e}")
