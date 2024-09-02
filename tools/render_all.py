import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("glb_path", type=str, help="Path to the glb file")
    parser.add_argument("blender36_exec_path", type=str, help="Path to the blender 3.6 executable")
    return parser.parse_args()

args = parse_args()
GLB_DIR = args.glb_path
# position to your blender executable, for example: xx/blender-3.6.4-linux-x64/blender
BLENDER_EXEC = args.blender36_exec_path

for file in os.listdir(GLB_DIR):
    if file.endswith(".glb"):
        glb_path = os.path.join(GLB_DIR, file)
        filename = file.split(".")[0]
        for render_type in ["mesh", "view"]:
            cmd = f"{BLENDER_EXEC} -b \
            utils/render_one.blend -P \
            utils/render_one.py {glb_path} {render_type}"
            os.system(cmd)