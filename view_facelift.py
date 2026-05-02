import open3d as o3d
from pathlib import Path

ply_path = Path(r"D:\Ishank\Face3D_AutoMask_Outpaint\3_facelift_output\clean_outpaint\gaussians.ply")

if not ply_path.exists():
    raise FileNotFoundError(f"PLY not found: {ply_path}")

print("Loading model...")
pcd = o3d.io.read_point_cloud(str(ply_path))

print("Number of points:", len(pcd.points))

# Optional: Improve visibility
pcd.estimate_normals()

o3d.visualization.draw_geometries(
    [pcd],
    window_name="FaceLift 3D Head (Point Cloud)",
    width=1200,
    height=800
)