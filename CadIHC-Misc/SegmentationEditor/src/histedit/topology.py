import numpy as np
import cv2
from scipy.ndimage import label

def extract_boundaries(segmentation_mask, simplify_tolerance=2.0):
    """
    Extract shared boundary vertices from a segmentation mask.
    Returns:
        vertices: List of [x, y] coordinates.
        regions: List of dicts, each containing:
            'original_id': The mask ID.
            'vertex_indices': List of lists of indices into `vertices` array, 
                              representing the outer contour and holes.
            'color': Display color.
    """
    height, width = segmentation_mask.shape
    unique_ids = np.unique(segmentation_mask)
    structure = np.array([[0,1,0],[1,1,1],[0,1,0]])
    
    # We will build a unified list of vertices to share them.
    # To do this efficiently, we'll first extract all contours for all regions.
    # Then we will snap close vertices together.
    
    all_contours = [] # list of (uid, contour_points)
    
    for uid in unique_ids:
        if uid == 0:
            continue
            
        bin_mask = (segmentation_mask == uid).astype(np.uint8)
        labeled_mask, num_features = label(bin_mask, structure=structure)
        
        for region_idx in range(1, num_features + 1):
            region_mask = (labeled_mask == region_idx).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                    
                # Simplify contour to reduce vertex count (critical for performance and dragging)
                simplified = cv2.approxPolyDP(contour, simplify_tolerance, True)
                if len(simplified) < 3:
                     continue
                
                # Reshape to (N, 2)
                pts = simplified.reshape(-1, 2)
                all_contours.append((uid, pts))
                
    # Now unify the vertices. If two vertices are very close, they map to the same shared vertex.
    # This allows dragging a boundary point to deform both regions.
    shared_vertices = []
    vertex_map = {} # maps quantized (x,y) to index in shared_vertices
    
    regions = []
    
    # Quantize grid size for snapping vertices together
    snap_dist = 3.0
    
    for uid, pts in all_contours:
        current_region_indices = []
        for pt in pts:
            x, y = pt[0], pt[1]
            
            # Find if there is a vertex close enough
            # For exact sharing we could just use (x,y), but cv2 contours on adjacent regions
            # might be off by 1 pixel. We snap them to a grid or search.
            # A simple spatial dictionary works well if we assume borders are exactly touching or within 1-2px
            qx, qy = int(np.round(x / snap_dist)), int(np.round(y / snap_dist))
            
            # check neighbors
            found_idx = -1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (qx+dx, qy+dy) in vertex_map:
                        # Check exact distance just in case
                        v_idx = vertex_map[(qx+dx, qy+dy)]
                        vx, vy = shared_vertices[v_idx]
                        if (x-vx)**2 + (y-vy)**2 <= snap_dist**2:
                            found_idx = v_idx
                            break
                if found_idx != -1:
                    break
                    
            if found_idx == -1:
                # Add new vertex
                found_idx = len(shared_vertices)
                shared_vertices.append([float(x), float(y)])
                vertex_map[(qx, qy)] = found_idx
                
            current_region_indices.append(found_idx)
            
        import pyray as pr
        color = pr.Color(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255), 255)
            
        regions.append({
            'original_id': uid,
            'vertex_indices': current_region_indices,
            'color': color
        })
        
    return shared_vertices, regions

def reconstruct_mask(vertices, regions, width, height):
    """
    Rasterize the regions defined by vertices back into a uint16 segmentation mask.
    """
    output_mask = np.zeros((height, width), dtype=np.uint16)
    
    for region in regions:
        uid = region['original_id']
        indices = region['vertex_indices']
        
        if len(indices) < 3:
            continue
            
        # Gather (x,y) coordinates for this poly
        poly_pts = []
        for idx in indices:
            poly_pts.append(vertices[idx])
            
        # cv2 fillPoly expects int32 coordinates in shape (N, 1, 2)
        pts = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
        
        cv2.fillPoly(output_mask, [pts], int(uid))
        # Polylines help cover exact mathematical edge boundaries depending on thickness
        cv2.polylines(output_mask, [pts], True, int(uid), thickness=2)
        
    # Now patch tiny 1-2 pixel rasterization gaps left behind
    # But ONLY in narrow interstitial boundary spaces between *different* masks, 
    # not dilating out into massive valid empty background spaces.
    zero_mask = (output_mask == 0)
    
    if np.any(zero_mask):
        from scipy.ndimage import distance_transform_edt
        
        # Compute distance to the *nearest* non-zero pixel
        global_mask = ~zero_mask
        dist, inds = distance_transform_edt(zero_mask, return_distances=True, return_indices=True)
        
        # We only want to fill pixels that are very close to an edge (<= 2 pixels)
        # AND we only want to fill them if they are in a "crack" between segmentations.
        # A simple heuristic for a crack vs open space is just strictly limiting to dist <= 1.5.
        # This will fill 1-pixel jagged gaps and touching boundaries, but will leave 
        # actual hollow spaces alone (since it only dilates 1 pixel inward).
        fill_candidates = zero_mask & (dist <= 1.5)
        
        # Apply the nearest labels where there are gaps
        output_mask[fill_candidates] = output_mask[tuple(inds[:, fill_candidates])]
        
    return output_mask
