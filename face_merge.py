# face_merge.py
import cv2
import dlib
import numpy as np

# Initialize dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ------------------------ #
# Get facial landmarks
def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("No face detected!")
    landmarks = predictor(gray, faces[0])
    points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]
    return points

# ------------------------ #
# Detect beard
def has_beard(img, landmarks, threshold=0.05):
    jaw_points = [landmarks[i] for i in range(0, 17)]
    hull = cv2.convexHull(np.array(jaw_points)).reshape(-1, 2)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0]

    dark_pixels = np.sum((y_channel < 80) & (mask == 255))
    total_pixels = np.sum(mask == 255)

    if total_pixels == 0:
        return False
    fraction_dark = dark_pixels / total_pixels
    return fraction_dark > threshold

# ------------------------ #
# Delaunay triangulation
def delaunay_triangulation(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        x = min(max(p[0], rect[0]+1), rect[0]+rect[2]-1)
        y = min(max(p[1], rect[1]+1), rect[1]+rect[3]-1)
        subdiv.insert((x, y))

    triangles = subdiv.getTriangleList()
    delaunay_tri = []
    for t in triangles:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        indices = []
        for pt in pts:
            for i, p in enumerate(points):
                if abs(pt[0]-p[0]) < 2 and abs(pt[1]-p[1]) < 2:
                    indices.append(i)
                    break
        if len(indices) == 3:
            delaunay_tri.append(tuple(indices))
    return delaunay_tri

# ------------------------ #
# Warp triangle
def warp_triangle(src, dst, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))

    t_src_rect = [(p[0]-r1[0], p[1]-r1[1]) for p in t_src]
    t_dst_rect = [(p[0]-r2[0], p[1]-r2[1]) for p in t_dst]

    src_crop = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if src_crop.size == 0:
        return

    M = cv2.getAffineTransform(np.float32(t_src_rect), np.float32(t_dst_rect))
    dst_patch = cv2.warpAffine(src_crop, M, (r2[2], r2[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(t_dst_rect), 255)

    dst_sub = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    dst_sub = cv2.bitwise_and(dst_sub, cv2.bitwise_not(mask_3ch))
    dst_sub = cv2.add(dst_sub, cv2.bitwise_and(dst_patch, mask_3ch))
    dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_sub

# ------------------------ #
# Histogram matching
def match_histogram(src, dst, landmarks):
    hull = cv2.convexHull(np.array(landmarks))
    hull = hull.reshape(-1, 2)

    mask = np.zeros(dst.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    dst_lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    matched = src_lab.copy()

    for i in range(3):
        src_chan = src_lab[:, :, i][mask==255]
        dst_chan = dst_lab[:, :, i][mask==255]

        src_hist, _ = np.histogram(src_chan.flatten(), 256, [0,256])
        dst_hist, _ = np.histogram(dst_chan.flatten(), 256, [0,256])

        src_cdf = src_hist.cumsum()
        dst_cdf = dst_hist.cumsum()
        if src_cdf[-1] > 0: src_cdf = src_cdf / src_cdf[-1]
        if dst_cdf[-1] > 0: dst_cdf = dst_cdf / dst_cdf[-1]

        lookup = np.interp(src_cdf, dst_cdf, np.arange(256))
        matched[:, :, i][mask==255] = np.interp(src_chan, np.arange(256), lookup).astype(np.uint8)

    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

# ------------------------ #
# Main merge_faces function
def merge_faces(src_img, dst_img):
    dst_img_out = dst_img.copy()
    src_points = get_landmarks(src_img)
    dst_points = get_landmarks(dst_img)

    # Beard handling
    if has_beard(src_img, src_points):
        skin_points = list(range(0, 17))  # exclude lower jaw
    else:
        skin_points = list(range(0, 68))

    h, w = dst_img.shape[:2]
    rect = (0, 0, w, h)
    triangles = delaunay_triangulation(rect, dst_points)

    warped_face = np.zeros_like(dst_img, dtype=np.uint8)
    for tri in triangles:
        t_src = [src_points[i] for i in tri]
        t_dst = [dst_points[i] for i in tri]
        warp_triangle(src_img, warped_face, t_src, t_dst)

    warped_face = match_histogram(warped_face, dst_img, [dst_points[i] for i in skin_points])

    # Seamless cloning
    hull = cv2.convexHull(np.array([dst_points[i] for i in skin_points])).reshape(-1, 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(hull), 255)
    mask = cv2.GaussianBlur(mask, (15, 15), 0)

    x, y, w_box, h_box = cv2.boundingRect(hull)
    center = (x + w_box // 2, y + h_box // 2)
    center = (min(max(center[0], 0), w-1), min(max(center[1], 0), h-1))

    output = cv2.seamlessClone(warped_face, dst_img_out, mask, center, cv2.NORMAL_CLONE)
    return output

