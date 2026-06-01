import cv2
import numpy as np
from settings import COARSE_SCALE, PAD, MIN_INLIERS, MAX_REFINE_SIZE


class ImageRegistration:

    def register(self, reference, moving, coarse_scale=COARSE_SCALE,
                 pad=PAD, min_inliers=MIN_INLIERS, max_refine_size=MAX_REFINE_SIZE):

        homography, info = self._estimate_partial_homography(reference, moving, coarse_scale=coarse_scale,
                                                             min_inliers=min_inliers)

        if homography is None:
            return None, None, None, info, None

        aligned, mask = self._warp_homography(moving, reference.shape, homography)
        bbox = self._mask_bbox(mask, pad=pad, image_shape=reference.shape[:2])

        refined_aligned_full, refined_mask_full = self._refine_with_optical_flow(reference, aligned, mask,
                                                                                 bbox, max_refine_size=max_refine_size)

        bbox = self._mask_bbox(refined_mask_full, pad=pad, image_shape=reference.shape[:2])

        return refined_aligned_full, refined_mask_full, homography, info, bbox

    def _estimate_partial_homography(self, reference, moving, coarse_scale, min_inliers):
        ref_small = cv2.resize(reference, None, fx=coarse_scale, fy=coarse_scale, interpolation=cv2.INTER_AREA)
        mov_small = cv2.resize(moving, None, fx=coarse_scale, fy=coarse_scale, interpolation=cv2.INTER_AREA)
        ref_gray = cv2.cvtColor(ref_small, cv2.COLOR_BGR2GRAY)
        mov_gray = cv2.cvtColor(mov_small, cv2.COLOR_BGR2GRAY)

        homography = None
        info = {
            "method": "sift",
            "matches": 0,
            "inliers": 0
        }

        sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01, edgeThreshold=10)
        mov_kp, mov_desc = sift.detectAndCompute(mov_gray, None)
        ref_kp, ref_desc = sift.detectAndCompute(ref_gray, None)

        if mov_desc is not None and ref_desc is not None and len(mov_kp) >= 4 and len(ref_kp) >= 4:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            knn_matches = matcher.knnMatch(mov_desc, ref_desc, k=2)
            good_matches = []
            for knn_match in knn_matches:
                first, second = knn_match
                if first.distance < 0.75 * second.distance:
                    good_matches.append(first)

            info["matches"] = len(good_matches)

            if len(good_matches) >= 4:
                mov_points = np.float32([mov_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                ref_points = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                scaled_homography, inlier_mask = cv2.findHomography(
                    mov_points,
                    ref_points,
                    cv2.RANSAC,
                    5.0
                )

                info["inliers"] = int(inlier_mask.sum())
                if info["inliers"] >= min_inliers:
                    homography = self._unscale_homography(scaled_homography, coarse_scale)

        return homography, info

    def _unscale_homography(self, scaled_homography, scale):
        scale_matrix = np.array([
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [0.0, 0.0, 1.0]
        ])
        homography = np.linalg.inv(scale_matrix) @ scaled_homography @ scale_matrix
        return homography / homography[2, 2]

    def _warp_homography(self, moving, reference_shape, homography):
        output_size = (reference_shape[1], reference_shape[0])
        aligned = cv2.warpPerspective(
            moving,
            homography,
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        source_mask = np.ones(moving.shape[:2], dtype=np.uint8) * 255
        mask = cv2.warpPerspective(
            source_mask,
            homography,
            output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return aligned, mask

    def _mask_bbox(self, mask, pad, image_shape):
        rows, cols = np.where(mask > 0)

        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1

        height, width = image_shape
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(height, y1 + pad)
        x1 = min(width, x1 + pad)
        return y0, x0, y1, x1

    def _refine_with_optical_flow(self, reference, aligned, mask,
                                  bbox, max_refine_size):
        y0, x0, y1, x1 = bbox
        reference_crop = reference[y0:y1, x0:x1]
        aligned_crop = aligned[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]

        height, width = mask_crop.shape

        scale = min(1.0, max_refine_size / float(max(height, width)))
        ref_gray = cv2.cvtColor(reference_crop, cv2.COLOR_BGR2GRAY)
        mov_gray = cv2.cvtColor(aligned_crop, cv2.COLOR_BGR2GRAY)

        if scale < 1.0:
            small_size = (int(round(width * scale)), int(round(height * scale)))
            ref_gray_small = cv2.resize(ref_gray, small_size, interpolation=cv2.INTER_AREA)
            mov_gray_small = cv2.resize(mov_gray, small_size, interpolation=cv2.INTER_AREA)
            mask_small = cv2.resize(mask_crop, small_size, interpolation=cv2.INTER_NEAREST)
        else:
            ref_gray_small = ref_gray
            mov_gray_small = mov_gray
            mask_small = mask_crop

        ref_gray_small = cv2.normalize(ref_gray_small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mov_gray_small = cv2.normalize(mov_gray_small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mov_gray_small = cv2.bitwise_and(mov_gray_small, mov_gray_small, mask=mask_small)

        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow_small = dis.calc(mov_gray_small, ref_gray_small, None)

        if scale < 1.0:
            flow = cv2.resize(flow_small, (width, height), interpolation=cv2.INTER_LINEAR) / scale
        else:
            flow = flow_small

        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
        map_y = (grid_y - flow[:, :, 1]).astype(np.float32)

        refined_aligned = cv2.remap(
            aligned_crop,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        refined_mask = cv2.remap(
            mask_crop,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        refined_aligned[refined_mask == 0] = 0

        refined_aligned_full = np.zeros_like(aligned)
        refined_mask_full = np.zeros_like(mask)
        refined_aligned_full[y0:y1, x0:x1] = refined_aligned
        refined_mask_full[y0:y1, x0:x1] = refined_mask

        return refined_aligned_full, refined_mask_full
