import cv2
import numpy as np
from typing import Optional


class DFTMarker:
    def _convert_img(self, img: np.ndarray):
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img_y = img_ycrcb[:, :, 0]
        # img_y = img_y.astype(np.float64)/255
        img_y = img_y.astype(np.float64)
        img_fft = np.fft.fft2(img_y)
        mag = np.fft.fftshift(np.abs(img_fft))
        ang = np.angle(img_fft)
        return mag, ang, img_ycrcb

    def _embed_to_empty(self, mark: np.ndarray, shape: list[int], r: int):
        w_mask = np.zeros(shape)
        l = len(mark)
        for i in range(l):
            x1 = int(shape[1] // 2 + np.round(r * np.sin(i * np.pi / l)))
            y1 = int(shape[0] // 2 + np.round(r * np.cos(i * np.pi / l)))
            w_mask[y1, x1] = mark[i]
        return w_mask

    def _get_radius(self, img: np.ndarray) -> int:
        shape = img.shape[:2]
        rad_mid = min(shape) // 4
        return rad_mid

    def embed(self, img: np.ndarray, mark: list, alpha: float, r: Optional[int] = None):
        if r is None:
            r = self._get_radius(img)
        mag, ang, img_ycrcb = self._convert_img(img)
        img_y = img_ycrcb[:, :, 0]
        l = len(mark)
        w_mask = np.zeros(mag.shape)
        for i in range(l):
            x1 = int(mag.shape[1] // 2 + np.round(r * np.sin(i * np.pi / l)))
            y1 = int(mag.shape[0] // 2 + np.round(r * np.cos(i * np.pi / l)))
            x2 = int(mag.shape[1] // 2 +
                     np.round(r * np.sin(i * np.pi / l + np.pi)))
            y2 = int(mag.shape[0] // 2 +
                     np.round(r * np.cos(i * np.pi / l + np.pi)))
            # Mean value of 3x3 area with center in (x1,y1)
            mean9_1 = img_y[x1 - 1:x1 + 2, y1 - 1:y1 + 2].mean()
            mean9_2 = img_y[x2 - 1:x2 + 2, y2 - 1:y2 + 2].mean()
            w_mask[y1, x1] = mark[i] * mean9_1
            w_mask[y2, x2] = mark[i] * mean9_2
        mag_m = mag + alpha * w_mask
        # Unifying magnitude and angle back to one complex number
        img_ic = np.fft.ifftshift(mag_m) * np.exp(1j * ang)
        img_ifft = np.fft.ifft2(img_ic)
        # img_y_m = (np.real(img_ifft)*255).clip(0,255).astype(np.uint8)
        img_y_m = (np.real(img_ifft)).clip(0, 255).astype(np.uint8)

        img_m_ycrcb = np.concatenate(
            [img_y_m[..., np.newaxis], img_ycrcb[:, :, 1:]], axis=2)
        img_m_bgr = cv2.cvtColor(img_m_ycrcb, cv2.COLOR_YCR_CB2BGR)
        return img_m_bgr

    def extract(self, img: np.ndarray, mark: np.ndarray,
                r: Optional[int] = None, search_range: int = 32):
        if r is None:
            r = self._get_radius(img)
        mag, ang, img_ycrcb = self._convert_img(img)
        radii = search_range * 2
        metric_arr = [None] * radii
        mm_arr = [None] * radii
        for c, sr in enumerate(range(r - search_range, r + search_range)):
            vec = self.extract_vec(mag, sr)
            empty_mask = self._embed_to_empty(mark, img.shape, sr)
            empty_vec = self.extract_vec(empty_mask, sr)
            mm_arr[c] = (vec, empty_vec)
            corr_array = self._rolling_corr(vec, empty_vec)
            metric_arr[c] = corr_array
        return max([np.nanmax(i)for i in metric_arr])
        return mm_arr, metric_arr

    def embed_resize(self, img: np.ndarray, mark: list, alpha: float, r: int):
        or_shape = img.shape
        if len(or_shape) == 3:
            or_shape = or_shape[:2]
        img512 = cv2.resize(img, (512, 512))
        embeded = self.embed(img512, mark, alpha, r)
        rback = cv2.resize(embeded, or_shape[::-1])
        return rback

    def _rolling_corr(self, a: np.ndarray, b: np.ndarray):
        res = np.array([
            np.corrcoef(np.roll(a, i), b)[0, 1] for i in range(len(a))
        ])
        return res

    def extract_vec(self, mag: np.ndarray, r):
        # mag,ang,ycrcb = self._convert_img(img)
        step = np.ceil(np.pi / (2 * np.arcsin(1 / (2 * r)))).astype(np.int32)
        # l = len(mark)
        out_vec = np.zeros((step,))
        w_mask = np.zeros(mag.shape, dtype=np.uint8)
        for i in range(np.ceil(step).astype(np.int32)):
            x1 = int(mag.shape[1] // 2 +
                     np.round(r * np.sin(i * np.pi / step)))
            y1 = int(mag.shape[0] // 2 +
                     np.round(r * np.cos(i * np.pi / step)))
            w_mask[y1, x1] = 255
            area3x3 = mag[y1 - 1:y1 + 2, x1 - 1:x1 + 2]
            out_vec[i] = area3x3.max()
        return out_vec
