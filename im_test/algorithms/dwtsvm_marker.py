import numpy as np
import cv2
import pywt
from sklearn.svm import SVC


class DWTSVMMarker:
    IMG_SIZE = (512, 512)

    def __init__(self, subband=7, threshold=35, wavelet_name='haar'):
        self.T = threshold
        self.subband = subband
        self.wl_name = wavelet_name
        self.metrics = {}

    @staticmethod
    def nc2(x: np.ndarray, y: np.ndarray):
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return np.correlate(x / norm_x, y / norm_y)[0]

    @staticmethod
    def np_mode(arr: np.ndarray):
        un = np.unique(arr, return_counts=True)
        max_idx = np.argwhere(un[1] == un[1].max())
        return un[0][max_idx].mean()

    @staticmethod
    def np_moment(data, order=1, center=None) -> np.ndarray:
        if center is None:
            center = data.mean()
        d = ((data - center)**order).mean()
        return d

    def _preproc_img(self, img: np.ndarray):
        if len(img.shape) == 3:
            if img.shape[2] == 1:
                return (img.squeeze(), None, None)
            elif img.shape[2] == 3:
                conv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                return conv[:, :, 0], conv[:, :, 1], conv[:, :, 2]
        elif len(img.shape) == 2:
            return (img, None, None)
        raise TypeError(f'Image has incompatible dimentions {img.shape}')

    def _restore_img(self,
                     channles: tuple[np.ndarray, np.ndarray | None, np.ndarray | None]):
        if channles[0] is None or channles[1] is None:
            if channles[0] is None or channles[1] is not None:
                raise TypeError(f'Something wrong with image')
            return channles[0]
        return cv2.cvtColor(np.array(channles).transpose(1, 2, 0), cv2.COLOR_YCrCb2BGR)

    def _dwt3sub(self, img: np.ndarray):
        # TODO: add support for any subband (idwt3sub too)
        ll, (hl, lh, hh) = pywt.dwt2(img, self.wl_name)
        ll1, (hl1, lh1, hh1) = pywt.dwt2(lh, self.wl_name)
        ll2, (hl2, lh2, hh2) = pywt.dwt2(lh1, self.wl_name)
        return {
            'll': ll,
            'hl': hl,
            'lh': lh,
            'hh': hh,
            'll1': ll1,
            'hl1': hl1,
            'lh1': lh1,
            'hh1': hh1,
            'll2': ll2,
            'hl2': hl2,
            'lh2': lh2,
            'hh2': hh2
        }

    def _idwt3sub(self, subbands: dict[str, np.ndarray]):
        s3 = pywt.idwt2((subbands['ll2'], (subbands[i]
                        for i in ['hl2', 'lh2', 'hh2'])), self.wl_name)
        s2 = pywt.idwt2(
            (subbands['ll1'], (subbands['hl1'], s3, subbands['hh1'])), self.wl_name)
        s1 = pywt.idwt2(
            (subbands['ll'], (subbands['hl'], s2, subbands['hh'])), self.wl_name)
        return s1

    def _cut_to_blocks(self, sb: np.ndarray
                       ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        windows = []
        # differece between largest and second largest elements in each window
        diff_maxes = []
        for i in range(0, sb.shape[0], 2):
            for j in range(0, sb.shape[1], 2):
                window = sb[i:i + 2, j:j + 2]
                psorted = np.partition(window.flatten(), -2)
                diff_maxes.append(psorted[-1] - psorted[-2])
                windows.append(window)
        return np.array(windows), np.array(diff_maxes)

    def _get_feature_set(self, block: np.ndarray):
        f2 = block.mean()
        f3 = block.std()
        f4 = block.var()
        f5 = self.np_mode(block)
        f6 = np.median(block)
        f7 = np.cov(block)
        f8 = self.np_moment(block, 5)
        f_c = [*block.ravel()]
        return np.array([f2, f3, f4, f5, f6, f7, f8, *f_c])

    def _embed_in_sb(self, sb: np.ndarray, mark: np.ndarray):
        # x500 times faster (600 microsec vs 600 nanosec):
        # windows = ar.reshape(ar.shape[0] // 2, 2, ar.shape[1] // 2, 2).transpose(0,2,1,3)
        # sb but cutted in 2x2 windows
        self.metrics['add_count'] = 0
        windows, diff_maxes = self._cut_to_blocks(sb)
        th = max(diff_maxes.mean(), self.T)
        # restored_sb = np.zeros(sb.shape)
        # this loop probably can be more efficient
        for i in range(windows.shape[0]):
            window = windows[i].ravel()
            argmax = np.argmax(window)
            if mark[i] == 1:
                if diff_maxes[i] < th:
                    window[argmax] = window[argmax] + self.T
                    self.metrics['add_count'] += 1
            else:
                window[argmax] -= diff_maxes[i]
        # put modified 2x2 tiles in 64x64 matrix
        restored_sb = windows.reshape(32, 32, 2, 2).transpose(
            0, 2, 1, 3).reshape(64, 64, -1).squeeze()
        return restored_sb

    def _to_uint8(self, arr: np.ndarray) -> np.ndarray:
        return arr.clip(0, 255).astype(np.uint8)

    def embed(self, img: np.ndarray, mark: np.ndarray,
              key_mark: np.ndarray, resize_back: bool = False):
        y, cr, cb = self._preproc_img(img)
        img_r = y
        dwt_d = self._dwt3sub(img_r)
        sb = dwt_d['hl2']
        sb_or_size = sb.shape
        sb = cv2.resize(sb, (64, 64))
        full_mark = np.concat([key_mark, mark])
        embedded_sb = self._embed_in_sb(sb, full_mark)
        embedded_sb = cv2.resize(embedded_sb, sb_or_size[::-1])
        dwt_d['hl2'] = embedded_sb
        restored_img = self._idwt3sub(dwt_d)
        return self._restore_img((self._to_uint8(restored_img), cr, cb))

    def extract(self, img: np.ndarray, key_mark: np.ndarray):
        y, cr, cb = self._preproc_img(img)
        # img_r = cv2.resize(img_p,self.IMG_SIZE)
        img_r = y
        dwt_sbs = self._dwt3sub(img_r)
        sb = dwt_sbs['hl2']
        sb = cv2.resize(sb, (64, 64))
        blocks, diffs = self._cut_to_blocks(sb)
        features = np.array([self._get_feature_set(i.ravel()) for i in blocks])
        train, test = features[:512], features[512:]
        svc = SVC()
        svc.fit(train, key_mark)
        train_pred = svc.predict(train)
        train_ber = (train_ber != key_mark).mean()
        self.metrics['train_ber'] = train_ber
        test_pred = svc.predict(test)
        return test_pred
