import cv2
import pywt
import numpy as np
import torch


class DWTTransform:
    def __init__(
        self,
        variant="LL",
        wavelet="sym4",
        mode="reflect",
        img_size=(224, 224),
        is_gray=False,
    ):
        self.variant = variant
        self.wavelet = wavelet
        self.mode = mode
        self.img_size = img_size
        self.is_gray = is_gray

    def normalize_resize(self, band):
        band = cv2.resize(band, self.img_size)
        band = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
        return band.astype(np.uint8)

    # =====================================================
    # RGB DWT
    # =====================================================
    def dwt_rgb(self, img):

        r, g, b = cv2.split(img)
        out = []

        for ch in (r, g, b):

            # DWT decomposition
            LL, (LH, HL, HH) = pywt.dwt2(
                ch,
                self.wavelet,
                self.mode
            )

            # Resize + normalize
            LL = self.normalize_resize(LL)
            LH = self.normalize_resize(LH)
            HL = self.normalize_resize(HL)
            HH = self.normalize_resize(HH)

            # -----------------------------------------
            # Select DWT representation
            # -----------------------------------------
            if self.variant == "LL":
                dwt_img = LL

            elif self.variant == "HH":
                dwt_img = HH

            elif self.variant == "LL_HH":
                dwt_img = cv2.addWeighted(
                    LL, 0.5,
                    HH, 0.5,
                    0
                )

            elif self.variant == "ALL":

                temp = cv2.addWeighted(
                    HL, 0.25,
                    HH, 0.25,
                    0
                )

                temp = cv2.addWeighted(
                    LH, 0.25,
                    temp, 1.0,
                    0
                )

                dwt_img = cv2.addWeighted(
                    LL, 0.25,
                    temp, 1.0,
                    0
                )

            else:
                raise ValueError(f"Unknown variant: {self.variant}")

            # -----------------------------------------
            # Figure-2 Fusion
            # Enhanced Channel = (Original + DWT)/2
            # -----------------------------------------
            fused = cv2.addWeighted(
                ch.astype(np.float32),
                0.5,
                dwt_img.astype(np.float32),
                0.5,
                0
            )

            fused = np.clip(fused, 0, 255).astype(np.uint8)

            out.append(fused)

        return cv2.merge(out)

    # =====================================================
    # GRAYSCALE DWT
    # =====================================================
    def dwt_gray(self, img):

        LL, (LH, HL, HH) = pywt.dwt2(
            img,
            self.wavelet,
            self.mode
        )

        LL = self.normalize_resize(LL)
        LH = self.normalize_resize(LH)
        HL = self.normalize_resize(HL)
        HH = self.normalize_resize(HH)

        if self.variant == "LL":
            dwt_img = LL

        elif self.variant == "HH":
            dwt_img = HH

        elif self.variant == "LL_HH":
            dwt_img = cv2.addWeighted(
                LL, 0.5,
                HH, 0.5,
                0
            )

        elif self.variant == "ALL":

            temp = cv2.addWeighted(
                HL, 0.25,
                HH, 0.25,
                0
            )

            temp = cv2.addWeighted(
                LH, 0.25,
                temp, 1.0,
                0
            )

            dwt_img = cv2.addWeighted(
                LL, 0.25,
                temp, 1.0,
                0
            )

        else:
            raise ValueError(f"Unknown variant: {self.variant}")

        # Figure-2 Fusion
        fused = cv2.addWeighted(
            img.astype(np.float32),
            0.5,
            dwt_img.astype(np.float32),
            0.5,
            0
        )

        fused = np.clip(fused, 0, 255).astype(np.uint8)

        return fused

    # =====================================================
    # CALL
    # =====================================================
    def __call__(self, img):

        img = np.array(img)

        if self.is_gray:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2GRAY
            )

            img = self.dwt_gray(img)

            img = np.expand_dims(img, axis=2)

        else:

            img = self.dwt_rgb(img)

        img = img.astype(np.float32) / 255.0

        img = torch.from_numpy(img).permute(2, 0, 1)

        return img