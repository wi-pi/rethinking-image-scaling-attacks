import cv2 as cv
import numpy as np
import pytest
import torch
import torchvision.transforms.functional as F
from PIL import Image

from scaleadv.models.layers import ScalingLayer
from scaleadv.scaling import ScalingAPI, ScalingLib, ScalingAlg, ShapeType
from scaleadv.scaling.backends.cv import ScalingBackendCV
from scaleadv.scaling.backends.pil import ScalingBackendPIL

lib_list = list(ScalingLib)
alg_list = list(ScalingAlg)
src_list = [(666, 555), (333, 444)]
tgt_list = [(555, 444), (333, 222), (111, 66)]


def _setup(src_shape: ShapeType, tgt_shape: ShapeType, lib: ScalingLib, alg: ScalingAlg):
    # Generate random image
    shape = src_shape + (3,)
    img = np.random.randint(256, size=shape, dtype=np.uint8)

    # API result
    api = ScalingAPI(src_shape, tgt_shape, lib, alg)
    api_img = F.to_tensor(img).numpy()
    api_img = api(api_img)
    return img, api_img, api


class TestScalingAPI(object):

    @pytest.mark.parametrize('src_shape', src_list)
    @pytest.mark.parametrize('tgt_shape', tgt_list)
    @pytest.mark.parametrize('alg', alg_list)
    def test_pil_api(self, src_shape: ShapeType, tgt_shape: ShapeType, alg: ScalingAlg):
        # API
        img, api_img, _ = _setup(src_shape, tgt_shape, ScalingLib.PIL, alg)

        # PIL
        pil_alg = ScalingBackendPIL.algorithms[alg]
        pil_img = Image.fromarray(img).resize(tgt_shape[::-1], resample=pil_alg)
        pil_img = F.to_tensor(pil_img).numpy()

        assert pil_img.shape == api_img.shape
        assert pil_img.shape[1:] == tgt_shape
        assert np.allclose(pil_img, api_img)

    @pytest.mark.parametrize('src_shape', src_list)
    @pytest.mark.parametrize('tgt_shape', tgt_list)
    @pytest.mark.parametrize('alg', alg_list)
    def test_cv_api(self, src_shape: ShapeType, tgt_shape: ShapeType, alg: ScalingAlg):
        # API
        img, api_img, api = _setup(src_shape, tgt_shape, ScalingLib.CV, alg)
        api_img = np.clip(api_img, 0, 1)

        # CV
        cv_alg = ScalingBackendCV.algorithms[alg]
        cv_img = cv.resize(img.astype(np.float32), tgt_shape[::-1], interpolation=cv_alg)
        cv_img = np.clip(cv_img, 0, 255)
        cv_img = F.to_tensor(cv_img.astype(np.uint8)).numpy()

        assert cv_img.shape == api_img.shape
        assert cv_img.shape[1:] == tgt_shape
        assert np.allclose(cv_img, api_img, atol=1.e-2)

    @pytest.mark.parametrize('src_shape', src_list)
    @pytest.mark.parametrize('tgt_shape', tgt_list)
    @pytest.mark.parametrize('lib', lib_list)
    @pytest.mark.parametrize('alg', alg_list)
    def test_matrix(self, src_shape: ShapeType, tgt_shape: ShapeType, lib: ScalingLib, alg: ScalingAlg):
        # By API
        img, api_img, api = _setup(src_shape, tgt_shape, lib, alg)

        # By matrix
        mat_img = F.to_tensor(img).numpy()
        mat_img = api.cl @ mat_img @ api.cr
        mat_img = np.clip(mat_img, 0, 1)

        diff = np.percentile(np.abs(mat_img - api_img), 98)
        assert np.isclose(diff, 0, atol=0.1)

    @pytest.mark.parametrize('src_shape', src_list)
    @pytest.mark.parametrize('tgt_shape', tgt_list)
    @pytest.mark.parametrize('lib', lib_list)
    @pytest.mark.parametrize('alg', alg_list)
    def test_mask_shape(self, src_shape: ShapeType, tgt_shape: ShapeType, lib: ScalingLib, alg: ScalingAlg):
        api = ScalingAPI(src_shape, tgt_shape, lib, alg)
        assert api.mask.shape == src_shape


class TestScalingLayer(object):

    @pytest.mark.parametrize('src_shape', src_list)
    @pytest.mark.parametrize('tgt_shape', tgt_list)
    @pytest.mark.parametrize('lib', lib_list)
    @pytest.mark.parametrize('alg', alg_list)
    def test_scaling_layer(self, src_shape: ShapeType, tgt_shape: ShapeType, lib: ScalingLib, alg: ScalingAlg):
        # Generate random image
        img = np.random.rand(3, *src_shape).astype(np.float32)

        # Normal scale
        api = ScalingAPI(src_shape, tgt_shape, lib, alg)
        api_img = api(img)

        # Scaling layer
        scaling = ScalingLayer.from_api(api).cuda()
        img = torch.tensor(img)[None, ...].cuda()
        scaling_img = scaling(img)[0].cpu().numpy()

        diff = np.percentile(np.abs(api_img - scaling_img), 98)
        assert np.allclose(diff, 0, atol=0.1)
