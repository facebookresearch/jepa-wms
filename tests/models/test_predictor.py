import unittest

import torch

from app.plan_common.models.AdaLN_vit import vit_predictor_AdaLN
from app.plan_common.models.vit import ViTPredictor
from src.models.ac_predictor import vit_ac_predictor


class TestAdaLNPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.embed_dim = 768
        self.predictor_embed_dim = 384
        self.action_dim = 7
        self.proprio_dim = 7
        self.num_frames = 4
        self.img_size = (256, 256)
        self.patch_size = 16
        self.grid_height = self.img_size[0] // self.patch_size
        self.grid_width = self.img_size[1] // self.patch_size

        self.predictor = vit_predictor_AdaLN(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=1,
            embed_dim=self.embed_dim,
            predictor_embed_dim=self.predictor_embed_dim,
            depth=2,
            num_heads=6,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            use_proprio=True,
            proprio_encoding="token",
            proprio_tokens=1,
            local_window=(1, self.grid_height, self.grid_width),
        )

    def test_adaln_predictor_batchsize_4(self):
        BS = 4
        T = self.num_frames
        H = self.grid_height
        W = self.grid_width

        # Visual input: B, T, V, H, W, D
        x = torch.rand((BS, T, 1, H, W, self.embed_dim))
        # Actions: B, T, A
        actions = torch.rand((BS, T, self.action_dim))
        # Proprioceptive: B, T, P
        proprio = torch.rand((BS, T, self.proprio_dim))

        visual_out, _, proprio_out = self.predictor(x, actions, proprio)

        self.assertIsInstance(visual_out, torch.Tensor)
        self.assertEqual(visual_out.size(), (BS, T, H * W, self.embed_dim))
        self.assertIsNotNone(proprio_out)
        self.assertEqual(proprio_out.size(), (BS, T, 1, self.predictor_embed_dim))

    def test_adaln_predictor_batchsize_1(self):
        BS = 1
        T = self.num_frames
        H = self.grid_height
        W = self.grid_width

        # Visual input: B, T, V, H, W, D
        x = torch.rand((BS, T, 1, H, W, self.embed_dim))
        # Actions: B, T, A
        actions = torch.rand((BS, T, self.action_dim))
        # Proprioceptive: B, T, P
        proprio = torch.rand((BS, T, self.proprio_dim))

        visual_out, _, proprio_out = self.predictor(x, actions, proprio)

        self.assertIsInstance(visual_out, torch.Tensor)
        self.assertEqual(visual_out.size(), (BS, T, H * W, self.embed_dim))
        self.assertIsNotNone(proprio_out)
        self.assertEqual(proprio_out.size(), (BS, T, 1, self.predictor_embed_dim))


class TestACPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.embed_dim = 768
        self.predictor_embed_dim = 384
        self.action_dim = 7
        self.proprio_dim = 7
        self.num_frames = 4
        self.img_size = (256, 256)
        self.patch_size = 16
        self.grid_height = self.img_size[0] // self.patch_size
        self.grid_width = self.img_size[1] // self.patch_size

        self.predictor = vit_ac_predictor(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=1,
            embed_dim=self.embed_dim,
            predictor_embed_dim=self.predictor_embed_dim,
            depth=2,
            num_heads=6,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            action_conditioning="token",
            proprio_tokens=1,
            proprio_encoder_inpred=True,
            action_encoder_inpred=True,
        )

    def test_ac_predictor_token_conditioning_batchsize_4(self):
        BS = 4
        T = self.num_frames
        H = self.grid_height
        W = self.grid_width
        N_ctxt = T * H * W

        # Visual input: B, N_ctxt, D
        x = torch.rand((BS, N_ctxt, self.embed_dim))
        # Actions: B, T, A
        actions = torch.rand((BS, T, self.action_dim))
        # Proprioceptive: B, T, P
        states = torch.rand((BS, T, self.proprio_dim))

        visual_out, action_out, proprio_out = self.predictor(x, actions, states)

        self.assertIsInstance(visual_out, torch.Tensor)
        self.assertEqual(visual_out.size(), (BS, T * H * W, self.embed_dim))
        self.assertIsNotNone(action_out)
        self.assertEqual(action_out.size(), (BS, T, 1, self.predictor_embed_dim))
        self.assertIsNotNone(proprio_out)
        self.assertEqual(proprio_out.size(), (BS, T, 1, self.predictor_embed_dim))

    def test_ac_predictor_token_conditioning_batchsize_1(self):
        BS = 1
        T = self.num_frames
        H = self.grid_height
        W = self.grid_width
        N_ctxt = T * H * W

        # Visual input: B, N_ctxt, D
        x = torch.rand((BS, N_ctxt, self.embed_dim))
        # Actions: B, T, A
        actions = torch.rand((BS, T, self.action_dim))
        # Proprioceptive: B, T, P
        states = torch.rand((BS, T, self.proprio_dim))

        visual_out, action_out, proprio_out = self.predictor(x, actions, states)

        self.assertIsInstance(visual_out, torch.Tensor)
        self.assertEqual(visual_out.size(), (BS, T * H * W, self.embed_dim))
        self.assertIsNotNone(action_out)
        self.assertEqual(action_out.size(), (BS, T, 1, self.predictor_embed_dim))
        self.assertIsNotNone(proprio_out)
        self.assertEqual(proprio_out.size(), (BS, T, 1, self.predictor_embed_dim))

    def test_ac_predictor_feature_conditioning(self):
        BS = 2
        T = self.num_frames
        H = self.grid_height
        W = self.grid_width
        N_ctxt = T * H * W
        action_emb_dim = 64
        proprio_emb_dim = 32

        predictor = vit_ac_predictor(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=1,
            embed_dim=self.embed_dim,
            predictor_embed_dim=self.predictor_embed_dim,
            depth=2,
            num_heads=6,
            action_dim=self.action_dim,
            proprio_dim=self.proprio_dim,
            action_conditioning="feature",
            action_emb_dim=action_emb_dim,
            proprio_emb_dim=proprio_emb_dim,
            proprio_encoder_inpred=True,
            action_encoder_inpred=True,
        )

        # Visual input: B, N_ctxt, D
        x = torch.rand((BS, N_ctxt, self.embed_dim))
        # Actions: B, T, A
        actions = torch.rand((BS, T, self.action_dim))
        # Proprioceptive: B, T, P
        states = torch.rand((BS, T, self.proprio_dim))

        visual_out, action_out, proprio_out = predictor(x, actions, states)

        self.assertIsInstance(visual_out, torch.Tensor)
        self.assertEqual(visual_out.size(), (BS, T * H * W, self.embed_dim))
        self.assertIsNotNone(action_out)
        self.assertEqual(action_out.size(), (BS, T, H * W, action_emb_dim))
        self.assertIsNotNone(proprio_out)
        self.assertEqual(proprio_out.size(), (BS, T, H * W, proprio_emb_dim))


class TestViTPredictor(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 384
        self.num_patches = 196
        self.num_frames = 4
        self.depth = 2
        self.heads = 6
        self.mlp_dim = 1024

    def test_vit_predictor_batchsize_4(self):
        predictor = ViTPredictor(
            num_patches=self.num_patches,
            num_frames=self.num_frames,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
        )

        BS = 4
        T = self.num_frames
        N = T * self.num_patches

        # Visual input: B, N, D
        x = torch.rand((BS, N, self.dim))

        output = predictor(x)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.size(), (BS, N, self.dim))

    def test_vit_predictor_batchsize_1(self):
        predictor = ViTPredictor(
            num_patches=self.num_patches,
            num_frames=self.num_frames,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
        )

        BS = 1
        T = self.num_frames
        N = T * self.num_patches

        # Visual input: B, N, D
        x = torch.rand((BS, N, self.dim))

        output = predictor(x)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.size(), (BS, N, self.dim))


if __name__ == "__main__":
    unittest.main()
