# Third-party notices

## XSoma

InstantAMP is derived from XSoma (https://github.com/nico-bohlinger/xsoma, MIT License, Copyright (c) 2026 Nico Bohlinger): the world
model, the BPTT policy update, the replay buffer, the optimizers, the critic, the motion prior, the reset banks and the trainer's
structure (`instantamp/*.py`), and `tools/convert_motions.py` and `tools/download_meshes.sh`. Its copyright notice is kept in
[`LICENSE`](LICENSE).

## XMimic, BlinkMimic, InstantTrack and supermimic_vibed

XSoma's world model, policy network, replay buffer, optimizers and rotation utilities (`instantamp/nets.py`, `instantamp/buffer.py`,
`instantamp/optim.py`, `instantamp/rot.py`) and the structure of the trainer come from XMimic
(https://github.com/nico-bohlinger/xmimic, MIT, same author as XSoma), whose code is in large part a port of BlinkMimic
(https://github.com/fuyichen0725/blinkmimic), which derives from InstantTrack (https://github.com/Huy-Quang-Dao/instantTrack)
and supermimic_vibed (https://github.com/zita-ch/supermimic_vibed). Those portions are used under their MIT Licenses:

```
MIT License

Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## mjlab and supermimic_AMP (the G1 AMP-Flat environment)

- `assets/g1_mjlab.xml` is the Unitree G1 MJCF of mjlab 1.2.0 (https://github.com/mujocolab/mjlab,
  `src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`, as supermimic_AMP ships it), unmodified (mjlab's main branch has since changed it). mjlab is distributed under the Apache License, Version 2.0
  ([`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)).
- `instantamp/env.py` and `instantamp/sim.py` reimplement, in JAX and MuJoCo Warp, the G1 AMP-Flat task of supermimic_AMP
  (https://github.com/JonnyDing/supermimic_AMP) and the mjlab components it builds on (the G1 actuator constants, the Unitree actuator
  model, the velocity command and the observation history). Formulas were ported; no code was copied.

## Unitree G1 description

The named G1 get-up task uses the Unitree G1 rev 1.0 asset from
[LeggedLab, commit f5f54fe](https://gitee.com/chaomingsanhua/legged_lab/tree/f5f54fe506d7c2e558e07163fd6624ae3d44bbed/source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/usd/g1_29dof_rev_1_0).
`tools/convert_leggedlab_g1.py` converts the USD joints, inertials, primitive collisions, and convex collision meshes to
`assets/g1_leggedlab.xml`. The conversion retains up to 64 source vertices per hull for MuJoCo Warp.
The Unitree G1 description is covered by the BSD notice below. The LeggedLab repository carries the Apache 2.0 license.

The G1 model in `assets/g1_mjlab.xml` and the meshes that `tools/download_meshes.sh` downloads (from Holosoma's repository,
https://github.com/amazon-far/holosoma, at a fixed commit; only rendering needs them, and they are not included here) derive from
Unitree's G1 description, distributed under the BSD 3-Clause License:

```
BSD 3-Clause License

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Motion data

`tools/download_data.sh` retrieves the G1 walk-and-run clips from LeggedLab (AMASS / ACCAD retargets under AMASS's license) and Unitree's G1 retarget of LAFAN1. LAFAN1 is by Ubisoft La Forge and carries the CC BY-NC-ND 4.0 license. [LeggedLab](https://github.com/zitongbai/legged_lab) · [Unitree LAFAN1 retarget](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)

The eleven Zenbot recovery motions originate from the LAFAN1 recording `fallAndGetUp1_subject1.bvh`. The LeggedLab `get_up_frax` pipeline retargeted selected frame ranges to the 43 kg Zenbot using Frax differential inverse kinematics and MuJoCo ground-contact refinement. InstantAMP converted those results to 50 Hz NPZ files in the robot's joint order. [Source file and clip hashes](training_logs/zenbot23_getup_standing_amp_pd/recovery_sources.json) · [LAFAN1 license](https://creativecommons.org/licenses/by-nc-nd/4.0/)

The six Zenbot standing motions were generated separately in MuJoCo using the Zenbot model. [Generator and recording method](tools/prepare_zenbot_standing_amp.py) · [Standing dataset manifest](data/zenbot23_walk_run_recovery/standing/manifest.json)

The Zenbot joint WM run uses 17 locomotion and standing clips from the Motion Train `zenbot_amp_frax` bank. The preparation script reads these clips and their weights at the recorded source revision and verifies the converted files against the training hashes. Access to the original motion data and its terms follows the source repository. [Source revision](https://github.com/ZPLcode/h1_train/tree/aafe0101ade37656f5390a7320bbd026f1f3c171) · [Preparation](tools/recipes/prepare_zenbot_wm_data.sh) · [Reference checksums](assets/zenbot_wm_reference.sha256)

## Dependencies

JAX, MuJoCo, MJX, MuJoCo Warp and NVIDIA Warp (Apache-2.0), NumPy and Pillow are installed from their own distributions under their
own licenses and are not redistributed here.
