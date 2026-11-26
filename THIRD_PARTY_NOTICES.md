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

Zenbot recovery references are derived from the LAFAN1 take `fallAndGetUp1_subject1.bvh` using the prepared LeggedLab FRAX bank. The 23-DoF gait references use Motion Train’s `zenbot_amp_frax` bank. The 74k release and the 26-DoF retarget use `zenbot_amp_frax_walk_sole8`. The preparation tools read separately obtained inputs and generate local 23- and 26-joint clips. Check the source dataset permissions before sharing recordings or converted motion files. [LAFAN1 source and license](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) · [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## Zenbot robot descriptions

The 23- and 26-DoF MJCF files in `assets/` come from Motion Train's RoboLab robot descriptions. The simulator uses actuator calibration from the corresponding models. The source project's BSD-3-Clause notice is included under `licenses/`. [Robot source revision](https://github.com/ZPLcode/h1_train/tree/4733fc9649dc3f2b99b2c3ba21fb2ec5e259e6ff/robolab/data/robots/roboparty) · [RoboLab BSD-3-Clause notice](licenses/Zenbot-BSD-3-Clause.txt)

The Zenbot23 and Zenbot26 dynamics sampling, shoulder/hip force schedule and mild-terrain parameters in `instantamp/zenbot_domain_randomization.py`, `instantamp/zenbot_push.py` and `instantamp/zenbot_terrain.py` follow Motion Train's local revision `ad923704f7fa1835730d04a10023b15d4e2b29e3`. They are implemented for JAX and native MuJoCo contact geometry. The source RoboLab notice above applies to the adapted task logic.

## Dependencies

JAX, MuJoCo, MJX, MuJoCo Warp and NVIDIA Warp (Apache-2.0), NumPy and Pillow are installed from their own distributions under their
own licenses and are not redistributed here.
