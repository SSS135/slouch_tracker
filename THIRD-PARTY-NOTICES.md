# Third-Party Notices

Slouch Tracker is licensed under the MIT License (see `LICENSE`,
Copyright (c) 2024 Slouch Tracker Contributors).

This application, its installer, and its distributed resources include
third-party software and machine-learning model files that are covered by
their own licenses. Those components remain under their respective licenses;
the notices below are provided to satisfy their attribution requirements. This
file covers material that is redistributed as part of a Slouch Tracker release
(bundled model files, the bundled ONNX Runtime, and code statically compiled
into the desktop binary), followed by a summary of the wider dependency tree.

Per-component sections list: name, version (where known), license,
attribution, and modifications.

---

## 1. Machine-learning models (OpenMMLab)

One ONNX model file is bundled in every release under
`src-tauri/resources/models/` and is loaded at runtime by the native ONNX
Runtime:

| File | Size | Role |
| --- | --- | --- |
| `rtmdet-nano.onnx` | ~4.1 MB | Person detection (RTMDet-nano) |

It derives from **OpenMMLab** projects:

- **RTMDet** — https://github.com/open-mmlab/mmdetection (MMDetection) — the
  detection architecture the bundled person detector is built on.
- **MMPose** — https://github.com/open-mmlab/mmpose (`projects/rtmpose`) — the
  project that distributes the RTMDet-nano person detector (at
  `projects/rtmpose/rtmdet/person/`) used by Slouch Tracker.

**License:** Apache License, Version 2.0.
**Attribution:** Copyright 2018-2020 OpenMMLab. All rights reserved.
(as stated in the MMPose and MMDetection `LICENSE` files).

> **Provenance.** The bundled model was exported from the Apache-2.0 OpenMMLab
> project **MMDetection** (RTMDet-nano person detector, the variant distributed
> with MMPose's `projects/rtmpose` demo pipeline). The maintainer has confirmed
> the export/re-export chain used MMDetection + MMPose only and did not install
> or import MMYOLO (a separate GPL-3.0 OpenMMLab project that also ships RTMDet
> configurations).

### 1a. `rtmdet-nano.onnx` — RTMDet-nano person detector

- **Upstream project:** MMDetection (RTMDet), used as the person detector for
  the RTMPose pipeline. The RTMDet-nano person detector is distributed by
  MMPose at `projects/rtmpose/rtmdet/person/` (config
  `rtmdet_nano_320-8xb32_coco-person.py`; the corresponding checkpoint
  published by OpenMMLab is
  `rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth`).
- **License:** Apache-2.0.
- **Modifications (bundled file is a modified export, not an upstream
  artifact):** the PyTorch checkpoint was exported to ONNX (opset 11,
  `producer_name = pytorch`, `producer_version = 2.4.1`) with detection
  post-processing baked in (graph outputs `dets` and `labels`) and with two
  additional intermediate P5 neck feature tensors exposed as extra graph
  outputs (`/bbox_head/cls_convs.2.1/pointwise_conv/activate/Mul_output_0` and
  `/bbox_head/reg_convs.2.1/pointwise_conv/activate/Mul_output_0`, 64 channels
  each) so the application can extract pooled RTMDet features. The ONNX file
  carries no `metadata_props`.

### 1c. Apache-2.0 obligations for the bundled model

Redistribution of the modified model export above is done under the Apache
License, Version 2.0. In accordance with that license this notices file:

1. retains the OpenMMLab attribution (Copyright 2018-2020 OpenMMLab);
2. provides the full text of the Apache License, Version 2.0 in Appendix A
   (also available at https://www.apache.org/licenses/LICENSE-2.0); and
3. states prominently that the bundled `.onnx` file is a **modified** export
   of the upstream RTMDet model (see the Modifications entry in section 1a).

---

## 2. NLF-L 3D human-pose model (isarandi/nlf)

One additional ONNX model file is bundled in every release under
`src-tauri/resources/models/` and loaded at runtime by the native ONNX Runtime
on the DirectML execution provider:

| File | Size | Role |
| --- | --- | --- |
| `nlf_l_crop_fp16.onnx` | ~233 MB | 3D human pose (Neural Localizer Fields, NLF-L); source of supplementary posture-depth features |

- **Upstream project:** **NLF (Neural Localizer Fields)** —
  https://github.com/isarandi/nlf
- **Paper:** István Sárándi and Gerard Pons-Moll, "Neural Localizer Fields for
  Continuous 3D Human Pose and Shape Estimation," NeurIPS 2024
  (arXiv:2407.07532).
- **Backbone lineage:** the NLF-L variant uses an **EfficientNetV2-L** backbone
  at 384 px input (per the paper: "We use EfficientNetV2-S (256 px) and L
  (384 px)"). EfficientNetV2 originates from Tan & Le (Google Research); NLF
  initializes it from ImageNet pretraining.
- **Code license:** the NLF source repository is licensed under the **MIT
  License**, Copyright (c) 2024 István Sárándi.
- **Model-weights license — IMPORTANT (non-commercial):** the released NLF
  model weights, from which the bundled ONNX file is derived, are — per the
  upstream project — **"released for noncommercial research use only."** This
  restriction applies to the pre-trained weights and therefore to the bundled
  `nlf_l_crop_fp16.onnx` file. The permissive MIT license covers the NLF
  *code*, not the *weights*. Any redistribution or commercial use of a Slouch
  Tracker build that includes this model file must comply with the NLF weights'
  non-commercial research-use restriction. (Contrast the OpenMMLab models in
  section 1, which are Apache-2.0 and carry no such restriction.)
- **Modifications (the bundled file is a modified export, not an upstream
  artifact):** the upstream NLF-L "crop" model was exported to ONNX and then
  converted to **fp16** weights, with float32 graph inputs/outputs preserved
  (`keep_io_types`). The application feeds a 384×384 RGB crop and reads the
  `coords3d_rel` and `uncertainty` outputs to derive scale/rotation-aware
  posture-depth features; only these derived features are stored, and the raw
  model outputs are not persisted.

---

## 3. ONNX Runtime (native, DirectML build)

- **Component:** Microsoft ONNX Runtime (`onnxruntime.dll` plus its companion
  `onnxruntime_providers_shared.dll`), the native inference engine bundled with
  the Windows release. This is the **DirectML build** of ONNX Runtime: it adds
  the DirectML execution provider (used to run the NLF-L model on the GPU) while
  retaining the CPU execution provider used — unchanged — for RTMDet.
- **Version:** ONNX Runtime 1.24.2 (DirectML build, per
  `src-tauri/resource-lock.json`). The staged binaries report file version
  `1.24.20260219.5.3ba85bd`.
- **License:** MIT License.
- **Attribution:** Copyright (c) Microsoft Corporation.
- **Bundled notices:** the ONNX Runtime license and its own third-party
  notices ship with the application under
  `src-tauri/resources/onnxruntime/notices/`:
  - `LICENSE` — the ONNX Runtime MIT license
  - `Privacy.md` — ONNX Runtime privacy statement
  - `ThirdPartyNotices.txt` — ONNX Runtime's upstream third-party notices
- **Upstream:** https://github.com/microsoft/onnxruntime
- **Modifications:** none; `onnxruntime.dll` and
  `onnxruntime_providers_shared.dll` are redistributed unmodified from the
  Microsoft.ML.OnnxRuntime.DirectML package.

The Rust binding to ONNX Runtime is the `ort` crate (version `2.0.0-rc.12`),
which is dual-licensed MIT OR Apache-2.0.

---

## 4. DirectML (Microsoft DirectX Machine Learning runtime)

- **Component:** `DirectML.dll` — Microsoft's DirectX Machine Learning
  hardware-acceleration library, bundled with the Windows release and loaded by
  the DirectML execution provider of ONNX Runtime to run the NLF-L model on the
  GPU.
- **Version:** Microsoft DirectML 1.15.4 (file version
  `1.15.4+241025-1615.1.dml-1.15.fac7597`), from the Microsoft.AI.DirectML
  package.
- **License:** proprietary **"Microsoft Software License Terms — Microsoft
  DirectX Machine Learning (DirectML)."** This is **not** an open-source
  license. It permits redistributing `DirectML.dll` as an integrated component
  within applications and services developed with machine-learning tools and
  frameworks that run on Windows/Xbox (as Slouch Tracker does), but prohibits
  distributing the library as a stand-alone offering. The full terms are
  available from Microsoft at https://aka.ms/directml.
- **Attribution:** Copyright (c) Microsoft Corporation.
- **Modifications:** none; redistributed unmodified.

---

## 5. mozjpeg / mozjpeg-sys (JPEG codec, compiled into the Windows binary)

MJPEG camera frames are decoded through the `nokhwa` capture stack, which on
Windows pulls in the mozjpeg JPEG codec. The mozjpeg C library is compiled
directly into the desktop binary.

- **Components:** `mozjpeg` (Rust wrapper, version 0.10.13) and `mozjpeg-sys`
  (native bindings/build, version 2.2.3), which build the mozjpeg /
  libjpeg-turbo C sources.
- **Licenses:** the mozjpeg / libjpeg-turbo sources are made available under a
  combination of the **IJG (Independent JPEG Group) license**, the
  **BSD-3-Clause** license (the libjpeg-turbo SIMD and modernized components),
  and the **Zlib** license.
- **Required acknowledgment (IJG):**

  > This software is based in part on the work of the Independent JPEG Group.

- **Upstream:** https://github.com/mozilla/mozjpeg and
  https://github.com/ImageOptim/mozjpeg-rust
- **Modifications:** none; redistributed/compiled unmodified.

---

## 6. Apache-2.0 licensed Rust dependencies (notable)

The following notable dependencies are used unmodified under the Apache
License, Version 2.0 (most are dual-licensed MIT OR Apache-2.0). Full license
text is in Appendix A.

| Component | Version | License |
| --- | --- | --- |
| `nokhwa` (camera capture) | 0.10.11 | Apache-2.0 |
| `nalgebra` (linear algebra) | 0.34.1 | Apache-2.0 |
| `tauri` (application framework) | 2.11.5 | Apache-2.0 OR MIT |
| `tauri-plugin-global-shortcut` | 2.3.1 | Apache-2.0 OR MIT |
| `tauri-plugin-dialog` | 2.7.1 | Apache-2.0 OR MIT |
| `tauri-plugin-log` | 2.9.0 | Apache-2.0 OR MIT |
| `ort` (ONNX Runtime binding) | 2.0.0-rc.12 | Apache-2.0 OR MIT |

Attribution: see each crate's repository for its authors and copyright lines.

---

## 7. MIT / ISC ecosystem dependencies

The broader Rust and TypeScript dependency trees (Tauri core and plugins,
`serde`, `rusqlite`, `ndarray`, `rmp-serde`, `sha2`, `zip`, `image`, the
`specta` / `tauri-specta` binding generators, the Svelte / Vite / Vitest
frontend toolchain, and their transitive dependencies) are predominantly
licensed under the **MIT** and **ISC** licenses, or are dual-licensed
**MIT OR Apache-2.0**. These permissive licenses require preservation of their
copyright and permission notices, which remain in the respective package
sources and distributed artifacts. All are used unmodified.

---

## 8. MPL-2.0 transitive dependencies (listed for completeness)

The following transitive crates are licensed under the **Mozilla Public
License, Version 2.0 (MPL-2.0)**. They are pulled in transitively (via the
Tauri / Webview styling stack) and are used **unmodified**; no MPL-covered
source files have been modified, so MPL-2.0's file-level source-disclosure
obligation is satisfied by pointing to the upstream sources. MPL-2.0 text is
available at https://www.mozilla.org/en-US/MPL/2.0/.

| Component | Version | License |
| --- | --- | --- |
| `cssparser` | 0.36.0 | MPL-2.0 |
| `cssparser-macros` | 0.6.1 | MPL-2.0 |
| `dtoa-short` | 0.3.5 | MPL-2.0 |
| `option-ext` | 0.2.0 | MPL-2.0 |
| `selectors` | 0.36.1 | MPL-2.0 |

---

## Appendix A — Apache License, Version 2.0

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

---

*This notices file was assembled for the public release of Slouch Tracker. It
covers redistributed model files, the bundled ONNX Runtime, code compiled into
the desktop binary, and a summary of the wider dependency licenses.*
