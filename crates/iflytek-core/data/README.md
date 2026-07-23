# EdgeEsr numerical tables

这些静态表来自参考仓库 `azazo1/iflytek-offline-asr-service` 的 commit `c624fd6138c1e67133fe7aa833c4bd8cd2970b8d`.

它们用于复现原始 xLite 的 FP16 GELU, exp2, reciprocal, reciprocal square root 和固定点声学前端行为. 运行时通过 `include_bytes!` 内置数据, 不加载 vendor C++, OpenCL 或 Python 组件.

| 文件 | SHA-256 |
| --- | --- |
| `original_frontend_tables.bin` | `c21f7b7ab1a1308080e24df8848efdc067af657977273227493629a41ace170d` |
| `xlite_gelu_fp16_lut.bin` | `d344d710d51c0818fa4f963d6cb00f1488e391d588e30848ca553d5554d22874` |
| `xlite_exp2_f32_periodic_corrections.bin` | `daf4ba40824b7e11355b89ab892a835bcf3b1e703db3d8637b24fcbfb9833c61` |
| `xlite_exp2_f32_subunit_corrections.bin` | `82ec5e2fa8da8042e35e35e2b94c2ce8149f80b360e92bcb43de37272fef2b2f` |
| `xlite_reciprocal_f32_corrections.bin` | `9e62e57b1413fcfbcf95d9809e463498a30a917fc6c8b351015bffd3e6855a79` |
| `xlite_rsqrt_f32_corrections.bin` | `23438926ae326d36feedfeb39979a8319f4481b5b17ce233acdf40cc9a1a3b34` |
