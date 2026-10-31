# PXCS 容器格式

本文记录 PXCS 的二进制布局、版本历史和兼容规则。所有整数均为小端序。

- 当前实现：1.4（最高支持 minor=4）
- 未加密归档写出 minor=2（与 1.2 完全一致）
- 含 jpegxl 条目的归档写出 minor=4（旧程序会报未知 codec）
- 加密归档写出 minor=3

## 1. 整体布局

### 1.1 未加密（1.0 – 1.2，以及 1.3 的未加密归档）

```text
+----------------------------+ 0
| Header           128 B     |
+----------------------------+ 128
| payload 0（按 8 字节对齐）  |
| payload 1                  |
| ...                        |
+----------------------------+ index_offset
| zlib(JSON Index)           |
+----------------------------+
| Footer            64 B     |
+----------------------------+ file_size
```

### 1.2 加密（1.3 起）

```text
+----------------------------+ 0
| Header           128 B     |
+----------------------------+ 128
| Crypto Header    128 B     |  明文，含 KDF 参数/盐/校验块/索引 MAC
+----------------------------+ 256  = data_offset
| payload 0（AEAD 密文）      |
| ...                        |
+----------------------------+ index_offset
| Index（见下）               |
+----------------------------+
| Footer            64 B     |
+----------------------------+ file_size
```

Index 在加密且文件名加密时是 `nonce(12) || AES-256-GCM(zlib(JSON))`，
否则仍是 `zlib(JSON)`（此时由 Crypto Header 里的 `index_mac` 保护完整性）。

## 2. Header（128 字节）

结构：`<4sHHIHIQQQ16s4s62sI`

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 4 | magic | `PXCS` |
| 4 | 2 | version_major | 必须为 1 |
| 6 | 2 | version_minor | 0..4，当前写出 2 / 3 / 4 |
| 8 | 4 | flags | 见下 |
| 12 | 2 | header_size | 固定 128 |
| 14 | 4 | entry_count | 条目数 |
| 18 | 8 | index_offset | Index 起始（逻辑偏移） |
| 26 | 8 | index_length | Index 长度 |
| 34 | 8 | data_offset | 128（未加密）或 256（加密） |
| 42 | 16 | archive_id | 随机 UUID，Header/Footer/Index 必须一致 |
| 58 | 4 | reserved4 | 必须为 0 |
| 62 | 62 | reserved62 | 必须为 0 |
| 124 | 4 | header_crc | 前 124 字节的 CRC32（本字段按 0 计算） |

flags：

| 位 | 名称 | 含义 |
|---|---|---|
| 0 | HAS_DELTA | 存在差分条目 |
| 1 | HAS_FILES | 存在非图片文件 |
| 2 | HAS_DUP | 存在重复条目 |
| 3 | HAS_RAW_IMAGES | 存在原始码流图片 |
| 4 | HAS_LOSSY_DELTA | 残差含量化（近似像素） |
| 5 | ENCRYPTED | 1.3：payload/索引使用 AEAD |

## 3. Footer（64 字节）

结构：`<8sQQQII16s8s`

| 偏移 | 大小 | 字段 |
|---|---|---|
| 0 | 8 | magic = `PXCSEND\0` |
| 8 | 8 | file_size（逻辑总长度） |
| 16 | 8 | index_offset |
| 24 | 8 | index_length |
| 32 | 4 | index_crc32（Index 密文/blobs 的 CRC32） |
| 36 | 4 | footer_crc32（本字段按 0 计算） |
| 40 | 16 | archive_id |
| 56 | 8 | reserved（必须为 0） |

## 4. Index（JSON + zlib）

```json
{
  "format": "PXCS",
  "version": 1,
  "archive_id": "<32 hex>",
  "created_by": "PXCS Python 1.3",
  "entries": [ ... ]
}
```

条目字段（缺省值用于向后兼容旧归档）：

| 字段 | 类型 | 说明 | 引入版本 |
|---|---|---|---|
| id | int | 连续，等于数组下标 | 1.0 |
| path | str | POSIX 相对路径 | 1.0 |
| kind | str | `image` / `file` | 1.0 |
| codec | str | pxc / pxc_residual / zlib_residual / webp_lossless / raw / jpegxl / store / zlib / dup | 1.0 |
| offset | int | payload 起始（逻辑偏移） | 1.0 |
| length | int | payload 在归档中占用的字节数（加密时含 nonce/tag） | 1.0 |
| plain_length | int? | 解密后的 payload 长度；**仅加密归档写入**，未加密时不出现该键（保持 1.2 索引字节不变）。缺省视为等于 length | 1.3 |
| size | int | 原始文件大小 | 1.0 |
| ref | int? | 引用条目 id（必须小于自身） | 1.0 |
| depth | int | 引用深度 0..255 | 1.0 |
| width/height/mode | | 图片属性 | 1.0 |
| quant | int | 残差量化步长，0=像素无损 | 1.2 |
| centered | bool | 残差以 128 为中心 | 1.2 |
| lossy | bool | 重建像素是近似值 | 1.2 |
| pixel_sha256 | str | 期望像素哈希（有损差分时为近似图哈希）。**raw 条目通常为空**：payload 就是源文件字节，字节级 SHA-256 已完整覆盖；空值表示不做像素级比对 | 1.0 |
| source_sha256 | str | 源文件字节哈希 | 1.0 |
| data_crc32 / data_sha256 | | **明文 payload** 的 CRC/SHA | 1.0 |
| raw_crc32 | int | 原始文件字节 CRC（raw/store/zlib） | 1.0 |
| mtime_ns | int | 修改时间 | 1.0 |
| reason | str | 编码决策原因，便于排查 | 1.2 |

加密归档下 `data_crc32/data_sha256` 仍然记录**明文**摘要：
读取时先 AEAD 解密，再校验，因此篡改密文会先被 GCM tag 拦住。

## 5. Crypto Header（128 字节，1.3）

结构：`<8sHHHH16sIII44s32s4sI`

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 8 | magic | `PXCSCRYP` |
| 8 | 2 | version | 1 |
| 10 | 2 | kdf_id | 1=Argon2id，2=scrypt |
| 12 | 2 | aead_id | 1=AES-256-GCM |
| 14 | 2 | flags | bit0=文件名已加密 |
| 16 | 16 | salt | 随机盐 |
| 32 | 4 | param1 | Argon2id: memory KiB；scrypt: log2(N) |
| 36 | 4 | param2 | Argon2id: iterations；scrypt: r |
| 40 | 4 | param3 | Argon2id: lanes；scrypt: p |
| 44 | 44 | check_blob | `nonce(12) || AEAD(token16)`，口令校验用 |
| 88 | 32 | index_mac | HMAC-SHA256(索引明文)，文件名未加密时使用 |
| 120 | 4 | reserved | 必须为 0 |
| 124 | 4 | header_crc | CRC32（本字段按 0 计算） |

密钥派生：

```text
master     = KDF(password_NFC, salt, params)
payload_key= HKDF-SHA256(master, info="pxcs/payload-key/v1", 32)
index_key  = HKDF-SHA256(master, info="pxcs/index-key/v1",   32)
check_key  = HKDF-SHA256(master, info="pxcs/check-key/v1",   32)
mac_key    = HKDF-SHA256(master, info="pxcs/index-mac/v1",   32)
```

AEAD 报文：

```text
payload = nonce(12) || AES-256-GCM(plaintext, aad = "PXCS1:payload" || archive_id)
index   = nonce(12) || AES-256-GCM(zlib(json), aad = "PXCS1:index" || archive_id)
check   = nonce(12) || AES-256-GCM(token16,  aad = "PXCS1:check"  || archive_id)
```

nonce 每条消息独立随机；条目数上限 1,000,000，远低于 96-bit 随机 nonce 的安全界。
`index_mac = HMAC-SHA256(mac_key, json_bytes)` 只在文件名未加密时写入，
用于防止改路径、改偏移、改长度。

## 6. 分卷（1.3）

分卷不改变上面的任何结构：把 1.1/1.2 的字节流按容量切开，每卷前面加 128 字节卷头。

卷头：`<8sHHIIQQ16s72sI`（128 字节）

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 8 | magic | `PXCSVOL\0` |
| 8 | 2 | version | 1 |
| 10 | 2 | flags | bit0=本卷是最后一卷 |
| 12 | 4 | volume_index | 1 起，必须与文件名一致 |
| 16 | 4 | volume_count | 总卷数 |
| 20 | 8 | logical_offset | 本卷首个字节在逻辑流中的偏移 |
| 28 | 8 | length | 本卷承载的逻辑字节数 |
| 36 | 16 | archive_id | 与内部归档一致 |
| 52 | 72 | reserved | 必须为 0 |
| 124 | 4 | header_crc | CRC32 |

约束（读取时逐条校验）：

- 文件名 `<base>.<3+位数字>`，卷号连续；
- `volume_index` 与文件名一致，`1..volume_count` 完整存在；
- `logical_offset[i] == sum(length[0..i-1])`（不允许空洞/重叠）；
- 文件实际大小 == `128 + length`；
- 只有最后一卷带 FINAL 标志；
- 所有卷 `archive_id` 相同。

读取时把 `(逻辑偏移, 长度)` 映射到 `(卷号, 卷内偏移)`，因此**单张图片的随机提取
不会因为分卷而失效**；跨卷的 payload 由读取层分段读出再拼接。

## 7. 版本历史与兼容规则

| 版本 | 变化 | 兼容策略 |
|---|---|---|
| 1.0 | 初始格式 | — |
| 1.1 | 差分/关键帧完善 | 读取 1.0 字段缺省 |
| 1.2 | 条目新增 `quant/centered/lossy`，量化残差 | 读取时给旧条目默认 0/False |
| 1.3 | 加密头、AEAD payload、`plain_length`、分卷 | 未加密仍写 minor=2；加密写 minor=3 |
| 1.4 | 新增 `jpegxl`：JPEG 无损重打包成 JPEG XL，可字节级还原原始文件 | 旧读者遇到该 codec 报错；`--no-jxl` 可退回 1.2 输出 |

解析规则：

1. `magic` 必须是 `PXCS`，`version_major` 必须是 1；
2. `version_minor` 不在 `(0,1,2,3,4)` 内 → `PxcVersionError`；
3. `flags & ENCRYPTED`：
   - 只有 `minor >= 3` 才允许；
   - `data_offset` 必须等于 256，且存在加密头；
   - 没有口令 → `PxcPasswordRequired`；口令错误/篡改 → `PxcAuthenticationError`；
4. 未加密：
   - `data_offset` 必须等于 128；
   - 加密相关代码完全不参与，旧归档行为不变；
5. 条目校验（偏移越界、ref 关系、路径规范化、codec 合法性）在解密索引之后统一执行。

写出版本选择：

- 无口令 → `minor = 2`（旧程序可读）；
- 有口令 → `minor = 3`；
- 是否分卷不影响写出的 minor。

## 8. 已知限制

- 分卷没有冗余，不能从丢失的卷恢复数据；
- 加密不隐藏归档总大小、条目数和每条 payload 的长度（只保护内容与文件名）；
- 索引整体仍受 256 MiB 上限约束；
- 单个 payload 必须能整体读入内存（大图依赖内存，而不是分卷）。
