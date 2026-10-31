"""PXCS 加密层：Argon2id/scrypt 派生 + AES-256-GCM 认证加密。

设计参照成熟实现的公开方案，不自创密码学：

- 口令派生：Argon2id（OWASP 推荐，``cryptography`` ≥ 44 提供），
  旧环境可回退到 scrypt（``hashlib.scrypt``，RFC 7914）；
- 认证加密：AES-256-GCM（NIST SP 800-38D），每条消息独立随机 96-bit nonce；
- 子密钥：HKDF-SHA256（RFC 5869）按用途分离 payload / index / MAC / check；
- 口令校验不使用哈希比对，而是解密一个 AEAD “校验块”；
- 每条 payload、索引、校验块都带附加认证数据（AAD），绑定归档 ``archive_id``，
  防止把别的归档的数据拼进来。

威胁模型：攻击者拿到归档文件但没有口令。文件结构、条目偏移、CRC 都是公开的，
机密性和完整性由 AEAD 保证；``index_mac`` 保护未加密文件名的索引（防止改路径/换偏移）。
"""
from __future__ import annotations

import hashlib
import os
import struct
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from pxc_core.errors import PxcError, PxcFormatError
from pxc_core.i18n import t
from pxc_core.utils import crc32

# ---------------------------------------------------------------- 常量

CRYPTO_MAGIC = b"PXCSCRYP"
CRYPTO_HEADER_SIZE = 128
CRYPTO_VERSION = 1

KDF_ARGON2ID = 1
KDF_SCRYPT = 2
KDF_NAMES = {KDF_ARGON2ID: "argon2id", KDF_SCRYPT: "scrypt"}

AEAD_AES256_GCM = 1
AEAD_NAMES = {AEAD_AES256_GCM: "aes-256-gcm"}

FLAG_NAMES_ENCRYPTED = 1 << 0

KEY_SIZE = 32
NONCE_SIZE = 12
TAG_SIZE = 16
SALT_SIZE = 16
CHECK_PLAINTEXT_SIZE = 16
CHECK_BLOB_SIZE = NONCE_SIZE + CHECK_PLAINTEXT_SIZE + TAG_SIZE  # 44

# 默认派生参数：256 MiB / 3 轮 / 4 lane，约 0.25s。
DEFAULT_ARGON2_MEMORY_KIB = 262144
DEFAULT_ARGON2_ITERATIONS = 3
DEFAULT_ARGON2_LANES = 4
DEFAULT_SCRYPT_LOG2_N = 17
DEFAULT_SCRYPT_R = 8
DEFAULT_SCRYPT_P = 1

# 读取归档时接受的参数范围，防止构造恶意 header 触发 OOM/长时间计算。
ARGON2_MEMORY_MIN = 8192
ARGON2_MEMORY_MAX = 2 * 1024 * 1024  # 2 GiB
ARGON2_ITERATIONS_MAX = 64
ARGON2_LANES_MAX = 16
SCRYPT_LOG2N_MIN = 10
SCRYPT_LOG2N_MAX = 22
SCRYPT_R_MAX = 16
SCRYPT_P_MAX = 8

CRYPTO_HEADER_STRUCT = struct.Struct("<8sHHHH16sIII44s32s4sI")
assert CRYPTO_HEADER_STRUCT.size == CRYPTO_HEADER_SIZE

_INDEX_AAD_TAG = b"PXCS1:index"
_PAYLOAD_AAD_TAG = b"PXCS1:payload"
_CHECK_AAD_TAG = b"PXCS1:check"


class PxcPasswordRequired(PxcError):
    """归档已加密，但没有提供口令。"""


class PxcAuthenticationError(PxcError):
    """认证失败：口令错误或数据被篡改/损坏。

    故意不区分“口令错误”和“数据损坏”，避免给出可区分的攻击信号。
    """


# ---------------------------------------------------------------- 依赖

def _cryptography():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.exceptions import InvalidTag
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的显式提示
        raise PxcError(t("crypto.missing_library")) from exc
    return AESGCM, hashes, HKDF, InvalidTag


def _argon2id_available() -> bool:
    try:
        from cryptography.hazmat.primitives.kdf.argon2 import Argon2id  # noqa: F401
    except ImportError:
        return False
    return True


def default_kdf() -> str:
    return "argon2id" if _argon2id_available() else "scrypt"


# ---------------------------------------------------------------- KDF

def normalize_password(password: str) -> bytes:
    """统一 NFC，避免 macOS/Windows 输入法产生的组合字符导致派生结果不同。"""
    return unicodedata.normalize("NFC", password).encode("utf-8")


def derive_key(
    password: str,
    salt: bytes,
    *,
    kdf_id: int,
    param1: int,
    param2: int,
    param3: int,
) -> bytes:
    """按归档 header 里的参数派生 32 字节主密钥。"""
    secret = normalize_password(password)
    if kdf_id == KDF_ARGON2ID:
        memory_kib, iterations, lanes = int(param1), int(param2), int(param3)
        if not (ARGON2_MEMORY_MIN <= memory_kib <= ARGON2_MEMORY_MAX):
            raise PxcFormatError(t("crypto.bad_kdf_param", name="memory", value=memory_kib))
        if not (1 <= iterations <= ARGON2_ITERATIONS_MAX):
            raise PxcFormatError(t("crypto.bad_kdf_param", name="iterations", value=iterations))
        if not (1 <= lanes <= ARGON2_LANES_MAX):
            raise PxcFormatError(t("crypto.bad_kdf_param", name="lanes", value=lanes))
        try:
            from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
        except ImportError as exc:  # pragma: no cover
            raise PxcError(t("crypto.missing_argon2")) from exc
        return Argon2id(
            length=KEY_SIZE,
            salt=salt,
            iterations=iterations,
            lanes=lanes,
            memory_cost=memory_kib,
        ).derive(secret)
    if kdf_id == KDF_SCRYPT:
        log2_n, r, p = int(param1), int(param2), int(param3)
        if not (SCRYPT_LOG2N_MIN <= log2_n <= SCRYPT_LOG2N_MAX):
            raise PxcFormatError(t("crypto.bad_kdf_param", name="log2N", value=log2_n))
        if not (1 <= r <= SCRYPT_R_MAX):
            raise PxcFormatError(t("crypto.bad_kdf_param", name="r", value=r))
        if not (1 <= p <= SCRYPT_P_MAX):
            raise PxcFormatError(t("crypto.bad_kdf_param", name="p", value=p))
        n = 1 << log2_n
        maxmem = 128 * n * r + 16 * 1024 * 1024
        return hashlib.scrypt(
            secret, salt=salt, n=n, r=r, p=p, dklen=KEY_SIZE, maxmem=maxmem
        )
    raise PxcFormatError(t("crypto.unknown_kdf", value=kdf_id))


def _hkdf(master: bytes, info: bytes, length: int = KEY_SIZE) -> bytes:
    _AESGCM, hashes, HKDF, _InvalidTag = _cryptography()
    return HKDF(
        algorithm=hashes.SHA256(), length=length, salt=b"", info=info
    ).derive(master)


# ---------------------------------------------------------------- 参数

@dataclass(frozen=True)
class CryptoParams:
    """写入归档时使用的派生参数。"""

    kdf_id: int = KDF_ARGON2ID
    param1: int = DEFAULT_ARGON2_MEMORY_KIB
    param2: int = DEFAULT_ARGON2_ITERATIONS
    param3: int = DEFAULT_ARGON2_LANES
    encrypt_filenames: bool = True

    @classmethod
    def from_config(
        cls,
        *,
        kdf: str = "auto",
        argon2_memory_kib: Optional[int] = None,
        argon2_iterations: Optional[int] = None,
        argon2_lanes: Optional[int] = None,
        scrypt_log2n: Optional[int] = None,
        scrypt_r: Optional[int] = None,
        scrypt_p: Optional[int] = None,
        encrypt_filenames: bool = True,
    ) -> "CryptoParams":
        name = (kdf or "auto").strip().lower()
        if name == "auto":
            name = default_kdf()
        if name == "argon2id":
            # 显式要求 Argon2id 但运行环境不支持时直接报错，不静默降级。
            if not _argon2id_available():
                raise PxcError(t("crypto.missing_argon2"))
            return cls(
                kdf_id=KDF_ARGON2ID,
                param1=int(argon2_memory_kib or DEFAULT_ARGON2_MEMORY_KIB),
                param2=int(argon2_iterations or DEFAULT_ARGON2_ITERATIONS),
                param3=int(argon2_lanes or DEFAULT_ARGON2_LANES),
                encrypt_filenames=bool(encrypt_filenames),
            )
        if name == "scrypt":
            return cls(
                kdf_id=KDF_SCRYPT,
                param1=int(scrypt_log2n or DEFAULT_SCRYPT_LOG2_N),
                param2=int(scrypt_r or DEFAULT_SCRYPT_R),
                param3=int(scrypt_p or DEFAULT_SCRYPT_P),
                encrypt_filenames=bool(encrypt_filenames),
            )
        raise PxcError(t("crypto.unknown_kdf_name", name=kdf))


# ---------------------------------------------------------------- 上下文

class CryptoContext:
    """一个归档的加密上下文；创建和读取共用。"""

    def __init__(
        self,
        password: str,
        archive_id: bytes,
        params: CryptoParams,
        salt: bytes,
        check_blob: bytes,
    ) -> None:
        if not password:
            raise PxcPasswordRequired(t("crypto.password_empty"))
        self.archive_id = bytes(archive_id)
        self.params = params
        self.salt = bytes(salt)
        self.check_blob = bytes(check_blob)
        self._payload_key = b""
        self._index_key = b""
        self._check_key = b""
        self._mac_key = b""
        self._derive(password)

    # ------------------------------------------------------------ 派生
    def _derive(self, password: str) -> None:
        master = derive_key(
            password,
            self.salt,
            kdf_id=self.params.kdf_id,
            param1=self.params.param1,
            param2=self.params.param2,
            param3=self.params.param3,
        )
        self._payload_key = _hkdf(master, b"pxcs/payload-key/v1")
        self._index_key = _hkdf(master, b"pxcs/index-key/v1")
        self._check_key = _hkdf(master, b"pxcs/check-key/v1")
        self._mac_key = _hkdf(master, b"pxcs/index-mac/v1")

    def _aad(self, tag: bytes) -> bytes:
        return tag + self.archive_id

    # ------------------------------------------------------------ AEAD
    def _seal(self, key: bytes, plaintext: bytes, aad: bytes) -> bytes:
        AESGCM, _h, _k, _i = _cryptography()
        nonce = os.urandom(NONCE_SIZE)
        return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)

    def _open(self, key: bytes, blob: bytes, aad: bytes) -> bytes:
        AESGCM, _h, _k, InvalidTag = _cryptography()
        if len(blob) < NONCE_SIZE + TAG_SIZE:
            raise PxcAuthenticationError(t("crypto.auth_failed"))
        nonce, body = blob[:NONCE_SIZE], blob[NONCE_SIZE:]
        try:
            return AESGCM(key).decrypt(nonce, body, aad)
        except InvalidTag as exc:
            raise PxcAuthenticationError(t("crypto.auth_failed")) from exc

    # ------------------------------------------------------------ payload
    def seal_payload(self, plaintext: bytes) -> bytes:
        return self._seal(self._payload_key, plaintext, self._aad(_PAYLOAD_AAD_TAG))

    def open_payload(self, blob: bytes) -> bytes:
        return self._open(self._payload_key, blob, self._aad(_PAYLOAD_AAD_TAG))

    # ------------------------------------------------------------ index
    def seal_index(self, plaintext: bytes) -> bytes:
        return self._seal(self._index_key, plaintext, self._aad(_INDEX_AAD_TAG))

    def open_index(self, blob: bytes) -> bytes:
        return self._open(self._index_key, blob, self._aad(_INDEX_AAD_TAG))

    def index_mac(self, index_raw: bytes) -> bytes:
        import hmac

        return hmac.new(self._mac_key, index_raw, hashlib.sha256).digest()

    def verify_index_mac(self, index_raw: bytes, expected: bytes) -> None:
        import hmac

        if not expected or not hmac.compare_digest(
            self.index_mac(index_raw), expected
        ):
            raise PxcAuthenticationError(t("crypto.auth_failed"))

    # ------------------------------------------------------------ check
    def build_check_blob(self) -> bytes:
        token = os.urandom(CHECK_PLAINTEXT_SIZE)
        return self._seal(self._check_key, token, self._aad(_CHECK_AAD_TAG))

    def verify_check(self) -> None:
        self._open(self._check_key, self.check_blob, self._aad(_CHECK_AAD_TAG))

    # ------------------------------------------------------------ misc
    @property
    def encrypt_filenames(self) -> bool:
        return bool(self.params.encrypt_filenames)

    def __repr__(self) -> str:  # 绝不输出口令
        return (
            f"CryptoContext(kdf={KDF_NAMES.get(self.params.kdf_id)}, "
            f"names_encrypted={self.encrypt_filenames})"
        )

def new_crypto_context(
    password: str,
    archive_id: bytes,
    params: CryptoParams,
) -> CryptoContext:
    """创建归档用：生成随机盐和校验块（密钥只派生一次）。"""
    salt = os.urandom(SALT_SIZE)
    ctx = CryptoContext(
        password, archive_id, params, salt, check_blob=b"\x00" * CHECK_BLOB_SIZE
    )
    ctx.check_blob = ctx.build_check_blob()
    return ctx


# ---------------------------------------------------------------- header

def pack_header(ctx: CryptoContext) -> bytes:
    flags = FLAG_NAMES_ENCRYPTED if ctx.encrypt_filenames else 0
    mac = b"\x00" * 32  # 索引 MAC 在收尾时回填，见 patch_header_mac
    zero = CRYPTO_HEADER_STRUCT.pack(
        CRYPTO_MAGIC,
        CRYPTO_VERSION,
        ctx.params.kdf_id,
        AEAD_AES256_GCM,
        flags,
        ctx.salt,
        ctx.params.param1,
        ctx.params.param2,
        ctx.params.param3,
        ctx.check_blob,
        mac,
        b"\x00" * 4,
        0,
    )
    return CRYPTO_HEADER_STRUCT.pack(
        CRYPTO_MAGIC,
        CRYPTO_VERSION,
        ctx.params.kdf_id,
        AEAD_AES256_GCM,
        flags,
        ctx.salt,
        ctx.params.param1,
        ctx.params.param2,
        ctx.params.param3,
        ctx.check_blob,
        mac,
        b"\x00" * 4,
        crc32(zero),
    )


def pack_header_with_mac(ctx: CryptoContext, index_mac: bytes) -> bytes:
    flags = FLAG_NAMES_ENCRYPTED if ctx.encrypt_filenames else 0
    raw = pack_header(ctx)
    zero = bytearray(raw)
    zero[88:120] = index_mac
    zero[124:128] = b"\x00\x00\x00\x00"
    return CRYPTO_HEADER_STRUCT.pack(
        CRYPTO_MAGIC,
        CRYPTO_VERSION,
        ctx.params.kdf_id,
        AEAD_AES256_GCM,
        flags,
        ctx.salt,
        ctx.params.param1,
        ctx.params.param2,
        ctx.params.param3,
        ctx.check_blob,
        index_mac,
        b"\x00" * 4,
        crc32(bytes(zero)),
    )


@dataclass
class CryptoHeader:
    version: int
    kdf_id: int
    aead_id: int
    flags: int
    salt: bytes
    param1: int
    param2: int
    param3: int
    check_blob: bytes
    index_mac: bytes

    @property
    def encrypt_filenames(self) -> bool:
        return bool(self.flags & FLAG_NAMES_ENCRYPTED)

    @property
    def kdf_name(self) -> str:
        return KDF_NAMES.get(self.kdf_id, str(self.kdf_id))

    @property
    def aead_name(self) -> str:
        return AEAD_NAMES.get(self.aead_id, str(self.aead_id))


def unpack_header(raw: bytes) -> CryptoHeader:
    if len(raw) != CRYPTO_HEADER_SIZE:
        raise PxcFormatError(t("crypto.header_truncated"))
    try:
        (
            magic,
            version,
            kdf_id,
            aead_id,
            flags,
            salt,
            param1,
            param2,
            param3,
            check_blob,
            index_mac,
            reserved,
            header_crc,
        ) = CRYPTO_HEADER_STRUCT.unpack(raw)
    except struct.error as exc:
        raise PxcFormatError(t("crypto.header_bad", reason=exc)) from exc
    if magic != CRYPTO_MAGIC:
        raise PxcFormatError(t("crypto.header_magic"))
    if version != CRYPTO_VERSION:
        raise PxcFormatError(
            t("crypto.header_version", version=version, supported=CRYPTO_VERSION)
        )
    if aead_id != AEAD_AES256_GCM:
        raise PxcFormatError(t("crypto.unknown_aead", value=aead_id))
    if kdf_id not in KDF_NAMES:
        raise PxcFormatError(t("crypto.unknown_kdf", value=kdf_id))
    if reserved != b"\x00" * 4:
        raise PxcFormatError(t("crypto.header_reserved"))
    zero = bytearray(raw)
    zero[124:128] = b"\x00\x00\x00\x00"
    if crc32(bytes(zero)) != header_crc:
        raise PxcFormatError(t("crypto.header_crc"))
    return CryptoHeader(
        version=version,
        kdf_id=kdf_id,
        aead_id=aead_id,
        flags=flags,
        salt=bytes(salt),
        param1=param1,
        param2=param2,
        param3=param3,
        check_blob=bytes(check_blob),
        index_mac=bytes(index_mac),
    )


def open_from_header(
    header: CryptoHeader,
    password: str,
    archive_id: bytes,
) -> CryptoContext:
    """按 header 派生密钥并验证校验块；口令错误/篡改都抛认证错误。"""
    params = CryptoParams(
        kdf_id=header.kdf_id,
        param1=header.param1,
        param2=header.param2,
        param3=header.param3,
        encrypt_filenames=header.encrypt_filenames,
    )
    ctx = CryptoContext(password, archive_id, params, header.salt, header.check_blob)
    ctx.verify_check()
    return ctx


def crypto_summary(header: CryptoHeader) -> str:
    """给 info 用的公开信息摘要（不含任何口令派生结果）。"""
    return t(
        "crypto.summary",
        kdf=header.kdf_name,
        aead=header.aead_name,
        p1=header.param1,
        p2=header.param2,
        p3=header.param3,
        names=t("crypto.names_encrypted") if header.encrypt_filenames
        else t("crypto.names_plain"),
    )


def password_strength_warning(password: str) -> Optional[str]:
    """返回口令强度提示（不阻断创建）。"""
    if len(password) < 8:
        return t("crypto.weak_password")
    return None


def describe_params(params: CryptoParams) -> dict[str, Any]:
    return {
        "kdf": KDF_NAMES.get(params.kdf_id, str(params.kdf_id)),
        "param1": params.param1,
        "param2": params.param2,
        "param3": params.param3,
        "encrypt_filenames": params.encrypt_filenames,
    }
