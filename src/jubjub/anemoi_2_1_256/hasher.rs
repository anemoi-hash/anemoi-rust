//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, STATE_WIDTH};
use super::{Jive, Sponge};

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// An Anemoi hash instantiation
pub struct AnemoiHash {
    state: [Felt; STATE_WIDTH],
    idx: usize,
}

impl Default for AnemoiHash {
    fn default() -> Self {
        Self {
            state: [Felt::zero(); STATE_WIDTH],
            idx: 0,
        }
    }
}

impl Sponge<Felt> for AnemoiHash {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut buf = [0u8; 32];
        for (i, chunk) in bytes.chunks(31).enumerate() {
            if i < num_elements - 1 {
                buf[0..31].copy_from_slice(chunk);
            } else {
                // The last chunk may be smaller than the others, which requires a special handling.
                // In this case, we also append a byte set to 1 to the end of the string, padding the
                // sequence in a way that adding additional trailing zeros will yield a different hash.
                let chunk_len = chunk.len();
                buf = [0u8; 32];
                buf[..chunk_len].copy_from_slice(chunk);
                // [Different to paper]: We pad the last chunk with 1 to prevent length extension attack.
                if chunk_len < 31 {
                    buf[chunk_len] = 1;
                }
            }

            // Convert the bytes into a field element and absorb it into the rate portion of the
            // state. An Anemoi permutation is applied to the internal state if all the the rate
            // registers have been filled with additional values. We then reset the insertion index.
            state[0] += Felt::read(&buf[..]).unwrap();
            apply_permutation(&mut state);
        }
        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        let mut digest_array = [state[0]; DIGEST_SIZE];
        apply_permutation(&mut state);
        digest_array[1] = state[0];

        Self::Digest::new(digest_array)
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        for &element in elems.iter() {
            state[0] += element;
            apply_permutation(&mut state);
        }

        state[STATE_WIDTH - 1] += Felt::one();

        // Squeezing phase

        let mut digest_array = [state[0]; DIGEST_SIZE];
        apply_permutation(&mut state);
        digest_array[1] = state[0];

        Self::Digest::new(digest_array)
    }

    // This will require 2 calls of the underlying Anemoi permutation.
    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        Self::hash_field(&Self::Digest::digests_to_elements(digests))
    }
}

impl Jive<Felt> for AnemoiHash {
    /// This instantiation of the Anemoi hash doesn't support the
    /// Jive compression mode due to its targeted security level,
    /// incompatible with the output length of Jive.
    ///
    /// Implementers aiming at using this instantiation in a setting
    /// where a compression mode would be required would need to
    /// implement it directly from the sponge construction provided.
    fn compress(_elems: &[Felt]) -> Vec<Felt> {
        unimplemented!()
    }

    /// This instantiation of the Anemoi hash doesn't support the
    /// Jive compression mode due to its targeted security level,
    /// incompatible with the output length of Jive.
    ///
    /// Implementers aiming at using this instantiation in a setting
    /// where a compression mode would be required would need to
    /// implement it directly from the sponge construction provided.
    fn compress_k(_elems: &[Felt], _k: usize) -> Vec<Felt> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    use super::super::BigInteger256;
    use super::*;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x6742fa627387cff7,
                0xe15dc56bb93d2de7,
                0xa8c64c4ffa1772e2,
                0x71e3cc8d5796a90b,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xd682118fd37f57b7,
                    0xc3565b3eca40765a,
                    0x25b23b836ea2f0b0,
                    0x6a0b2d76c611a5f3,
                ])),
                Felt::new(BigInteger256([
                    0xf2afd5daa4fce9fe,
                    0x3f354bf35c3d70d6,
                    0x84c0589f52bf6306,
                    0x350080ce2be07bf2,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe826f8f54ab4fae4,
                    0xc956fb3187d7ceac,
                    0x5bc5c8e0d8b32a42,
                    0x6ba803d763a9dcdc,
                ])),
                Felt::new(BigInteger256([
                    0x998cab7eeb21d486,
                    0x78a59e14ac20b923,
                    0x40432539ac63f578,
                    0x37397b6e7829893b,
                ])),
                Felt::new(BigInteger256([
                    0x4f00b886c4341cab,
                    0x64241a37323968f5,
                    0xe14c84ee0ad8d1aa,
                    0x68ee779038c5b8e4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x61023461213df241,
                    0xd7853710b38c3a97,
                    0xf3b0ab803e505f7b,
                    0x10cf765cd2a53699,
                ])),
                Felt::new(BigInteger256([
                    0x1accdd20ae793f21,
                    0xcfd9993e5305ccf0,
                    0xbe3e520887d55f2d,
                    0x09562014969985d1,
                ])),
                Felt::new(BigInteger256([
                    0x2666a87bf96d58a2,
                    0x0264253880be37f6,
                    0xf61f396e4f25c251,
                    0x1117d938ed083d45,
                ])),
                Felt::new(BigInteger256([
                    0xa6cf66b6a10a123c,
                    0x44366370d113c36f,
                    0x3fbedbd833210303,
                    0x4b155a23fa9943dd,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x1b9c9d1b196a1e86,
                    0x961b3b0ad21a861e,
                    0xd3d99fd0562a80b6,
                    0x5ebac5eca406f79f,
                ])),
                Felt::new(BigInteger256([
                    0x44720b6becd08609,
                    0xea919f50284d2ba2,
                    0xdbe3eeb74ecd0a93,
                    0x0790602f552ecaba,
                ])),
                Felt::new(BigInteger256([
                    0xd13a8c6f82029fc2,
                    0x36ffba76446b4d36,
                    0x9fab0ca8bb28325a,
                    0x171a153ba72767c3,
                ])),
                Felt::new(BigInteger256([
                    0xf79f0c922382bfbd,
                    0x9d89c09cf5eaf328,
                    0x257d6165d46f01cf,
                    0x1fbacd6fb757c7cf,
                ])),
                Felt::new(BigInteger256([
                    0x842855a6f945d8ca,
                    0xa67601abfed51def,
                    0x7898caee05225269,
                    0x3a8ea5b7639adcf5,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa5ccdcd49cc51f2c,
                    0x6efb26be428c3063,
                    0x6ae7933a48e0fbae,
                    0x5355eab51e0d3ac9,
                ])),
                Felt::new(BigInteger256([
                    0x13d256fe3a13a8af,
                    0x57008f32c309332b,
                    0xd6caa1327e93e5ae,
                    0x05d78447eff10eff,
                ])),
                Felt::new(BigInteger256([
                    0x70d61e2efcda0aab,
                    0xa21e934bfd1edb9b,
                    0x19812f2832af82b8,
                    0x648c6dac763d17c3,
                ])),
                Felt::new(BigInteger256([
                    0x525d562bc40d6373,
                    0x794a5961cc9cf98b,
                    0xe1fe234e7b3a53d8,
                    0x0917cbc9bc85c52e,
                ])),
                Felt::new(BigInteger256([
                    0xf9acefe3f7db0ff9,
                    0xfd1c152141618675,
                    0x815a45cfbbb2d767,
                    0x63f4c9112aadff70,
                ])),
                Felt::new(BigInteger256([
                    0x4146589edd51fa98,
                    0xe5ab3231b078472f,
                    0xfecbaf2103f2fb7e,
                    0x54c584c9feda75a2,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x983623796f622394,
                    0x60d9b6e110c78927,
                    0x37fc89d57946267a,
                    0x30d49b75b9db8616,
                ])),
                Felt::new(BigInteger256([
                    0xeb9274a96695dd29,
                    0x0d9a7248e7dc8912,
                    0x6f1908a4acf84061,
                    0x292c35911380bd93,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6101332fafb4488b,
                    0x4d7b902967df986b,
                    0xe0b307ce87cb8d7d,
                    0x2a1ff40c2ae8d97b,
                ])),
                Felt::new(BigInteger256([
                    0xc7ab2379306e4c2b,
                    0x8557e3b6b094a50a,
                    0xedd2311f7db94ed5,
                    0x052e864957bc63cd,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa8affaca27d0d859,
                    0x77cd3b35f35b272b,
                    0xc376b696491a4dfc,
                    0x30820520376b8cb4,
                ])),
                Felt::new(BigInteger256([
                    0x2eb2b0a3c2115300,
                    0x68a12e92fa866fd2,
                    0x46a6009b05f4d3c1,
                    0x0b4af117d1876f5c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x87c99b46e9d198aa,
                    0x6912a8b8ddb8f08c,
                    0x9aa3dbc0b1a4b1c5,
                    0x23ea836d067289ba,
                ])),
                Felt::new(BigInteger256([
                    0x3436cfb295018d2d,
                    0x1af10425cba0cc3c,
                    0x1222e0a884bcad58,
                    0x052cae1509d9bc78,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb9016d111726c8ce,
                    0xe948299b8381bccc,
                    0x4c686fda9eb9b821,
                    0x0884b4c38779cfc3,
                ])),
                Felt::new(BigInteger256([
                    0xa6661a243fa60213,
                    0x9fc8148d5a9366ab,
                    0x1d40cd235ea45b49,
                    0x5a7781cbfa1432de,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcd9318d1af025eca,
                    0x7bb0156024c833be,
                    0x4823f621f93e2508,
                    0x0b22a5edd7f4a430,
                ])),
                Felt::new(BigInteger256([
                    0x90b7266294382712,
                    0x1cbccdc10385c633,
                    0xa9a27a7a8e0ae65b,
                    0x287481a468d10bd1,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x0dcfc249b1e785df,
                    0xdd7285064bcf1618,
                    0x49f5124bab1b5045,
                    0x041653895545d559,
                ])),
                Felt::new(BigInteger256([
                    0x186ccb25466c3384,
                    0x8d176702a6af7a42,
                    0x45d429bc332da1ae,
                    0x5e1f8516024aa156,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4bed3edc4a12a769,
                    0xbcca6327be39d3d5,
                    0x8d245df254b2a592,
                    0x064e38bac95b26db,
                ])),
                Felt::new(BigInteger256([
                    0x3680a676a9fe0891,
                    0x13b8c27542cc2b42,
                    0xbe46c2310bddea94,
                    0x17205f5f2ccbfed5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xb96c25aa66897f59,
                    0x9c5d0d542dae8662,
                    0xe21f60bf99d44c14,
                    0x0ab80987da5c01d4,
                ])),
                Felt::new(BigInteger256([
                    0xa4d98dc57d07d6ff,
                    0xb62e20e8b1272d13,
                    0xf6b22b6a876164ef,
                    0x46a634c3c34cde1c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x4fd9643b9de4acaf,
                    0x5d77f13fdb149f34,
                    0x43a6c65b82a2c3a6,
                    0x5e3fe6881e7f761b,
                ])),
                Felt::new(BigInteger256([
                    0x2065b802aac1edbb,
                    0xaacee936fe6c1070,
                    0x423b2ebed6fe2075,
                    0x2130cb8f1a579e32,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }
}
