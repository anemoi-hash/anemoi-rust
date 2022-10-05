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
                0x272308c9a3847d35,
                0x565a6d2929764fd7,
                0xfd93e63947ed21c6,
                0x099e399a63ceb48a,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x719f159ae101656d,
                    0x5274765c54f1576e,
                    0x3037d6155fad1e1b,
                    0x0d2658d18bec8efa,
                ])),
                Felt::new(BigInteger256([
                    0x6613afbfdc3e8ed0,
                    0x94da4a3b1d8e710d,
                    0x17ef0269ed3efd51,
                    0x1f836d0dde4c48f6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x14c71dee1166ea55,
                    0xb1f1c6ed99a0089b,
                    0x82aaf349ce9a8079,
                    0x177a4ee1247046bd,
                ])),
                Felt::new(BigInteger256([
                    0x4f8f2176dbf60a29,
                    0x7f29bef9b70725c1,
                    0x90cdfb3c8f5d03d9,
                    0x32a20ef667148700,
                ])),
                Felt::new(BigInteger256([
                    0xc28c55009f19d1ac,
                    0xa0b9ba782d035aea,
                    0xd8f2b4f46e9aa680,
                    0x37a7097748b4e590,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x97221e307b20e460,
                    0x011bdc2ef675a64f,
                    0x8f0b85695fdde117,
                    0x17a9f2f540b89575,
                ])),
                Felt::new(BigInteger256([
                    0xb6199aa69eb383ff,
                    0x2d6fe5d6f6d25c0d,
                    0x5e02ea9b611e184d,
                    0x128142f57c1b058c,
                ])),
                Felt::new(BigInteger256([
                    0x01fb6eda17b3a2e9,
                    0xa9672c6569a5e1fb,
                    0x54c692b5354167c8,
                    0x193dba2c2befdf01,
                ])),
                Felt::new(BigInteger256([
                    0x5e8c7c1b333ea806,
                    0xa681dfd4a512a7f6,
                    0xdd1b9ee68aefa962,
                    0x2124326974037d1b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8059d57fdf5b0865,
                    0x55163410a3951ba3,
                    0xa7b02f5e52b427f0,
                    0x1ce8bf7c100b8659,
                ])),
                Felt::new(BigInteger256([
                    0x4fc441e3477c40e2,
                    0xa46d68f4758b4154,
                    0x9e548ac1a4841015,
                    0x177ba521d36c4b84,
                ])),
                Felt::new(BigInteger256([
                    0x21269c8a2f20c617,
                    0x2f1dc24a229935d6,
                    0x238aea63b82479ca,
                    0x1977976322b5f302,
                ])),
                Felt::new(BigInteger256([
                    0x0fb217cd96417acd,
                    0x0f1fc5523cbcb1ad,
                    0x7fd1b2d50745a77e,
                    0x229b9cb7713e1b75,
                ])),
                Felt::new(BigInteger256([
                    0xc18138b5efed5d45,
                    0x86abe32176689011,
                    0x06d0ba9e0f8ca63e,
                    0x3a7e840eb20b0c76,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x3601071a023a3e99,
                    0x02b5e06e936f1e3d,
                    0x1793f5c7fe976b80,
                    0x2428ff01fd17fd8b,
                ])),
                Felt::new(BigInteger256([
                    0x97386a8fe1a71868,
                    0x940370623132d101,
                    0x3abeabc7910d3316,
                    0x109a3241653e3574,
                ])),
                Felt::new(BigInteger256([
                    0x037c8d66e021a742,
                    0x840778f2b3b1bae7,
                    0x61672ed0a5c13765,
                    0x140e11e366b62b8b,
                ])),
                Felt::new(BigInteger256([
                    0x7e80b78e325c2277,
                    0x364d24df76876305,
                    0x9630cdba95c66a3f,
                    0x13be78111233efee,
                ])),
                Felt::new(BigInteger256([
                    0xf307f1cdf21e12f1,
                    0x21ba7f2619333b2d,
                    0x20fdbfa6f7b9502a,
                    0x24ffd79ac80d6852,
                ])),
                Felt::new(BigInteger256([
                    0x4cd1369b9fb66eb2,
                    0x040602b384d1909d,
                    0xf61d84cf3c046660,
                    0x2bcd1fb4ec98fb51,
                ])),
            ],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0x2b954d8d8cd7fec1,
                    0x6afa618939e13cc0,
                    0x56ff5c57cdd91232,
                    0x02b0e686cc332362,
                ])),
                Felt::new(BigInteger256([
                    0x6eb34fc5b1973e1e,
                    0x3547add0cbcb6843,
                    0x25cb6ce96996f7bb,
                    0x3693b102033fcfc0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7ee88182cd38fe66,
                    0x862729c79298dca7,
                    0x20be9dfcb711b1b4,
                    0x0d0e91103fefd468,
                ])),
                Felt::new(BigInteger256([
                    0x2ae245a1691802f5,
                    0x69f61bea6397db6c,
                    0x0a4e7a1fd85e08bd,
                    0x0940c3b5efb5be8f,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x07b0696b165ca2fc,
                    0x4d5802c20049ad42,
                    0x106cc2550caedaae,
                    0x2ce7ef94ac4405cd,
                ])),
                Felt::new(BigInteger256([
                    0xfef570613467db16,
                    0x75e24ee7ab90f553,
                    0x99373be1badb7bae,
                    0x2f73c0ea0309f413,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x02cf064a0913eaaa,
                    0xde692e5d4d253652,
                    0xec47042918041606,
                    0x2cab21107e7b11f4,
                ])),
                Felt::new(BigInteger256([
                    0xb5d91b9f24dddddb,
                    0x62a72439aa0d0f11,
                    0x2a6dde73210bd853,
                    0x37cf53b203f30292,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x3c69f8cbb541aa49,
                    0x77a43f73043fd182,
                    0xac0d4ce4685d68b5,
                    0x09cb7c479b10db4f,
                ])),
                Felt::new(BigInteger256([
                    0x24c226c189e31ae9,
                    0x78aad301dd388fa2,
                    0x15c10ae419dd5c7d,
                    0x10b8cfcc73f95d83,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xbc49d69cb9d0cd27,
                    0x5deee460b659fafa,
                    0xb77e4d05121fb320,
                    0x372a97e2bb3b360e,
                ])),
                Felt::new(BigInteger256([
                    0x597e0ddf31e36062,
                    0x69ba065202eef2b6,
                    0x86cb2ad3978512c0,
                    0x230dffb266782ec8,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x66d54d5b296a7b88,
                    0x240c193aadc36ff1,
                    0xf531ed1accd1b401,
                    0x185326af59285da5,
                ])),
                Felt::new(BigInteger256([
                    0x1b8c92d8466acaf3,
                    0x2d2d4edba46fb076,
                    0x5d3d72fbcf9dc82c,
                    0x09d1fde50607d69c,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x6e24183463208100,
                    0xe9b6af8653b09ca5,
                    0x979848cf67a0ea0d,
                    0x0d60c59b9038bcd1,
                ])),
                Felt::new(BigInteger256([
                    0xb74231840ecd0f22,
                    0x9c79068883d5d4a3,
                    0x64ae4ae64733d0ff,
                    0x1228e096da629564,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x1db4a32ed5144c17,
                    0x0b26f79163630cd9,
                    0x1e3a5801dc7a6b62,
                    0x35eed03f9b350bab,
                ])),
                Felt::new(BigInteger256([
                    0xf8b305cc8afc8e25,
                    0x47336b4adc45f038,
                    0xfc62baa6df99dfe3,
                    0x2a95bf476a6785b0,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x7a64da37431ae9c8,
                    0x9d71a4b3c8b4495a,
                    0xb4aabbd710e672fe,
                    0x0064f11e2540159d,
                ])),
                Felt::new(BigInteger256([
                    0x800df6800887fadc,
                    0x17fdce578413eba0,
                    0xdad72646745566b6,
                    0x065e95ef21b915c0,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }
}
