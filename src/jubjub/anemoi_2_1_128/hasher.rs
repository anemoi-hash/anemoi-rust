//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{apply_permutation, DIGEST_SIZE, NUM_COLUMNS, STATE_WIDTH};
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

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
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

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // We use internally the Jive compression method, as compressing the digests
        // through the Sponge construction would require two internal permutation calls.
        let result = Self::compress(&Self::Digest::digests_to_elements(digests));
        Self::Digest::new(result.try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiHash {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        apply_permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        // This instantiation only supports Jive-2 compression mode.
        assert!(k == 2);

        Self::compress(elems)
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
                0x2333e028a2c28cbf,
                0xfe6f46a0ce0beccb,
                0x1c5c5075bb64325d,
                0x2b77126cd4777006,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x90afcbc4f0c5f077,
                    0x0940eb17cb403e9d,
                    0x01621995276ae677,
                    0x542c12468cfa2fae,
                ])),
                Felt::new(BigInteger256([
                    0xc1953fe8f173887a,
                    0x992f9a97690e54a6,
                    0xa8b553add55edbb8,
                    0x3a9c613778dbe943,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x37d698f1a540cd6c,
                    0xf9ae144d330dff6c,
                    0x3ecb42e97f17d106,
                    0x411899c70298a1a7,
                ])),
                Felt::new(BigInteger256([
                    0x223e93c95f1f6a7b,
                    0xab8a77ba06bc75a3,
                    0xb01c572f65a5d407,
                    0x69e68fcabe6ef6ee,
                ])),
                Felt::new(BigInteger256([
                    0xf53f7e212bf93f79,
                    0x7f9504437d335461,
                    0xdc925d87d398ab18,
                    0x30c6fe470bdd151a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x06e90a7d9b169290,
                    0xc40511a06e636677,
                    0x34fff69e4a66a46b,
                    0x18b7e62a6fba8a9b,
                ])),
                Felt::new(BigInteger256([
                    0xc803094cf2da2193,
                    0x738758563677e360,
                    0xd524f6b280d708ce,
                    0x08a48616af5e8e1f,
                ])),
                Felt::new(BigInteger256([
                    0xfe730b031eb3ac5f,
                    0x4792dd0fed2d11dd,
                    0xe703e7eb494720cd,
                    0x4aa4083105713002,
                ])),
                Felt::new(BigInteger256([
                    0x9f1c27ba0c7c1c5b,
                    0x8395d487e942e59d,
                    0xe06db9d430cb691c,
                    0x4f4e79fc5b8f1da4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2182e5a8e98f89e7,
                    0x999be15fe49c48f6,
                    0xa3ccdd03cd685ded,
                    0x189ac63918ba85cb,
                ])),
                Felt::new(BigInteger256([
                    0xa73c8cadca7101d2,
                    0xcca7c3417a539c94,
                    0x614e839091485da8,
                    0x4754dca5cb718d03,
                ])),
                Felt::new(BigInteger256([
                    0x530f707c33b7ad73,
                    0xcd31ce30f0ea09de,
                    0x855f070ef0b44763,
                    0x16c7dc7707a8200d,
                ])),
                Felt::new(BigInteger256([
                    0xf97af64ea61aed37,
                    0x731856e5ac931163,
                    0xa45d528dfe43c8e2,
                    0x0a8651abc852f075,
                ])),
                Felt::new(BigInteger256([
                    0xcdb6b6d7ccd5d438,
                    0xa8811d3da30e837c,
                    0x938e7bbf542181c0,
                    0x2d55d8356609c6c6,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x407a5c1a15ba3d29,
                    0x1874ef298048f44d,
                    0xfc29ecdaa0e85b42,
                    0x2744023c60f46da5,
                ])),
                Felt::new(BigInteger256([
                    0x4f1c38b0919ddccc,
                    0xd336b1168429bbe2,
                    0xba4039465f862fd8,
                    0x627429185a941ccb,
                ])),
                Felt::new(BigInteger256([
                    0x55fd426c1b611f6c,
                    0xa99ec261ca7a3895,
                    0x74d297974d71b97d,
                    0x4a10b5227b5c84b4,
                ])),
                Felt::new(BigInteger256([
                    0x002a00a13efa5b79,
                    0x6a787672570c7496,
                    0x181a0c4dde0aead1,
                    0x4e6ac03de86fb28e,
                ])),
                Felt::new(BigInteger256([
                    0x65b2f2d0a9b946a1,
                    0xe4208f18ff815610,
                    0xfb165fe5b1a0a03e,
                    0x3dbd1a1be0e1903d,
                ])),
                Felt::new(BigInteger256([
                    0x60266df9968f153a,
                    0x201f86a86a6b079d,
                    0x5b695e2d4fdca393,
                    0x14d17e0ae807124a,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xa8ff0e52ff3f965f,
                0x632978eadf3b665e,
                0xc251aabf3c86aaee,
                0x3107c0f2b8423ba0,
            ]))],
            [Felt::new(BigInteger256([
                0x2f49b6fa5224957a,
                0xabe184b8859812d6,
                0x2f5018e00f40206f,
                0x0523c501e6634c0f,
            ]))],
            [Felt::new(BigInteger256([
                0x39e34d1336d0f37f,
                0x9851f65f23320bf6,
                0xa03b17d81560548d,
                0x1e48f959b1e024bc,
            ]))],
            [Felt::new(BigInteger256([
                0x65e4242e8d99d87f,
                0x16685377165198cc,
                0x0001b7076f2c1b33,
                0x4d496e06a8189c03,
            ]))],
            [Felt::new(BigInteger256([
                0x9dc2ec0375df08fa,
                0xbd41914fa1c1599b,
                0x55b26d55ac8bae07,
                0x669ffa50e71b7b23,
            ]))],
            [Felt::new(BigInteger256([
                0x18d021c28dc3c0b8,
                0x65f71c79812bc3ae,
                0xc1ff2673efe254a4,
                0x65d19773f45e73eb,
            ]))],
            [Felt::new(BigInteger256([
                0xc7d9fffb056d1686,
                0xd5b6c5635ea9a047,
                0x4e538d9ca2901b1e,
                0x33cc1d1c043c5e50,
            ]))],
            [Felt::new(BigInteger256([
                0x52a89b2b216757d8,
                0x966f5d6c58a6289d,
                0x25d9b54ca2a948ec,
                0x0d2f6060b36c6ba3,
            ]))],
            [Felt::new(BigInteger256([
                0xb58daf9dd383ae22,
                0x6ba9f67bfd342b8b,
                0x6b40eb7c9b64d940,
                0x550ed1e7c440323b,
            ]))],
            [Felt::new(BigInteger256([
                0xf72949503e25c433,
                0xd75039dcc5b34247,
                0xd1bdffc741e35046,
                0x50a2f1f623844121,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/Nashtare/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0x8cc0c0fcfed536cf,
                    0xba3a44d280e17ec8,
                    0xcacfb932e94eede9,
                    0x3d148b3417f094e1,
                ])),
                Felt::new(BigInteger256([
                    0x1882b1aa11a04b55,
                    0xb2cfbbbb55e7efb7,
                    0xade65a1e252f0cd9,
                    0x1777f3f89cdeb8af,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4b5bd01a6f3a4fd9,
                    0x3bc5929972e0c4c6,
                    0xf5b160796de1d12b,
                    0x449b8c5b97a47a4f,
                ])),
                Felt::new(BigInteger256([
                    0xeafde8b5a39bd311,
                    0xe301cb41eb588718,
                    0x75556a30b33bf25a,
                    0x706e41ef682b0d73,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x99fef359597268ba,
                    0x97973daaedc81d0e,
                    0xe2c352ab00cfd9ff,
                    0x3d4227ba795f957e,
                ])),
                Felt::new(BigInteger256([
                    0xe2049dba17eddd8a,
                    0xb88b3327a8668b2a,
                    0x3284aba9e0589ab2,
                    0x70356e0948ff59d3,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x91b59575d0401b1d,
                    0xd8dd6b91c8ef0c6e,
                    0x62a1f2a0b30e0f21,
                    0x688d7970f8c9aed6,
                ])),
                Felt::new(BigInteger256([
                    0x087fb188ea810cf7,
                    0x62848fc2f392fe41,
                    0x30ac0fe009468c32,
                    0x56ba5e9675357078,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xffa5bb5976edf34d,
                    0x1e3afe3c14ecb8ab,
                    0x47c4b76b36a91a6d,
                    0x21447409ff30304b,
                ])),
                Felt::new(BigInteger256([
                    0x1846fea547de13d2,
                    0x506c848e9148a6f3,
                    0x807b230c5425ad2e,
                    0x363b747b0c77f7cc,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xbc399b1bf190415b,
                    0xefab90a357f22351,
                    0xd157ff4cc8039e12,
                    0x48cc2a389e89bac8,
                ])),
                Felt::new(BigInteger256([
                    0x57acbd67d34a5c49,
                    0x75684153f8639dd9,
                    0x091d390b88b4e98a,
                    0x5e1cfaa509c2965e,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xa122c2f0fddee7ac,
                0x12e1783c5aa80b91,
                0xa59503aa8faf18ba,
                0x66311754e5d9f789,
            ]))],
            [Felt::new(BigInteger256([
                0x9d9b7583d960a9ea,
                0x8f356cd779c2b3a9,
                0x28d17a845184f65e,
                0x4bdc9517c024c520,
            ]))],
            [Felt::new(BigInteger256([
                0x441b8497e3cd22c9,
                0x83ebe6e5f6abc3be,
                0x54e358b28407ea3b,
                0x404656c534aaa060,
            ]))],
            [Felt::new(BigInteger256([
                0x9de92a84fd8ae54e,
                0x6f21ba9743dffa60,
                0x1def87e72032a404,
                0x0140c86a3e50b9f6,
            ]))],
            [Felt::new(BigInteger256([
                0x5a24720461093504,
                0x83285c6472cd950c,
                0xa2a8f6f4c5829fd5,
                0x7349f6cc08d3ca78,
            ]))],
            [Felt::new(BigInteger256([
                0xafbed753a1ab3fd9,
                0xecd90a5ef0b3b210,
                0xa558ea4f9933bf26,
                0x643b242e39baf165,
            ]))],
            [Felt::new(BigInteger256([
                0x672643d9ae04fe39,
                0x94a77a53c8f56208,
                0x0c839abc7bec2214,
                0x5cf5cd58616579c9,
            ]))],
            [Felt::new(BigInteger256([
                0xfadc7e118a611d36,
                0x398779ace407896e,
                0x4a4dee9d224bedf1,
                0x38e0e4247da106c5,
            ]))],
            [Felt::new(BigInteger256([
                0x2117961f841280e8,
                0x94da5375ca64f686,
                0x619b76ae062a2573,
                0x407c3b8f2893b3e3,
            ]))],
            [Felt::new(BigInteger256([
                0x21ddee8c74553b49,
                0xc450f76f4ec22679,
                0x7418c9e9be388da8,
                0x369750c1564f941b,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiHash::compress_k(input, 2));
        }
    }
}
