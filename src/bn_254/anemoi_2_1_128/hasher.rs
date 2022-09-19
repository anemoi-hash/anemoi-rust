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

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
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

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        AnemoiDigest::new(state[..DIGEST_SIZE].try_into().unwrap())
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
                0x534faa60bb5c3e54,
                0xa2a6f66fc13b7ce3,
                0x8425a15e9af628b9,
                0x14d3bcbb9784dc00,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x9021412fe3e87ed8,
                    0x0c3bac0908b74e98,
                    0x8a75698a1eca957f,
                    0x15baeb7c2a608c9b,
                ])),
                Felt::new(BigInteger256([
                    0x0085550aab170597,
                    0x25c749e7e79a87e6,
                    0x529ecad6674c1cae,
                    0x16b8bad1247c7f06,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x2332630329b48510,
                    0xfa25b89f9f460753,
                    0xe63fa8868f197cae,
                    0x10b440699f1990e8,
                ])),
                Felt::new(BigInteger256([
                    0xa18e0d128d39b275,
                    0x53b10b2836c29d71,
                    0xd3b25f82296061f5,
                    0x220d7d178c80f25a,
                ])),
                Felt::new(BigInteger256([
                    0x104cbe7c585b25c7,
                    0x51281067cc4bed07,
                    0x4dde4f9101778872,
                    0x0bd7a64ca24256d8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xfda444bb7b9680b8,
                    0x3d62dad2e2a176ce,
                    0x65588fac0a1675bb,
                    0x0b6dcdc59af5059c,
                ])),
                Felt::new(BigInteger256([
                    0xa6d6168e14a059f7,
                    0x1470dfeaa32889eb,
                    0x1df36088cd0d3ce1,
                    0x273ebcbd5277a971,
                ])),
                Felt::new(BigInteger256([
                    0xc24e08a0038c5d96,
                    0x06af3817fc020709,
                    0x151be4f078c64c57,
                    0x09047e7b076ff325,
                ])),
                Felt::new(BigInteger256([
                    0x4b9c0fd5c50c8657,
                    0x25adf2cf9e3a9187,
                    0x697f729a6808ef2c,
                    0x0d5c8e5ec320e33a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x77cd9b51c1e07220,
                    0xd63828580a2e9918,
                    0xbb76d1a2b4e43be0,
                    0x036d0deff9a74bdf,
                ])),
                Felt::new(BigInteger256([
                    0x8667e5aa353f74d4,
                    0x8210ff3c370bb5ae,
                    0x1140e6b11c86eb01,
                    0x2528e96ec019898a,
                ])),
                Felt::new(BigInteger256([
                    0xe0842f4f16715691,
                    0x76ec5dde775318ae,
                    0x4804fdbe1e28c991,
                    0x2393d8938dd4e02f,
                ])),
                Felt::new(BigInteger256([
                    0x3ee06e7ac61a7a95,
                    0x69d982a59544be8a,
                    0x51691a90a235c2b1,
                    0x0c7307a1b4944ef7,
                ])),
                Felt::new(BigInteger256([
                    0x300c9d728b2ddc2f,
                    0x4efd1cc279721599,
                    0xc7afabb3d926ffa5,
                    0x193a08a379d00882,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd8acb9b71686e22c,
                    0x96a4b37b056d189a,
                    0x80fd6ee627eba927,
                    0x0fbde360ca2f3c7e,
                ])),
                Felt::new(BigInteger256([
                    0xd42046933b8347a8,
                    0x0ebb75bb7a02ea0e,
                    0xae998b409a3eac78,
                    0x1ee7e8b129ffb18b,
                ])),
                Felt::new(BigInteger256([
                    0x912101114d054b8a,
                    0x2bc62b6faa5e0479,
                    0xb13116a7eb467f7a,
                    0x206396bdefecb347,
                ])),
                Felt::new(BigInteger256([
                    0x3c9734b65174a4b9,
                    0x89f9d89669248aaa,
                    0x48ebea2e437f099e,
                    0x2adc9e26b5b1597b,
                ])),
                Felt::new(BigInteger256([
                    0x2d9aaf43f34ed2ad,
                    0xb9e6df742954f8e5,
                    0xa229f6e414732840,
                    0x1961b71438296901,
                ])),
                Felt::new(BigInteger256([
                    0x4f81aadb6d943945,
                    0x63902334e0acc1c3,
                    0x531f61a58269c7ce,
                    0x164b6ee72b7ed73e,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xa4404cd4d964c552,
                0x898423c5407d466d,
                0x7eb4d1dec2067ae2,
                0x0e2fc879f0595ae6,
            ]))],
            [Felt::new(BigInteger256([
                0x1134ede91527414d,
                0xfdf793c4869aa7d7,
                0x7551229cd4d6ed3f,
                0x1b5437b9b5c2194f,
            ]))],
            [Felt::new(BigInteger256([
                0x14d4294a7678af60,
                0x5c57a5f23f92a784,
                0x6009d39fc6ce98b5,
                0x2ae51ccdf59074de,
            ]))],
            [Felt::new(BigInteger256([
                0x5901cec16865cf75,
                0x21819512c78297e6,
                0x1f42bacd9b29a361,
                0x08694b359d654431,
            ]))],
            [Felt::new(BigInteger256([
                0x93fcc3ad077959af,
                0x12a7566d6e568a3c,
                0x5c1cf6b21c1c657a,
                0x073c2bc79f673464,
            ]))],
            [Felt::new(BigInteger256([
                0xab6115dff65b5b4e,
                0x5d7df3a2a1b59ab6,
                0xacd83734dcd74f29,
                0x25f65bb819c6ae65,
            ]))],
            [Felt::new(BigInteger256([
                0xb90d30abe71a03ae,
                0x7bd4e98546c306de,
                0x815d30b0fe980517,
                0x189f2be38deda3c3,
            ]))],
            [Felt::new(BigInteger256([
                0x2c1b335d0470df53,
                0xe9a75ffc7505bf4f,
                0x4ce5780202b1de13,
                0x2ccc9b4f5f28a648,
            ]))],
            [Felt::new(BigInteger256([
                0xe9e78cccfca77ff8,
                0xea4b13a2110467ad,
                0x341c57f917fa6d81,
                0x0bb3f465a8de30e0,
            ]))],
            [Felt::new(BigInteger256([
                0x65fe5a6a25f7e250,
                0xc3deb57fbf62d29a,
                0x2606f9f002599cd8,
                0x25ebe0fd1c009354,
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
                    0x2fa5197af13cd657,
                    0xa25e0f684820800e,
                    0x0c8e74955cec0bfe,
                    0x1e46ee9610c76b44,
                ])),
                Felt::new(BigInteger256([
                    0xeb25e655c11694ec,
                    0x57dbd3fe84f0d478,
                    0x0936638085407800,
                    0x23f6a5abeeb14fb1,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4ab9a8ece1023c24,
                    0xb1156dcd11a9d15b,
                    0x02f79cb8fb157cd3,
                    0x1b763f762fb339da,
                ])),
                Felt::new(BigInteger256([
                    0xc20f249d5be41e70,
                    0xeb33967358bbdcac,
                    0xb96c25beedd516d5,
                    0x1ce9e708b4a53900,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x789fa14aa0c3f886,
                    0xf092ed3049b6987e,
                    0xac104a50dd1573bf,
                    0x189c45bbc07dda3d,
                ])),
                Felt::new(BigInteger256([
                    0x10acdf270755059c,
                    0x51f5dae408a655c1,
                    0xb4bbd76cce7a5672,
                    0x18a2a4ac129014bc,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6037ffa1bc918493,
                    0xce60abf8154d21aa,
                    0x680365dd7d6f0000,
                    0x1d7ee8bdae70c264,
                ])),
                Felt::new(BigInteger256([
                    0xc2d4c9a9901fc098,
                    0x5eb2054a9a552fa1,
                    0xdc54304ddd89d58c,
                    0x2afaf17b2e78e58f,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc1eeab5b916fac0b,
                    0x5840cb7658257233,
                    0x969b4e914eaeef14,
                    0x265120ca323929ef,
                ])),
                Felt::new(BigInteger256([
                    0x0d5a1eb97cc3759f,
                    0xc82c7307515342f3,
                    0xb7cedc8b3c5d3bb4,
                    0x1dc4881e99e515bb,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x164c4cdef4f8fb4a,
                    0x544df6bc7150d751,
                    0x124bb25ee429a560,
                    0x017d90990fd050f6,
                ])),
                Felt::new(BigInteger256([
                    0x7449ff331b707bbc,
                    0x5d9dcd3b2bcf63fa,
                    0xf5540982d4b4e91b,
                    0x0281a159f1ccd463,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x10dcf05d158fc61d,
                0xa95040eb07f21d42,
                0xc8e7f13572ab0e21,
                0x06a50b2182a9605c,
            ]))],
            [Felt::new(BigInteger256([
                0x4101eec26ae2e9ed,
                0xbe83544c1869323f,
                0xb6cf7fbf69807d5b,
                0x07fc3d84a88992c9,
            ]))],
            [Felt::new(BigInteger256([
                0xf7730421eb7ccaed,
                0x22aab07b0fb68568,
                0x188c2da3221e31a6,
                0x0ebd1a7d78ea99e0,
            ]))],
            [Felt::new(BigInteger256([
                0x4b730478f5f443ef,
                0xe69b7ee55a2b1869,
                0x467f8187a79acbad,
                0x03c7c31d8114eab6,
            ]))],
            [Felt::new(BigInteger256([
                0x0fb99d2fd323686f,
                0x25c093a1ed40683c,
                0x5dfe379d3a5eebe0,
                0x17ac8cb36a348501,
            ]))],
            [Felt::new(BigInteger256([
                0xf5d733110c182519,
                0x7cb8ce28f4f8a250,
                0x9103d8d9060d833a,
                0x18d77585d3d93a44,
            ]))],
            [Felt::new(BigInteger256([
                0x0e7d1bd45c77a6fa,
                0xd3d15dc994e9e3bb,
                0xc853dc0e00af660d,
                0x068a01fe0602a8a7,
            ]))],
            [Felt::new(BigInteger256([
                0xfcdb6801904742b9,
                0xc08b4ad1fbd570bd,
                0xc5aae7489e576ce0,
                0x0e415baccc3ca067,
            ]))],
            [Felt::new(BigInteger256([
                0x45d6ae6c063b195e,
                0x8f56fecc407a41d5,
                0xe14190b1485085b7,
                0x12fbc4e6d5a38126,
            ]))],
            [Felt::new(BigInteger256([
                0x51643577a9a0605b,
                0x14e9903c1d15be12,
                0x31ccecd2e477c002,
                0x0bb180d48e323d72,
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
