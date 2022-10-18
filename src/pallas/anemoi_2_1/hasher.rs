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
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0xd04377ddbba5372d,
                0x0562fe8efd298586,
                0x4cf43165334d026c,
                0x069866c4bade5c04,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x8a799bd89963da17,
                    0x120b451976ea24e5,
                    0x770d6e114c2d00b0,
                    0x0c5fd960f40ecfee,
                ])),
                Felt::new(BigInteger256([
                    0x8682438e5e3ba915,
                    0x8c508d3e8246f35c,
                    0x40f14cdc523d368e,
                    0x29f64bce7172d81c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x75fef41d464876c1,
                    0x5bc82e25febe9fa3,
                    0x2b3513cd12b5a138,
                    0x19c9dd5307e26c4e,
                ])),
                Felt::new(BigInteger256([
                    0x47392d1402797921,
                    0xcef479e45cd8ecea,
                    0x8a3d18bfae37a555,
                    0x12a019a1e4c6af5d,
                ])),
                Felt::new(BigInteger256([
                    0xab507501f0d0dca8,
                    0x61de006e186310d9,
                    0x597d7fb74498765b,
                    0x1507f69b16f5399a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x0539af984f4205bc,
                    0x122910a4f5da4ba3,
                    0x40ef3f1b18acf4fa,
                    0x018a427e2b858260,
                ])),
                Felt::new(BigInteger256([
                    0x8c69c67e3b3babe8,
                    0xcab44e10e646105c,
                    0x08d59c5aa7340767,
                    0x1ffd58377302ea91,
                ])),
                Felt::new(BigInteger256([
                    0xab9dcb8777e79a37,
                    0x6191861c5c3c4387,
                    0xbd3811797043a681,
                    0x0fcd847a561561de,
                ])),
                Felt::new(BigInteger256([
                    0xe420326007ba6dee,
                    0xe958c104666be904,
                    0x365872bb94e1c9b1,
                    0x324b184d64a4265c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x13de732f3b331b5a,
                    0x94c5eefb33b52ec2,
                    0xf501cd66effa4053,
                    0x1e8bda2b7152fbfa,
                ])),
                Felt::new(BigInteger256([
                    0x1e82414d03983e99,
                    0x299a00ed2206ca98,
                    0xabdd25001b392980,
                    0x37e88e9862d01c94,
                ])),
                Felt::new(BigInteger256([
                    0x040acbcba43f9c04,
                    0x6abf886bc4882ee4,
                    0xcf7da0fcf7f44c7c,
                    0x194bef1924336f8f,
                ])),
                Felt::new(BigInteger256([
                    0xbefbacd3e601a1ed,
                    0xa10c145f8e335a09,
                    0x34fb685753957c1a,
                    0x132d9145563b7362,
                ])),
                Felt::new(BigInteger256([
                    0x3b06df9500bb2861,
                    0x5445f1dc0ae69b74,
                    0x98bda9c6b4e23d50,
                    0x2c0bac8df77d5e71,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc478bf06f3947de5,
                    0x2df4495a546e8258,
                    0xd36480de14fa9cff,
                    0x306c29b54da67c67,
                ])),
                Felt::new(BigInteger256([
                    0x3e1470d039a777f7,
                    0xf80d6c949301bd9a,
                    0x8acac429725d0556,
                    0x0b6a4c7cca32674b,
                ])),
                Felt::new(BigInteger256([
                    0xaee115e803030ff8,
                    0x8013c4e03f392bc5,
                    0xcfd58a065f47478e,
                    0x22c049293b42fe25,
                ])),
                Felt::new(BigInteger256([
                    0x8f6c7adce74d74f0,
                    0x9012453f5aa0af78,
                    0xa2bae3d8bfe5e3fb,
                    0x2de74455adf7eba7,
                ])),
                Felt::new(BigInteger256([
                    0x0c2927f13db21a37,
                    0x8b03877e0878dc18,
                    0xf63e41e934fb5fa4,
                    0x33bd77c6e65ac85d,
                ])),
                Felt::new(BigInteger256([
                    0x8bdfe901c4182f11,
                    0x0db74bd55f669111,
                    0x92a4155b674eac63,
                    0x2c21d943853afe4e,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xd9ea61dc8b36ff9b,
                0x9055341480a84ca2,
                0x8ebdea1c22182b20,
                0x25284cd8a4260fb5,
            ]))],
            [Felt::new(BigInteger256([
                0x8210e24cfca07a7e,
                0x138008c868674c8a,
                0x44f09ab486ec3de6,
                0x341b4ce244945278,
            ]))],
            [Felt::new(BigInteger256([
                0xcff60955f7fe5e94,
                0x701a12c7b3573a45,
                0x27c01a3d9336895f,
                0x0d22b6be266ccbda,
            ]))],
            [Felt::new(BigInteger256([
                0x5b0af5e149ded1f8,
                0x72cdf711225a6831,
                0x04a08ca18d9ca09f,
                0x3ce8d91bb607793a,
            ]))],
            [Felt::new(BigInteger256([
                0x6cc9e7a0256aac64,
                0xb209374a271fed54,
                0x46aef09e6dd1021a,
                0x04b4fdddad5a81f5,
            ]))],
            [Felt::new(BigInteger256([
                0x725d226894e2909c,
                0x49f19a003bc846ad,
                0x4288fd3170238bf8,
                0x3ce15d98593f9d20,
            ]))],
            [Felt::new(BigInteger256([
                0x1d45ef737342ae39,
                0xdb7daae65e6f4332,
                0x8d442d49ff377f18,
                0x00ad25579985b5e4,
            ]))],
            [Felt::new(BigInteger256([
                0x642a4e807a8dde6a,
                0x73ba001f2d91defe,
                0x375fdfc228d3b785,
                0x1aeb74f7ff2746b9,
            ]))],
            [Felt::new(BigInteger256([
                0x79e87f9f46cdeb80,
                0xba55f467acd296d9,
                0xffd29c0a68bf33d5,
                0x3168e6099cf0d129,
            ]))],
            [Felt::new(BigInteger256([
                0x66dbdac2d7e881bc,
                0x306d1c40cefdb278,
                0xe3435c74578ae016,
                0x335f52088503774b,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiHash::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![
                Felt::new(BigInteger256([
                    0xf1ffc36bd30f2bfb,
                    0x47c20d8b084c3878,
                    0x9abe08f88cb653f3,
                    0x327567291c39bf19,
                ])),
                Felt::new(BigInteger256([
                    0x328612a9df43d1fc,
                    0xe875b987d90a7bb0,
                    0xad48f8bb00fc7aa7,
                    0x2491c5f55065f256,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xa65b9762b0413c9c,
                    0x5f68c867bb3c08d4,
                    0xfda7a53f4a25bab9,
                    0x23f0ee4b3c85d9ab,
                ])),
                Felt::new(BigInteger256([
                    0xe1dd80040e17605d,
                    0x0ec10aa8666dbb6a,
                    0x93d14bda89e083bd,
                    0x3aff7261a30e38e8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xdceef77db009f558,
                    0x360c57c127c4f36c,
                    0x9141b74c414d1488,
                    0x0ce7a51117a54419,
                ])),
                Felt::new(BigInteger256([
                    0xb91eab91b8f6b2a0,
                    0x6d8f087cf251763e,
                    0x201bb4565d198c39,
                    0x30fe7f41e8a05d5b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6b2cced5b298446f,
                    0xcbfe2b049baf394e,
                    0x95c32af349cd3bcf,
                    0x1726a6f9729e1aff,
                ])),
                Felt::new(BigInteger256([
                    0xcf50c6bab1614d4d,
                    0xa97139f6c5ef27bf,
                    0x7d370304ef24cd53,
                    0x05296bb78bff588a,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x8d4d5e6b4d4fa04a,
                    0x1a48135ed15d46eb,
                    0x01d81d32491d4014,
                    0x25977d8760dcaa16,
                ])),
                Felt::new(BigInteger256([
                    0xed316bceea5819ff,
                    0x6e7473aad39d780e,
                    0x06aed8033a9a98f4,
                    0x30be89ed1703369c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x9adf33bd27adc9b1,
                    0x03475a5491fea73d,
                    0xc9d1bcbd45fa2691,
                    0x209482596da7487e,
                ])),
                Felt::new(BigInteger256([
                    0xe5ce2cc9af45ebfc,
                    0xa825e3518a690e10,
                    0xae5e25f82e9aeae3,
                    0x0430cc33878daafc,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x1d78490f971cb1d8,
                0xef1ae50e41d33ec2,
                0x3ca2e0a9b3ba4b44,
                0x2c51d4f14c16b79e,
            ]))],
            [Felt::new(BigInteger256([
                0xf2c852bb302f6e53,
                0xcebafb80b343b09f,
                0xaf7ba56fba32296c,
                0x2f76d927a60254c7,
            ]))],
            [Felt::new(BigInteger256([
                0xc6e3bfda4bf1012c,
                0xd4590e9d32002238,
                0xdc5d56db7b964f5d,
                0x24f4e7b33ec174f6,
            ]))],
            [Felt::new(BigInteger256([
                0xeb68a18c8ef96b0b,
                0x0ba2c8cdd2b76180,
                0x741e789bbe169fb3,
                0x2e2895d110a18265,
            ]))],
            [Felt::new(BigInteger256([
                0xaa6d9c4ee450a2a2,
                0xb826185ca9eb701a,
                0x6688235e1d8a2aca,
                0x3431335b7ac4498b,
            ]))],
            [Felt::new(BigInteger256([
                0x646ab20da5e5aa58,
                0xd7d210c3b7bb7dca,
                0xb6d1a140d88ab329,
                0x36760650cf4b0fd0,
            ]))],
            [Felt::new(BigInteger256([
                0x4e853e79aa835c7b,
                0xc1d5572e488e31d1,
                0x3fd5a1e624b6f294,
                0x0ca6d1cdc4e5234a,
            ]))],
            [Felt::new(BigInteger256([
                0xfb08f1b9dd1ed1ca,
                0x677549c378824031,
                0x00ccfaa837f9f97f,
                0x300539726892e28c,
            ]))],
            [Felt::new(BigInteger256([
                0x3698b01cb3a61e88,
                0x71e1fe688c29aa1c,
                0xb41d6bc33e409de5,
                0x38971467b537a037,
            ]))],
            [Felt::new(BigInteger256([
                0x46c4aecf26648f6f,
                0xe9088f4e4b892cb9,
                0x826e608bce7d170a,
                0x1c9a6a13042ac705,
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
