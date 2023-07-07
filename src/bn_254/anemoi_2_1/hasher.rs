//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiBn254_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiBn254_2_1 {
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
            AnemoiBn254_2_1::permutation(&mut state);
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
            AnemoiBn254_2_1::permutation(&mut state);
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

impl Jive<Felt> for AnemoiBn254_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiBn254_2_1::permutation(&mut state);

        vec![state[0] + state[1] + elems[0] + elems[1]]
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
    use ark_ff::to_bytes;

    #[test]
    fn test_anemoi_hash() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0x49d9864e11ab079b,
                0x575d57b0fce07868,
                0x4614e1a827ec7e93,
                0x18a2ae92e5f073bb,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xd1c135a0b8914e83,
                    0xc6aea489aaa4da89,
                    0xd5cb50ea9ffb3de7,
                    0x173b342730ad2777,
                ])),
                Felt::new(BigInteger256([
                    0x1d6278c775483b85,
                    0x70c4bd55c3db9b04,
                    0xfccdfb068e850c6f,
                    0x139e0c930cee25d7,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe96c828d87b303f4,
                    0xfd0c576e094b77dd,
                    0x41cd3f1079c02800,
                    0x2bb378710de1073b,
                ])),
                Felt::new(BigInteger256([
                    0xff1a87fe3483b5b6,
                    0x73b94349b6433ec6,
                    0xd4ebef5ddb048e13,
                    0x2262d4077b800925,
                ])),
                Felt::new(BigInteger256([
                    0x0718ebadc771c456,
                    0x124acef3414a1545,
                    0x3f6b1890d437007f,
                    0x2637c5ace42a411c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x758891b710abf3e3,
                    0xeac9e349a3a51b3e,
                    0xbeff5822c9c1cc53,
                    0x279b605c226fba18,
                ])),
                Felt::new(BigInteger256([
                    0xf48eb77484970a12,
                    0xf85b303c0b02636d,
                    0x90b55873d126de81,
                    0x22816a5f9195f62e,
                ])),
                Felt::new(BigInteger256([
                    0x787031393e37ab56,
                    0xcb883c9f63ddd636,
                    0x550ea4151bdc0f7a,
                    0x0379db5cecef6c66,
                ])),
                Felt::new(BigInteger256([
                    0x730502e43c83ffdb,
                    0xe44f2bd69d88c14f,
                    0x215699ef7909dbf6,
                    0x02f4147b26b528db,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb3ff9816128b4be0,
                    0x22d430f52c689700,
                    0x2de70d8f09a3485d,
                    0x007d0bf653b0a202,
                ])),
                Felt::new(BigInteger256([
                    0x0beab19e9f80803f,
                    0xffb50911d58ae88c,
                    0x35ccb063ef309165,
                    0x276b90ec836bedc0,
                ])),
                Felt::new(BigInteger256([
                    0x516f69124503a7c2,
                    0xb93fc258bcb0af15,
                    0xc5b21fe6fc6ae5db,
                    0x0d66a137d8a01851,
                ])),
                Felt::new(BigInteger256([
                    0xa029465ea90e5274,
                    0xeecd3a066a65824b,
                    0xf3196832b20f4f36,
                    0x241bd510b68ef5de,
                ])),
                Felt::new(BigInteger256([
                    0x8fb9062e620cc767,
                    0xae79b0e6f678a8c5,
                    0xa741dff3bce490fd,
                    0x15a9efbff87906e4,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x938ed88edf5f2dd9,
                    0xd3dbc617a46012c7,
                    0xf3aab620b9d86e3b,
                    0x0e8a036f1e626d4b,
                ])),
                Felt::new(BigInteger256([
                    0x27146d573d72c9b0,
                    0x5c21e29e395dc8a6,
                    0x80401ecc98bc77ef,
                    0x18801ebc999b98f0,
                ])),
                Felt::new(BigInteger256([
                    0xf298e2ca92449e1b,
                    0xb01b5db2d11dcfd9,
                    0x5fae5e0d07e7ca0d,
                    0x2188687f6f718719,
                ])),
                Felt::new(BigInteger256([
                    0x2eed3521ab6f09c8,
                    0x018a608d9da7dd25,
                    0xc658bc7283c7515b,
                    0x049a7a5d75918175,
                ])),
                Felt::new(BigInteger256([
                    0x323c7b56a78d1d01,
                    0x44a7954ecbd955c7,
                    0x89a90daeaf411ef9,
                    0x0ab5c36dae117d8d,
                ])),
                Felt::new(BigInteger256([
                    0x3ef8f2cc28765b85,
                    0xeaab7bbcdbdffe48,
                    0xba603eb5b90e96da,
                    0x0dd85281afda0fb0,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x2436613fb251474b,
                0x121ce48a7acb6c14,
                0x1ea072123ff869a7,
                0x0eb1b7a1d1ca6603,
            ]))],
            [Felt::new(BigInteger256([
                0x853547d79171709e,
                0x9d5e3fb4a590033c,
                0xe233eb7f374cb2a9,
                0x04c6df19d19620a7,
            ]))],
            [Felt::new(BigInteger256([
                0xa8dcb891ac0390f3,
                0x5e02b67c4ada27bf,
                0x84e61d7b5ffbb9b3,
                0x072faa87b654c43d,
            ]))],
            [Felt::new(BigInteger256([
                0xe9cbe2475d324587,
                0x558d343fbdd38b2a,
                0x2dae7a03bf97c74a,
                0x1cd06eaa5dd28207,
            ]))],
            [Felt::new(BigInteger256([
                0xd9a051ffe228573b,
                0x2b0065a45bab9af6,
                0xd84dee161e57c32f,
                0x04e57b49edf2ae57,
            ]))],
            [Felt::new(BigInteger256([
                0xc0d993e47ea1f0e2,
                0xea3cae23cdea333a,
                0x7a6b8560b8f2994c,
                0x13198efce05dbbd0,
            ]))],
            [Felt::new(BigInteger256([
                0x2cb503a6df3fe9f2,
                0x0c764432f38da164,
                0xe39818a64c88d5f8,
                0x14e405094f73d4ca,
            ]))],
            [Felt::new(BigInteger256([
                0x80864920e95af735,
                0xa922a2b0ba8c9436,
                0x66f1b70a5c3f65f6,
                0x2c2cb224bb5517d6,
            ]))],
            [Felt::new(BigInteger256([
                0xa11d1386abe33db7,
                0x664532a5f60a54e9,
                0x94ceadb4dd3f8dfe,
                0x069180d44c680d65,
            ]))],
            [Felt::new(BigInteger256([
                0xe9eceeb014af7ee1,
                0xe491b964e529af14,
                0x4b72e48a327d68b0,
                0x0f6c74028e7ac883,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiBn254_2_1::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::one()],
            vec![Felt::one(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x2436613fb251474b,
                0x121ce48a7acb6c14,
                0x1ea072123ff869a7,
                0x0eb1b7a1d1ca6603,
            ]))],
            [Felt::new(BigInteger256([
                0x853547d79171709e,
                0x9d5e3fb4a590033c,
                0xe233eb7f374cb2a9,
                0x04c6df19d19620a7,
            ]))],
            [Felt::new(BigInteger256([
                0xa8dcb891ac0390f3,
                0x5e02b67c4ada27bf,
                0x84e61d7b5ffbb9b3,
                0x072faa87b654c43d,
            ]))],
            [Felt::new(BigInteger256([
                0xe9cbe2475d324587,
                0x558d343fbdd38b2a,
                0x2dae7a03bf97c74a,
                0x1cd06eaa5dd28207,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiBn254_2_1::hash(&bytes).to_elements());
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
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xbc45060e66ee58ff,
                0x94bf27cd2882d256,
                0xc7752dfd884a1bb1,
                0x233960037e0fbb3b,
            ]))],
            [Felt::new(BigInteger256([
                0xf6f6c00125acf3bf,
                0xe0cc31a4b937b98f,
                0x2b1043e94492ab33,
                0x2b19e4a3fde0d749,
            ]))],
            [Felt::new(BigInteger256([
                0x453287459a866eca,
                0x032f4e437226b414,
                0x4cda55451b23d1fd,
                0x02cc0358f9a1e954,
            ]))],
            [Felt::new(BigInteger256([
                0xb0efa9fa23f15b80,
                0x22da2636bc0de64c,
                0x17d86a7dfae5172b,
                0x27c1e8899a1d3839,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiBn254_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiBn254_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
