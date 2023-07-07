//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiVesta_2_1, Jive, Sponge};
use super::{DIGEST_SIZE, STATE_WIDTH};
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiVesta_2_1 {
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
            AnemoiVesta_2_1::permutation(&mut state);
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
            AnemoiVesta_2_1::permutation(&mut state);
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

impl Jive<Felt> for AnemoiVesta_2_1 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiVesta_2_1::permutation(&mut state);

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
                0x32e63e318a598dff,
                0xea794f429cb10676,
                0xbc1ae9469a3350a2,
                0x33b4a24325df197f,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0xfb8d75e5c0d08f9d,
                    0x9e3b9fa813f4acc0,
                    0xb238060b8575cf78,
                    0x087eeb72190c9a0a,
                ])),
                Felt::new(BigInteger256([
                    0xed3149f19ccacb9e,
                    0x00a162dd4e92ff86,
                    0x37847474d7bc3946,
                    0x09dccf858ffd8e7d,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xef08db05c9bcf2a8,
                    0xca85153df4a2e0ca,
                    0xdfb9461081e9f29e,
                    0x0cef1176102dc156,
                ])),
                Felt::new(BigInteger256([
                    0x5f858fd350213127,
                    0x83d1c988fe846091,
                    0xd5bd08b6f04be361,
                    0x0a6398ad06cab47c,
                ])),
                Felt::new(BigInteger256([
                    0xec1263782a036e22,
                    0x2946971204165bbb,
                    0x5efb9e1dd87c48d5,
                    0x03be8990caa19f2e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xc7c416b86ddc3fe4,
                    0x51195e7a2caea4af,
                    0xe9c5b26ad250149f,
                    0x321cc3ef185f0252,
                ])),
                Felt::new(BigInteger256([
                    0x76e5cf241be9352a,
                    0x48f4a54b548b8c0a,
                    0x577473d4209a7256,
                    0x2ba0963d37dbb314,
                ])),
                Felt::new(BigInteger256([
                    0x4b5fb722317b698e,
                    0x52185736f8e53ddc,
                    0xbd487508f3bcc567,
                    0x2200723d64d5904f,
                ])),
                Felt::new(BigInteger256([
                    0x236b30e927749499,
                    0x008e91806a505aa0,
                    0x13f2b9b27addd94a,
                    0x059df199c6d8550c,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd865dae564ef35ca,
                    0x383fcfd2e33407a7,
                    0xaa7c641b73c2cc47,
                    0x332e389693d75bd3,
                ])),
                Felt::new(BigInteger256([
                    0x6fba0adb6baafdb0,
                    0x758b49f67cd2d174,
                    0x331c4a02db13d9af,
                    0x07e83022981a45ab,
                ])),
                Felt::new(BigInteger256([
                    0xd050e6fff3e14603,
                    0x9a8ce98d7f1a748b,
                    0xd34818ab46f136de,
                    0x09e3718edce39c98,
                ])),
                Felt::new(BigInteger256([
                    0xef1602b1024b612b,
                    0x9af37815cd784c89,
                    0x80ec6cff33089451,
                    0x1a1027b71f57477b,
                ])),
                Felt::new(BigInteger256([
                    0xc64acf1681715622,
                    0xf9818b05f42e5b26,
                    0xec9d95cc2a11fcb9,
                    0x3234f91b11c3b8c0,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xe96d88b6ba814eae,
                    0xd4c9c2595c2b141d,
                    0x7def7afe61f73015,
                    0x0edff6d5b9146546,
                ])),
                Felt::new(BigInteger256([
                    0x5cbd34137d4bc42a,
                    0x395d3a6da70c74d9,
                    0x758f84fcc83db5d0,
                    0x36e68b3ec76940f4,
                ])),
                Felt::new(BigInteger256([
                    0x477afdd6ba143c91,
                    0x20334a0a689b48ce,
                    0xf513c30c941f7253,
                    0x26de3fe70632c963,
                ])),
                Felt::new(BigInteger256([
                    0x1c78e09488a59525,
                    0x8cd38e9174bc45e6,
                    0xca1e50e95422380d,
                    0x2a74a19fca8a49cb,
                ])),
                Felt::new(BigInteger256([
                    0xcc1ecd530615ee3c,
                    0xa5fa74d84b71a27e,
                    0x46166bfadf8c2731,
                    0x21c1406e62f53f6a,
                ])),
                Felt::new(BigInteger256([
                    0x543a95e20c6db46c,
                    0xcebee9cf2b7389b3,
                    0xdb74ff6429fa26c4,
                    0x081a585604177134,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xe4484e4d9cb56ab3,
                0x4c1e1eab5f3a5d35,
                0xdb539d68d7aef2fb,
                0x328395f3ab3baedd,
            ]))],
            [Felt::new(BigInteger256([
                0x073adf8531a94728,
                0x58921a70ecf0f156,
                0x9616137969de4c6b,
                0x33726ef1ad99329b,
            ]))],
            [Felt::new(BigInteger256([
                0xf5838cbab6cb93f6,
                0xfc076c5e318b7274,
                0x54d86ddf65dfa99f,
                0x14a3fbf609635e45,
            ]))],
            [Felt::new(BigInteger256([
                0xeb51c8928746dabb,
                0xc393ab0999339fd6,
                0xc8e891c3008346da,
                0x2277aceab8f1d548,
            ]))],
            [Felt::new(BigInteger256([
                0x2683dd8f92daa90a,
                0x5c10807a44045bc9,
                0xed6d8ad017fc1b39,
                0x162c027b7ea5dd66,
            ]))],
            [Felt::new(BigInteger256([
                0xa1f2c3b61fbc03c0,
                0x3c380ed2103dc97c,
                0xb1f0cc57eec39882,
                0x2359e3f87491b3ef,
            ]))],
            [Felt::new(BigInteger256([
                0xb7054df4dfc3ebcb,
                0xc63cae095426a9f3,
                0x99cd3b358369a9b5,
                0x0610eba468449472,
            ]))],
            [Felt::new(BigInteger256([
                0x476850d43685258a,
                0x3ce80141784e3401,
                0x399824cfbdafb615,
                0x0915b3eb5c527f1a,
            ]))],
            [Felt::new(BigInteger256([
                0xbedc01b78b9c7b60,
                0x1f7a56408422cac4,
                0xbb0e9ecf25e4b7e0,
                0x0f1453e12065395b,
            ]))],
            [Felt::new(BigInteger256([
                0x5589cef52ad60413,
                0xd3965e031deb009f,
                0x66641895c62a99a8,
                0x3a28bc687d671902,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiVesta_2_1::hash_field(input).to_elements());
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
                0xe4484e4d9cb56ab3,
                0x4c1e1eab5f3a5d35,
                0xdb539d68d7aef2fb,
                0x328395f3ab3baedd,
            ]))],
            [Felt::new(BigInteger256([
                0x073adf8531a94728,
                0x58921a70ecf0f156,
                0x9616137969de4c6b,
                0x33726ef1ad99329b,
            ]))],
            [Felt::new(BigInteger256([
                0xf5838cbab6cb93f6,
                0xfc076c5e318b7274,
                0x54d86ddf65dfa99f,
                0x14a3fbf609635e45,
            ]))],
            [Felt::new(BigInteger256([
                0xeb51c8928746dabb,
                0xc393ab0999339fd6,
                0xc8e891c3008346da,
                0x2277aceab8f1d548,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 62];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiVesta_2_1::hash(&bytes).to_elements());
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
                0xb0bef530b1acffc8,
                0x978f22d8bea06a6c,
                0x0f3d84a00ae6331d,
                0x120dde22f35ce658,
            ]))],
            [Felt::new(BigInteger256([
                0x58199a59c7511623,
                0x9f425e9d6f10caa4,
                0xf6d306deeb6df5cf,
                0x359a8a69413df7ff,
            ]))],
            [Felt::new(BigInteger256([
                0x00623a18ed447b31,
                0x4420a69249bcc35f,
                0x9fe40f4794583bef,
                0x1aa499dd2395dd60,
            ]))],
            [Felt::new(BigInteger256([
                0x54b0433b1e57b347,
                0x24ffd53eace33d0d,
                0x63cb75fe76ec600d,
                0x04f63365ea724a17,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_2_1::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_2_1::compress_k(input, 2));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(
                expected,
                AnemoiVesta_2_1::merge(&[
                    AnemoiDigest::new([input[0]]),
                    AnemoiDigest::new([input[1]])
                ])
                .to_elements()
            );
        }
    }
}
