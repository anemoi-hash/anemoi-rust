//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiPallas_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiPallas_4_3 {
    type Digest = AnemoiDigest;

    fn hash(bytes: &[u8]) -> Self::Digest {
        // Compute the number of field elements required to represent this
        // sequence of bytes.
        let num_elements = if bytes.len() % 31 == 0 {
            bytes.len() / 31
        } else {
            bytes.len() / 31 + 1
        };

        let sigma = if num_elements % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        // Initialize the internal hash state to all zeroes.
        let mut state = [Felt::zero(); STATE_WIDTH];

        // Absorption phase

        // Break the string into 31-byte chunks, then convert each chunk into a field element,
        // and absorb the element into the rate portion of the state. The conversion is
        // guaranteed to succeed as we spare one last byte to ensure this can represent a valid
        // element encoding.
        let mut i = 0;
        let mut num_hashed = 0;
        let mut buf = [0u8; 32];
        for chunk in bytes.chunks(31) {
            if num_hashed + i < num_elements - 1 {
                buf[..31].copy_from_slice(chunk);
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
            state[i] += Felt::read(&buf[..]).unwrap();
            i += 1;
            if i % RATE_WIDTH == 0 {
                AnemoiPallas_4_3::permutation(&mut state);
                i = 0;
                num_hashed += RATE_WIDTH;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            AnemoiPallas_4_3::permutation(&mut state);
        }

        // Squeezing phase

        // Finally, return the first DIGEST_SIZE elements of the state.
        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn hash_field(elems: &[Felt]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        let sigma = if elems.len() % RATE_WIDTH == 0 {
            Felt::one()
        } else {
            Felt::zero()
        };

        let mut i = 0;
        for &element in elems.iter() {
            state[i] += element;
            i += 1;
            if i % RATE_WIDTH == 0 {
                AnemoiPallas_4_3::permutation(&mut state);
                i = 0;
            }
        }

        // We then add sigma to the last register of the capacity.
        state[STATE_WIDTH - 1] += sigma;

        // If the message length is not a multiple of RATE_WIDTH, we append 1 to the rate cell
        // next to the one where we previously appended the last message element. This is
        // guaranted to be in the rate registers (i.e. to not require an extra permutation before
        // adding this constant) if sigma is equal to zero. We then apply a final Anemoi permutation
        // to the whole state.
        if sigma.is_zero() {
            state[i] += Felt::one();
            AnemoiPallas_4_3::permutation(&mut state);
        }

        // Squeezing phase

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }

    fn merge(digests: &[Self::Digest; 2]) -> Self::Digest {
        // initialize state to all zeros
        let mut state = [Felt::zero(); STATE_WIDTH];

        // 2*DIGEST_SIZE < RATE_SIZE so we can safely store
        // the digests into the rate registers at once
        state[0..DIGEST_SIZE].copy_from_slice(digests[0].as_elements());
        state[DIGEST_SIZE..2 * DIGEST_SIZE].copy_from_slice(digests[0].as_elements());

        // Apply internal Anemoi permutation
        AnemoiPallas_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiPallas_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiPallas_4_3::permutation(&mut state);

        let mut result = [Felt::zero(); NUM_COLUMNS];
        for (i, r) in result.iter_mut().enumerate() {
            *r = elems[i] + elems[i + NUM_COLUMNS] + state[i] + state[i + NUM_COLUMNS];
        }

        result.to_vec()
    }

    fn compress_k(elems: &[Felt], k: usize) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);
        assert!(STATE_WIDTH % k == 0);
        assert!(k % 2 == 0);

        let mut state = elems.try_into().unwrap();
        AnemoiPallas_4_3::permutation(&mut state);

        let mut result = vec![Felt::zero(); STATE_WIDTH / k];
        let c = result.len();
        for (i, r) in result.iter_mut().enumerate() {
            for j in 0..k {
                *r += elems[i + c * j] + state[i + c * j];
            }
        }

        result
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
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
            vec![Felt::new(BigInteger256([
                0xc6693736b48d2e54,
                0x82c3d3eab61a6910,
                0xd7992e5e0829e231,
                0x3e8b454a663b44ed,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x7c15c32e4b64cf25,
                    0x886315f0beb8c1f2,
                    0x232c3750d3e94319,
                    0x3c2629e17b555618,
                ])),
                Felt::new(BigInteger256([
                    0xc540befb2e124af6,
                    0x336d1ad436fd375f,
                    0x79addc4b70deb310,
                    0x033f551083a8d935,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xd51efe2564e1b4e9,
                    0xb4e514f883ee0b15,
                    0xebb42660447650bd,
                    0x3d9ad66da20be1ae,
                ])),
                Felt::new(BigInteger256([
                    0xace6b6e7905fb97e,
                    0xd3fdf3067f05fe3c,
                    0xd3b1f972dcd7409b,
                    0x0580e914f6275ab3,
                ])),
                Felt::new(BigInteger256([
                    0xb19f0fb4d3ed6fcf,
                    0x05a5b7b21cdec0cf,
                    0xa83432e145e43f90,
                    0x3f55320eca2b20de,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb1281da0a909b226,
                    0xa18ca4d72a9d315b,
                    0x87efa2ab21c13fe7,
                    0x0a733c1a5b2b21ad,
                ])),
                Felt::new(BigInteger256([
                    0x1482a2108cc30769,
                    0x53bd72ac006dddd8,
                    0xe628f6f109e013f8,
                    0x2eb4d2ad2ec4eea5,
                ])),
                Felt::new(BigInteger256([
                    0x53d706b2084089cd,
                    0xb900c51fa00891ed,
                    0x02b0e73ef75c6bd2,
                    0x0945479c0ba9db0a,
                ])),
                Felt::new(BigInteger256([
                    0xf7e0a8f7cf423006,
                    0xc26623fffd692244,
                    0xff2544f3047c9988,
                    0x337f4d7268eea5aa,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xb27f2796bf3e71da,
                    0x3f181e592724b50e,
                    0x6f570649a9030f3e,
                    0x2adf1ba01b979f48,
                ])),
                Felt::new(BigInteger256([
                    0x4f7192ed11771ce6,
                    0x1c730ead7376c47d,
                    0x1021b0b06a25c0f9,
                    0x0cd5b43e469f47d4,
                ])),
                Felt::new(BigInteger256([
                    0x83afa24882e7b711,
                    0xe043215ff583559b,
                    0xf42bc33487c333fa,
                    0x00cc6e02e690067e,
                ])),
                Felt::new(BigInteger256([
                    0xa393c1ebb8cd6c46,
                    0xa80f3b6e295c522d,
                    0x22d3e32b13837632,
                    0x268414f25ae36edc,
                ])),
                Felt::new(BigInteger256([
                    0xe53982c58a85ab78,
                    0x642ffec2d4310f19,
                    0xb95ea8158602303c,
                    0x3ee0de47af92c550,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x4556d4db5d3dd827,
                    0xfaa3b012bd47f4f0,
                    0xec922338e87e838b,
                    0x13495d73806cda16,
                ])),
                Felt::new(BigInteger256([
                    0x6e2c6e96aaee948a,
                    0x0b7d65f9df2bb0c0,
                    0x8c137c700cb6bc3d,
                    0x0c19fb5e00f87a91,
                ])),
                Felt::new(BigInteger256([
                    0xc8710475362f6236,
                    0x0ffa6a840cc3b4dd,
                    0x5c56a8063f53d171,
                    0x2ccb4d9f02c6f575,
                ])),
                Felt::new(BigInteger256([
                    0x038463fd7ae5a871,
                    0xbd9a5b3122202577,
                    0xfdc456599ff4adc3,
                    0x29b31330ad273993,
                ])),
                Felt::new(BigInteger256([
                    0xd1b9ccd040e77329,
                    0xcecfd0001059affd,
                    0x5c0e75977bf615dc,
                    0x00201a42be25cdd8,
                ])),
                Felt::new(BigInteger256([
                    0xfdcde15410900de9,
                    0x8f99b09999d1566d,
                    0x628626dd6cfe22f4,
                    0x090754d2c44cf5ee,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xe979abe1477e7097,
                0x082c73331981065a,
                0xa53ae8e5bf41f50e,
                0x1c3a43df513eb626,
            ]))],
            [Felt::new(BigInteger256([
                0x0a92d9b13d71bca8,
                0x898030e0ea51d444,
                0xe4d7d3fa7cb99deb,
                0x09ca2791d0e5f8ce,
            ]))],
            [Felt::new(BigInteger256([
                0x281001237f51710f,
                0x5905904a282dc728,
                0x435919cb90cc1b7d,
                0x2e7f6d6d21a5422a,
            ]))],
            [Felt::new(BigInteger256([
                0xc3734db26abf8234,
                0x56f4a3e33990d6d7,
                0x1e0816da766d6553,
                0x00136de296a6e88c,
            ]))],
            [Felt::new(BigInteger256([
                0x99ccd25fa159275c,
                0x1ecf5fd38a550925,
                0x24da422fb5aaaee5,
                0x36af0064a4321aa8,
            ]))],
            [Felt::new(BigInteger256([
                0x03ab5f3522341bf3,
                0xbfe8f094da889637,
                0xf0f79ddf7738e7fc,
                0x2256e7e71928edbd,
            ]))],
            [Felt::new(BigInteger256([
                0xb8f64d1444c406e4,
                0x48833a7cb1cbc7b1,
                0x2cf6e88c66859265,
                0x198c764f7d8cb3c8,
            ]))],
            [Felt::new(BigInteger256([
                0x93373c4873f2d8fe,
                0xae3f89ecc08a9cb1,
                0xe383c3264c35a324,
                0x3ebbd87bf5a319ef,
            ]))],
            [Felt::new(BigInteger256([
                0xf67988fa8b425ce9,
                0xc611509c6fa6f2cb,
                0xac671d3e990a07c2,
                0x3a8396f738ba34c1,
            ]))],
            [Felt::new(BigInteger256([
                0x892e69707c3915d6,
                0x99af0b9e53162b85,
                0xb7860ae0c9451c1f,
                0x2f3d6e13d7470f7d,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiPallas_4_3::hash_field(input).to_elements());
        }
    }

    #[test]
    fn test_anemoi_hash_bytes() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(); 4],
            vec![Felt::one(); 4],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0xe979abe1477e7097,
                0x082c73331981065a,
                0xa53ae8e5bf41f50e,
                0x1c3a43df513eb626,
            ]))],
            [Felt::new(BigInteger256([
                0x0a92d9b13d71bca8,
                0x898030e0ea51d444,
                0xe4d7d3fa7cb99deb,
                0x09ca2791d0e5f8ce,
            ]))],
            [Felt::new(BigInteger256([
                0x281001237f51710f,
                0x5905904a282dc728,
                0x435919cb90cc1b7d,
                0x2e7f6d6d21a5422a,
            ]))],
            [Felt::new(BigInteger256([
                0xc3734db26abf8234,
                0x56f4a3e33990d6d7,
                0x1e0816da766d6553,
                0x00136de296a6e88c,
            ]))],
        ];

        // The inputs can all be represented with at least 1 byte less than the field size,
        // hence computing the Anemoi hash digest from the byte sequence yields the same
        // result as treating the inputs as field elements.
        for (input, expected) in input_data.iter().zip(output_data) {
            let mut bytes = [0u8; 124];
            bytes[0..31].copy_from_slice(&to_bytes!(input[0]).unwrap()[0..31]);
            bytes[31..62].copy_from_slice(&to_bytes!(input[1]).unwrap()[0..31]);
            bytes[62..93].copy_from_slice(&to_bytes!(input[2]).unwrap()[0..31]);
            bytes[93..124].copy_from_slice(&to_bytes!(input[3]).unwrap()[0..31]);

            assert_eq!(expected, AnemoiPallas_4_3::hash(&bytes).to_elements());
        }
    }

    #[test]
    fn test_anemoi_jive() {
        // Generated from https://github.com/anemoi-hash/anemoi-hash/
        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [
                Felt::new(BigInteger256([
                    0xa36a133d7ea0a395,
                    0x17fad0ccc8456451,
                    0x0394145b48ce1f9f,
                    0x0b9bb94407e29a8a,
                ])),
                Felt::new(BigInteger256([
                    0xd8b3a5c8439f877c,
                    0xc69b8aff768c9cc4,
                    0x2ce22998ba69d3ea,
                    0x16dd3900eec4d4c5,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xf14f2da692b4cc6a,
                    0xd5f84af5b8ace9ab,
                    0x6c0a9912048572ce,
                    0x1d6ba8f4d245adbe,
                ])),
                Felt::new(BigInteger256([
                    0x0c09e6b3fa22fec2,
                    0xc3f7a38d77d28a8e,
                    0x13c8195e6642ac41,
                    0x32a07284c40a7813,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xe707b1d1273454a2,
                    0x36920d5ff26f9ebd,
                    0x838c59713e57a9cf,
                    0x1923d1304b631174,
                ])),
                Felt::new(BigInteger256([
                    0x945a7a5b97fc816d,
                    0xd92f15f8f939abc8,
                    0x1ba5a6478bdc5d09,
                    0x074c88ef9d7cc596,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0x476bc3b43674575b,
                    0x6ac6216084f8c901,
                    0x44950cd4f2423c19,
                    0x2d6aca4859dddd50,
                ])),
                Felt::new(BigInteger256([
                    0x40065d71e21b03e6,
                    0x410844271c7cbc7d,
                    0xe752a69782132530,
                    0x3b0d56de34e68c71,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x7c1db905c2402b11,
                0xde965bcc3ed20116,
                0x30763df40337f389,
                0x2278f244f6a76f4f,
            ]))],
            [Felt::new(BigInteger256([
                0x642be36d8cd7cb2b,
                0x77a9558727327b1e,
                0x7fd2b2706ac81f10,
                0x100c1b79965025d1,
            ]))],
            [Felt::new(BigInteger256([
                0x7b622c2cbf30d60f,
                0x0fc12358eba94a86,
                0x9f31ffb8ca3406d9,
                0x20705a1fe8dfd70a,
            ]))],
            [Felt::new(BigInteger256([
                0xee44f039188f5b40,
                0x8987cc8b98288c62,
                0x2be7b36c74556149,
                0x287821268ec469c2,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiPallas_4_3::compress_k(input, 4));
        }
    }
}
