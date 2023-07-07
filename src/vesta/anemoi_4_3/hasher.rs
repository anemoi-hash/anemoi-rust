//! Sponge trait implementation for Anemoi

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::digest::AnemoiDigest;
use super::{AnemoiVesta_4_3, Jive, Sponge};
use super::{DIGEST_SIZE, NUM_COLUMNS, RATE_WIDTH, STATE_WIDTH};
use crate::traits::Anemoi;

use super::Felt;
use super::{One, Zero};

use ark_ff::FromBytes;

impl Sponge<Felt> for AnemoiVesta_4_3 {
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
                AnemoiVesta_4_3::permutation(&mut state);
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
            AnemoiVesta_4_3::permutation(&mut state);
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
                AnemoiVesta_4_3::permutation(&mut state);
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
            AnemoiVesta_4_3::permutation(&mut state);
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
        AnemoiVesta_4_3::permutation(&mut state);

        Self::Digest::new(state[..DIGEST_SIZE].try_into().unwrap())
    }
}

impl Jive<Felt> for AnemoiVesta_4_3 {
    fn compress(elems: &[Felt]) -> Vec<Felt> {
        assert!(elems.len() == STATE_WIDTH);

        let mut state = elems.try_into().unwrap();
        AnemoiVesta_4_3::permutation(&mut state);

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
        AnemoiVesta_4_3::permutation(&mut state);

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
                0x1c0b410d25e72664,
                0xc87293069d5dc7bf,
                0x6334f6db15bad8bd,
                0x1ae225d76d673796,
            ]))],
            vec![
                Felt::new(BigInteger256([
                    0x03b240805380ac57,
                    0x008272b92a7446e8,
                    0x59d476b1cc4b5439,
                    0x2da5f3913b21e965,
                ])),
                Felt::new(BigInteger256([
                    0x56e359298456b132,
                    0x2f57dca3c5d30c31,
                    0x7b0d584693ec065e,
                    0x2da76125a16f6eae,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xcf96359459bf32a0,
                    0xe2617ad699cca772,
                    0x5e2b8302bc38dec2,
                    0x0c38ea204cb7184e,
                ])),
                Felt::new(BigInteger256([
                    0xff92dff8fcaa41a5,
                    0xe31585941cc2052a,
                    0x694276956465ac01,
                    0x08eb7a1b22756273,
                ])),
                Felt::new(BigInteger256([
                    0xaab553fd5893d126,
                    0x66b7857b686c2f5e,
                    0xaeeb5beac615f095,
                    0x159a66c794e390b8,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0xf9630d14c8585360,
                    0x1f20b99b7c688015,
                    0x59eda9bb6d06d133,
                    0x202f5b25f773cb2a,
                ])),
                Felt::new(BigInteger256([
                    0x43da2f0b91ed395c,
                    0x704e697e65b6ea0e,
                    0xeb2b32d0762aa23f,
                    0x3197ce359ec57ba8,
                ])),
                Felt::new(BigInteger256([
                    0x0f1d6ba7a922e301,
                    0x0eb81488ae97f618,
                    0x5cc98924d3c48414,
                    0x2af66209e7ea05d8,
                ])),
                Felt::new(BigInteger256([
                    0x697acf9c8e7541b7,
                    0xb18922fe7c65243e,
                    0x70c8f0fff074b5ad,
                    0x3c31db2a94f5f75e,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x6e8bc55100da8155,
                    0xf949279aac688a8f,
                    0x5f308fc3e91630fc,
                    0x0b3318a55544d52d,
                ])),
                Felt::new(BigInteger256([
                    0x96e16577f43f1555,
                    0xa1dac655e5d7cc2a,
                    0xc6d71da578c72dca,
                    0x1f3a7cf5beac95c0,
                ])),
                Felt::new(BigInteger256([
                    0x2a625cceb35cb915,
                    0xfc1f4544e92322bf,
                    0xf45c5182a4c66ce2,
                    0x3f3552cc44270d3b,
                ])),
                Felt::new(BigInteger256([
                    0xa11671653ac93f78,
                    0x72effcb8f9676bac,
                    0xed1f7171189631d2,
                    0x388f22922908f07e,
                ])),
                Felt::new(BigInteger256([
                    0x50b6d691f784235f,
                    0x25c1cb16e9f9457a,
                    0xf1739a937b37a843,
                    0x0aa9a6dcc96c4b1b,
                ])),
            ],
            vec![
                Felt::new(BigInteger256([
                    0x28dc92a7fe950867,
                    0x43e8bfed9fb100db,
                    0x99fb42c9225dd8c0,
                    0x0c81a3633bb72f21,
                ])),
                Felt::new(BigInteger256([
                    0x0fd8d5189adc69dc,
                    0x9850d4d296d5d440,
                    0x70bbaec593ba23b8,
                    0x0e8bef0d0e49433c,
                ])),
                Felt::new(BigInteger256([
                    0x113ba1c31fe6ff78,
                    0xf78252beb9e2b596,
                    0x1ba12b28828f38e3,
                    0x07bba1ff3d465540,
                ])),
                Felt::new(BigInteger256([
                    0x2321255dc36e1269,
                    0x3903551614bf30d3,
                    0x49af816fb22f9203,
                    0x0f8390e5075cdb21,
                ])),
                Felt::new(BigInteger256([
                    0xea48986f7788f5ae,
                    0x52c288388832426b,
                    0x6bf23779b7e29694,
                    0x312f818d28a60a37,
                ])),
                Felt::new(BigInteger256([
                    0xcfbdcd9a56e38ad2,
                    0x872069727e2564bc,
                    0xf1100329d76581ba,
                    0x04f5950f9d50a011,
                ])),
            ],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x4fb8d48032be2c98,
                0x720609b24a0cc38b,
                0xd1412e1a82a764df,
                0x1e374221105209fc,
            ]))],
            [Felt::new(BigInteger256([
                0xdf6d850bb7003e14,
                0x4ee9df639011ec3a,
                0x6cb94cc454354445,
                0x06c83f8590308b17,
            ]))],
            [Felt::new(BigInteger256([
                0xf04ab3efd810fc7b,
                0xf8da437797acd205,
                0xfbb6e0d8fa20b4c5,
                0x3424018ecefee6ae,
            ]))],
            [Felt::new(BigInteger256([
                0x4e8d9826b78723b7,
                0x773dc8a6a6735162,
                0x3c7f2ae9d397dd59,
                0x055615b70b2c2adc,
            ]))],
            [Felt::new(BigInteger256([
                0xab9c0e86174a7b50,
                0x6cb446bd73374db4,
                0x6a0d44ab31dc4ea1,
                0x02846982e67f8fb1,
            ]))],
            [Felt::new(BigInteger256([
                0x88ff4c9fbe818a4d,
                0x043c84d2c8c47c22,
                0x501a0dd859796e51,
                0x30f889f0b1480116,
            ]))],
            [Felt::new(BigInteger256([
                0x7987e1253342f480,
                0xd8bec3031d1c7973,
                0xb9061ec558a668d1,
                0x0d4076b145839981,
            ]))],
            [Felt::new(BigInteger256([
                0xc90d9f1971432cb9,
                0x384f1f28a62e80ea,
                0x52aebc7a3c0070a8,
                0x3d2de8b5e828799c,
            ]))],
            [Felt::new(BigInteger256([
                0x531287e4aa787884,
                0x2ed5ae7fc98bef9e,
                0xf818950c70a044a2,
                0x1bbe2ebc3a725d34,
            ]))],
            [Felt::new(BigInteger256([
                0x7a2f26fba575a11b,
                0x592c73c3c82787ac,
                0x11bb31b028531431,
                0x0b1fb5301ed4f984,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected, AnemoiVesta_4_3::hash_field(input).to_elements());
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
                0x4fb8d48032be2c98,
                0x720609b24a0cc38b,
                0xd1412e1a82a764df,
                0x1e374221105209fc,
            ]))],
            [Felt::new(BigInteger256([
                0xdf6d850bb7003e14,
                0x4ee9df639011ec3a,
                0x6cb94cc454354445,
                0x06c83f8590308b17,
            ]))],
            [Felt::new(BigInteger256([
                0xf04ab3efd810fc7b,
                0xf8da437797acd205,
                0xfbb6e0d8fa20b4c5,
                0x3424018ecefee6ae,
            ]))],
            [Felt::new(BigInteger256([
                0x4e8d9826b78723b7,
                0x773dc8a6a6735162,
                0x3c7f2ae9d397dd59,
                0x055615b70b2c2adc,
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

            assert_eq!(expected, AnemoiVesta_4_3::hash(&bytes).to_elements());
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
                    0x4bbd8ff6b764f1e0,
                    0x07768ef94fb7273b,
                    0x15809bab6e4ef821,
                    0x10c3ea8d8cb77810,
                ])),
                Felt::new(BigInteger256([
                    0xd0f40923ddd86202,
                    0xf07c7a94784cdbba,
                    0x8eb63acbc82a73f6,
                    0x384be9a394e2fc29,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xa500dfb25ac95a80,
                    0xdad3f54f7953b9c4,
                    0xe069f1e4ef3e98a0,
                    0x3466760acfe18aeb,
                ])),
                Felt::new(BigInteger256([
                    0x2904b244ffcd6f44,
                    0xa4842149c5c64dc2,
                    0xfc4d72c04553c211,
                    0x195a7d19cdf4dc39,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xd675302645e12b2b,
                    0x77966152684c1e3d,
                    0x91121f9ea4ee5001,
                    0x130cdb1d2906847f,
                ])),
                Felt::new(BigInteger256([
                    0xa911a5b8e45d77bc,
                    0xa976d4b3ae2826ef,
                    0x6254d63c1e17ef23,
                    0x32779554aaa40dd6,
                ])),
            ],
            [
                Felt::new(BigInteger256([
                    0xcf112915c250fb8f,
                    0xacdc96cb52b6d457,
                    0x6adf0a4bb82995d1,
                    0x3362e38d41468af1,
                ])),
                Felt::new(BigInteger256([
                    0x3dc5187c56466239,
                    0xa868344ca9f40170,
                    0xcd36f85b917b3795,
                    0x38204fc64c01993e,
                ])),
            ],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_4_3::compress(input));
        }

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_4_3::compress_k(input, 2));
        }

        let input_data = [
            vec![Felt::zero(), Felt::zero(), Felt::zero(), Felt::zero()],
            vec![Felt::one(), Felt::one(), Felt::one(), Felt::one()],
            vec![Felt::zero(), Felt::zero(), Felt::one(), Felt::one()],
            vec![Felt::one(), Felt::one(), Felt::zero(), Felt::zero()],
        ];

        let output_data = [
            [Felt::new(BigInteger256([
                0x906aadf9953d53e1,
                0xd5ac7091be6f5a18,
                0xa436d67736796c17,
                0x090fd431219a7439,
            ]))],
            [Felt::new(BigInteger256([
                0x41bea6d65a96c9c3,
                0x5d117d9d35855ea9,
                0xdcb764a534925ab2,
                0x0dc0f3249dd66725,
            ]))],
            [Felt::new(BigInteger256([
                0xf33feabe2a3ea2e6,
                0xfec69d0a0cdf9c4f,
                0xf366f5dac3063f24,
                0x05847071d3aa9255,
            ]))],
            [Felt::new(BigInteger256([
                0x808f567118975dc7,
                0x32fe321bf3162cea,
                0x381602a749a4cd67,
                0x2b8333538d482430,
            ]))],
        ];

        for (input, expected) in input_data.iter().zip(output_data) {
            assert_eq!(expected.to_vec(), AnemoiVesta_4_3::compress_k(input, 4));
        }
    }
}
